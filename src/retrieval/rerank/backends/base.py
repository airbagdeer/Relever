"""The reranker backend contract, plus the batching machinery every backend shares.

A backend turns ``(query, [doc, ...])`` into one score per document. Higher is
more relevant; the absolute scale is model-specific and meaningless across models
(see ``reranker_registry`` — P(yes) in [0,1], a raw logit, and a log-odds
difference all appear in this study).

Two invariants every pointwise backend must uphold, because violating either
fails *silently* rather than raising:

  1. **Padding side.** Decoder rerankers read ``logits[:, -1, :]``. With right
     padding, the last position is a pad token and every score is garbage.
     :func:`assert_padding_side` makes the tokenizer's actual state an assertion
     rather than an assumption.
  2. **fp32 logit read.** bf16 has ~3 significant decimal digits; mxbai's own card
     prints ``[12., 0.5, 11.125, 0.8125]``. Reading logits in bf16 manufactures
     ties, which then get broken arbitrarily by the sort. Always ``.float()``
     before extracting a score.

Batching is by **token budget** rather than fixed sequence count: documents are
length-bucketed so a batch of short paragraphs holds more rows than a batch of
long ones. This keeps padding waste low and memory roughly constant, which is what
lets the same code run an 8B model and a 33M model without per-model tuning.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

# Fallback when a backend cannot estimate token counts cheaply.
_CHARS_PER_TOKEN = 3.7


@dataclass
class TruncationCounter:
    """Tracks how often the tokenizer had to cut a document.

    Surfaced in the run log and asserted against ``RerankerSpec.expected_truncation``
    in the tests. A model that silently truncates is a model whose numbers mean
    something different from what the table claims, so this is never left implicit.
    """

    n_docs: int = 0
    n_truncated: int = 0

    def observe(self, n_docs: int, n_truncated: int) -> None:
        self.n_docs += n_docs
        self.n_truncated += n_truncated

    @property
    def rate(self) -> float:
        return self.n_truncated / self.n_docs if self.n_docs else 0.0

    def log(self, alias: str) -> None:
        logger.info(
            "[%s] truncation: %d / %d documents (%.3f%%)",
            alias, self.n_truncated, self.n_docs, 100.0 * self.rate,
        )


class Reranker(Protocol):
    """Scores every document against one query. Pointwise: the score is set-independent.

    The returned list is in the same order as ``docs`` — backends that reorder
    internally (length bucketing) must scatter their results back before returning.
    ``truncation`` is part of the contract: the driver logs it and warns when a model
    truncated more than its registry entry declares.
    """

    alias: str
    truncation: TruncationCounter

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        ...


@dataclass(frozen=True)
class ListwiseItem:
    """One (query, candidate-list) reranking job for a listwise model."""

    event_id: int
    retriever: str
    query: str
    doc_ids: tuple[str, ...]   # the retriever's top-K, in retrieval rank order
    docs: tuple[str, ...]      # aligned with doc_ids


@dataclass(frozen=True)
class ListwiseResult:
    event_id: int
    retriever: str
    order: tuple[str, ...]          # permutation of the input doc_ids, best first
    scores: tuple[float, ...] | None  # None for models that emit only an ordering


class ListwiseReranker(Protocol):
    """Reorders a candidate list as a set. The score of a document depends on which
    other documents it is shown with, so there is no reusable per-document score and
    the pooled cache does not apply.

    The interface is deliberately *many* items at once rather than one: rank_zephyr
    parallelises across queries (``batch_size=32``), not across its 9 sequential
    sliding windows, so a per-query call would leave the GPU almost idle.
    """

    alias: str
    truncation: TruncationCounter

    def rerank_many(self, items: Sequence[ListwiseItem]) -> list[ListwiseResult]:
        ...


@dataclass
class ScoreHealth:
    """Tracks how well a reranker's scores actually *order* the top of the pool.

    Ranking metrics are computed by a STABLE sort, so documents with identical scores
    keep their first-stage order. A reranker whose top-K scores are all tied is therefore
    indistinguishable from the retriever — and it fails silently: the numbers look
    plausible, every test passes, and the reranker column is quietly a copy of the
    baseline.

    This is not hypothetical. mxbai-rerank-large-v2 shipped 421 of 2758 documents at
    exactly 1.0 because sentence-transformers applied its default Sigmoid to the model's
    log-odds, saturating everything confident. Caught only by inspecting raw parquet.

    ``distinct@K`` = distinct scores among the top-K, divided by K. 1.0 means the reranker
    fully determines that prefix; low values mean the stable sort is doing the ordering.
    """

    n_queries: int = 0
    sum_at_10: float = 0.0
    sum_at_100: float = 0.0
    sum_at_500: float = 0.0
    worst_at_10: float = 1.0

    def observe(self, scores: Sequence[float]) -> None:
        if not scores:
            return
        top = sorted(scores, reverse=True)
        self.n_queries += 1
        for k, attr in ((10, "sum_at_10"), (100, "sum_at_100"), (500, "sum_at_500")):
            prefix = top[:k]
            ratio = len(set(prefix)) / len(prefix)
            setattr(self, attr, getattr(self, attr) + ratio)
        self.worst_at_10 = min(self.worst_at_10, len(set(top[:10])) / len(top[:10]))

    def _mean(self, attr: str) -> float:
        return getattr(self, attr) / self.n_queries if self.n_queries else 1.0

    def log(self, alias: str) -> None:
        if not self.n_queries:
            return
        logger.info(
            "[%s] score health over %d queries: distinct@10=%.3f distinct@100=%.3f "
            "distinct@500=%.3f (worst distinct@10=%.3f)",
            alias, self.n_queries, self._mean("sum_at_10"), self._mean("sum_at_100"),
            self._mean("sum_at_500"), self.worst_at_10,
        )

    def check(self, alias: str, min_distinct_at_10: float = 0.9) -> str | None:
        """Return a problem description, or None. Callers should fail loudly."""
        if not self.n_queries:
            return None
        d10 = self._mean("sum_at_10")
        if d10 < min_distinct_at_10:
            return (
                f"[{alias}] only {d10:.3f} of the top-10 scores are distinct (want "
                f">= {min_distinct_at_10}). The stable sort is returning first-stage order "
                f"for the tied block, so this reranker's column would silently duplicate "
                f"the retriever. Likely a saturating activation (sigmoid/softmax) or a "
                f"dtype too coarse for the score's dynamic range."
            )
        return None


def assert_padding_side(tokenizer, expected: str, alias: str) -> None:
    """Fail loudly if the tokenizer is not padding on the side the score read needs.

    This is the single highest-risk detail in the harness: a decoder reranker with
    right padding returns plausible-looking numbers that are entirely noise.
    """
    actual = getattr(tokenizer, "padding_side", None)
    if actual != expected:
        raise ValueError(
            f"[{alias}] tokenizer.padding_side is {actual!r}, expected {expected!r}. "
            f"Decoder rerankers read logits[:, -1, :]; the wrong padding side reads a "
            f"pad position and silently corrupts every score."
        )


def estimate_tokens(text: str) -> int:
    """Cheap length proxy used only to *order* docs into buckets, never to truncate."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def length_bucketed_batches(
    docs: Sequence[str],
    token_budget: int,
    max_batch_size: int,
    per_doc_overhead: int = 0,
) -> Iterator[tuple[list[int], list[str]]]:
    """Yield ``(original_indices, docs)`` batches ordered by ascending length.

    Sorting by length before batching means each batch pads to roughly its own
    longest member instead of to the global maximum. Indices are returned so the
    caller can scatter scores back into the input order — the caller must never
    assume batch order equals input order.
    """
    if not docs:
        return
    order = sorted(range(len(docs)), key=lambda i: estimate_tokens(docs[i]))
    batch_idx: list[int] = []
    batch_docs: list[str] = []
    batch_max_len = 0

    for i in order:
        doc_len = estimate_tokens(docs[i]) + per_doc_overhead
        new_max = max(batch_max_len, doc_len)
        # Cost of the batch if we add this doc: padded to the new longest member.
        if batch_idx and (new_max * (len(batch_idx) + 1) > token_budget
                          or len(batch_idx) >= max_batch_size):
            yield batch_idx, batch_docs
            batch_idx, batch_docs = [], []
            new_max = doc_len
        batch_idx.append(i)
        batch_docs.append(docs[i])
        batch_max_len = new_max

    if batch_idx:
        yield batch_idx, batch_docs


def score_with_oom_backoff(
    score_batch_fn,
    docs: Sequence[str],
    initial_chunk: int,
    alias: str,
    min_chunk: int = 1,
) -> list[float]:
    """Score ``docs`` in chunks of ``initial_chunk``, halving the chunk on CUDA OOM.

    The whole bucket is re-scored at the smaller chunk size, so a retry never hands
    the GPU a chunk larger than the one that just failed. Anything that is not an
    OOM propagates. An OOM at ``min_chunk`` is a configuration error (a document too
    long for the model) and must surface rather than be papered over.
    """
    import torch  # noqa: PLC0415 - keep base importable in venvs without torch

    def run(chunk: int) -> list[float]:
        out: list[float] = []
        for i in range(0, len(docs), chunk):
            out.extend(score_batch_fn(docs[i:i + chunk]))
        return out

    chunk = max(min_chunk, initial_chunk)
    while True:
        try:
            return run(chunk)
        except torch.cuda.OutOfMemoryError:
            if chunk <= min_chunk:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            chunk = max(min_chunk, chunk // 2)
            logger.warning("[%s] CUDA OOM; re-scoring bucket at chunk=%d", alias, chunk)


def enable_tf32_if_fp32(dtype_name: str, device: str | None) -> bool:
    """Turn on TF32 matmul for fp32 inference on CUDA. Returns whether it was enabled.

    Several rerankers must run in fp32 because their score is read straight off the model
    and fp16/bf16 quantise it onto a grid coarse enough to tie hundreds of documents (see
    ``ScoreHealth``). Strict fp32 matmul on an A100 is ~8x slower than fp16 tensor cores --
    measured: bge-m3 went from 36 q/min to 4.7 q/min.

    TF32 rounds the matmul *inputs* to a 10-bit mantissa but accumulates and stores in
    fp32. The output tensor therefore keeps full fp32 granularity, which is exactly the
    property the score needs; only the last couple of bits of each product are lost. It
    recovers most of the tensor-core throughput.

    Recorded in the run manifest, because scores computed with and without it are not
    bit-identical and must not be interleaved in one cache.
    """
    import torch  # noqa: PLC0415

    if dtype_name != "float32" or not str(device).startswith("cuda"):
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 matmul enabled for float32 inference (fp32 accumulate + fp32 output)")
    return True


def resolve_dtype(name: str, device: str | None = None):
    """Map a registry dtype name to a torch dtype, downgrading half precision on CPU.

    fp16 matmul on CPU is slow and numerically fragile; the cluster runs bf16 on A100 but
    the smoke tests and any local check run on CPU. fp32 is strictly more precise, so the
    downgrade can only make a CPU score *more* faithful, never less.
    """
    import torch  # noqa: PLC0415

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
    if device is not None and str(device).startswith("cpu") and dtype is not torch.float32:
        logger.info("device=cpu: using float32 instead of %s", name)
        return torch.float32
    return dtype


def resolve_device() -> str:
    """CUDA -> MPS -> CPU, matching the fallback chain used across this repo."""
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
