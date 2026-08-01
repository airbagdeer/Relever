"""rank_zephyr_7b_v1_full: generative listwise reranking via sliding windows (rank_llm).

RankZephyr consumes a window of numbered passages and *generates* a permutation string
(``[2] > [1] > [4] > ...``). It has no scoring head, so it cannot be run pointwise and
produces no reusable per-document score — only an ordering.

We drive rank_llm's ``RankListwiseOSLLM`` directly with ``vllm_batched=True`` rather than
the convenience wrapper ``ZephyrReranker``. The wrapper hard-defaults ``vllm_batched=False``,
which silently loads the model through plain HuggingFace ``transformers`` and calls
``generate()`` one query at a time — ~14 GB resident, a single query in flight, ~27 s/query.
``vllm_batched=True`` instead builds vLLM's ``LLM`` engine (``gpu_memory_utilization=0.93``,
CUDA graphs) and processes every request's window-N step as one batched generation. On this
workload (18 retrievers x 1100 queries, 9 windows each) that is a >10x speedup. The RankZephyr
protocol is otherwise byte-identical: generation-based (``use_logits=False``), greedy
(``run_llm_batched`` hard-codes ``SamplingParams(temperature=0.0)``), ``variable_passages=True``,
``context_size=4096``, ``window_size=20``, ``step=10``.

    from rank_llm.rerank.listwise import RankListwiseOSLLM
    from rank_llm.rerank import PromptMode
    reranker = RankListwiseOSLLM(model=..., vllm_batched=True, window_size=20, ...)
    results = reranker.rerank_batch(requests, rank_start=0, rank_end=100, window_size=20, step=10)
    # -> list[Result]; Result.candidates is the reordered list

Three properties that shape this backend and belong in the paper, not in a footnote we
forgot to write:

* **9 sequential windows at depth 100** (``(100-20)/10 + 1``). Within one query the windows
  are strictly sequential — window *i+1*'s input is window *i*'s output. Parallelism is
  *across queries*: vLLM batches many queries' window-*i* generations together, which is
  exactly why ``rerank_many`` hands the engine a large batch at once.
* **``context_size=4096`` truncates hard.** 20 passages x ~475 tokens is ~9.5k tokens, so
  rank_llm shrinks each passage to roughly 190 tokens. RankZephyr sees about 40% of each
  v2 paragraph. This is a property of the model, declared as ``expected_truncation=1.0``
  in the registry and reported as a finding.
* **``num_passes=1``.** RankZephyr-rho (3 sequential passes) buys +0.004 / +0.010 nDCG@10
  on DL19/DL20 for 3x the compute. Not worth it here.

Requires ``.venv-rankllm`` (rank-llm 0.20.3 + vLLM 0.6.6.post1, the stable v0 engine, on
Python 3.11). ``rank_llm`` raises if vLLM is absent — there is no HF-``generate`` fallback
worth using here (it is the slow path this backend exists to avoid).

License: MIT.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence

from .base import ListwiseItem, ListwiseReranker, ListwiseResult, TruncationCounter
from reranker_registry import RerankerSpec

logger = logging.getLogger(__name__)

WINDOW_SIZE = 20
STRIDE = 10
CONTEXT_SIZE = 4096
# ZephyrReranker's default system message; kept verbatim so the prompt matches the card.
SYSTEM_MESSAGE = (
    "You are RankLLM, an intelligent assistant that can rank passages based on "
    "their relevancy to the query"
)


class RankZephyrReranker(ListwiseReranker):
    def __init__(
        self,
        spec: RerankerSpec,
        device: str | None = None,
        num_gpus: int = 1,
        **_ignored,
    ):
        try:
            from rank_llm.rerank import PromptMode  # noqa: PLC0415
            from rank_llm.rerank.listwise import RankListwiseOSLLM  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - venv-dependent
            raise ImportError(
                "rank_llm is not installed. Run this backend with .venv-rankllm "
                "(python 3.11, rank-llm 0.20.3 + vLLM 0.6.6.post1). See the rerank "
                "README's isolated-venvs section for the exact stack."
            ) from exc

        self.alias = spec.alias
        self.spec = spec
        self.truncation = TruncationCounter()

        # vllm_batched=True is the whole point: it builds vLLM's v0 LLM engine instead of the
        # HF sequential fallback. use_logits=False keeps the generation-based RankZephyr
        # protocol (the model writes a permutation string, not first-token logits).
        self.reranker = RankListwiseOSLLM(
            model=spec.hf_id,
            context_size=CONTEXT_SIZE,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            device=device or "cuda",
            num_gpus=num_gpus,
            variable_passages=True,
            window_size=WINDOW_SIZE,
            system_message=SYSTEM_MESSAGE,
            use_logits=False,
            vllm_batched=True,
        )

        # Greedy decoding, per the RankZephyr paper. rank_llm's vLLM batched path
        # (run_llm_batched) hard-codes SamplingParams(temperature=0.0); confirm it from source
        # so a version that changed it fails loudly instead of silently sampling -- the bug
        # that made the first run 19,800 identity no-ops.
        self._assert_greedy()

        logger.info(
            "[%s] rank_zephyr ready (vLLM batched): window=%d step=%d context=%d num_gpus=%d "
            "gpu_mem_util=0.93 use_logits=False | greedy (temperature=0, source-confirmed)",
            spec.alias, WINDOW_SIZE, STRIDE, CONTEXT_SIZE, num_gpus,
        )

    def _assert_greedy(self) -> None:
        """Confirm the engine decodes greedily by inspecting rank_llm's source.

        The vLLM batched path builds ``SamplingParams(..., temperature=0.0)`` inline in
        ``run_llm_batched``; there is no runtime attribute to read, so we check the source
        text. A rank_llm that dropped the hard-coded 0.0 would fall through to vLLM's
        temperature=1.0 default and sample non-reproducibly -- refuse rather than risk it."""
        try:
            src = inspect.getsource(type(self.reranker))
        except (OSError, TypeError):
            src = ""
        if "temperature=0.0" not in src and "temperature=0," not in src:
            raise RuntimeError(
                f"[{self.alias}] could not confirm greedy decoding: RankListwiseOSLLM's source "
                f"no longer visibly hard-codes temperature=0 in its vLLM sampling params. "
                f"Refusing to run rather than risk stochastic, non-reproducible rankings."
            )

    def rerank_many(self, items: Sequence[ListwiseItem]) -> list[ListwiseResult]:
        from rank_llm.data import Candidate, Query, Request  # noqa: PLC0415

        if not items:
            return []
        depth = len(items[0].docs)
        if any(len(it.docs) != depth for it in items):
            # vLLM's sliding_windows_batched requires every request in the batch to carry the
            # same number of candidates, so a ragged batch would raise deep inside rank_llm.
            raise ValueError(f"[{self.alias}] all items in a batch must share one depth.")

        requests = [
            Request(
                query=Query(text=it.query, qid=f"{it.retriever}:{it.event_id}"),
                candidates=[
                    # score is the retriever's; rank_llm uses it only to order the input,
                    # which is already in rank order. docid carries our paragraph id back.
                    Candidate(docid=doc_id, score=float(depth - i), doc={"text": doc})
                    for i, (doc_id, doc) in enumerate(zip(it.doc_ids, it.docs))
                ],
            )
            for it in items
        ]

        results = self.reranker.rerank_batch(
            requests, rank_start=0, rank_end=depth, window_size=WINDOW_SIZE, step=STRIDE
        )
        if len(results) != len(items):
            raise ValueError(
                f"[{self.alias}] rank_llm returned {len(results)} results for {len(items)} requests."
            )

        out: list[ListwiseResult] = []
        for item, result in zip(items, results):
            order = tuple(str(c.docid) for c in result.candidates)
            if sorted(order) != sorted(item.doc_ids):
                # rank_llm repairs malformed generations, but a silent id drop would
                # corrupt the metrics. ~0.2% of windows generate malformed output.
                raise ValueError(
                    f"[{self.alias}] q{item.event_id}/{item.retriever}: reranked ids are not a "
                    f"permutation of the input ({len(order)} returned, {depth} expected)."
                )
            # Every passage is word-capped and then shrunk to fit the 4096 context.
            self.truncation.observe(depth, depth)
            out.append(ListwiseResult(item.event_id, item.retriever, order, None))
        return out
