"""Raw-``transformers`` sequence-classification backend: bge-reranker-v2-m3.

Implements the card's ``AutoModelForSequenceClassification`` recipe verbatim:

    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

Two deliberate deviations, both documented in ``reranker_registry``:

1. ``max_length=1088`` rather than the card's 512. The card's value is an example,
   not a contract — the bge-m3 backbone has ``max_position_embeddings=8194`` and
   long context is its headline feature. 1088 leaves the full 1024-token document
   budget intact for 475-token paragraphs.
2. ``truncation="only_second"`` rather than ``truncation=True`` (``longest_first``).
   With a <=32-token query and a ~475-token document these produce identical output,
   but ``only_second`` *guarantees* the query is never cut rather than relying on it
   happening to be the shorter sequence.

We use raw transformers rather than ``FlagEmbedding`` because ``FlagReranker`` would
drag in a dependency that pins an older ``transformers`` — the same reasoning that
keeps ``pylate`` out of the main venv.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from .base import (
    Reranker,
    TruncationCounter,
    enable_tf32_if_fp32,
    length_bucketed_batches,
    resolve_device,
    resolve_dtype,
    score_with_oom_backoff,
)
from reranker_registry import RerankerSpec

logger = logging.getLogger(__name__)

# Padded tokens per forward pass. Chosen so a bf16/fp16 568M model stays well inside
# an 80GB A100 even at the 1088-token ceiling; the OOM guard halves it if not.
DEFAULT_TOKEN_BUDGET = 262_144
MAX_BATCH_SIZE = 512


class HFSeqClsReranker(Reranker):
    def __init__(
        self,
        spec: RerankerSpec,
        device: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        if spec.max_length is None:
            raise ValueError(f"[{spec.alias}] hf_seq_cls requires an explicit max_length.")

        self.alias = spec.alias
        self.spec = spec
        self.max_length: int = spec.max_length
        self.device = device or resolve_device()
        self.token_budget = token_budget
        self.truncation = TruncationCounter()
        self.tf32 = enable_tf32_if_fp32(spec.dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec.hf_id, dtype=resolve_dtype(spec.dtype, self.device)
        )
        self.model.to(self.device).eval()

        logger.info(
            "[%s] seq-cls ready: max_length=%d, dtype=%s, device=%s, num_labels=%d",
            spec.alias, self.max_length, spec.dtype, self.device, self.model.config.num_labels,
        )

    def _pair_overhead(self, query: str) -> int:
        """Non-document tokens per pair: the query plus XLM-R's 4 specials
        (``<s> q </s></s> d </s>`` -- ``spec.template_overhead``)."""
        query_len = len(self.tokenizer(query, add_special_tokens=False, verbose=False)["input_ids"])
        return query_len + self.spec.template_overhead

    def _count_truncated(self, docs: Sequence[str], query: str) -> int:
        available = self.max_length - self._pair_overhead(query)
        lens = self.tokenizer(list(docs), add_special_tokens=False, verbose=False)["input_ids"]
        return sum(1 for ids in lens if len(ids) > available)

    @torch.no_grad()
    def _score_batch(self, query: str, batch_docs: Sequence[str]) -> list[float]:
        inputs = self.tokenizer(
            [query] * len(batch_docs),
            list(batch_docs),
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        # .float() before leaving the GPU: fp16 logits have ~3 significant digits and
        # would manufacture ties that the downstream sort then breaks arbitrarily.
        logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
        return logits.cpu().tolist()

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        if instruction is not None:
            raise ValueError(f"[{self.alias}] has no instruction slot.")
        if not docs:
            return []

        self.truncation.observe(len(docs), self._count_truncated(docs, query))

        # The query + specials ride along with every document in a batch, so they count
        # against the token budget. Omitting them makes batches larger than intended and
        # provokes avoidable OOM retries.
        overhead = self._pair_overhead(query)
        scores = [0.0] * len(docs)
        for idx, batch_docs in length_bucketed_batches(
            docs, self.token_budget, MAX_BATCH_SIZE, per_doc_overhead=overhead
        ):
            batch_scores = score_with_oom_backoff(
                lambda d: self._score_batch(query, d),
                batch_docs,
                initial_chunk=len(batch_docs),
                alias=self.alias,
            )
            # length_bucketed_batches reorders by length; scatter back to input order.
            for i, s in zip(idx, batch_scores):
                scores[i] = s
        return scores
