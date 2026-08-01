"""jina-reranker-v3: listwise "last but not late" interaction, one forward pass.

Mechanism (from the repo's ``modeling.py``): the prompt packs the query instruction, then
``<passage id="i">{doc}<|embed_token|></passage>`` for every document, then
``<query>{query}<|rerank_token|></query>`` last. A single forward pass yields a contextual
embedding at each ``<|embed_token|>`` and a query embedding at ``<|rerank_token|>``; both
go through a projector and the score is their cosine similarity. ``lm_head`` is
``nn.Identity`` — nothing is generated.

    model = AutoModel.from_pretrained('jinaai/jina-reranker-v3', dtype="auto", trust_remote_code=True)
    results = model.rerank(query, documents, max_doc_length=1024)
    # -> [{'document': str, 'relevance_score': float, 'index': int, 'embedding': None}, ...]

Facts that shape this backend, all verified against ``modeling.py``:

* **``.rerank()`` is the only public API.** ``compute_score()`` does not exist, and the
  repo ships no sentence-transformers config.
* **One query at a time.** There is no batched-query entry point.
* **``block_size = 125``.** With 100 candidates (the listwise depth) everything fits in a
  single forward pass, so no cross-block score gluing occurs — documents all attend to
  one another, which is the entire point of the model.
* **Attention is causal**, so document *i* attends to *1..i-1* but not to *i+1..n*. Scores
  therefore depend on document *order*. We feed the retriever's rank order, which is the
  realistic deployment; the paper reports random order scores ~0.6-1.0 nDCG@10 higher,
  a caveat worth stating rather than silently exploiting.
* ``relevance_score`` is a raw cosine in [-1, 1], **not** the hosted API's rescaled score.
  Local and API scores are not comparable.

License: CC BY-NC 4.0 (non-commercial) — used for offline research evaluation only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .base import ListwiseItem, ListwiseReranker, ListwiseResult, TruncationCounter, resolve_device
from reranker_registry import RerankerSpec

logger = logging.getLogger(__name__)

# Per-document token cap. The model's own default is 2048; 1024 matches this study's
# document budget and still leaves >99.9% of the v2 paragraphs untruncated.
MAX_DOC_LENGTH = 1024
MAX_QUERY_LENGTH = 512
# Beyond this, modeling.py chunks into blocks whose scores share only an averaged query
# vector and whose documents never attend across the boundary. At depth 100 we never
# reach it; if a caller ever does, that is a protocol change and must be loud.
BLOCK_SIZE = 125


class JinaV3Reranker(ListwiseReranker):
    def __init__(self, spec: RerankerSpec, device: str | None = None, **_ignored):
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self.alias = spec.alias
        self.spec = spec
        self.device = device or resolve_device()
        self.truncation = TruncationCounter()

        self.model = AutoModel.from_pretrained(
            spec.hf_id, dtype="auto", trust_remote_code=True
        )
        self.model.to(self.device).eval()
        # `rerank()` truncates internally at max_doc_length and reports nothing. We
        # measure independently so this model is held to the same "no silent caps"
        # standard as every other backend, rather than being taken at its word.
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
        logger.info("[%s] jina-v3 listwise ready on %s (max_doc_length=%d)",
                    spec.alias, self.device, MAX_DOC_LENGTH)

    def _count_truncated(self, docs: Sequence[str]) -> int:
        lens = self.tokenizer(list(docs), add_special_tokens=False, verbose=False)["input_ids"]
        return sum(1 for ids in lens if len(ids) > MAX_DOC_LENGTH)

    def rerank_many(self, items: Sequence[ListwiseItem]) -> list[ListwiseResult]:
        """One forward pass per item. ``rerank()`` accepts a single query only."""
        results: list[ListwiseResult] = []
        for item in items:
            n = len(item.docs)
            if n > BLOCK_SIZE:
                raise ValueError(
                    f"[{self.alias}] {n} documents exceeds block_size={BLOCK_SIZE}. "
                    f"modeling.py would split them into blocks that never attend to each "
                    f"other and whose scores share only an averaged query vector — the "
                    f"listwise premise would silently no longer hold."
                )
            ranked = self.model.rerank(
                item.query,
                list(item.docs),
                max_doc_length=MAX_DOC_LENGTH,
                max_query_length=MAX_QUERY_LENGTH,
            )
            # rerank() returns best-first with `index` pointing back into `documents`.
            order = tuple(item.doc_ids[int(r["index"])] for r in ranked)
            scores = tuple(float(r["relevance_score"]) for r in ranked)
            if len(order) != n:
                raise ValueError(
                    f"[{self.alias}] q{item.event_id}/{item.retriever}: rerank() returned "
                    f"{len(order)} of {n} documents."
                )
            self.truncation.observe(n, self._count_truncated(item.docs))
            results.append(
                ListwiseResult(item.event_id, item.retriever, order, scores)
            )
        return results
