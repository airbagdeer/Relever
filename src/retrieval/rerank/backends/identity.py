"""Debug backend: assigns every document the same score.

This is not a model. It is the regression harness for the *plumbing*.

Because ``load_reranked_list`` stable-sorts by ``-score``, a constant score leaves the
retriever's original rank order untouched. So a full run of ``identity-constant``
through the real pipeline — pool scoring, parquet write, cache read, join, tiebreak,
metric computation — must reproduce ``summary/evaluation/LEADERBOARD.md`` to the
digit. Any deviation means the harness reorders documents somewhere it shouldn't.

It costs seconds (no model is loaded), so it is cheap to run before every real sweep,
and it is the only test that covers the tiebreak path against a known-good answer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .base import Reranker, TruncationCounter
from reranker_registry import RerankerSpec

logger = logging.getLogger(__name__)

CONSTANT_SCORE = 0.0


class IdentityReranker(Reranker):
    def __init__(self, spec: RerankerSpec, device: str | None = None, **_ignored):
        self.alias = spec.alias
        self.spec = spec
        # No model runs, so the device is nominal — but the driver records it in the
        # cache manifest, and a constant score is device-independent by construction.
        self.device = device or "cpu"
        self.tf32 = False
        self.truncation = TruncationCounter()
        logger.info("[%s] identity backend: every document scores %.1f", spec.alias, CONSTANT_SCORE)

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        if instruction is not None:
            raise ValueError(f"[{self.alias}] has no instruction slot.")
        self.truncation.observe(len(docs), 0)
        return [CONSTANT_SCORE] * len(docs)
