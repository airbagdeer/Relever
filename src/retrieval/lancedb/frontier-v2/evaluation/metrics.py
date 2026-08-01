"""Pure information-retrieval metric functions.

Every function takes a binary relevance vector ``rel`` in *rank order* (``rel[i]``
is 1 if the document at rank ``i+1`` is relevant, else 0) and, where the metric
needs it, the total number of relevant documents ``R`` for the query.

Conventions (standard TREC-style):
  * ``R`` is the size of the (pooled) relevant set and may exceed ``len(rel)``;
    relevant documents that were never retrieved simply contribute zero. This is
    what caps Recall@k and R-precision when ``R`` is large.
  * Precision@k divides by the cutoff ``k`` even when fewer than ``k`` documents
    were retrieved (the missing slots count as non-relevant).
  * nDCG is **binary** (gain = relevance label); the ideal ranking places
    ``min(R, k)`` relevant documents first.
  * Queries with ``R == 0`` have several metrics undefined; these functions
    return ``0.0`` for them, and the evaluator excludes ``R == 0`` queries from
    all aggregates (so the 0.0 is never averaged in).

This module is intentionally dependency-light (numpy only) and free of any
project imports so it is trivially unit-testable in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _prefix_relevant(rel: Sequence[int]) -> np.ndarray:
    """Cumulative count of relevant docs: ``out[i]`` = #relevant in ``rel[:i+1]``."""
    arr = np.asarray(rel, dtype=np.int64)
    if arr.size == 0:
        return arr
    return np.cumsum(arr)


def _relevant_in_top_k(prefix: np.ndarray, k: int) -> int:
    """#relevant among the first ``k`` ranks, given a prefix-sum array."""
    if k <= 0 or prefix.size == 0:
        return 0
    idx = min(k, prefix.size) - 1
    return int(prefix[idx])


def precision_at_k(rel: Sequence[int], k: int) -> float:
    """(#relevant in top-k) / k. Divides by the cutoff k regardless of list length."""
    if k <= 0:
        return 0.0
    return _relevant_in_top_k(_prefix_relevant(rel), k) / k


def recall_at_k(rel: Sequence[int], R: int, k: int) -> float:
    """(#relevant in top-k) / R. Returns 0.0 when R == 0."""
    if R <= 0:
        return 0.0
    return _relevant_in_top_k(_prefix_relevant(rel), k) / R


def r_precision(rel: Sequence[int], R: int) -> float:
    """Precision at rank R (== Recall at rank R, the break-even point)."""
    if R <= 0:
        return 0.0
    return _relevant_in_top_k(_prefix_relevant(rel), R) / R


def reciprocal_rank(rel: Sequence[int]) -> float:
    """1 / rank of the first relevant document; 0.0 if none retrieved."""
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def success_at_k(rel: Sequence[int], k: int) -> float:
    """1.0 if at least one relevant document appears in the top-k, else 0.0."""
    return 1.0 if _relevant_in_top_k(_prefix_relevant(rel), k) > 0 else 0.0


def average_precision(rel: Sequence[int], R: int) -> float:
    """Average precision: mean of Precision@i over the ranks i where rel[i]==1,
    normalised by R (relevant docs never retrieved contribute 0). Returns 0.0
    when R == 0."""
    if R <= 0:
        return 0.0
    arr = np.asarray(rel, dtype=np.int64)
    if arr.size == 0:
        return 0.0
    ranks = np.arange(1, arr.size + 1)
    prefix = np.cumsum(arr)
    precision_at_hits = (prefix / ranks)[arr == 1]
    return float(precision_at_hits.sum() / R)


def dcg_at_k(rel: Sequence[int], k: int) -> float:
    """Binary DCG@k = sum_{i<=k} rel_i / log2(i+1) (rank i is 1-indexed)."""
    if k <= 0:
        return 0.0
    arr = np.asarray(rel, dtype=np.float64)[:k]
    if arr.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, arr.size + 2))
    return float((arr * discounts).sum())


def ndcg_at_k(rel: Sequence[int], R: int, k: int) -> float:
    """Binary nDCG@k. The ideal ranking puts min(R, k) relevant docs first, so
    IDCG@k = sum_{i<=min(R,k)} 1/log2(i+1). Returns 0.0 when R == 0."""
    if k <= 0 or R <= 0:
        return 0.0
    ideal_hits = min(R, k)
    if ideal_hits <= 0:
        return 0.0
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(rel, k) / idcg


def compute_query_metrics(
    rel: Sequence[int],
    R: int,
    k_values: Sequence[int],
    success_k_values: Sequence[int],
) -> dict[str, float]:
    """Compute the full metric suite for one query's ranked relevance vector.

    Returns a flat dict with keys like ``ndcg@10``, ``precision@50``,
    ``recall@100``, ``success@5`` plus scalar ``mrr``, ``map`` (AP) and
    ``r_precision``. ``rel`` is the model's ranked list relevance vector; ``R``
    is the pooled relevant count for the query.
    """
    out: dict[str, float] = {}
    for k in k_values:
        out[f"ndcg@{k}"] = ndcg_at_k(rel, R, k)
        out[f"precision@{k}"] = precision_at_k(rel, k)
        out[f"recall@{k}"] = recall_at_k(rel, R, k)
    for k in success_k_values:
        out[f"success@{k}"] = success_at_k(rel, k)
    out["mrr"] = reciprocal_rank(rel)
    out["map"] = average_precision(rel, R)
    out["r_precision"] = r_precision(rel, R)
    return out
