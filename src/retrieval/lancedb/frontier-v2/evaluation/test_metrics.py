"""Unit tests for the pure IR metric functions (metrics.py).

Expected values are hand-computed (see the worked arithmetic in comments) and,
for nDCG, additionally cross-checked against scikit-learn's ``ndcg_score`` which
uses the same gain/discount convention.

Run: uv run pytest src/retrieval/lancedb/frontier-v2/evaluation/test_metrics.py -v
"""

from __future__ import annotations

import math

import pytest

import metrics as m

# Canonical example used across several tests:
#   rel = [0, 1, 0, 1, 1], R = 3  (3 relevant total, all retrieved)
REL = [0, 1, 0, 1, 1]
R = 3


# --------------------------------------------------------------------------- #
# Precision@k
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "k, expected",
    [
        (1, 0.0),        # 0 relevant in top-1
        (2, 1 / 2),      # 1 relevant in top-2
        (3, 1 / 3),      # 1 relevant in top-3
        (5, 3 / 5),      # 3 relevant in top-5
        (10, 3 / 10),    # only 5 retrieved; missing slots non-relevant -> 3/10
    ],
)
def test_precision_at_k(k, expected):
    assert m.precision_at_k(REL, k) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Recall@k
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "k, expected",
    [
        (1, 0.0),
        (2, 1 / 3),
        (5, 3 / 3),      # all relevant retrieved by rank 5
        (10, 3 / 3),
    ],
)
def test_recall_at_k(k, expected):
    assert m.recall_at_k(REL, R, k) == pytest.approx(expected)


@pytest.mark.unit
def test_recall_capped_when_R_exceeds_retrieved():
    # Only 1 of 4 relevant docs retrieved -> recall caps at 0.25 no matter the depth.
    rel = [1, 0]
    assert m.recall_at_k(rel, R=4, k=2) == pytest.approx(0.25)
    assert m.recall_at_k(rel, R=4, k=500) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# R-precision  (== precision at rank R)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_r_precision():
    # precision@3 = 1/3
    assert m.r_precision(REL, R) == pytest.approx(1 / 3)


@pytest.mark.unit
def test_r_precision_R_exceeds_list():
    # rel=[1,0], R=4 -> sum(rel[:4]) / 4 = 1/4
    assert m.r_precision([1, 0], R=4) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Reciprocal rank / MRR
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_reciprocal_rank():
    assert m.reciprocal_rank(REL) == pytest.approx(0.5)        # first hit at rank 2
    assert m.reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)  # first hit at rank 1
    assert m.reciprocal_rank([0, 0, 0]) == 0.0                 # no hit


# --------------------------------------------------------------------------- #
# Success@k / hit-rate
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("k, expected", [(1, 0.0), (2, 1.0), (5, 1.0)])
def test_success_at_k(k, expected):
    assert m.success_at_k(REL, k) == expected


# --------------------------------------------------------------------------- #
# Average precision / MAP
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_average_precision():
    # relevant at ranks 2,4,5 -> P@2=1/2, P@4=2/4, P@5=3/5
    # AP = (0.5 + 0.5 + 0.6) / 3 = 1.6/3
    assert m.average_precision(REL, R) == pytest.approx(1.6 / 3)


@pytest.mark.unit
def test_average_precision_unretrieved_relevant_count_as_zero():
    # rel=[1,0], R=4: only the rank-1 hit contributes -> AP = (1/1)/4 = 0.25
    assert m.average_precision([1, 0], R=4) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# nDCG@k (binary)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_dcg_at_k():
    # DCG@5 = 1/log2(3) + 1/log2(5) + 1/log2(6)
    expected = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6)
    assert m.dcg_at_k(REL, 5) == pytest.approx(expected)


@pytest.mark.unit
def test_ndcg_at_5():
    dcg = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 ideal hits
    assert m.ndcg_at_k(REL, R, 5) == pytest.approx(dcg / idcg)


@pytest.mark.unit
def test_ndcg_at_1_first_doc_irrelevant():
    assert m.ndcg_at_k(REL, R, 1) == pytest.approx(0.0)


@pytest.mark.unit
def test_ndcg_idcg_uses_min_R_k():
    # rel=[1,0], R=4, k=4: DCG=1; IDCG@4 = sum_{i=1..4} 1/log2(i+1)
    dcg = 1.0
    idcg = sum(1 / math.log2(i + 1) for i in range(1, 5))
    assert m.ndcg_at_k([1, 0], R=4, k=4) == pytest.approx(dcg / idcg)


@pytest.mark.unit
def test_ndcg_matches_sklearn():
    # Cross-check against sklearn on a fully-retrieved ranking (same convention).
    from sklearn.metrics import ndcg_score

    rel = [0, 1, 0, 1, 1, 0, 1]
    Rloc = sum(rel)
    for k in (1, 3, 5, 7):
        true_rel = [rel]            # graded/binary gains in *current* order
        scores = [list(range(len(rel), 0, -1))]  # strictly decreasing -> preserves order
        expected = ndcg_score(true_rel, scores, k=k)
        assert m.ndcg_at_k(rel, Rloc, k) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_R_zero_returns_zero_everywhere():
    rel = [0, 0, 0]
    assert m.recall_at_k(rel, 0, 5) == 0.0
    assert m.r_precision(rel, 0) == 0.0
    assert m.average_precision(rel, 0) == 0.0
    assert m.ndcg_at_k(rel, 0, 5) == 0.0


@pytest.mark.unit
def test_all_zero_relevance_with_positive_R():
    rel = [0, 0, 0, 0]
    assert m.precision_at_k(rel, 4) == 0.0
    assert m.recall_at_k(rel, 5, 4) == 0.0
    assert m.reciprocal_rank(rel) == 0.0
    assert m.success_at_k(rel, 4) == 0.0
    assert m.average_precision(rel, 5) == 0.0
    assert m.ndcg_at_k(rel, 5, 4) == 0.0


@pytest.mark.unit
def test_empty_ranking():
    assert m.precision_at_k([], 10) == 0.0
    assert m.recall_at_k([], 5, 10) == 0.0
    assert m.reciprocal_rank([]) == 0.0
    assert m.ndcg_at_k([], 5, 10) == 0.0


@pytest.mark.unit
def test_compute_query_metrics_keys_and_values():
    out = m.compute_query_metrics(REL, R, k_values=(1, 5), success_k_values=(1, 5))
    assert out["precision@5"] == pytest.approx(0.6)
    assert out["recall@5"] == pytest.approx(1.0)
    assert out["ndcg@1"] == pytest.approx(0.0)
    assert out["success@1"] == 0.0
    assert out["success@5"] == 1.0
    assert out["mrr"] == pytest.approx(0.5)
    assert out["map"] == pytest.approx(1.6 / 3)
    assert out["r_precision"] == pytest.approx(1 / 3)
