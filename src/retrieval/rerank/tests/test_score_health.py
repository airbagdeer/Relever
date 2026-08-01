"""A reranker whose top-10 scores are tied is a reranker that isn't reranking.

Ranking metrics come from a STABLE sort, so tied documents keep their first-stage order.
A saturating activation (sigmoid, softmax) or a dtype too coarse for the score's dynamic
range therefore produces a results column that silently duplicates the retriever. Every
test passes; every number looks plausible; the paper is wrong.

This happened: `mxbai-rerank-large-v2` shipped 421 of 2758 documents at exactly 1.0
because sentence-transformers applies a default `Sigmoid()` unless the repo pins
otherwise (ms-marco pins `Identity`; mxbai does not). The values below are the real ones
read back from the cluster's parquet.
"""

from __future__ import annotations

import pytest
import reranker_registry as rr
from backends.base import ScoreHealth

# Verbatim from data/frontier_rerank_scores/mxbai-rerank-large-v2/default/0000.parquet,
# scored with the default (sigmoid) activation.
MXBAI_SATURATED_TOP = [1.0] * 20 + [0.99609375] * 20 + [0.9921875] * 20
# Same model, activation_fn=Identity(), measured locally.
MXBAI_LOGODDS = [10.125, 8.125, 2.0, -2.9375, -4.375, 1.5, 11.25, 0.5, 3.75, -1.25]
# ms-marco, which pins Identity in its config.json.
MS_MARCO_LOGITS = [9.218911, -4.0780287, 3.11, -1.4, 0.5, 7.7, -2.2, 6.1, -8.0, 2.9]


@pytest.mark.unit
def test_saturated_scores_are_flagged():
    """The exact failure that reached the cluster."""
    h = ScoreHealth()
    for _ in range(8):
        h.observe(MXBAI_SATURATED_TOP)

    problem = h.check("mxbai-rerank-large-v2")
    assert problem is not None
    assert "distinct" in problem
    assert "saturating activation" in problem


@pytest.mark.unit
def test_healthy_scores_pass():
    for scores in (MXBAI_LOGODDS, MS_MARCO_LOGITS):
        h = ScoreHealth()
        for _ in range(8):
            h.observe(scores)
        assert h.check("m") is None


@pytest.mark.unit
def test_distinct_ratios_are_computed_over_the_top_k():
    """ScoreHealth sorts descending, so the top-K is the K largest scores."""
    h = ScoreHealth()
    # top-10 after sorting = [5,5,5,5,5,5,4,3,2,1] -> 5 distinct of 10
    h.observe([5.0] * 6 + [4.0, 3.0, 2.0, 1.0] + [i / 100.0 for i in range(90)])
    assert h._mean("sum_at_10") == pytest.approx(0.5)
    assert h._mean("sum_at_100") > 0.9   # the long tail is all unique


@pytest.mark.unit
def test_all_tied_is_the_worst_case():
    h = ScoreHealth()
    h.observe([0.0] * 500)
    assert h._mean("sum_at_10") == pytest.approx(0.1)   # one distinct value in 10
    assert h.worst_at_10 == pytest.approx(0.1)
    assert h.check("x") is not None


@pytest.mark.unit
def test_empty_and_short_inputs_do_not_crash():
    h = ScoreHealth()
    h.observe([])
    assert h.n_queries == 0
    assert h.check("x") is None          # nothing observed -> no verdict
    h.observe([1.0, 2.0])                # fewer than 10 docs
    assert h.check("x") is None


@pytest.mark.unit
def test_identity_control_is_exempt_in_the_registry():
    """identity-constant is all-tied on purpose. The gate must not abort it, and no real
    reranker may claim the same exemption."""
    assert rr.spec_for("identity-constant").expects_tied_scores is True
    for alias in rr.RERANKER_ALIASES:
        assert rr.spec_for(alias).expects_tied_scores is False, alias


@pytest.mark.unit
def test_mxbai_is_fp32_because_bf16_cannot_resolve_its_log_odds():
    """bf16 carries ~8 mantissa bits. mxbai's card prints [12., 0.5, 11.125, 0.8125],
    a ~0.06-0.125 grid -- roughly 150 bins across a top-500's log-odds span."""
    for alias in ("mxbai-rerank-large-v2", "ms-marco-MiniLM-L12-v2", "bge-reranker-v2-m3"):
        assert rr.spec_for(alias).dtype == "float32", alias
