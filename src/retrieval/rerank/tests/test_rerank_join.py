"""Unit tests for the rerank join: sort, tiebreak, tail preservation, loud failures.

These are pure — they build ``RankedList``s by hand and never touch a model or the disk.
The properties tested here are the ones whose violation would produce plausible-looking
but wrong numbers rather than a crash.
"""

from __future__ import annotations

import pytest
from data_loading import RankedList, apply_listwise_permutation, rerank_ranked_list
from metrics import r_precision


def make_base(ids: list[str], labels: dict[str, int], alias: str = "retriever-x") -> RankedList:
    """A first-stage list in rank order, with descending retrieval scores."""
    return RankedList(
        alias=alias,
        event_id=42,
        ids=ids,
        rel=[1 if labels.get(i) == 1 else 0 for i in ids],
        scores=[float(len(ids) - n) for n in range(len(ids))],
        n_retrieved=len(ids),
        n_missing_labels=0,
    )


@pytest.mark.unit
def test_constant_scores_reproduce_first_stage_order():
    """The tiebreak contract: a reranker with no opinion must change nothing.

    This is what makes the `identity-constant` regression against the published
    leaderboard meaningful. If `sorted` were not stable, or if the key were not the
    original rank, the whole harness would silently permute ties.
    """
    ids = [f"d{i}" for i in range(20)]
    labels = {i: 0 for i in ids}
    base = make_base(ids, labels)
    cached = dict.fromkeys(ids, 0.0)

    out = rerank_ranked_list(base, cached, rerank_depth=20, labels=labels)
    assert out.ids == base.ids


@pytest.mark.unit
def test_ties_break_by_original_rank():
    """Equal scores must preserve relative retrieval order, not reverse or shuffle it."""
    ids = ["a", "b", "c", "d"]
    labels = dict.fromkeys(ids, 0)
    base = make_base(ids, labels)
    # b and c tie; a is worst, d is best.
    cached = {"a": 0.1, "b": 0.5, "c": 0.5, "d": 0.9}

    out = rerank_ranked_list(base, cached, rerank_depth=4, labels=labels)
    assert out.ids == ["d", "b", "c", "a"]  # b before c: original order


@pytest.mark.unit
def test_only_the_prefix_is_reordered_tail_is_untouched():
    """Standard rerank-top-K protocol. Ranks past K keep the retriever's order."""
    ids = [f"d{i}" for i in range(10)]
    labels = dict.fromkeys(ids, 0)
    base = make_base(ids, labels)
    # Ascending scores over the prefix -> the prefix must come back reversed.
    cached = {f"d{i}": float(i) for i in range(10)}

    out = rerank_ranked_list(base, cached, rerank_depth=4, labels=labels)
    assert out.ids[:4] == ["d3", "d2", "d1", "d0"]
    assert out.ids[4:] == ids[4:]


@pytest.mark.unit
def test_perfect_reranker_yields_r_precision_one():
    """A reranker that floats every positive to the top must score P@R = 1.0."""
    ids = [f"d{i}" for i in range(10)]
    positives = {"d7", "d8", "d9"}
    labels = {i: (1 if i in positives else 0) for i in ids}
    base = make_base(ids, labels)
    assert r_precision(base.rel, R=3) == 0.0  # positives start at the bottom

    cached = {i: (1.0 if i in positives else 0.0) for i in ids}
    out = rerank_ranked_list(base, cached, rerank_depth=10, labels=labels)
    assert r_precision(out.rel, R=3) == 1.0


@pytest.mark.unit
def test_missing_score_raises_rather_than_scoring_zero():
    """A candidate with no cached score means the cache and the dump disagree.

    Treating it as rel=0 (or score=0) would quietly demote a real document. It must be
    an error the operator sees.
    """
    ids = ["a", "b", "c"]
    labels = dict.fromkeys(ids, 0)
    base = make_base(ids, labels)
    cached = {"a": 1.0, "c": 0.5}  # 'b' absent

    with pytest.raises(KeyError, match="no cached score"):
        rerank_ranked_list(base, cached, rerank_depth=3, labels=labels, reranker="rr")


@pytest.mark.unit
def test_rerank_depth_must_be_positive():
    ids = ["a", "b"]
    labels = dict.fromkeys(ids, 0)
    with pytest.raises(ValueError, match="must be positive"):
        rerank_ranked_list(make_base(ids, labels), {"a": 1.0, "b": 0.0}, 0, labels)


@pytest.mark.unit
def test_relevance_vector_follows_the_new_order():
    """rel must be recomputed from the permuted ids, not carried over positionally."""
    ids = ["a", "b", "c"]
    labels = {"a": 0, "b": 1, "c": 0}
    base = make_base(ids, labels)
    assert base.rel == [0, 1, 0]

    cached = {"a": 0.0, "b": 1.0, "c": 0.5}
    out = rerank_ranked_list(base, cached, rerank_depth=3, labels=labels)
    assert out.ids == ["b", "c", "a"]
    assert out.rel == [1, 0, 0]


# --------------------------------------------------------------------------- #
# Listwise permutations
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_listwise_permutation_applies_and_preserves_tail():
    ids = [f"d{i}" for i in range(10)]
    labels = dict.fromkeys(ids, 0)
    base = make_base(ids, labels)
    order = list(reversed(ids[:4]))

    out = apply_listwise_permutation(base, order, labels)
    assert out.ids[:4] == ["d3", "d2", "d1", "d0"]
    assert out.ids[4:] == ids[4:]


@pytest.mark.unit
def test_listwise_permutation_with_dropped_id_raises():
    """rank_zephyr generates a permutation string; ~0.2% of windows are malformed.

    rank_llm repairs most of it, but a dropped id must never reach the metrics.
    """
    ids = [f"d{i}" for i in range(5)]
    labels = dict.fromkeys(ids, 0)
    base = make_base(ids, labels)

    with pytest.raises(ValueError, match="not a permutation"):
        apply_listwise_permutation(base, ["d0", "d1", "d2", "d3", "d3"], labels)


@pytest.mark.unit
def test_listwise_permutation_with_hallucinated_id_raises():
    ids = [f"d{i}" for i in range(5)]
    labels = dict.fromkeys(ids, 0)
    base = make_base(ids, labels)

    with pytest.raises(ValueError, match="not a permutation"):
        apply_listwise_permutation(base, ["d0", "d1", "d2", "d3", "ZZZ"], labels)
