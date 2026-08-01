"""Unit tests for the v2 per-relation union dedup in run_batch_retrieval.

The v2 change replaced the old "retrieve 2000/method, truncate to 1000 unique
total" logic with a union that keeps EVERY unique paragraph from both methods.
These tests pin that contract: cross-method overlap is removed (the two stored
lists are disjoint by id), within-method duplicates collapse, ordering is
preserved, and nothing is truncated.
"""

from run_batch_retrieval import deduplicate_results


def _hit(rid, score=1.0):
    return {"id": rid, "score": score, "payload": {"content": f"para-{rid}"}}


def _ids(results):
    return [r["id"] for r in results]


def test_no_overlap_keeps_everything():
    splade = [_hit("a"), _hit("b")]
    bm25 = [_hit("c"), _hit("d")]

    final_splade, final_bm25 = deduplicate_results(splade, bm25)

    assert _ids(final_splade) == ["a", "b"]
    assert _ids(final_bm25) == ["c", "d"]
    # Full union preserved, nothing dropped.
    assert len(final_splade) + len(final_bm25) == 4


def test_cross_method_overlap_removed_and_disjoint():
    splade = [_hit("a"), _hit("b"), _hit("c")]
    bm25 = [_hit("b"), _hit("c"), _hit("d")]

    final_splade, final_bm25 = deduplicate_results(splade, bm25)

    # BM25 keeps all its unique hits; shared b/c are attributed to BM25.
    assert _ids(final_bm25) == ["b", "c", "d"]
    # SPLADE keeps only what BM25 didn't already return.
    assert _ids(final_splade) == ["a"]
    # The two stored lists must be disjoint by id (deduped per relation).
    assert set(_ids(final_splade)).isdisjoint(_ids(final_bm25))
    # Union covers exactly the distinct ids.
    assert set(_ids(final_splade)) | set(_ids(final_bm25)) == {"a", "b", "c", "d"}


def test_within_method_duplicates_collapse_preserving_first():
    splade = [_hit("a"), _hit("a"), _hit("b"), _hit("a")]
    bm25 = []

    final_splade, final_bm25 = deduplicate_results(splade, bm25)

    assert _ids(final_splade) == ["a", "b"]
    assert final_bm25 == []


def test_nothing_truncated_for_large_disjoint_inputs():
    splade = [_hit(f"s{i}") for i in range(1000)]
    bm25 = [_hit(f"b{i}") for i in range(1000)]

    final_splade, final_bm25 = deduplicate_results(splade, bm25)

    # No optimal-K cap: the full 1000 + 1000 union survives.
    assert len(final_splade) == 1000
    assert len(final_bm25) == 1000
    assert len(set(_ids(final_splade)) | set(_ids(final_bm25))) == 2000


def test_none_ids_skipped():
    splade = [_hit("a"), {"id": None, "score": 1.0, "payload": {}}, _hit("b")]
    bm25 = [{"score": 1.0, "payload": {}}, _hit("c")]  # missing id key entirely

    final_splade, final_bm25 = deduplicate_results(splade, bm25)

    assert _ids(final_splade) == ["a", "b"]
    assert _ids(final_bm25) == ["c"]


def test_order_preserved_across_subquery_concatenation():
    # Mimics flattened batches: instantiated sub-query hits first, then generic.
    splade = [_hit("inst1"), _hit("inst2"), _hit("generic1"), _hit("inst1")]
    bm25 = [_hit("bm_a"), _hit("bm_b")]

    final_splade, final_bm25 = deduplicate_results(splade, bm25)

    assert _ids(final_splade) == ["inst1", "inst2", "generic1"]
    assert _ids(final_bm25) == ["bm_a", "bm_b"]
