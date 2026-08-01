"""The reported listwise depth must come from the data, not from a constant.

`apply_listwise_permutation` derives the prefix length from the permutation itself, so the
*metrics* stay correct even if a shard is rerun at a different `--depth`. What breaks is
the label: the row would still say `@100`. A mislabelled column in a paper is a wrong
number, so `evaluate_rerank` reads `doc["depth"]` off disk and cross-checks every file.
"""

from __future__ import annotations

import pytest
import score_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(score_store, "FRONTIER_RERANK_SCORES_PATH", tmp_path)
    return score_store


@pytest.mark.unit
def test_write_listwise_records_the_depth_actually_used(store):
    order = [f"d{i}" for i in range(50)]
    store.write_listwise("rr", "retr", 7, depth=50, order=order, scores=None)

    doc = store.read_listwise("rr", "retr", 7)
    assert doc["depth"] == 50
    assert len(doc["order"]) == 50
    assert doc["scores"] is None       # rank_zephyr emits an ordering, not scores


@pytest.mark.unit
def test_listwise_scores_are_persisted_when_the_model_emits_them(store):
    store.write_listwise("rr", "retr", 7, depth=3, order=["a", "b", "c"], scores=[0.9, 0.5, 0.1])
    doc = store.read_listwise("rr", "retr", 7)
    assert doc["scores"] == [0.9, 0.5, 0.1]


@pytest.mark.unit
def test_a_depth_that_disagrees_with_the_permutation_is_detectable(store):
    """`_listwise_depth` in evaluate_rerank rejects this. Reproduce the inconsistency the
    check exists to catch: `depth` says 100 but only 3 ids were stored."""
    store.write_listwise("rr", "retr", 7, depth=100, order=["a", "b", "c"], scores=None)
    doc = store.read_listwise("rr", "retr", 7)
    assert doc["depth"] != len(doc["order"])


@pytest.mark.unit
def test_missing_permutation_raises_with_the_command_to_run(store):
    with pytest.raises(FileNotFoundError, match="rerank_listwise.py"):
        store.read_listwise("rr", "retr", 999)


@pytest.mark.unit
def test_listwise_exists_is_per_retriever(store):
    store.write_listwise("rr", "retrA", 1, depth=2, order=["a", "b"])
    assert store.listwise_exists("rr", "retrA", 1)
    assert not store.listwise_exists("rr", "retrB", 1)
