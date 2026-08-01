"""Tests for the per-model cap in aggregate_candidates.

Each model's <alias>_<qid>.json stores the full retrieval depth
(retrieve_core.DEFAULT_K = 3000), but the deduped all_candidates_<qid>.json
union must include only each model's top `per_model_k` (by rank, default 500).

    uv run pytest src/retrieval/lancedb/frontier-v2/test_aggregate_candidates.py
"""

from __future__ import annotations

import json

import pytest

import _bootstrap  # noqa: F401  (sets sys.path so config / sibling modules import)
import retrieve_core
import aggregate_candidates as agg


def _write_model_file(qdir, alias: str, eid: int, n: int) -> None:
    """Write a per-model file with candidates ranked 1..n (1 = best)."""
    cands = [
        {"id": f"{alias}-{r}", "rank": r, "score": 1.0 - r / 1e6}
        for r in range(1, n + 1)
    ]
    p = qdir / f"{alias}_{retrieve_core.qid_str(eid)}.json"
    p.write_text(json.dumps({
        "event_id": eid, "query": "q", "model": alias,
        "k": n, "n_candidates": n, "candidates": cands,
    }), encoding="utf-8")


def _setup_query(tmp_path, monkeypatch, eid: int, depth: int, aliases):
    monkeypatch.setattr(retrieve_core, "FRONTIER_CANDIDATES_DIR", tmp_path)
    qdir = tmp_path / retrieve_core.qid_str(eid)
    qdir.mkdir(parents=True)
    id_to_text: dict[str, str] = {}
    for alias in aliases:
        _write_model_file(qdir, alias, eid, depth)
        id_to_text.update({f"{alias}-{r}": f"text-{alias}-{r}" for r in range(1, depth + 1)})
    return qdir, id_to_text


def _read_union(qdir, eid: int):
    out = qdir / f"all_candidates_{retrieve_core.qid_str(eid)}.json"
    return json.loads(out.read_text(encoding="utf-8"))


def test_caps_each_model_to_top_k(tmp_path, monkeypatch):
    """With per_model_k=500 over depth-600 files, only ranks 1..500 survive per model."""
    eid = 7
    qdir, id_to_text = _setup_query(tmp_path, monkeypatch, eid, depth=600, aliases=["modelA", "modelB"])

    written, missing = agg._aggregate_one_query(
        eid, "q", id_to_text, overwrite=True, per_model_k=500)

    assert written and missing == 0
    ids = {c["id"] for c in _read_union(qdir, eid)["candidates"]}
    # top-500 kept, 501..600 dropped, for BOTH models
    assert {"modelA-1", "modelA-500", "modelB-1", "modelB-500"} <= ids
    assert not ({"modelA-501", "modelA-600", "modelB-501", "modelB-600"} & ids)
    assert len(ids) == 1000  # 500 each, disjoint id namespaces


def test_no_cap_unions_everything(tmp_path, monkeypatch):
    """per_model_k=None reproduces the old behavior: union the full per-model depth."""
    eid = 8
    qdir, id_to_text = _setup_query(tmp_path, monkeypatch, eid, depth=600, aliases=["modelA"])

    agg._aggregate_one_query(eid, "q", id_to_text, overwrite=True, per_model_k=None)

    cands = _read_union(qdir, eid)["candidates"]
    assert len(cands) == 600


def test_dedup_keeps_provenance_within_cap(tmp_path, monkeypatch):
    """A paragraph in both models' top-K is one union row carrying both models' provenance."""
    eid = 9
    monkeypatch.setattr(retrieve_core, "FRONTIER_CANDIDATES_DIR", tmp_path)
    qdir = tmp_path / retrieve_core.qid_str(eid)
    qdir.mkdir(parents=True)
    # modelA: shared-1 at rank 3; modelB: shared-1 at rank 700 (beyond the cap)
    (qdir / f"modelA_{retrieve_core.qid_str(eid)}.json").write_text(json.dumps({
        "model": "modelA", "candidates": [
            {"id": "shared-1", "rank": 3, "score": 0.9},
            {"id": "a-only", "rank": 1, "score": 1.0},
        ]}), encoding="utf-8")
    (qdir / f"modelB_{retrieve_core.qid_str(eid)}.json").write_text(json.dumps({
        "model": "modelB", "candidates": [
            {"id": "b-only", "rank": 1, "score": 1.0},
            {"id": "shared-1", "rank": 700, "score": 0.1},  # dropped by cap=500
        ]}), encoding="utf-8")
    id_to_text = {"shared-1": "s", "a-only": "a", "b-only": "b"}

    agg._aggregate_one_query(eid, "q", id_to_text, overwrite=True, per_model_k=500)

    by_id = {c["id"]: c for c in _read_union(qdir, eid)["candidates"]}
    # shared-1 is in modelA's top-500 (rank 3) but NOT modelB's (rank 700 > 500),
    # so it stays in the union with ONLY modelA provenance.
    assert set(by_id) == {"shared-1", "a-only", "b-only"}
    assert set(by_id["shared-1"]["retrieved_by"]) == {"modelA"}
    assert by_id["shared-1"]["n_models"] == 1
