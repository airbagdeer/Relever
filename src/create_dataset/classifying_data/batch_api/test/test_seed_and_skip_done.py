"""Unit tests for label reuse: `seed-from-old` + `build --skip-done`.

These exercise the reuse path (reuse an earlier run's labels for unchanged paragraphs, matched by
candidate id, and send only genuinely-new paragraphs to the batch API) entirely on tiny local
fixtures — no OpenAI API calls. Module path globals (``OUT_ROOT`` / ``CANDIDATES_DIR`` /
``RELATIONS_PATH`` / ``MANIFEST_PATH``) are monkeypatched onto a tmp dir.

Run: uv run pytest src/create_dataset/classifying_data/batch_api/test/test_seed_and_skip_done.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# Make `src.*` importable when pytest collects this file directly (mirrors the CLI module, which
# inserts the repo root on sys.path so it also works as a plain script).
for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (_p / "pyproject.toml").is_file():
        sys.path.insert(0, str(_p))
        break

from src.create_dataset.classifying_data.batch_api import label_frontier_batch as lfb  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _write_candidates(cand_dir: Path, qid: str, event_id: int, cands: list[tuple[str, str]]) -> None:
    (cand_dir / qid).mkdir(parents=True, exist_ok=True)
    data = {"event_id": event_id, "candidates": [{"id": i, "text": t} for i, t in cands]}
    (cand_dir / qid / f"all_candidates_{qid}.json").write_text(json.dumps(data))


def _write_old_preds(old_root: Path, qid: str, preds: list[tuple[str, int, int]]) -> None:
    (old_root / qid).mkdir(parents=True, exist_ok=True)
    with (old_root / qid / f"predictions_{qid}.jsonl").open("w") as fh:
        for cid, ev, pr in preds:
            fh.write(json.dumps({"id": cid, "event_id": ev, "prediction": pr}) + "\n")


def _read_preds(qid: str) -> list[dict]:
    p = lfb.predictions_path(qid)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _read_input_ids(qid: str) -> list[str]:
    p = lfb.input_path(qid)
    if not p.exists():
        return []
    return [json.loads(line)["custom_id"] for line in p.read_text().splitlines() if line.strip()]


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tmp OUT_ROOT + CANDIDATES_DIR + old predictions root, with module globals patched.

    Layout:
      0000 — existing query: candidates c1..c4 (c4 new); old preds label c1,c2 (0/1),
             c3 is -1 (re-classify), cX is stale (candidate removed).
      0001 — brand-new query: candidates n1,n2; no old predictions.
      0002 — fully reused: candidates c1,c2; old preds cover both (0/1), nothing new.
    """
    out_root = tmp_path / "out"
    cand_dir = tmp_path / "candidates"
    old_root = out_root / "old"
    out_root.mkdir()
    cand_dir.mkdir()

    _write_candidates(cand_dir, "0000", 0, [("c1", "t1"), ("c2", "t2"), ("c3", "t3"), ("c4", "t4")])
    _write_candidates(cand_dir, "0001", 1, [("n1", "u1"), ("n2", "u2")])
    _write_candidates(cand_dir, "0002", 2, [("c1", "t1"), ("c2", "t2")])

    _write_old_preds(old_root, "0000", [("c1", 0, 0), ("c2", 0, 1), ("c3", 0, -1), ("cX", 0, 0)])
    _write_old_preds(old_root, "0002", [("c1", 2, 0), ("c2", 2, 1)])
    # 0001 intentionally has no old predictions.

    relations = [
        {"event_id": ev, "sentence": "s", "abstract_entity_type": "a",
         "constant_entity": "c", "relation": "r"} for ev in (0, 1, 2)
    ]
    rel_path = tmp_path / "queries.json"
    rel_path.write_text(json.dumps(relations))

    monkeypatch.setattr(lfb, "OUT_ROOT", out_root)
    monkeypatch.setattr(lfb, "CANDIDATES_DIR", cand_dir)
    monkeypatch.setattr(lfb, "RELATIONS_PATH", rel_path)
    monkeypatch.setattr(lfb, "MANIFEST_PATH", out_root / "_manifest.json")
    return {"out_root": out_root, "cand_dir": cand_dir, "old_root": old_root}


def _seed(old_root: Path, *, force: bool = False, reuse_failed: bool = False) -> None:
    lfb.cmd_seed_from_old(argparse.Namespace(
        old_root=str(old_root), qids=None, force=force, reuse_failed=reuse_failed))


def _build(skip_done: bool) -> dict:
    lfb.cmd_build(argparse.Namespace(
        qids=None, limit=None, force=False, workers=1,
        model="gpt-5.2", effort="medium", variant="short2b",
        max_tokens=16000, skip_done=skip_done))
    return json.loads((lfb.MANIFEST_PATH).read_text())


# --------------------------------------------------------------------------- #
# seed-from-old
# --------------------------------------------------------------------------- #
def test_seed_reuses_valid_labels_and_drops_stale_and_failed(env):
    _seed(env["old_root"])

    preds = {r["id"]: r["prediction"] for r in _read_preds("0000")}
    assert preds == {"c1": 0, "c2": 1}          # c1/c2 reused
    assert "c3" not in preds                     # -1 dropped -> will be re-classified
    assert "cX" not in preds                     # stale (candidate removed) dropped

    assert _read_preds("0001") == []             # new query: nothing to seed
    assert {r["id"] for r in _read_preds("0002")} == {"c1", "c2"}  # fully reused


def test_seed_reuse_failed_flag_keeps_minus_one(env):
    _seed(env["old_root"], reuse_failed=True)
    preds = {r["id"]: r["prediction"] for r in _read_preds("0000")}
    assert preds == {"c1": 0, "c2": 1, "c3": -1}  # -1 kept when explicitly reusing failures


def test_seed_does_not_clobber_existing_without_force(env):
    _seed(env["old_root"])
    # Simulate a real fetched prediction landing after seeding.
    lfb.predictions_path("0000").write_text(json.dumps({"id": "real", "event_id": 0, "prediction": 1}) + "\n")
    _seed(env["old_root"])                        # second seed, no --force
    assert _read_preds("0000") == [{"id": "real", "event_id": 0, "prediction": 1}]
    _seed(env["old_root"], force=True)            # --force overwrites
    assert {r["id"] for r in _read_preds("0000")} == {"c1", "c2"}


# --------------------------------------------------------------------------- #
# build --skip-done (via _build_qid, the parallel-safe worker)
# --------------------------------------------------------------------------- #
def test_build_qid_skip_done_emits_only_new(env):
    _seed(env["old_root"])
    lfb._build_init("SYS", {0: {"sentence": "s"}}, {
        "model": "m", "endpoint": lfb.ENDPOINT, "max_completion_tokens": 16000, "effort": "medium",
    }, True)
    res = lfb._build_qid("0000")
    assert res["n_total"] == 4                    # full unique-candidate count preserved
    assert res["n_requests"] == 2                 # only c3 (re-classify) + c4 (new)
    assert res["reused"] == 2
    assert set(_read_input_ids("0000")) == {"c3", "c4"}


def test_build_qid_without_skip_done_emits_all(env):
    _seed(env["old_root"])                        # predictions exist, but skip_done off -> ignored
    lfb._build_init("SYS", {0: {"sentence": "s"}}, {
        "model": "m", "endpoint": lfb.ENDPOINT, "max_completion_tokens": 16000, "effort": "medium",
    }, False)
    res = lfb._build_qid("0000")
    assert res["n_requests"] == 4
    assert set(_read_input_ids("0000")) == {"c1", "c2", "c3", "c4"}


# --------------------------------------------------------------------------- #
# full cmd_build --skip-done: manifest accounting + fully-reused completeness
# --------------------------------------------------------------------------- #
def test_cmd_build_skip_done_manifest_and_complete_flag(env):
    _seed(env["old_root"])
    m = _build(skip_done=True)
    q = m["queries"]

    assert q["0000"]["n_candidates"] == 4 and q["0000"]["n_requests"] == 2
    assert q["0001"]["n_candidates"] == 2 and q["0001"]["n_requests"] == 2   # new query: all new
    # Fully-reused query: no new requests, marked complete, and must not be submittable.
    assert q["0002"]["n_requests"] == 0
    assert q["0002"]["complete"] is True
    assert set(_read_input_ids("0001")) == {"n1", "n2"}
    assert _read_input_ids("0002") == []
