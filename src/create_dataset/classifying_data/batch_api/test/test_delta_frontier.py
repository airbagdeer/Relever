"""Unit tests for the delta labeler (``label_frontier_delta_batch``).

Everything runs on tiny local fixtures — no OpenAI API calls, no real dataset. The pure core
(``union_labels`` / ``needed_ids_for_R`` / ``compute_delta`` / ``resolve_texts``) is exercised with
explicit tmp paths; ``cmd_build`` is exercised end-to-end with the module path/threshold globals
monkeypatched onto a tmp tree (``POOL_DEPTH`` / ``MAX_RANK`` shrunk so a top-500 pool isn't needed).

Run: uv run pytest src/create_dataset/classifying_data/batch_api/test/test_delta_frontier.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# Make `src.*` importable when pytest collects this file directly.
for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (_p / "pyproject.toml").is_file():
        sys.path.insert(0, str(_p))
        break

from src.create_dataset.classifying_data.batch_api import label_frontier_delta_batch as d  # noqa: E402
from src.create_dataset.classifying_data.batch_api import label_frontier_batch as lfb  # noqa: E402


# --------------------------------------------------------------------------- #
# fixture writers
# --------------------------------------------------------------------------- #
def _write_model(cand_dir: Path, qid: str, alias: str, ranked_ids: list[str]) -> None:
    """One per-model ranked file: rank == position (1-based)."""
    (cand_dir / qid).mkdir(parents=True, exist_ok=True)
    cands = [{"id": cid, "rank": i + 1, "score": 1.0 / (i + 1)} for i, cid in enumerate(ranked_ids)]
    doc = {"event_id": int(qid), "model": alias, "candidates": cands}
    (cand_dir / qid / f"{alias}_{qid}.json").write_text(json.dumps(doc))


def _write_preds(root: Path, qid: str, preds: dict[str, int]) -> None:
    (root / qid).mkdir(parents=True, exist_ok=True)
    with (root / qid / f"predictions_{qid}.jsonl").open("w") as fh:
        for cid, pr in preds.items():
            fh.write(json.dumps({"id": cid, "event_id": int(qid), "prediction": pr}) + "\n")


def _write_splade(splade_dir: Path, name: str, splade: dict[str, str], bm25: dict[str, str]) -> None:
    splade_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "event_id": 0,
        "splade_results": [{"id": i, "payload": {"content": c}} for i, c in splade.items()],
        "bm25_results": [{"id": i, "payload": {"content": c}} for i, c in bm25.items()],
    }
    (splade_dir / f"{name}.json").write_text(json.dumps(doc))


# --------------------------------------------------------------------------- #
# union_labels / needed_ids_for_R
# --------------------------------------------------------------------------- #
def test_union_labels_merges_main_and_delta(tmp_path):
    main, delta = tmp_path / "main", tmp_path / "delta"
    _write_preds(main, "0000", {"a": 1, "b": 0})
    _write_preds(delta, "0000", {"c": 1})
    assert d.union_labels("0000", main_dir=main, delta_root=delta) == {"a": 1, "b": 0, "c": 1}


def test_needed_ids_is_union_of_each_model_topR(tmp_path):
    cand = tmp_path / "cand"
    _write_model(cand, "0000", "mA", ["a", "b", "c", "d"])
    _write_model(cand, "0000", "mB", ["a", "x", "c", "y"])
    assert d.needed_ids_for_R("0000", 2, candidates_dir=cand, max_rank=10) == {"a", "b", "x"}
    assert d.needed_ids_for_R("0000", 3, candidates_dir=cand, max_rank=10) == {"a", "b", "c", "x"}


# --------------------------------------------------------------------------- #
# compute_delta
# --------------------------------------------------------------------------- #
def _delta(cand, main, delta, qid, **kw):
    kw.setdefault("overhead", 0)  # exact top-R unless a test overrides
    return d.compute_delta(qid, candidates_dir=cand, main_dir=main, delta_root=delta,
                           pool_depth=2, max_rank=10, **kw)


def test_no_delta_when_R_at_or_below_pool_depth(tmp_path):
    cand, main, delta = tmp_path / "c", tmp_path / "m", tmp_path / "d"
    _write_model(cand, "0000", "mA", ["a", "b", "c", "d"])
    _write_preds(main, "0000", {"a": 1, "b": 1})  # R == 2 == pool_depth -> not triggered
    info = _delta(cand, main, delta, "0000")
    assert info.R == 2 and info.n_delta == 0 and info.delta_ids == []


def test_delta_is_needed_minus_labeled(tmp_path):
    cand, main, delta = tmp_path / "c", tmp_path / "m", tmp_path / "d"
    _write_model(cand, "0000", "mA", ["a", "b", "c", "d", "e"])
    _write_model(cand, "0000", "mB", ["a", "x", "y", "d", "z"])
    _write_preds(main, "0000", {"a": 1, "b": 1, "x": 1})  # R == 3 > pool_depth 2
    info = _delta(cand, main, delta, "0000")
    assert info.R == 3
    # needed = ranks 1..3 across models = {a,b,c,x,y}; labeled = {a,b,x}; delta = {c,y}
    assert info.delta_ids == ["c", "y"] and info.n_needed == 5 and not info.r_capped


def test_cat_and_mouse_grows_then_converges(tmp_path):
    cand, main, delta = tmp_path / "c", tmp_path / "m", tmp_path / "d"
    _write_model(cand, "0000", "mA", ["a", "b", "c", "d", "e"])
    _write_model(cand, "0000", "mB", ["a", "x", "y", "d", "z"])
    _write_preds(main, "0000", {"a": 1, "b": 1, "x": 1})            # R = 3 -> delta {c,y}
    assert _delta(cand, main, delta, "0000").delta_ids == ["c", "y"]

    _write_preds(delta, "0000", {"c": 1, "y": 0})                   # a delta positive lands -> R = 4
    info2 = _delta(cand, main, delta, "0000")
    assert info2.R == 4 and info2.delta_ids == ["d"]                # ranks 1..4 pull in d

    _write_preds(delta, "0000", {"c": 1, "y": 0, "d": 0})           # label d -> nothing new
    info3 = _delta(cand, main, delta, "0000")
    assert info3.R == 4 and info3.delta_ids == []                   # converged


def test_skip_done_excludes_ids_labeled_in_either_file(tmp_path):
    cand, main, delta = tmp_path / "c", tmp_path / "m", tmp_path / "d"
    _write_model(cand, "0000", "mA", ["a", "b", "c", "d"])
    _write_preds(main, "0000", {"a": 1, "b": 1, "c": 1})            # R = 3, c already labeled in main
    _write_preds(delta, "0000", {"d": 0})                          # d already labeled in delta
    info = _delta(cand, main, delta, "0000")
    # needed ranks 1..3 = {a,b,c}; all labeled -> delta empty (c not re-emitted)
    assert info.delta_ids == []


def test_R_is_clamped_to_max_rank(tmp_path):
    cand, main, delta = tmp_path / "c", tmp_path / "m", tmp_path / "d"
    _write_model(cand, "0000", "mA", ["a", "b", "c", "d", "e"])
    _write_preds(main, "0000", {"a": 1, "b": 1, "d": 1, "e": 1})    # R = 4
    info = d.compute_delta("0000", candidates_dir=cand, main_dir=main, delta_root=delta,
                           pool_depth=2, max_rank=3, overhead=0)
    assert info.R == 4 and info.r_capped is True and info.rank_target == 3
    # rcap = 3 -> needed = {a,b,c} only; c is the sole unlabeled one
    assert info.n_needed == 3 and info.delta_ids == ["c"]


def test_overhead_extends_rank_cutoff(tmp_path):
    cand, main, delta = tmp_path / "c", tmp_path / "m", tmp_path / "d"
    _write_model(cand, "0000", "mA", ["a", "b", "c", "d", "e", "f"])
    _write_model(cand, "0000", "mB", ["a", "x", "y", "z", "g", "h"])
    _write_preds(main, "0000", {"a": 1, "b": 1, "x": 1})  # R = 3
    base = _delta(cand, main, delta, "0000", overhead=0)   # rank<=3 -> {a,b,c,x,y} - {a,b,x}
    assert base.rank_target == 3 and base.delta_ids == ["c", "y"]
    over = _delta(cand, main, delta, "0000", overhead=0.2)  # rank<=ceil(3.6)=4
    # ranks 1..4: mA{a,b,c,d}, mB{a,x,y,z} -> {a,b,c,d,x,y,z}; minus {a,b,x} = {c,d,y,z}
    assert over.rank_target == 4 and over.delta_ids == ["c", "d", "y", "z"]


# --------------------------------------------------------------------------- #
# resolve_texts
# --------------------------------------------------------------------------- #
def test_resolve_texts_scans_splade_and_bm25_and_skips_empty(tmp_path):
    sp = tmp_path / "splade"
    _write_splade(sp, "0000", splade={"a": "AAA", "d": "   "}, bm25={"b": "BBB"})
    _write_splade(sp, "0001", splade={}, bm25={"c": "CCC"})
    got = d.resolve_texts({"a", "b", "c", "d", "z"}, splade_dir=sp)
    assert got == {"a": "AAA", "b": "BBB", "c": "CCC"}  # d empty, z absent


# --------------------------------------------------------------------------- #
# _write_batch_input
# --------------------------------------------------------------------------- #
def test_write_batch_input_shape_and_skips_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DELTA_ROOT", tmp_path / "delta")
    cfg = d._config()
    qinfo = {"sentence": "S", "abstract_entity_type": "A", "constant_entity": "K", "relation": "REL"}
    n, unresolved = d._write_batch_input(
        "0000", ["c", "y", "z"], {"c": "C text"}, qinfo, "SYS-PROMPT", cfg)
    assert (n, unresolved) == (1, 2)  # only c has text

    lines = (tmp_path / "delta" / "0000" / "batch_input_0000.jsonl").read_text().splitlines()
    assert len(lines) == 1
    req = json.loads(lines[0])
    assert req["custom_id"] == "c"
    assert req["body"]["model"] == cfg["model"]
    assert req["body"]["messages"][0]["content"] == "SYS-PROMPT"
    assert "C text" in req["body"]["messages"][1]["content"]
    assert "Sentence: S" in req["body"]["messages"][1]["content"]


# --------------------------------------------------------------------------- #
# cmd_build end-to-end (tmp globals; POOL_DEPTH/MAX_RANK shrunk)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def build_env(tmp_path, monkeypatch):
    cand = tmp_path / "candidates"
    main = tmp_path / "main"
    delta_root = tmp_path / "main" / "extended_pooling_delta"
    splade = tmp_path / "splade"

    _write_model(cand, "0000", "mA", ["a", "b", "c", "d", "e"])
    _write_model(cand, "0000", "mB", ["a", "x", "y", "d", "z"])
    _write_preds(main, "0000", {"a": 1, "b": 1, "x": 1})  # R = 3 (> pool_depth 2)
    _write_splade(splade, "s0", splade={"c": "C-text", "y": "Y-text", "d": "D-text"}, bm25={})

    rel = tmp_path / "relations.json"
    rel.write_text(json.dumps([{"event_id": 0, "sentence": "s", "abstract_entity_type": "a",
                                "constant_entity": "k", "relation": "r"}]))

    monkeypatch.setattr(d, "CANDIDATES_DIR", cand)
    monkeypatch.setattr(d, "MAIN_CLASSIFIED_DIR", main)
    monkeypatch.setattr(d, "DELTA_ROOT", delta_root)
    monkeypatch.setattr(d, "SPLADE_V2_DIR", splade)
    monkeypatch.setattr(d, "MANIFEST_PATH", delta_root / "_manifest.json")
    monkeypatch.setattr(d, "POOL_DEPTH", 2)
    monkeypatch.setattr(d, "MAX_RANK", 10)
    monkeypatch.setattr(lfb, "RELATIONS_PATH", rel)
    return {"cand": cand, "main": main, "delta_root": delta_root, "splade": splade}


def _build(**kw):
    ns = argparse.Namespace(qids=kw.get("qids"), limit=kw.get("limit"),
                            force=kw.get("force", False), overhead=kw.get("overhead", 0.0))
    d.cmd_build(ns)
    return json.loads((d.MANIFEST_PATH).read_text())


def _input_ids(delta_root: Path, qid: str) -> set[str]:
    p = delta_root / qid / f"batch_input_{qid}.jsonl"
    return {json.loads(li)["custom_id"] for li in p.read_text().splitlines() if li.strip()}


def test_cmd_build_writes_delta_input_and_manifest(build_env):
    m = _build()
    e = m["queries"]["0000"]
    assert e["R_at_build"] == 3 and e["n_delta"] == 2 and e["n_requests"] == 2
    assert e["input_built"] is True
    assert _input_ids(build_env["delta_root"], "0000") == {"c", "y"}  # ranks 1..3 minus pool


def test_cmd_build_is_idempotent_when_R_unchanged(build_env):
    _build()
    m2 = _build()  # nothing changed -> no rebuild
    assert m2["queries"]["0000"]["n_requests"] == 2
    assert _input_ids(build_env["delta_root"], "0000") == {"c", "y"}


def test_cmd_build_next_round_after_delta_labels_land(build_env):
    _build()
    # Simulate a fetch: c,y labeled in the delta predictions (c positive -> R grows to 4).
    _write_preds(build_env["delta_root"], "0000", {"c": 1, "y": 0})
    m = _build()
    e = m["queries"]["0000"]
    assert e["R_at_build"] == 4 and e["n_requests"] == 1
    assert _input_ids(build_env["delta_root"], "0000") == {"d"}  # ranks 1..4 minus labeled


def test_cmd_build_with_overhead_labels_deeper(build_env):
    m = _build(overhead=0.2)  # R=3 -> rank_target = ceil(3.6) = 4
    e = m["queries"]["0000"]
    assert e["R_at_build"] == 3 and e["n_requests"] == 3 and m["config"]["overhead"] == 0.2
    # ranks 1..4: mA{a,b,c,d}, mB{a,x,y,d} -> {a,b,c,d,x,y}; minus pool {a,b,x} = {c,d,y}
    assert _input_ids(build_env["delta_root"], "0000") == {"c", "d", "y"}


def test_cmd_build_rebuilds_when_overhead_changes(build_env):
    _build(overhead=0.0)
    assert _input_ids(build_env["delta_root"], "0000") == {"c", "y"}
    m = _build(overhead=0.2)  # overhead changed -> rebuild deeper even though R is unchanged
    assert _input_ids(build_env["delta_root"], "0000") == {"c", "d", "y"}
    assert m["queries"]["0000"]["n_requests"] == 3
