"""Concurrent shards must not race on staging files.

The scorer runs one process per GPU. Per-query outputs are sharded by
``event_id % num_shards`` and never collide, but the run manifest is a single file that
*every* shard writes at startup. With a fixed ``<name>.tmp`` staging path this races:

    shard A: create <name>.tmp
    shard B: create <name>.tmp   (truncates A's)
    shard A: rename <name>.tmp -> <name>     (tmp is now gone)
    shard B: rename <name>.tmp -> <name>     -> FileNotFoundError

This happened on the cluster: 3 of 4 shards died on the first reranker whose manifest
did not already exist. Single-shard smoke runs cannot catch it, so it gets real processes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest
import score_store

WRITERS = 8
CONFIG = {
    "reranker": "m", "variant": "default", "dtype": "bfloat16",
    "device_kind": "cuda", "tf32": False, "max_length": 1152, "padding_side": "left",
}


def _write_manifest(root: str, barrier) -> str:
    """Run in a child process, mirroring one shard's startup."""
    import score_store as ss  # noqa: PLC0415  (fresh import per process)

    ss.FRONTIER_RERANK_SCORES_PATH = Path(root)
    barrier.wait()                       # maximise the overlap
    try:
        ss.check_or_write_manifest("m", "default", CONFIG)
        return "ok"
    except Exception as exc:             # noqa: BLE001 - the failure we are testing for
        return f"{type(exc).__name__}: {exc}"


def _write_scores(args) -> str:
    root, event_id, barrier = args
    import score_store as ss  # noqa: PLC0415

    ss.FRONTIER_RERANK_SCORES_PATH = Path(root)
    barrier.wait()
    try:
        ss.write_scores("m", "default", event_id, [f"d{i}" for i in range(50)],
                        [float(i) for i in range(50)])
        return "ok"
    except Exception as exc:             # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


@pytest.mark.unit
def test_concurrent_manifest_writers_all_succeed(tmp_path):
    """Every shard writes the same manifest at startup. All must survive.

    A plain ``ctx.Barrier`` cannot be pickled to a spawned child, so the barrier is a
    Manager proxy. It exists to force the writers to overlap; without it the race is
    real but rarely observed, which is exactly how it reached the cluster.
    """
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(WRITERS)
        with ctx.Pool(WRITERS) as pool:
            results = pool.starmap(_write_manifest, [(str(tmp_path), barrier)] * WRITERS)

    assert results == ["ok"] * WRITERS, f"a shard died on the manifest: {results}"

    manifest = json.loads((tmp_path / "m" / "default" / "_manifest.json").read_text())
    assert manifest == CONFIG            # last writer wins; payloads are identical


@pytest.mark.unit
def test_concurrent_score_writers_leave_no_staging_files(tmp_path, monkeypatch):
    """Different shards write different queries; none may leave a .tmp behind."""
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(WRITERS)
        with ctx.Pool(WRITERS) as pool:
            results = pool.map(_write_scores, [(str(tmp_path), i, barrier) for i in range(WRITERS)])

    assert results == ["ok"] * WRITERS, f"a shard died writing scores: {results}"

    written = sorted((tmp_path / "m" / "default").glob("*.parquet"))
    assert len(written) == WRITERS
    assert not list((tmp_path / "m" / "default").glob("*.tmp")), "staging file leaked"

    # monkeypatch, never a bare assignment: a leaked module-global here would silently
    # redirect every later test's cache lookups at a temp dir.
    monkeypatch.setattr(score_store, "FRONTIER_RERANK_SCORES_PATH", tmp_path)
    assert len(score_store.read_scores("m", "default", 0)) == 50


@pytest.mark.unit
def test_staging_path_is_unique_per_call(tmp_path):
    """The property the fix rests on: two writers never share a staging filename."""
    target = tmp_path / "x" / "0000.parquet"
    paths = {score_store._staging_path(target) for _ in range(16)}
    assert len(paths) == 16
    for p in paths:
        assert Path(p).parent == target.parent, "staging file must sit beside its target"
        assert Path(p).name != target.name + ".tmp", "the old, collision-prone name"


@pytest.mark.unit
def test_failed_write_removes_its_staging_file(tmp_path, monkeypatch):
    """A crash mid-write must not leave a half-file that resume would accept."""
    monkeypatch.setattr(score_store, "FRONTIER_RERANK_SCORES_PATH", tmp_path)
    with pytest.raises(ValueError, match="ids vs"):
        score_store.write_scores("m", "default", 0, ["a", "b"], [1.0])  # length mismatch
    # the guard fires before any file is created
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.parquet"))
