"""Unit tests for the checkpoint store and bounded drop-table logic."""

from pathlib import Path

import lancedb
import pytest

from checkpoint import CheckpointStore, drop_target_tables


def _seed_table(db, name: str, n_rows: int) -> None:
    db.create_table(
        name,
        data=[{"id": str(i), "text": "x", "vector": [0.0, 0.0, 0.0]} for i in range(n_rows)],
        mode="overwrite",
    )


def test_is_done_requires_state_table_and_row_count_match(tmp_path: Path) -> None:
    db = lancedb.connect(tmp_path / "lance")
    cp = CheckpointStore(tmp_path / "state")

    table_name = "thesis-qdrant-classified-fake-paragraph"
    _seed_table(db, table_name, n_rows=10)

    assert cp.is_done("fake", U=10, db=db, table_name=table_name) is False, (
        "Without a state record, is_done must be False."
    )

    cp.mark_running("fake", U=10)
    assert cp.is_done("fake", U=10, db=db, table_name=table_name) is False

    cp.mark_done("fake", rows_written=10, U=10)
    assert cp.is_done("fake", U=10, db=db, table_name=table_name) is True

    # Table missing after being marked done -> not done.
    db.drop_table(table_name)
    assert cp.is_done("fake", U=10, db=db, table_name=table_name) is False

    # Row count drift -> not done.
    _seed_table(db, table_name, n_rows=9)
    assert cp.is_done("fake", U=10, db=db, table_name=table_name) is False


def test_mark_failed_persists_error(tmp_path: Path) -> None:
    cp = CheckpointStore(tmp_path / "state")
    cp.mark_running("fake", U=5)
    cp.mark_failed("fake", error="OOM", U=5)

    cp2 = CheckpointStore(tmp_path / "state")
    state = cp2.get("fake")
    assert state.status == "failed"
    assert state.error == "OOM"


def test_drop_target_tables_only_touches_listed_names(tmp_path: Path) -> None:
    db = lancedb.connect(tmp_path / "lance")

    targets = [
        "thesis-qdrant-classified-gte-small-paragraph",
        "thesis-qdrant-classified-bge-m3-paragraph",
    ]
    untouched = [
        "thesis-some-other-table",
        "thesis-qdrant-classified-jina-embeddings-v3-sentence",  # different granularity
        "thesis-qdrant-classified-not-in-list-paragraph",
    ]
    for name in targets + untouched:
        _seed_table(db, name, n_rows=2)

    dropped = drop_target_tables(db, targets)

    assert sorted(dropped) == sorted(targets)
    remaining = set(db.table_names())
    for name in untouched:
        assert name in remaining, f"{name} should not have been dropped"
    for name in targets:
        assert name not in remaining, f"{name} should have been dropped"


def test_drop_target_tables_silent_on_missing(tmp_path: Path) -> None:
    db = lancedb.connect(tmp_path / "lance")
    dropped = drop_target_tables(db, ["missing-table"])
    assert dropped == []


def test_state_keyed_by_table_so_granularities_dont_collide(tmp_path: Path) -> None:
    """The same alias indexed for paragraph/sentence/proposition must keep
    separate state — that's why we key by table_name, not alias.
    """
    cp = CheckpointStore(tmp_path / "state")
    para = "thesis-qdrant-classified-gte-small-paragraph"
    sent = "thesis-qdrant-classified-gte-small-sentence"

    cp.mark_done(para, rows_written=100, U=100)
    cp.mark_running(sent, U=500)

    cp2 = CheckpointStore(tmp_path / "state")
    assert cp2.get(para).status == "done"
    assert cp2.get(para).U == 100
    assert cp2.get(sent).status == "running"
    assert cp2.get(sent).U == 500


def test_concurrent_writes_do_not_lose_entries(tmp_path: Path) -> None:
    """Two CheckpointStore instances writing different keys interleaved must
    both end up persisted in state.json (no last-writer-wins on the file)."""
    import threading

    cp_a = CheckpointStore(tmp_path / "state")
    cp_b = CheckpointStore(tmp_path / "state")

    aliases = [f"thesis-qdrant-classified-m{i}-paragraph" for i in range(20)]

    def write_half(cp: CheckpointStore, names: list[str]) -> None:
        for name in names:
            cp.mark_running(name, U=42)
            cp.mark_done(name, rows_written=42, U=42)

    half = len(aliases) // 2
    t1 = threading.Thread(target=write_half, args=(cp_a, aliases[:half]))
    t2 = threading.Thread(target=write_half, args=(cp_b, aliases[half:]))
    t1.start(); t2.start()
    t1.join(); t2.join()

    cp_check = CheckpointStore(tmp_path / "state")
    for name in aliases:
        st = cp_check.get(name)
        assert st.status == "done", f"{name} lost in concurrent write race"
        assert st.U == 42
