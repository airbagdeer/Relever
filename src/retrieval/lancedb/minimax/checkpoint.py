"""Per-model checkpoint state and bounded drop-table helpers.

Resume contract:
- A model is considered "done" if state.json says so AND its LanceDB table
  exists with row count == U (the unique paragraph count of the corpus).
- A model that crashed mid-encode is restarted: we drop its table by exact
  name (never via a wildcard) and re-encode from scratch.
- Tables outside the explicit `target_table_names` list are never touched.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import lancedb

logger = logging.getLogger(__name__)

STATE_DIRNAME = "_minimax_checkpoint"
STATE_FILENAME = "state.json"
LOCK_FILENAME = "state.lock"


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout_s: float = 60.0) -> Iterator[None]:
    """Advisory exclusive file lock via fcntl. Used to make read-modify-write
    of the shared state.json safe across concurrent indexer processes.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire {lock_path} within {timeout_s}s"
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _db_tables(db: lancedb.DBConnection) -> set[str]:
    """Return the set of existing table names in the LanceDB connection.

    Wraps the LanceDB API to be tolerant of the version churn between the
    deprecated `db.table_names()` (returns list[str]) and the newer
    `db.list_tables()` (returns a `ListTablesResponse` with a `.tables` field).
    """
    listed = db.list_tables()
    if hasattr(listed, "tables"):
        return set(listed.tables)
    if isinstance(listed, list):
        return set(listed)
    # Fallback: assume iterable of strings.
    return set(listed)


@dataclass
class ModelState:
    status: str = "pending"  # pending | running | done | failed
    rows_written: int = 0
    U: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


class CheckpointStore:
    """Concurrent-safe per-(alias, granularity) checkpoint state.

    State is keyed by `table_name` (which already encodes alias + granularity
    + any suffix) so a single state.json holds independent state for
    paragraph / sentence / proposition runs of the same model. Reads and
    writes go through an fcntl advisory lock so multiple indexer processes
    (e.g. one per GPU under the `parallel` schedule) can't race on the
    shared state.json.
    """

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / STATE_FILENAME
        self.lock_path = self.state_dir / LOCK_FILENAME

    def _read_locked(self) -> dict[str, ModelState]:
        out: dict[str, ModelState] = {}
        if not self.state_path.exists():
            return out
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read checkpoint state %s: %s", self.state_path, exc)
            return out
        for key, payload in raw.items():
            if isinstance(payload, dict):
                out[key] = ModelState(
                    status=str(payload.get("status", "pending")),
                    rows_written=int(payload.get("rows_written", 0)),
                    U=int(payload.get("U", 0)),
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    error=payload.get("error"),
                )
        return out

    def _write_locked(self, state: dict[str, ModelState]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in state.items()}, f, indent=2)
        tmp.replace(self.state_path)

    def _update(self, key: str, mutate) -> None:
        """Atomically read state, mutate the entry for `key`, write back."""
        with _file_lock(self.lock_path):
            state = self._read_locked()
            state[key] = mutate(state.get(key))
            self._write_locked(state)

    def get(self, key: str) -> ModelState:
        with _file_lock(self.lock_path):
            return self._read_locked().get(key, ModelState())

    def is_done(
        self,
        key: str,
        U: int,
        db: lancedb.DBConnection,
        table_name: str,
    ) -> bool:
        st = self.get(key)
        if st.status != "done":
            return False
        if st.U != U or st.rows_written != U:
            return False
        if table_name not in _db_tables(db):
            return False
        try:
            tbl = db.open_table(table_name)
            return len(tbl) == U
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Could not open table %s for checkpoint validation: %s", table_name, exc)
            return False

    def mark_running(self, key: str, U: int) -> None:
        def _m(_prev: Optional[ModelState]) -> ModelState:
            return ModelState(
                status="running",
                rows_written=0,
                U=U,
                started_at=time.time(),
            )
        self._update(key, _m)

    def mark_done(self, key: str, rows_written: int, U: int) -> None:
        def _m(prev: Optional[ModelState]) -> ModelState:
            base = prev or ModelState(U=U)
            return ModelState(
                status="done",
                rows_written=rows_written,
                U=U,
                started_at=base.started_at,
                finished_at=time.time(),
            )
        self._update(key, _m)

    def mark_failed(self, key: str, error: str, U: int) -> None:
        def _m(prev: Optional[ModelState]) -> ModelState:
            base = prev or ModelState(U=U)
            return ModelState(
                status="failed",
                rows_written=base.rows_written,
                U=U,
                started_at=base.started_at,
                finished_at=time.time(),
                error=error,
            )
        self._update(key, _m)


def drop_target_tables(
    db: lancedb.DBConnection,
    target_table_names: Iterable[str],
) -> list[str]:
    """Drop only the named tables that currently exist. Return names dropped.

    This is bounded by the explicit list — never globbed, never wildcarded.
    Other tables in the LanceDB are not touched.
    """
    existing = _db_tables(db)
    dropped: list[str] = []
    for name in target_table_names:
        if name in existing:
            try:
                db.drop_table(name)
                dropped.append(name)
                logger.info("Dropped LanceDB table: %s", name)
            except (FileNotFoundError, ValueError, OSError) as exc:
                logger.warning("Failed to drop %s: %s", name, exc)
    return dropped


def default_state_dir(lancedb_path: Path) -> Path:
    return lancedb_path / STATE_DIRNAME
