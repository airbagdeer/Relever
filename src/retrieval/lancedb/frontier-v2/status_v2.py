"""frontier-v2 indexing status / row-count / disk-location report.

For each of the 18 models, prints its checkpoint status, whether its LanceDB
table exists at its routed root (home vs scratch), the table row count, and
whether it matches the corpus size U. Also prints free space on both roots.

    uv run python src/retrieval/lancedb/frontier-v2/status_v2.py
    uv run python src/retrieval/lancedb/frontier-v2/status_v2.py --table-suffix _smoke
"""

from __future__ import annotations

import argparse
import shutil

import lancedb

import _bootstrap  # noqa: F401
from checkpoint import CheckpointStore, _db_tables, default_state_dir
from config import LANCEDB_PATH, LANCEDB_SCRATCH_PATH, get_classified_collection_name
from model_registry import ALL_ALIASES, root_for, spec_for


def _free_gb(path) -> float:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="frontier-v2 indexing status report.")
    parser.add_argument("--table-suffix", type=str, default="")
    args = parser.parse_args()

    checkpoint = CheckpointStore(default_state_dir(LANCEDB_PATH))

    # One connection per distinct root.
    dbs: dict[str, lancedb.DBConnection] = {}
    tables_by_root: dict[str, set[str]] = {}
    for root in {str(LANCEDB_PATH), str(LANCEDB_SCRATCH_PATH)}:
        try:
            dbs[root] = lancedb.connect(root)
            tables_by_root[root] = _db_tables(dbs[root])
        except (OSError, ValueError):
            tables_by_root[root] = set()

    print(f"home    ({LANCEDB_PATH}): {_free_gb(LANCEDB_PATH):.0f}G free")
    print(f"scratch ({LANCEDB_SCRATCH_PATH}): {_free_gb(LANCEDB_SCRATCH_PATH):.0f}G free")
    print()
    print(f"{'alias':<34} {'backend':<8} {'root':<8} {'status':<8} {'rows':>10} {'U':>10}  ok")
    print("-" * 92)

    n_done = 0
    for alias in ALL_ALIASES:
        spec = spec_for(alias)
        root = root_for(alias)
        root_key = str(root)
        root_label = "home" if root == LANCEDB_PATH else "scratch"
        table_name = get_classified_collection_name(alias, "paragraph") + args.table_suffix

        st = checkpoint.get(table_name)
        rows = 0
        U = st.U
        if table_name in tables_by_root.get(root_key, set()):
            try:
                rows = len(dbs[root_key].open_table(table_name))
            except (FileNotFoundError, ValueError, KeyError):
                rows = -1
        ok = "yes" if (U > 0 and rows == U and st.status == "done") else ""
        if ok:
            n_done += 1
        print(f"{alias:<34} {spec.backend:<8} {root_label:<8} {st.status:<8} "
              f"{rows:>10} {U:>10}  {ok}")

    print("-" * 92)
    print(f"{n_done}/{len(ALL_ALIASES)} models complete (rows == U, status == done)"
          + (f" [suffix={args.table_suffix!r}]" if args.table_suffix else ""))


if __name__ == "__main__":
    main()
