"""Count unique paragraphs across all relations (cross-relation dedup check).

This is the "deduplicate across all relations" verification for the v2 dataset.
It scans the per-event retrieval output (``{event_id}.json`` files written by
``run_batch_retrieval.py``) and counts the **global** set of unique paragraph
``id``s across every query's SPLADE + BM25 results — exactly the dedup key the
indexer (``corpus_loader.py``) uses, so this
predicts the indexed paragraph universe without paying for any embedding.

It is **report-only**: it never retrieves, re-runs, or modifies anything. If the
unique count comes up short of the 1M target it prints the number and a rough
hint about how far ``--per-method-limit`` would need to grow, then stops — you
decide whether to bump the budget and re-run into a clean output dir.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/splade_wikipedia_v2"
TARGET_UNIQUE = 1_000_000


def _event_ids(data: dict) -> List[str]:
    """All non-null paragraph ids in an event payload (SPLADE + BM25)."""
    results = data.get("splade_results", []) + data.get("bm25_results", [])
    return [r.get("id") for r in results if r.get("id") is not None]


def count_unique(output_dir: Path) -> dict:
    """Scan per-event files and return aggregate stats (no side effects)."""
    files = sorted(
        output_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
    )

    global_ids: set = set()
    n_events = 0
    skipped = 0
    sum_event_unique = 0  # sum of per-event unique counts (pre cross-relation dedup)
    per_event_unique: List[int] = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        if not (isinstance(data, dict) and "splade_results" in data and "bm25_results" in data):
            skipped += 1
            continue

        ids = _event_ids(data)
        event_unique = set(ids)
        per_event_unique.append(len(event_unique))
        sum_event_unique += len(event_unique)
        global_ids.update(event_unique)
        n_events += 1

    mean_unique = (sum_event_unique / n_events) if n_events else 0.0
    return {
        "n_events": n_events,
        "skipped": skipped,
        "sum_event_unique": sum_event_unique,
        "global_unique": len(global_ids),
        "mean_unique_per_query": mean_unique,
    }


def main(output_dir: Path, target: int, per_method_limit: int) -> None:
    if not output_dir.exists():
        raise SystemExit(f"Output dir {output_dir} does not exist — run retrieval first.")

    stats = count_unique(output_dir)
    g = stats["global_unique"]

    print(f"Scanned:                 {output_dir}")
    print(f"Events (valid files):    {stats['n_events']}")
    if stats["skipped"]:
        print(f"Skipped (non-event/bad): {stats['skipped']}")
    print(f"Mean unique / query:     {stats['mean_unique_per_query']:.1f}")
    print(f"Sum of per-query unique: {stats['sum_event_unique']:,}  (before cross-relation dedup)")
    print(f"Global unique paragraphs:{g:>12,}  (after cross-relation dedup)")
    if stats["sum_event_unique"]:
        overlap = 1 - g / stats["sum_event_unique"]
        print(f"Cross-relation overlap:  {overlap * 100:.1f}% of slots were duplicates across queries")
    print(f"Target:                  {target:>12,}")

    if g >= target:
        print(f"\n✅ Cleared the target: {g:,} >= {target:,} unique paragraphs.")
        return

    short_by = target - g
    factor = target / g if g else float("inf")
    print(f"\n❌ Short by {short_by:,} ({g / target * 100:.1f}% of target).")
    if math.isfinite(factor):
        suggested = math.ceil(per_method_limit * factor / 50) * 50  # round up to a tidy number
        print(
            f"   Rough guide: clearing 1M would need ~{factor:.2f}x more unique "
            f"paragraphs/query.\n"
            f"   That points to roughly --per-method-limit {suggested} (was "
            f"{per_method_limit}), assuming cross-query overlap stays similar.\n"
            f"   This is only a report — nothing was re-run. Decide whether to raise "
            f"the budget and re-run retrieval into a clean output dir."
        )
    else:
        print("   No paragraphs found — check that retrieval actually produced output.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Count globally-unique paragraphs across all per-event retrieval "
                    "files (cross-relation dedup / 1M check). Report-only."
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Per-event retrieval output dir to scan (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--target", type=int, default=TARGET_UNIQUE,
                    help=f"Unique-paragraph target to check against (default: {TARGET_UNIQUE})")
    ap.add_argument("--per-method-limit", type=int, default=1000,
                    help="The --per-method-limit used for retrieval; only used to size the "
                         "'how much to grow' hint when short (default: 1000)")
    args = ap.parse_args()
    main(args.output_dir, args.target, args.per_method_limit)
