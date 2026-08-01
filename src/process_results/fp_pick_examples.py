"""Candidate browser for the paper's verbatim FP example snippets (human-in-the-loop).

Lists labeled false positives of the 2-system subset for the requested cells so a
human can pick the 2--3 snippets quoted in the error-analysis subsection. Not part
of the build chain — the chosen snippets are hardcoded in LaTeX with a provenance
comment (event, chunk id, cell, per-system ranks).

    uv run python src/process_results/fp_pick_examples.py --event 259 --cells 111 101 001
    uv run python src/process_results/fp_pick_examples.py --all-events --cells 001 --top 15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.create_dataset.classifying_data.batch_api.label_fp_taxonomy_batch import (  # noqa: E402
    _texts_for_event,
)
from src.create_dataset.classifying_data.fp_subset_report import (  # noqa: E402
    _cell,
    _cell_str,
    join_predictions,
    load_population,
)
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (  # noqa: E402
    FP_TAXONOMY_PROMPT_VERSION,
)
from src.retrieval.config import FP_TAXONOMY_DATA_PATH  # noqa: E402

V2_RELATIONS_PATH = PROJECT_ROOT / "data" / "queries" / "queries.json"

# Markers whose density flags reference/navigation dumps rather than prose.
_CITE_MARKERS = ("Retrieved ", "Archived from", "ISBN", "doi:", '"^"', "^ ")


def _clutter(text: str) -> int:
    return sum(text.count(m) for m in _CITE_MARKERS)


def _one_line(text: str, max_chars: int) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:max_chars] + ("…" if len(flat) > max_chars else "")


def _query_sentence(event_id: int) -> str:
    try:
        rows = json.loads(V2_RELATIONS_PATH.read_text())
        for r in rows:
            if int(r.get("event_id", -1)) == event_id:
                return (r.get("sentence") or r.get("event") or "")[:120]
    except Exception:
        pass
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", type=int, default=None, help="restrict to one event_id")
    ap.add_argument("--all-events", action="store_true")
    ap.add_argument("--cells", nargs="+", default=["111", "101", "001"],
                    help="cells as I/R/C bitstrings, e.g. 101")
    ap.add_argument("--max-chars", type=int, default=350)
    ap.add_argument("--top", type=int, default=12, help="candidates shown per cell")
    ap.add_argument("--population",
                    default=str(FP_TAXONOMY_DATA_PATH / "subset_2sys" / "subset_population.jsonl"))
    args = ap.parse_args()
    if args.event is None and not args.all_events:
        raise SystemExit("Pass --event <id> or --all-events.")

    wanted = {tuple(ch == "1" for ch in c) for c in args.cells}
    pop = load_population(Path(args.population))
    joined, _ = join_predictions(pop, FP_TAXONOMY_PROMPT_VERSION)
    if args.event is not None:
        joined = [r for r in joined if int(r["event_id"]) == args.event]
    matches = [r for r in joined if _cell(r) in wanted]
    if not matches:
        raise SystemExit("No labeled pairs match the requested cells/event.")

    texts_cache: dict[int, dict[str, str]] = {}
    by_event: dict[int, set[str]] = {}
    for r in matches:
        by_event.setdefault(int(r["event_id"]), set()).add(r["id"])
    for eid, ids in by_event.items():
        texts_cache[eid] = _texts_for_event(eid, ids)

    for cell in sorted(wanted, reverse=True):
        rows = [r for r in matches if _cell(r) == cell]
        scored = []
        for r in rows:
            text = texts_cache.get(int(r["event_id"]), {}).get(r["id"], "")
            scored.append((_clutter(text), len(text), r, text))
        scored.sort(key=lambda t: (t[0], t[1]))
        print(f"\n{'=' * 90}\nCELL {_cell_str(cell)} — {len(rows)} candidates "
              f"(sorted clean-first, short-first; showing {min(args.top, len(scored))})")
        for clutter, _, r, text in scored[: args.top]:
            ranks = ", ".join(f"{a}@{k}" for a, k in sorted(r["systems"].items()))
            q = _query_sentence(int(r["event_id"]))
            print(f"\n  event {r['event_id']}  R={r['R_q']}  chunk {r['id'][:8]}  "
                  f"[{ranks}]  clutter={clutter}")
            if q:
                print(f"  query: {q}")
            print(f"  text : {_one_line(text, args.max_chars)}")


if __name__ == "__main__":
    main()
