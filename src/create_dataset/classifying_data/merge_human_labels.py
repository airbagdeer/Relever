"""Merge human label exports (from the labeling tool) into data/judge_validation/short/human__gold.jsonl.

The HTML labeler exports the COMPLETE set of labels made on a device. Labeling is
split across devices (laptop = paragraphs 0-99, phone = 100-199) and happens over
several sittings, so this merges any number of export files — plus the existing
human__gold.jsonl — by id (last input wins), validates ids against _sample.json,
and reports coverage. Re-run it whenever you export again.

    uv run python src/create_dataset/classifying_data/merge_human_labels.py \
        --inputs ~/Downloads/human__gold*.jsonl ~/Downloads/*.ndjson
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data/judge_validation/short"
SAMPLE_PATH = RAW_DIR / "_sample.json"
OUT_PATH = RAW_DIR / "human__gold.jsonl"


def _read_records(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  (skip malformed line in {path.name})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="export files to merge in")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--no-accumulate", action="store_true",
                    help="ignore the existing human__gold.jsonl (start fresh from --inputs only)")
    args = ap.parse_args()

    sample = json.loads(SAMPLE_PATH.read_text())
    order = {rec["id"]: i for i, rec in enumerate(sample)}  # id -> sample index

    out_path = Path(args.out)
    sources: list[Path] = []
    if out_path.exists() and not args.no_accumulate:
        sources.append(out_path)  # existing first, so new exports override
    sources.extend(Path(p) for p in args.inputs)

    merged: dict[str, dict] = {}
    unknown: set[str] = set()
    bad = 0
    for src in sources:
        if not src.exists():
            print(f"WARNING: input not found: {src}")
            continue
        recs = _read_records(src)
        kept = 0
        for r in recs:
            rid = r.get("id")
            pred = r.get("prediction")
            if rid not in order:
                unknown.add(str(rid))
                continue
            if pred not in (0, 1):
                bad += 1
                continue
            merged[rid] = {"id": rid, "prediction": int(pred), "brief_reason": r.get("brief_reason", "")}
            kept += 1
        print(f"  {src.name}: {len(recs)} records, {kept} valid")

    # write sorted by sample order
    rows = sorted(merged.values(), key=lambda r: order[r["id"]])
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))

    # coverage
    first = sum(1 for r in rows if order[r["id"]] < 100)
    second = sum(1 for r in rows if order[r["id"]] >= 100)
    pos = sum(1 for r in rows if r["prediction"] == 1)
    print(f"\nMerged -> {out_path}")
    print(f"  {len(rows)}/{len(sample)} labeled  ({pos} positive, {len(rows)-pos} negative)")
    print(f"  by half: 0-99 -> {first}/100   |   100-199 -> {second}/100")
    if unknown:
        print(f"  WARNING: {len(unknown)} id(s) not in _sample.json (ignored): "
              f"{', '.join(list(unknown)[:3])}{'...' if len(unknown) > 3 else ''}")
    if bad:
        print(f"  ({bad} record(s) skipped — prediction not 0/1, e.g. skipped items)")


if __name__ == "__main__":
    main()
