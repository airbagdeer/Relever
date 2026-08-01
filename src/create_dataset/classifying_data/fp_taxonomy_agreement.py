"""Classifier–human agreement on the FP pilot (v3 three-bit presence schema).

Joins the hand-labeled gold (``fp_human__gold.jsonl``, exported by the labeling tool)
with the batch classifier's predictions (per-event ``fp_taxonomy_<qid>.jsonl``, current
prompt version) and reports, for each of the three presence bits
(instance_present / relation_present / constant_present):

  * percent agreement + Cohen's kappa (the appendix's validation numbers),
  * the 2x2 confusion (human rows x classifier columns),

plus the exact-match agreement + kappa over the full 8-cell pattern.

Outputs land under ``summary/fp_taxonomy/`` (pilot_agreement_<version>.md + CSV),
per the summary-layout convention.

    uv run python src/create_dataset/classifying_data/fp_taxonomy_agreement.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.create_dataset.classifying_data.batch_api.label_fp_taxonomy_batch import (  # noqa: E402
    PILOT_DIR,
    load_predictions_for_event,
)
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (  # noqa: E402
    BITS,
    FP_TAXONOMY_PROMPT_VERSION,
)
from src.retrieval.config import SUMMARY_FP_TAXONOMY_PATH  # noqa: E402

DEFAULT_GOLD = PILOT_DIR / "fp_human__gold.jsonl"


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Unweighted Cohen's kappa over (rater_a, rater_b) label pairs."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def load_gold(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if all(isinstance(r.get(b), bool) for b in BITS):
                rows.append(r)
    return rows


def join_with_classifier(gold: list[dict], version: str) -> tuple[list[dict], int]:
    """Attach the classifier's bits to each gold row; count misses."""
    cache: dict[int, dict[str, dict]] = {}
    joined, missing = [], 0
    for g in gold:
        eid = int(g["event_id"])
        if eid not in cache:
            cache[eid] = load_predictions_for_event(eid)
        rec = cache[eid].get(g["id"])
        if rec is None or rec.get("prompt_version") != version \
                or not all(isinstance(rec.get(b), bool) for b in BITS):
            missing += 1
            continue
        joined.append({**g, **{f"clf_{b}": rec[b] for b in BITS}})
    return joined, missing


def _cell(r: dict, prefix: str = "") -> str:
    return "/".join("1" if r[f"{prefix}{b}"] else "0" for b in BITS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--version", default=FP_TAXONOMY_PROMPT_VERSION,
                    help="classifier prompt version to compare against")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise SystemExit(f"Gold file not found: {gold_path} — label with the HTML tool "
                         "and move the export there first.")
    gold = load_gold(gold_path)
    joined, missing = join_with_classifier(gold, args.version)
    n = len(joined)
    if n == 0:
        raise SystemExit(f"No overlap between gold ({len(gold)} rows) and classifier "
                         f"predictions under version {args.version}.")

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit(f"# FP presence-bits pilot agreement — classifier {args.version} vs human gold")
    emit()
    emit(f"- gold labels: {len(gold)} ({gold_path.name}); joined with classifier: {n}"
         + (f"; missing classifier predictions: {missing}" if missing else ""))
    emit()
    emit("## Per-bit agreement (human vs classifier)")
    emit()
    emit("| bit | agreement | kappa | human yes | clf yes |")
    emit("|---|---|---|---|---|")
    per_bit_stats: dict[str, tuple[float, float]] = {}
    for b in BITS:
        pairs = [(str(r[b]), str(r[f"clf_{b}"])) for r in joined]
        pct = sum(1 for a, c in pairs if a == c) / n
        kap = cohen_kappa(pairs)
        per_bit_stats[b] = (pct, kap)
        emit(f"| {b} | {pct:.1%} | {kap:.3f} | "
             f"{sum(1 for r in joined if r[b])} | {sum(1 for r in joined if r[f'clf_{b}'])} |")
    cell_pairs = [(_cell(r), _cell(r, 'clf_')) for r in joined]
    cell_pct = sum(1 for a, c in cell_pairs if a == c) / n
    cell_kap = cohen_kappa(cell_pairs)
    emit()
    emit(f"- **exact 8-cell match: {cell_pct:.1%}  ·  Cohen's kappa = {cell_kap:.3f}** (n={n})")
    emit()
    emit("## Per-bit confusion (human rows x classifier columns)")
    emit()
    for b in BITS:
        tp = sum(1 for r in joined if r[b] and r[f"clf_{b}"])
        fn = sum(1 for r in joined if r[b] and not r[f"clf_{b}"])
        fp = sum(1 for r in joined if not r[b] and r[f"clf_{b}"])
        tn = sum(1 for r in joined if not r[b] and not r[f"clf_{b}"])
        emit(f"**{b}**: yes/yes {tp} · yes/no {fn} · no/yes {fp} · no/no {tn}")
    emit()
    emit("## Cell marginals (human vs classifier)")
    emit()
    hc = Counter(_cell(r) for r in joined)
    cc = Counter(_cell(r, "clf_") for r in joined)
    emit("| cell (instance/relation/constant) | human | classifier |")
    emit("|---|---|---|")
    for cell in sorted(set(hc) | set(cc), key=lambda s: -(hc.get(s, 0) + cc.get(s, 0))):
        emit(f"| {cell} | {hc.get(cell, 0)} | {cc.get(cell, 0)} |")

    SUMMARY_FP_TAXONOMY_PATH.mkdir(parents=True, exist_ok=True)
    md_path = SUMMARY_FP_TAXONOMY_PATH / f"pilot_agreement_{args.version}.md"
    md_path.write_text("\n".join(lines) + "\n")

    csv_path = SUMMARY_FP_TAXONOMY_PATH / f"pilot_agreement_{args.version}.csv"
    with csv_path.open("w") as fh:
        fh.write("bit,agreement,kappa\n")
        for b in BITS:
            pct, kap = per_bit_stats[b]
            fh.write(f"{b},{pct:.4f},{kap:.4f}\n")
        fh.write(f"exact_cell,{cell_pct:.4f},{cell_kap:.4f}\n")

    disagreements = [r for r in joined if _cell(r) != _cell(r, "clf_")]
    dis_path = SUMMARY_FP_TAXONOMY_PATH / f"pilot_disagreements_{args.version}.jsonl"
    with dis_path.open("w") as fh:
        for r in disagreements:
            fh.write(json.dumps(r) + "\n")

    print(f"\n-> {md_path}\n-> {csv_path}\n-> {dis_path} ({len(disagreements)} cell disagreements)")


if __name__ == "__main__":
    main()
