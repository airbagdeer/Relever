"""Per-system 8-cell FP composition for a subset run (e.g. the two flagship systems).

Joins a subset population (``extract_fp_population.py`` JSONL, each pair carrying the
systems that retrieved it) with the per-event v5 taxonomy predictions, and reports —
for each system separately and for the union — the distribution over the eight
presence cells (instance / relation / constant), the per-bit marginals, and the
headline "instance + constant present, relation absent" share (the "right entities,
wrong relation" reading).

Also breaks the union composition down by R_q bucket (``subset_by_rbucket.csv``)
and the per-(pair, system) slots by first-stage rank bin (``subset_by_rank.csv``).

Outputs under ``summary/fp_taxonomy/subset_2sys/`` (``subset_composition.csv`` +
``subset_by_rbucket.csv`` + ``subset_by_rank.csv`` + ``REPORT.md``), per the
summary-layout convention.

    uv run python src/create_dataset/classifying_data/fp_subset_report.py \\
        --population data/fp_taxonomy/subset_2sys/subset_population.jsonl
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
    CELL_GLOSS,
    load_predictions_for_event,
)
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (  # noqa: E402
    BITS,
    FP_TAXONOMY_PROMPT_VERSION,
)
from src.retrieval.config import FP_TAXONOMY_DATA_PATH, SUMMARY_FP_TAXONOMY_PATH  # noqa: E402

# All 8 cells in a fixed, readable order (instance, relation, constant).
CELLS: tuple[tuple[bool, bool, bool], ...] = (
    (True, True, True), (True, False, True), (False, True, True), (True, True, False),
    (False, False, True), (True, False, False), (False, True, False), (False, False, False),
)

# Display order of the R_q buckets stored per pair (names produced by
# extract_fp_population.r_bucket_name from eval_config.R_BUCKETS).
R_BUCKET_ORDER: tuple[str, ...] = ("1-4", "5-9", "10-49", "50-99", "100-499", "500+")

# First-stage rank bins for the per-(pair, system) slot breakdown. Ranks are the raw
# 1-based retrieval ranks stored in the population; in ``topk_negatives`` mode they
# exceed 10 whenever positives crowd the top of the list.
RANK_BINS: tuple[str, ...] = ("1-3", "4-10", ">10")


def _cell(rec: dict) -> tuple[bool, bool, bool]:
    a, b, c = (bool(rec[k]) for k in BITS)
    return (a, b, c)


def _cell_str(cell: tuple[bool, bool, bool]) -> str:
    return "/".join("1" if v else "0" for v in cell)


def load_population(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Population not found: {path} — run extract_fp_population.py first.")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def join_predictions(pop: list[dict], version: str) -> tuple[list[dict], int]:
    """Attach each pair's v5 bits from the per-event store; count unclassified."""
    cache: dict[int, dict[str, dict]] = {}
    joined, missing = [], 0
    for p in pop:
        eid = int(p["event_id"])
        if eid not in cache:
            cache[eid] = load_predictions_for_event(eid)
        rec = cache[eid].get(p["id"])
        if rec is None or rec.get("prompt_version") != version \
                or not all(isinstance(rec.get(b), bool) for b in BITS):
            missing += 1
            continue
        joined.append({**p, **{b: rec[b] for b in BITS}})
    return joined, missing


def _composition(recs: list[dict]) -> dict:
    """8-cell counts/shares, per-bit marginals, and headline shares for a set of pairs."""
    n = len(recs)
    cells = Counter(_cell(r) for r in recs)
    marg = {b: sum(1 for r in recs if r[b]) / n if n else 0.0 for b in BITS}
    # "right entities, wrong relation": instance + constant present, relation absent.
    right_ent_wrong_rel = cells.get((True, False, True), 0) / n if n else 0.0
    all_present = cells.get((True, True, True), 0) / n if n else 0.0
    # Both queried entity slots filled (instance AND constant), any relation bit.
    inst_and_const = (cells.get((True, True, True), 0)
                      + cells.get((True, False, True), 0)) / n if n else 0.0
    return {
        "n": n,
        "cells": {c: cells.get(c, 0) for c in CELLS},
        "shares": {c: (cells.get(c, 0) / n if n else 0.0) for c in CELLS},
        "marginals": marg,
        "right_entities_wrong_relation": right_ent_wrong_rel,
        "all_present": all_present,
        "instance_and_constant": inst_and_const,
    }


def composition_by(recs: list[dict], keys_fn) -> dict[str, dict]:
    """Group records by ``keys_fn`` (returns an iterable of group keys per record —
    duplicates allowed, so a pair retrieved by both systems counts one slot per
    occurrence) and compute the composition of each group."""
    groups: dict[str, list[dict]] = {}
    for r in recs:
        for k in keys_fn(r):
            groups.setdefault(k, []).append(r)
    return {k: _composition(v) for k, v in groups.items()}


def _rank_bin(rank: int) -> str:
    if rank <= 3:
        return "1-3"
    if rank <= 10:
        return "4-10"
    return ">10"


def _write_breakdown_csv(path: Path, comps: dict[str, dict], order: tuple[str, ...]) -> None:
    """Same row schema as subset_composition.csv, plus the derived headline rows."""
    with path.open("w") as fh:
        fh.write("group,n,metric,value_count,value_share\n")
        for name in order:
            if name not in comps:
                continue
            comp = comps[name]
            for c in CELLS:
                fh.write(f"{name},{comp['n']},cell_{_cell_str(c)},"
                         f"{comp['cells'][c]},{comp['shares'][c]:.4f}\n")
            for b in BITS:
                fh.write(f"{name},{comp['n']},marg_{b},,{comp['marginals'][b]:.4f}\n")
            fh.write(f"{name},{comp['n']},right_entities_wrong_relation,,"
                     f"{comp['right_entities_wrong_relation']:.4f}\n")
            fh.write(f"{name},{comp['n']},all_present,,{comp['all_present']:.4f}\n")
            fh.write(f"{name},{comp['n']},instance_and_constant,,"
                     f"{comp['instance_and_constant']:.4f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population",
                    default=str(FP_TAXONOMY_DATA_PATH / "subset_2sys" / "subset_population.jsonl"))
    ap.add_argument("--version", default=FP_TAXONOMY_PROMPT_VERSION)
    ap.add_argument("--out-dir", default=str(SUMMARY_FP_TAXONOMY_PATH / "subset_2sys"))
    args = ap.parse_args()

    pop = load_population(Path(args.population))
    joined, missing = join_predictions(pop, args.version)
    if not joined:
        raise SystemExit(f"No pairs classified under {args.version} yet — fetch the batch first.")

    systems = sorted({a for p in pop for a in p["systems"]})
    groups: dict[str, list[dict]] = {"union": joined}
    for s in systems:
        groups[s] = [r for r in joined if s in r["systems"]]
    comps = {name: _composition(recs) for name, recs in groups.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV: one row per (group, cell) + marginal/headline rows ------------------
    csv_path = out_dir / "subset_composition.csv"
    with csv_path.open("w") as fh:
        fh.write("group,n,metric,value_count,value_share\n")
        for name, comp in comps.items():
            for c in CELLS:
                fh.write(f"{name},{comp['n']},cell_{_cell_str(c)},"
                         f"{comp['cells'][c]},{comp['shares'][c]:.4f}\n")
            for b in BITS:
                fh.write(f"{name},{comp['n']},marg_{b},,"
                         f"{comp['marginals'][b]:.4f}\n")
            fh.write(f"{name},{comp['n']},right_entities_wrong_relation,,"
                     f"{comp['right_entities_wrong_relation']:.4f}\n")

    # ---- breakdown CSVs: union × R bucket, (pair, system) slots × rank bin ---------
    rb_comps = composition_by(joined, lambda r: (r["r_bucket"],))
    rank_comps = composition_by(
        joined, lambda r: tuple(_rank_bin(int(k)) for k in r["systems"].values()))
    rb_csv = out_dir / "subset_by_rbucket.csv"
    rank_csv = out_dir / "subset_by_rank.csv"
    _write_breakdown_csv(rb_csv, rb_comps, R_BUCKET_ORDER)
    _write_breakdown_csv(rank_csv, rank_comps, RANK_BINS)

    # ---- REPORT.md ----------------------------------------------------------------
    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)

    emit("# FP composition — two flagship systems (10 highest-scored negatives)")
    emit()
    emit(f"Judge: gpt-5.2:medium × prompt {args.version}. Population: "
         f"`{Path(args.population).name}` ({len(pop):,} unique pairs; classified "
         f"{len(joined):,}" + (f", {missing:,} missing" if missing else "") + ").")
    emit("Per system = that system's 10 highest-scored judged-negative paragraphs per "
         "query (its own top-10 negatives); union = deduplicated across the two.")
    emit()

    disp = {"colbert-v2": "ColBERTv2 (late-interaction, top nDCG@10)",
            "ICT-TIME-and-Querit-embedding-v1": "ICT-Querit-v1 (dense, top nDCG@10)",
            "union": "Union (dedup)"}
    order = ["colbert-v2", "ICT-TIME-and-Querit-embedding-v1", "union"]
    order = [g for g in order if g in comps] + [g for g in comps if g not in order]

    emit("## Per-bit marginals (share of the system's top-10 negatives containing each component)")
    emit()
    emit("| system | n | instance | relation | constant |")
    emit("|---|---|---|---|---|")
    for g in order:
        c = comps[g]
        emit(f"| {disp.get(g, g)} | {c['n']:,} | {c['marginals']['instance_present']:.1%} "
             f"| {c['marginals']['relation_present']:.1%} "
             f"| {c['marginals']['constant_present']:.1%} |")
    emit()

    emit("## 8-cell composition (instance/relation/constant)")
    emit()
    header = "| cell | " + " | ".join(disp.get(g, g).split(" (")[0] for g in order) + " |"
    emit(header)
    emit("|" + "---|" * (len(order) + 1))
    for cell in CELLS:
        row = [f"{comps[g]['shares'][cell]:.1%}" for g in order]
        emit(f"| {_cell_str(cell)} — {CELL_GLOSS[cell]} | " + " | ".join(row) + " |")
    emit()

    emit("## Headline")
    emit()
    for g in order:
        c = comps[g]
        emit(f"- **{disp.get(g, g)}**: right-entities-wrong-relation (1/0/1) "
             f"{c['right_entities_wrong_relation']:.1%}; all-three-present (1/1/1) "
             f"{c['all_present']:.1%}; both-entity-slots-present (1/1/1 + 1/0/1) "
             f"{c['instance_and_constant']:.1%}; relation absent overall "
             f"{1 - c['marginals']['relation_present']:.1%}.")
    emit()

    def emit_breakdown(title: str, note: str, comps_by: dict[str, dict],
                       group_order: tuple[str, ...], group_col: str) -> None:
        emit(f"## {title}")
        emit()
        emit(note)
        emit()
        emit(f"| {group_col} | n | " + " | ".join(_cell_str(c) for c in CELLS)
             + " | I∧C |")
        emit("|" + "---|" * (len(CELLS) + 3))
        for g in group_order:
            if g not in comps_by:
                continue
            c = comps_by[g]
            cells_row = " | ".join(f"{c['shares'][cell]:.1%}" for cell in CELLS)
            emit(f"| {g} | {c['n']:,} | {cells_row} | {c['instance_and_constant']:.1%} |")
        emit()

    emit_breakdown(
        "Composition × R bucket (union pairs)",
        "Cell columns follow the fixed order above (instance/relation/constant); "
        "I∧C = both queried entity slots present (1/1/1 + 1/0/1).",
        rb_comps, R_BUCKET_ORDER, "R bucket")
    emit_breakdown(
        "Composition × first-stage rank (per-system slots)",
        "One slot per (pair, system) placement; ranks are the raw 1-based retrieval "
        "ranks, so in `topk_negatives` mode the >10 bin holds negatives that sit "
        "below interleaved positives — it exists only for queries whose top ranks "
        "contain positives (higher-R_q queries), which confounds it with R_q.",
        rank_comps, RANK_BINS, "rank bin")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

    # ---- console ------------------------------------------------------------------
    print(f"Classified {len(joined):,}/{len(pop):,} pairs (v{args.version}); "
          f"{missing:,} missing.")
    for g in order:
        c = comps[g]
        print(f"\n[{g}] n={c['n']:,}  instance {c['marginals']['instance_present']:.0%} · "
              f"relation {c['marginals']['relation_present']:.0%} · "
              f"constant {c['marginals']['constant_present']:.0%}")
        for cell in CELLS:
            if c["shares"][cell] > 0:
                print(f"    {_cell_str(cell)} {c['shares'][cell]:5.1%}  {CELL_GLOSS[cell]}")
    print(f"\n[rank bins]  " + "  ".join(
        f"{b}: n={rank_comps[b]['n']:,} I∧C {rank_comps[b]['instance_and_constant']:.1%}"
        for b in RANK_BINS if b in rank_comps))
    print(f"[R buckets]  " + "  ".join(
        f"{b}: I∧C {rb_comps[b]['instance_and_constant']:.1%}"
        for b in R_BUCKET_ORDER if b in rb_comps))
    print(f"\n-> {out_dir / 'REPORT.md'}\n-> {csv_path}\n-> {rb_csv}\n-> {rank_csv}")


if __name__ == "__main__":
    main()
