"""Roll up the FP-taxonomy classifications into the paper's composition tables.

Joins the deduplicated FP population (``data/fp_taxonomy/fp_population.jsonl``) with
the per-event taxonomy predictions (``data/fp_taxonomy/<qid>/fp_taxonomy_<qid>.jsonl``)
and writes, under ``summary/fp_taxonomy/``:

  fp_composition_by_system.csv   per-system category shares of its top-10 FP slots
                                 (systems ordered by nDCG@10), + wrong_relation
                                 sub-flag split and coverage counts
  fp_composition_overall.csv     overall + dense vs late-interaction rows, at both
                                 the unique-pair level (each mistake once) and the
                                 slot level (weighted by how many systems made it)
  fp_composition_by_r_bucket.csv category shares per R_q bucket (unique pairs)
  REPORT.md                      headline numbers + a short interpretation

Slot-level = one count per (system, query, paragraph) top-10 occurrence: what a
user actually sees in a ranking. Unique-pair level = one count per distinct
(query, paragraph) mistake. Both are reported; the paper's per-system figure is
slot-level by construction (a system's own top-10).

    uv run python src/retrieval/lancedb/frontier-v2/evaluation/analysis_fp_taxonomy.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import _bootstrap  # noqa: F401
import eval_config as cfg
from config import FP_TAXONOMY_DATA_PATH, SUMMARY_FP_TAXONOMY_PATH

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (  # noqa: E402
    CATEGORIES,
    FP_TAXONOMY_PROMPT_VERSION,
    SUB_FLAGS,
)

POPULATION_PATH = FP_TAXONOMY_DATA_PATH / "fp_population.jsonl"


def _load_population() -> list[dict]:
    return [json.loads(l) for l in POPULATION_PATH.read_text().splitlines() if l.strip()]


def _load_taxonomy(version: str) -> dict[tuple[int, str], dict]:
    """(event_id, chunk_id) -> prediction record, current prompt version only."""
    out: dict[tuple[int, str], dict] = {}
    for d in sorted(FP_TAXONOMY_DATA_PATH.iterdir()):
        if not (d.is_dir() and d.name.isdigit()):
            continue
        f = d / f"fp_taxonomy_{d.name}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("prompt_version") == version:
                out[(int(r["event_id"]), r["id"])] = r
    return out


def _ndcg10_order() -> dict[str, float]:
    """alias -> nDCG@10 from the published leaderboard."""
    out: dict[str, float] = {}
    with (cfg.SUMMARY_DIR / "leaderboard.csv").open() as fh:
        for row in csv.DictReader(fh):
            out[row["model"]] = float(row["ndcg@10"])
    return out


def _shares(counts: Counter, n: int) -> dict[str, float]:
    return {c: (counts.get(c, 0) / n if n else 0.0) for c in CATEGORIES}


def _wr_flag_counts(recs: list[dict]) -> Counter:
    return Counter(r.get("sub_flag") or "null"
                   for r in recs if r["category"] == "wrong_relation")


def main(version: str = FP_TAXONOMY_PROMPT_VERSION) -> None:
    pop = _load_population()
    tax = _load_taxonomy(version)
    ndcg = _ndcg10_order()
    li_aliases = {m.alias for m in cfg.MODELS if m.retriever_type == "late_interaction"}

    # join
    joined: list[dict] = []
    n_missing = 0
    for p in pop:
        rec = tax.get((p["event_id"], p["id"]))
        if rec is None:
            n_missing += 1
            continue
        joined.append({**p, "category": rec["category"], "sub_flag": rec["sub_flag"]})
    if not joined:
        raise SystemExit(f"No classified pairs under prompt version {version}.")

    SUMMARY_FP_TAXONOMY_PATH.mkdir(parents=True, exist_ok=True)

    # ---- per-system (slot-level: each system counts every pair in its own top-10) --
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in joined:
        for alias in r["systems"]:
            by_system[alias].append(r)

    sys_rows: list[dict] = []
    for alias in sorted(cfg.MODEL_ALIASES, key=lambda a: -ndcg.get(a, 0.0)):
        recs = by_system.get(alias, [])
        n = len(recs)
        cats = Counter(r["category"] for r in recs)
        flags = _wr_flag_counts(recs)
        row = {
            "model": alias,
            "retriever_type": cfg.MODEL_BY_ALIAS[alias].retriever_type,
            "ndcg@10": round(ndcg.get(alias, float("nan")), 4),
            "n_fp_slots_classified": n,
            **{f"share_{c}": round(v, 4) for c, v in _shares(cats, n).items()},
            **{f"share_wr_{f}": round(flags.get(f, 0) / n, 4) if n else 0.0
               for f in SUB_FLAGS},
        }
        sys_rows.append(row)
    with (SUMMARY_FP_TAXONOMY_PATH / "fp_composition_by_system.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sys_rows[0].keys()))
        w.writeheader()
        w.writerows(sys_rows)

    # ---- overall (unique-pair and slot level), overall + by family ----------------
    def _overall_row(name: str, recs: list[dict], weighted: bool) -> dict:
        if weighted:  # slot level: one count per (system, pair) occurrence
            expanded: list[dict] = []
            for r in recs:
                expanded.extend([r] * len(r["systems"]))
            recs = expanded
        n = len(recs)
        cats = Counter(r["category"] for r in recs)
        flags = _wr_flag_counts(recs)
        return {
            "population": name,
            "level": "slots" if weighted else "unique_pairs",
            "n": n,
            **{f"share_{c}": round(v, 4) for c, v in _shares(cats, n).items()},
            **{f"share_wr_{f}": round(flags.get(f, 0) / n, 4) if n else 0.0
               for f in SUB_FLAGS},
        }

    li_recs = [r for r in joined if any(a in li_aliases for a in r["systems"])]
    dense_recs = [r for r in joined
                  if any(a not in li_aliases for a in r["systems"])]
    li_slot = [dict(r, systems={a: k for a, k in r["systems"].items() if a in li_aliases})
               for r in joined if any(a in li_aliases for a in r["systems"])]
    dense_slot = [dict(r, systems={a: k for a, k in r["systems"].items() if a not in li_aliases})
                  for r in joined if any(a not in li_aliases for a in r["systems"])]
    overall_rows = [
        _overall_row("all", joined, weighted=False),
        _overall_row("all", joined, weighted=True),
        _overall_row("late_interaction", li_recs, weighted=False),
        _overall_row("late_interaction", li_slot, weighted=True),
        _overall_row("dense", dense_recs, weighted=False),
        _overall_row("dense", dense_slot, weighted=True),
    ]
    with (SUMMARY_FP_TAXONOMY_PATH / "fp_composition_overall.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(overall_rows[0].keys()))
        w.writeheader()
        w.writerows(overall_rows)

    # ---- by R_q bucket (unique pairs) ----------------------------------------------
    bucket_rows = []
    order = {f"{lo}-{hi}" if hi else f"{lo}+": i for i, (lo, hi) in enumerate(cfg.R_BUCKETS)}
    for bucket in sorted({r["r_bucket"] for r in joined}, key=lambda b: order.get(b, 99)):
        recs = [r for r in joined if r["r_bucket"] == bucket]
        cats = Counter(r["category"] for r in recs)
        bucket_rows.append({
            "r_bucket": bucket, "n_unique_pairs": len(recs),
            **{f"share_{c}": round(v, 4) for c, v in _shares(cats, len(recs)).items()},
        })
    with (SUMMARY_FP_TAXONOMY_PATH / "fp_composition_by_r_bucket.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bucket_rows[0].keys()))
        w.writeheader()
        w.writerows(bucket_rows)

    # ---- REPORT.md -----------------------------------------------------------------
    all_u = overall_rows[0]
    all_s = overall_rows[1]
    wr_u = all_u["share_wrong_relation"]
    entity_u = all_u["share_wrong_relation"] + all_u["share_constant_only"]
    lines = [
        f"# False-positive taxonomy — composition report ({version})",
        "",
        f"Population: {len(pop):,} unique (query, paragraph) pairs "
        f"(top-10 judged-negative slots of all {len(cfg.MODEL_ALIASES)} systems, "
        f"deduplicated); classified: {len(joined):,}"
        + (f"; missing: {n_missing:,}" if n_missing else "") + ".",
        "",
        "## Headline (unique pairs)",
        "",
    ]
    for c in CATEGORIES:
        lines.append(f"- {c}: **{all_u[f'share_{c}']:.1%}**"
                     + (f" (different relation asserted "
                        f"{all_u['share_wr_different_relation_asserted']:.1%}, "
                        f"no relation asserted "
                        f"{all_u['share_wr_no_relation_asserted']:.1%})"
                        if c == "wrong_relation" else ""))
    lines += [
        "",
        f"Slot-level (what a top-10 actually contains): wrong_relation "
        f"{all_s['share_wrong_relation']:.1%}, wrong_entities "
        f"{all_s['share_wrong_entities']:.1%}, constant_only "
        f"{all_s['share_constant_only']:.1%}, looks_positive "
        f"{all_s['share_looks_positive']:.1%}, off_target "
        f"{all_s['share_off_target']:.1%}.",
        "",
        "## Interpretation",
        "",
        f"Right-entities-wrong-relation accounts for {wr_u:.1%} of distinct top-10 "
        f"false positives; together with constant-entity-only paragraphs "
        f"({all_u['share_constant_only']:.1%}), entity-anchored confusions cover "
        f"{entity_u:.1%}. wrong_entities ({all_u['share_wrong_entities']:.1%}) and "
        f"off_target ({all_u['share_off_target']:.1%}) make up the rest; "
        f"looks_positive ({all_u['share_looks_positive']:.1%}) is the label-noise "
        f"audit rate, reported separately.",
        "",
        "See fp_composition_by_system.csv for the per-system breakdown "
        "(figure: paper/figures/fig_fp_taxonomy.pdf).",
        "",
    ]
    (SUMMARY_FP_TAXONOMY_PATH / "REPORT.md").write_text("\n".join(lines))

    print(f"Joined {len(joined):,}/{len(pop):,} pairs (version {version}; "
          f"{n_missing:,} unclassified).")
    print(f"Unique-pair shares: " + " · ".join(
        f"{c} {all_u[f'share_{c}']:.1%}" for c in CATEGORIES))
    print(f"-> {SUMMARY_FP_TAXONOMY_PATH}/")


if __name__ == "__main__":
    main()
