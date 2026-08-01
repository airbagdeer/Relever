"""Reproduce the FP-judge selection significance tests (McNemar, exact).

The judge-selection prose in ``summary/fp_taxonomy/model_compare.md`` records three
paired comparisons of exact-8-cell correctness against the 100-pair human gold:

  * gpt-5.2 x v5 vs gpt-5.6-luna x v5   (expected discordants 20 vs 5,  p = 0.0041)
  * gpt-5.2 x v5 vs gpt-5.6-luna x v4   (expected discordants 20 vs 8,  p = 0.0357)
  * gpt-5.2 x v5 vs gpt-5.2 x v3        (prompt iteration, 14 vs 7,     p ~= 0.19)

This script recomputes them from the per-pair sync outputs under
``data/fp_taxonomy/validation/model_compare/`` (McNemar's paired test, exact two-sided
binomial on the discordant pairs — same construction as ``agreement_significance.py``)
and HARD-FAILS if any reproduced discordant count differs from the documented one,
so the paper's appendix numbers stay regenerable.

Outputs ``summary/fp_taxonomy/model_compare_significance.{md,csv}``.

    uv run python src/create_dataset/classifying_data/fp_mcnemar.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.create_dataset.classifying_data.batch_api.label_fp_taxonomy_batch import (  # noqa: E402
    PILOT_DIR,
)
from src.create_dataset.classifying_data.fp_taxonomy_agreement import (  # noqa: E402
    load_gold,
)
from src.create_dataset.classifying_data.fp_taxonomy_prompt import BITS  # noqa: E402
from src.retrieval.config import SUMMARY_FP_TAXONOMY_PATH  # noqa: E402

MODEL_COMPARE_DIR = PILOT_DIR / "model_compare"

# (config_a, config_b, expected (a_only, b_only) discordants from model_compare.md).
COMPARISONS: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("gpt-5.2__v5", "gpt-5.6-luna__v5", (20, 5)),
    ("gpt-5.2__v5", "gpt-5.6-luna__v4", (20, 8)),
    ("gpt-5.2__v5", "gpt-5.2__v3", (14, 7)),
)


def _cell(rec: dict) -> tuple[bool, ...]:
    return tuple(bool(rec[b]) for b in BITS)


def load_config_cells(name: str) -> dict[str, tuple[bool, ...]]:
    path = MODEL_COMPARE_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing per-pair output: {path}")
    cells: dict[str, tuple[bool, ...]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            cells[rec["id"]] = _cell(rec)
    return cells


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact binomial p over the discordant pairs (McNemar's test)."""
    n_disc = a_only + b_only
    if n_disc == 0:
        return 1.0
    return binomtest(min(a_only, b_only), n_disc, 0.5).pvalue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=str(PILOT_DIR / "fp_human__gold.jsonl"))
    args = ap.parse_args()

    gold = load_gold(Path(args.gold))
    gold_cells = {g["id"]: _cell(g) for g in gold}
    if not gold_cells:
        raise SystemExit(f"No gold rows loaded from {args.gold}")

    rows: list[dict] = []
    failures: list[str] = []
    for cfg_a, cfg_b, (exp_a, exp_b) in COMPARISONS:
        cells_a = load_config_cells(cfg_a)
        cells_b = load_config_cells(cfg_b)
        ids = [i for i in gold_cells if i in cells_a and i in cells_b]
        both = a_only = b_only = neither = 0
        for i in ids:
            ok_a = cells_a[i] == gold_cells[i]
            ok_b = cells_b[i] == gold_cells[i]
            both += ok_a and ok_b
            a_only += ok_a and not ok_b
            b_only += ok_b and not ok_a
            neither += not ok_a and not ok_b
        p = mcnemar_exact(a_only, b_only)
        rows.append({
            "config_a": cfg_a, "config_b": cfg_b, "n": len(ids),
            "both_correct": both, "a_only": a_only, "b_only": b_only,
            "both_wrong": neither, "p_exact": p,
        })
        if (a_only, b_only) != (exp_a, exp_b):
            failures.append(f"{cfg_a} vs {cfg_b}: reproduced discordants "
                            f"{a_only}/{b_only}, documented {exp_a}/{exp_b}")

    md = ["# FP judge selection — McNemar exact tests (reproduction)", "",
          "Exact-8-cell correctness vs the 100-pair human gold; McNemar's paired test,",
          "two-sided exact binomial over the discordant pairs. Reproduces the numbers",
          "recorded in `model_compare.md` (script: `fp_mcnemar.py`).", "",
          "| config A | config B | n | both correct | A only | B only | both wrong | p (exact) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config_a']} | {r['config_b']} | {r['n']} | {r['both_correct']} "
                  f"| {r['a_only']} | {r['b_only']} | {r['both_wrong']} | {r['p_exact']:.4f} |")
        print(f"{r['config_a']} vs {r['config_b']}: {r['a_only']}/{r['b_only']} discordant, "
              f"p = {r['p_exact']:.4f} (n={r['n']})")

    SUMMARY_FP_TAXONOMY_PATH.mkdir(parents=True, exist_ok=True)
    md_path = SUMMARY_FP_TAXONOMY_PATH / "model_compare_significance.md"
    md_path.write_text("\n".join(md) + "\n")
    csv_path = SUMMARY_FP_TAXONOMY_PATH / "model_compare_significance.csv"
    with csv_path.open("w") as fh:
        fh.write("config_a,config_b,n,both_correct,a_only,b_only,both_wrong,p_exact\n")
        for r in rows:
            fh.write(f"{r['config_a']},{r['config_b']},{r['n']},{r['both_correct']},"
                     f"{r['a_only']},{r['b_only']},{r['both_wrong']},{r['p_exact']:.6f}\n")
    print(f"\n-> {md_path}\n-> {csv_path}")

    if failures:
        raise SystemExit("SELF-CHECK FAILED — reproduced counts diverge from "
                         "model_compare.md:\n  " + "\n  ".join(failures))
    print("Self-check OK: all discordant counts match model_compare.md.")


if __name__ == "__main__":
    main()
