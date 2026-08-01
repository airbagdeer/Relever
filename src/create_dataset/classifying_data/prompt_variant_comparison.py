"""Compare gpt-5.2:medium prompt variants against the human gold.

Variants (each a per-id 0/1 jsonl produced by estimate_frontier_cost.py):
- short      — data/judge_validation/short/gpt-5.2__medium.jsonl        (baseline)
- decomposed — data/judge_validation/decomposed/gpt-5.2__medium.jsonl
- short2     — data/judge_validation/short2/gpt-5.2__medium.jsonl (hardened)
- short2b    — data/judge_validation/short2b/gpt-5.2__medium.jsonl (+lexical guard)

Reports %agree_human (and Cohen kappa) on the FULL 200-pair gold, on the paper's
tuning/held-out halves (--split-at 100: first 100 gold pairs in annotation order =
tuning, the rest = held-out), and on the FROZEN contested subset
(summary/judge_validation/short/contested_ids.json, written by disagreement_analysis.py),
plus paired McNemar exact tests + bootstrap CIs for short -> short2b (the headline
change) on each of those subsets.

Usage:
    uv run python src/create_dataset/classifying_data/prompt_variant_comparison.py
    uv run python .../prompt_variant_comparison.py --split-at 0   # disable the split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.config import PROJECT_ROOT, SUMMARY_PATH  # noqa: E402

try:
    from . import frontier_cost_report as report_mod
    from . import agreement_significance as sig
except ImportError:  # run as a plain script
    import frontier_cost_report as report_mod  # type: ignore
    import agreement_significance as sig  # type: ignore

REPORT_DIR = SUMMARY_PATH / "judge_validation" / "short"
CONTESTED_PATH = REPORT_DIR / "contested_ids.json"
HUMAN_PATH = PROJECT_ROOT / "data/judge_validation/short/human__gold.jsonl"

VARIANTS = {
    "short (baseline)": PROJECT_ROOT / "data/judge_validation/short/gpt-5.2__medium.jsonl",
    "decomposed": PROJECT_ROOT / "data/judge_validation/decomposed/gpt-5.2__medium.jsonl",
    "short2 (hardened)": PROJECT_ROOT / "data/judge_validation/short2/gpt-5.2__medium.jsonl",
    "short2b (+lexical guard)": PROJECT_ROOT / "data/judge_validation/short2b/gpt-5.2__medium.jsonl",
}

# Per-variant result directories (drives the open-source models x variants section).
VARIANT_DIRS = {
    "short": PROJECT_ROOT / "data/judge_validation/short",
    "short2": PROJECT_ROOT / "data/judge_validation/short2",
    "short2b": PROJECT_ROOT / "data/judge_validation/short2b",
    "decomposed": PROJECT_ROOT / "data/judge_validation/decomposed",
}
OS_SUFFIX = "__open_source.jsonl"


def load_preds(path: Path) -> dict[str, int]:
    return {
        r["id"]: int(r["prediction"])
        for r in report_mod._read_jsonl(path)
        if r.get("prediction") in (0, 1)
    }


def load_human() -> dict[str, int]:
    """Gold labels in file order == annotation order (the split relies on this)."""
    return {
        r["id"]: int(r["prediction"])
        for r in report_mod._read_jsonl(HUMAN_PATH)
        if r.get("prediction") in (0, 1)
    }


def split_gold(gold: dict[str, int], n: int) -> tuple[set[str], set[str]]:
    """(tuning, held-out) id sets: the first n gold pairs in file order vs the rest."""
    ids = list(gold)
    return set(ids[:n]), set(ids[n:])


def kappa(preds: dict[str, int], gold: dict[str, int], subset: set[str] | None) -> float:
    ids = sorted(set(preds) & set(gold) & (subset if subset is not None else set(gold)))
    if not ids:
        return float("nan")
    return round(float(cohen_kappa_score([gold[i] for i in ids], [preds[i] for i in ids])), 3)


def agree(preds: dict[str, int], gold: dict[str, int], subset: set[str] | None) -> tuple[float, int]:
    p = preds if subset is None else {i: v for i, v in preds.items() if i in subset}
    frac, n = report_mod.pct_agreement(p, gold)
    return (round(frac * 100, 1) if n else float("nan"), n)


def mcnemar(base: dict[str, int], var: dict[str, int], gold: dict[str, int],
            subset: set[str] | None) -> dict:
    ids = set(base) & set(var) & set(gold)
    if subset is not None:
        ids &= subset
    ids = sorted(ids)
    bc = sig.correctness(base, gold)
    vc = sig.correctness(var, gold)
    bvec = [bc[i] for i in ids]
    vvec = [vc[i] for i in ids]
    b = sum(1 for i in ids if bc[i] == 1 and vc[i] == 0)  # base right, variant wrong
    c = sum(1 for i in ids if bc[i] == 0 and vc[i] == 1)  # base wrong, variant right
    rng = np.random.default_rng(0)
    lo, hi = sig.bootstrap_ci_delta(bvec, vvec, 10000, rng)
    return {
        "n": len(ids),
        "base_to_var_regressions": b,
        "base_to_var_fixes": c,
        "delta_pp": round((sum(vvec) - sum(bvec)) / len(ids) * 100, 1) if ids else float("nan"),
        "delta_ci95": (round(lo, 1), round(hi, 1)),
        "mcnemar_p": round(sig.mcnemar_exact_p(b, c), 4),
    }


def os_aliases_in(dirs) -> list[str]:
    """Open-source model aliases discovered as '<alias>__open_source.jsonl' across dirs."""
    aliases: set[str] = set()
    for d in dirs:
        if not d.exists():
            continue
        for f in d.glob(f"*{OS_SUFFIX}"):
            aliases.add(f.name[: -len(OS_SUFFIX)])
    return sorted(aliases)


def os_models_x_variants(human: dict[str, int], contested: set[str]) -> dict:
    """For every OS model x prompt variant, %agree_human on full overlap + contested.

    Returns {alias: {variant: {"full": (pct, n), "contested": (pct, n)}}}.
    """
    table: dict[str, dict] = {}
    for alias in os_aliases_in(VARIANT_DIRS.values()):
        for variant, d in VARIANT_DIRS.items():
            path = d / f"{alias}{OS_SUFFIX}"
            if not path.exists():
                continue
            preds = load_preds(path)
            table.setdefault(alias, {})[variant] = {
                "full": agree(preds, human, None),
                "contested": agree(preds, human, contested),
            }
    return table


def print_os_table(table: dict, variant_keys: list[str]) -> None:
    if not table:
        print("\n(no open-source __open_source.jsonl results found yet across variant dirs)")
        return
    print("\nOpen-source models — %agree_human  (full / contested), per prompt variant:")
    print(f"{'model':<26}" + "".join(f"{v:>20}" for v in variant_keys))
    for alias, by_var in table.items():
        row = f"{alias:<26}"
        for v in variant_keys:
            cell = by_var.get(v)
            if cell:
                (af, _), (ac, _) = cell["full"], cell["contested"]
                row += f"{f'{af}/{ac}':>20}"
            else:
                row += f"{'-':>20}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-at", type=int, default=100,
                    help="tuning/held-out split point in gold annotation order (0 disables)")
    args = ap.parse_args()

    human = load_human()
    contested = set(json.loads(CONTESTED_PATH.read_text()))
    preds = {name: load_preds(p) for name, p in VARIANTS.items()}

    tune: set[str] | None = None
    held: set[str] | None = None
    if args.split_at:
        tune, held = split_gold(human, args.split_at)
        tp, hp = sum(human[i] for i in tune), sum(human[i] for i in held)
        print(f"split at {args.split_at}: tuning {len(tune)} ({tp} pos/{len(tune) - tp} neg), "
              f"held-out {len(held)} ({hp} pos/{len(held) - hp} neg)")

    print(f"human gold: {len(human)} labels | contested set: {len(contested)} ids\n")
    header = f"{'variant':<26} {'%agree_full':>12} {'n':>4} {'kappa':>6}"
    if tune is not None:
        header += f" {'%tune':>6} {'k_tune':>7} {'%held':>6} {'k_held':>7}"
    header += f" {'%agree_contested':>18} {'n':>4}"
    print(header)
    for name, pr in preds.items():
        af, nf = agree(pr, human, None)
        ac, nc = agree(pr, human, contested)
        row = f"{name:<26} {af:>12} {nf:>4} {kappa(pr, human, None):>6}"
        if tune is not None:
            row += (f" {agree(pr, human, tune)[0]:>6} {kappa(pr, human, tune):>7}"
                    f" {agree(pr, human, held)[0]:>6} {kappa(pr, human, held):>7}")
        row += f" {ac:>18} {nc:>4}"
        print(row)

    base = preds["short (baseline)"]
    var = preds["short2b (+lexical guard)"]
    subsets: list[tuple[str, set[str] | None]] = [("pooled (all 200)", None)]
    if tune is not None:
        subsets += [("tuning half", tune), ("held-out half", held)]
    subsets.append(("contested subset", contested))
    print("\nPaired McNemar — short (baseline) -> short2b (+lexical guard), vs human gold:")
    for label, subset in subsets:
        m = mcnemar(base, var, human, subset)
        print(f"  [{label}] n={m['n']}  fixes={m['base_to_var_fixes']}  "
              f"regressions={m['base_to_var_regressions']}  "
              f"Δ={m['delta_pp']:+} pp (95% CI {m['delta_ci95'][0]:+}..{m['delta_ci95'][1]:+})  "
              f"McNemar p={m['mcnemar_p']}")

    # Open-source models x prompt variants (added to the comparison).
    os_table = os_models_x_variants(human, contested)
    print_os_table(os_table, list(VARIANT_DIRS))

    # persist a small machine-readable rollup
    rollup = {
        "n_human": len(human),
        "n_contested": len(contested),
        "split_at": args.split_at or None,
        "agree": {
            name: {
                "full": agree(pr, human, None),
                "kappa_full": kappa(pr, human, None),
                **({"tune": agree(pr, human, tune),
                    "kappa_tune": kappa(pr, human, tune),
                    "held": agree(pr, human, held),
                    "kappa_held": kappa(pr, human, held)} if tune is not None else {}),
                "contested": agree(pr, human, contested),
            } for name, pr in preds.items()
        },
        "mcnemar_short_to_short2b": {
            "pooled": mcnemar(base, var, human, None),
            **({"tune": mcnemar(base, var, human, tune),
                "held": mcnemar(base, var, human, held)} if tune is not None else {}),
            "contested": mcnemar(base, var, human, contested),
        },
        "os_models_x_variants": os_table,
    }
    out = REPORT_DIR / "prompt_variant_comparison.json"
    out.write_text(json.dumps(rollup, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
