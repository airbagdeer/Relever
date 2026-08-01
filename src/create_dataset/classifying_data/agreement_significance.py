"""Significance testing for labeler agreement with the frozen frontier consensus.

The n=100/200 sample is small, so a few-paragraph swing in %agree_consensus can be
noise. This module quantifies that:

  1. Single-run: each candidate labeler's agreement with the frozen consensus
     (CONSENSUS_LABELERS majority vote), with a bootstrap 95% CI.
  2. Paired (a baseline dir vs a variant dir, e.g. short vs decomposed prompt): for
     every config present in BOTH, McNemar's exact test on the paired per-paragraph
     correctness vectors (vs the same consensus) + a bootstrap CI on the agreement
     delta. This is the headline test for "did the decomposed prompt help?".

Reuses frontier_cost_report's majority_vote / config_key / CONSENSUS_LABELERS so the
gold is computed identically to the report.

Usage:
    # single-run CIs for the base experiment
    uv run python -m src.create_dataset.classifying_data.agreement_significance \
        --baseline-dir data/judge_validation/short
    # paired short-vs-decomposed (consensus gold taken from the base dir)
    uv run python -m src.create_dataset.classifying_data.agreement_significance \
        --baseline-dir data/judge_validation/short \
        --variant-dir data/judge_validation/decomposed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from . import frontier_cost_report as report_mod
except ImportError:  # run as a plain script
    import frontier_cost_report as report_mod  # type: ignore


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_preds(path: Path) -> dict[str, int]:
    preds: dict[str, int] = {}
    if not path.exists():
        return preds
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("prediction") in (0, 1):
            preds[r["id"]] = int(r["prediction"])
    return preds


def load_labelers(raw_dir: Path) -> dict[str, dict[str, int]]:
    """{labeler_key -> {id -> 0/1}} for every config jsonl + opus in a directory."""
    labelers: dict[str, dict[str, int]] = {}
    for f in sorted(raw_dir.glob("*__*.jsonl")):
        if f.name.startswith("batch_"):
            continue
        model, _, effort = f.stem.rpartition("__")
        if not model:
            continue
        preds = _read_preds(f)
        if preds:
            labelers[report_mod.config_key(model, effort)] = preds
    opus = _read_preds(raw_dir / f"{report_mod.OPUS_LABELER}.jsonl")
    if opus:
        labelers[report_mod.OPUS_LABELER] = opus
    return labelers


def build_consensus(consensus_dir: Path, names: list[str]) -> dict[str, int]:
    """Frozen majority vote over the CONSENSUS_LABELERS present in consensus_dir."""
    labelers = load_labelers(consensus_dir)
    present = [n for n in names if n in labelers]
    missing = [n for n in names if n not in labelers]
    if missing:
        print(f"WARNING: consensus labelers missing from {consensus_dir}: {missing}")
    return report_mod.majority_vote(labelers, present)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def correctness(preds: dict[str, int], gold: dict[str, int]) -> dict[str, int]:
    """{id -> 1 if preds matches gold else 0} over the shared ids."""
    return {i: int(preds[i] == gold[i]) for i in (set(preds) & set(gold))}


def bootstrap_ci_mean(values: list[int], B: int, rng: np.random.Generator) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    means = arr[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(np.percentile(means, 2.5) * 100), float(np.percentile(means, 97.5) * 100)


def bootstrap_ci_delta(
    base: list[int], var: list[int], B: int, rng: np.random.Generator
) -> tuple[float, float]:
    b = np.asarray(base, dtype=float)
    v = np.asarray(var, dtype=float)
    n = b.size
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(B, n))
    deltas = v[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(deltas, 2.5) * 100), float(np.percentile(deltas, 97.5) * 100)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact (binomial) McNemar p-value on discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "<1e-4"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def fmt_sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def single_run_table(
    labelers: dict[str, dict[str, int]], gold: dict[str, int], B: int, rng: np.random.Generator,
    consensus_names: set[str],
) -> str:
    rows = []
    for name, preds in labelers.items():
        c = correctness(preds, gold)
        if not c:
            continue
        vals = list(c.values())
        agree = 100.0 * sum(vals) / len(vals)
        lo, hi = bootstrap_ci_mean(vals, B, rng)
        tag = " (consensus member)" if name in consensus_names else ""
        rows.append((name + tag, len(vals), agree, lo, hi))
    rows.sort(key=lambda r: r[2], reverse=True)
    md = [
        "## Agreement with the frozen consensus (single run, bootstrap 95% CI)\n",
        "| labeler | n | %agree_consensus | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for name, n, agree, lo, hi in rows:
        md.append(f"| {name} | {n} | {agree:.1f} | [{lo:.1f}, {hi:.1f}] |")
    md.append("")
    return "\n".join(md)


def paired_table(
    base: dict[str, dict[str, int]], var: dict[str, dict[str, int]], gold: dict[str, int],
    B: int, rng: np.random.Generator,
) -> str:
    md = [
        "## Paired comparison — baseline vs variant (McNemar + bootstrap Δ CI)\n",
        "For each config in BOTH runs, per-paragraph correctness vs the same frozen "
        "consensus, on the ids both runs share. Δ = variant − baseline (pp). "
        "McNemar exact two-sided on discordant pairs.\n",
        "| config | n | baseline %agree | variant %agree | Δ pp | 95% CI (Δ) | McNemar p | sig |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    shared_configs = sorted(set(base) & set(var))
    if not shared_configs:
        md.append("| _(no configs present in both dirs)_ | | | | | | | |")
        return "\n".join(md) + "\n"
    for cfg in shared_configs:
        bc = correctness(base[cfg], gold)
        vc = correctness(var[cfg], gold)
        ids = sorted(set(bc) & set(vc))
        if not ids:
            continue
        bvec = [bc[i] for i in ids]
        vvec = [vc[i] for i in ids]
        n = len(ids)
        base_a = 100.0 * sum(bvec) / n
        var_a = 100.0 * sum(vvec) / n
        delta = var_a - base_a
        # discordant pairs: b = base-correct & var-wrong; c = base-wrong & var-correct
        b = sum(1 for i in ids if bc[i] == 1 and vc[i] == 0)
        c = sum(1 for i in ids if bc[i] == 0 and vc[i] == 1)
        p = mcnemar_exact_p(b, c)
        lo, hi = bootstrap_ci_delta(bvec, vvec, B, rng)
        md.append(
            f"| {cfg} | {n} | {base_a:.1f} | {var_a:.1f} | {delta:+.1f} | "
            f"[{lo:+.1f}, {hi:+.1f}] | {fmt_p(p)} | {fmt_sig(p)} |"
        )
    md.append("")
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", default="data/judge_validation/short")
    ap.add_argument("--variant-dir", default=None,
                    help="if set, also run the paired baseline-vs-variant comparison")
    ap.add_argument("--consensus-dir", default=None,
                    help="dir holding the frozen consensus labels (default: baseline-dir)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output markdown path (default: summary/<dir-name>/...)")
    args = ap.parse_args()

    base_dir = (_PROJECT_ROOT / args.baseline_dir).resolve()
    consensus_dir = (_PROJECT_ROOT / (args.consensus_dir or args.baseline_dir)).resolve()
    rng = np.random.default_rng(args.seed)
    consensus_names = set(report_mod.CONSENSUS_LABELERS)

    gold = build_consensus(consensus_dir, report_mod.CONSENSUS_LABELERS)
    print(f"Frozen consensus: {len(gold)} decided ids over {report_mod.CONSENSUS_LABELERS}")

    base_labelers = load_labelers(base_dir)
    md = [
        "# Agreement significance — vs frozen frontier consensus\n",
        f"Consensus = majority vote of **{', '.join(report_mod.CONSENSUS_LABELERS)}** "
        f"({len(gold)} decided ids, from `{consensus_dir.name}`).",
        f"Bootstrap resamples: {args.bootstrap}. Significance: `***` p<0.001, `**` p<0.01, "
        "`*` p<0.05, `ns` not significant.\n",
        single_run_table(base_labelers, gold, args.bootstrap, rng, consensus_names),
    ]

    target_dir = base_dir
    if args.variant_dir:
        var_dir = (_PROJECT_ROOT / args.variant_dir).resolve()
        var_labelers = load_labelers(var_dir)
        md.append(f"_Baseline dir: `{base_dir.name}` · Variant dir: `{var_dir.name}`_\n")
        md.append(paired_table(base_labelers, var_labelers, gold, args.bootstrap, rng))
        target_dir = var_dir

    if args.out:
        out_path = (_PROJECT_ROOT / args.out).resolve()
    else:
        rel = target_dir.relative_to(_PROJECT_ROOT / "data")
        out_path = _PROJECT_ROOT / "summary" / rel / "AGREEMENT_SIGNIFICANCE.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
