"""Judge test-retest: self-consistency of the production judge across 3 batch re-runs.

The paper's Limitations section states "the full labeling run is single-pass, so
run-to-run judge variability is unmeasured". This report measures it: the production
labeler (``gpt-5.2`` @ ``reasoning_effort="medium"``, ``short2b`` prompt, one pair per
request, OpenAI Batch API) was re-run 3 independent times on the 200-pair validation
sample (identical request bodies, distinct batch jobs), and this script computes the
cross-run agreement plus each run's agreement with the human gold.

Inputs:
  - data/judge_validation/test_retest/{run1b,run2,run3}/predictions.jsonl (the 3 re-runs)
  - data/judge_validation/short2b/gpt-5.2__medium.jsonl  (sync run behind the paper's 96.0%)
  - data/judge_validation/test_retest/june_batch/predictions.jsonl (June batch run; secondary reference)
  - data/judge_validation/short/human__gold.jsonl        (200-pair human gold, 76 pos / 124 neg)

Outputs:
  - summary/evaluation/JUDGE_SELF_CONSISTENCY.md
  - summary/evaluation/judge_self_consistency.csv

Join convention matches the paper's 2026-07-14 decision (keep prediction in {0,1},
count ALL 200 sampled pairs — no duplicate exclusion — intersect ids), so numbers are
comparable to the paper's 96.0% / kappa 0.914 baseline; a built-in self-check aborts
if that baseline does not reproduce.

Usage:
    uv run python -m src.create_dataset.classifying_data.batch_api.test.judge_test_retest_report
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent      # .../classifying_data/batch_api/test
_PROJECT_ROOT = _PKG_DIR.parents[4]             # repo root
sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from src.create_dataset.classifying_data import frontier_cost_pricing as pricing_mod  # noqa: E402
from src.create_dataset.classifying_data.frontier_cost_report import (  # noqa: E402
    _read_jsonl,
    fleiss_over,
    majority_vote,
    pct_agreement,
)

MODEL = "gpt-5.2"
# A first third run stalled in OpenAI's batch queue (0/200 after ~12h) and was cancelled;
# run1b is its byte-identical resubmission. The analyzed trio is 1b/2/3.
RUN_TAGS = ["run1b", "run2", "run3"]

FCE_DIR = _PROJECT_ROOT / "data/judge_validation/short"
RETEST_DIR = _PROJECT_ROOT / "data/judge_validation/test_retest"
SAMPLE_PATH = FCE_DIR / "_sample.json"
GOLD_PATH = FCE_DIR / "human__gold.jsonl"
SYNC0_PATH = _PROJECT_ROOT / "data/judge_validation/short2b/gpt-5.2__medium.jsonl"
JUN18_PATH = RETEST_DIR / "june_batch/predictions.jsonl"

OUT_DIR = _PROJECT_ROOT / "summary/evaluation"
OUT_MD = OUT_DIR / "JUDGE_SELF_CONSISTENCY.md"
OUT_CSV = OUT_DIR / "judge_self_consistency.csv"

# The paper baseline this analysis must reproduce before any other number is trusted:
# sync short2b run vs human gold over ALL 200 pairs (paper convention since 2026-07-14,
# no duplicate exclusion) = 192/200, kappa 0.914.
SELF_CHECK_AGREE, SELF_CHECK_N, SELF_CHECK_KAPPA = 192, 200, 0.914

# Pre-registered decision rule (plan): results are "in favor of the paper" iff
# mean pairwise agreement >= 93% AND Fleiss' kappa >= 0.80 across the 3 runs.
FAVORABLE_MIN_PAIRWISE = 0.93
FAVORABLE_MIN_FLEISS = 0.80


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_labels(path: Path) -> tuple[dict[str, int], int]:
    """{id -> 0/1} over all 200 pairs (paper convention); also returns parsed row count."""
    rows = _read_jsonl(path)
    labels = {r["id"]: int(r["prediction"]) for r in rows if r.get("prediction") in (0, 1)}
    return labels, len(rows)


def kappa_on_shared(a: dict[str, int], b: dict[str, int]) -> tuple[float, int]:
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan"), 0
    ya = [a[i] for i in shared]
    yb = [b[i] for i in shared]
    try:
        return float(cohen_kappa_score(ya, yb)), len(shared)
    except Exception:
        return float("nan"), len(shared)


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return center - half, center + half


def run_meta(tag: str) -> dict:
    """Line counts / batch-discounted cost for one re-run.

    The raw ``batch_state.json`` (provider-side batch ids) is not distributed;
    when it is absent the id/timestamp columns degrade to "n/a".
    """
    run_dir = RETEST_DIR / tag
    state_path = run_dir / "batch_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    preds = _read_jsonl(run_dir / "predictions.jsonl")
    disc = pricing_mod.batch_discount_for(MODEL)
    return {
        "tag": tag,
        "batch_id": state.get("batch_id", "n/a"),
        "submitted_at": state.get("submitted_at", "n/a"),
        "n_lines": len(preds),
        "n_unparseable": sum(1 for p in preds if p.get("prediction") == -1),
        "batch_cost_usd": sum(p.get("cost_uncached", 0.0) for p in preds) * disc,
    }


def fmt_pct(p: float) -> str:
    return "n/a" if math.isnan(p) else f"{100 * p:.1f}%"


def fmt_k(k: float) -> str:
    return "n/a" if math.isnan(k) else f"{k:.3f}"


def main() -> None:
    # ---- load ---------------------------------------------------------------- #
    runs: dict[str, dict[str, int]] = {}
    n_lines: dict[str, int] = {}
    for tag in RUN_TAGS:
        path = RETEST_DIR / tag / "predictions.jsonl"
        if not path.exists():
            raise SystemExit(
                f"Missing {path} — run: uv run python -m "
                f"src.create_dataset.classifying_data.batch_api.test.label_batch_openai "
                f"fetch --run-tag {tag}"
            )
        runs[tag], n_lines[tag] = load_labels(path)

    sync0, _ = load_labels(SYNC0_PATH)
    jun18, jun18_lines = load_labels(JUN18_PATH)  # 1 corrupt line -> 199 of 200 parsed
    gold, _ = load_labels(GOLD_PATH)
    sample = {r["id"]: r for r in json.loads(SAMPLE_PATH.read_text())}

    # ---- self-check: reproduce the paper baseline ----------------------------- #
    p_sync, n_sync = pct_agreement(sync0, gold)
    k_sync, _ = kappa_on_shared(sync0, gold)
    agree_sync = round(p_sync * n_sync)
    if (agree_sync, n_sync) != (SELF_CHECK_AGREE, SELF_CHECK_N) or abs(k_sync - SELF_CHECK_KAPPA) > 0.0005:
        raise SystemExit(
            f"Self-check FAILED: sync0 vs gold = {agree_sync}/{n_sync}, kappa {k_sync:.4f}; "
            f"expected {SELF_CHECK_AGREE}/{SELF_CHECK_N}, kappa {SELF_CHECK_KAPPA}. "
            "Join convention no longer matches the paper baseline — do not trust the numbers."
        )
    print(f"self-check OK: sync0 vs gold = {agree_sync}/{n_sync} ({fmt_pct(p_sync)}), kappa {fmt_k(k_sync)}")

    # ---- cross-run agreement (primary) ----------------------------------------- #
    csv_rows: list[dict] = []
    pair_stats: list[tuple[str, str, float, float, int]] = []
    pooled_succ = 0
    pooled_n = 0
    for a, b in combinations(RUN_TAGS, 2):
        p, n = pct_agreement(runs[a], runs[b])
        k, _ = kappa_on_shared(runs[a], runs[b])
        pair_stats.append((a, b, p, k, n))
        pooled_succ += round(p * n)
        pooled_n += n
        csv_rows.append({"metric": "pct_agreement", "rater_a": a, "rater_b": b, "value": round(p, 4), "n": n})
        csv_rows.append({"metric": "cohen_kappa", "rater_a": a, "rater_b": b, "value": round(k, 4), "n": n})

    mean_pairwise = sum(p for _, _, p, _, _ in pair_stats) / len(pair_stats)
    lo_pool, hi_pool = wilson_ci(pooled_succ, pooled_n)
    fleiss, fleiss_n = fleiss_over(runs, RUN_TAGS)

    t_ids = sorted(set.intersection(*[set(runs[t]) for t in RUN_TAGS]))
    unanimous = [i for i in t_ids if len({runs[t][i] for t in RUN_TAGS}) == 1]
    flips = [i for i in t_ids if i not in set(unanimous)]
    unanimity = len(unanimous) / len(t_ids) if t_ids else float("nan")

    csv_rows += [
        {"metric": "mean_pairwise_pct", "rater_a": "run1+run2+run3", "rater_b": "", "value": round(mean_pairwise, 4), "n": pooled_n},
        {"metric": "wilson95_low", "rater_a": "pooled_pairs", "rater_b": "", "value": round(lo_pool, 4), "n": pooled_n},
        {"metric": "wilson95_high", "rater_a": "pooled_pairs", "rater_b": "", "value": round(hi_pool, 4), "n": pooled_n},
        {"metric": "fleiss_kappa", "rater_a": "run1+run2+run3", "rater_b": "", "value": round(fleiss, 4), "n": fleiss_n},
        {"metric": "unanimity_rate", "rater_a": "run1+run2+run3", "rater_b": "", "value": round(unanimity, 4), "n": len(t_ids)},
    ]

    # ---- vs human gold ---------------------------------------------------------- #
    majority = majority_vote(runs, RUN_TAGS)
    majority = {i: v for i, v in majority.items() if i in set(t_ids)}  # 3 votes per id, no ties
    gold_refs: list[tuple[str, dict[str, int]]] = (
        [(t, runs[t]) for t in RUN_TAGS]
        + [("majority-of-3", majority), ("sync0 (paper 96.0%/κ0.914)", sync0), ("june-batch", jun18)]
    )
    gold_stats: list[tuple[str, float, float, int]] = []
    for name, preds in gold_refs:
        p, n = pct_agreement(preds, gold)
        k, _ = kappa_on_shared(preds, gold)
        gold_stats.append((name, p, k, n))
        csv_rows.append({"metric": "pct_vs_gold", "rater_a": name, "rater_b": "human:gold", "value": round(p, 4), "n": n})
        csv_rows.append({"metric": "kappa_vs_gold", "rater_a": name, "rater_b": "human:gold", "value": round(k, 4), "n": n})

    # ---- secondary pairwise: new runs vs the two pre-existing single runs -------- #
    ref_stats: list[tuple[str, str, float, float, int]] = []
    for ref_name, ref in (("sync0", sync0), ("june-batch", jun18)):
        for t in RUN_TAGS:
            p, n = pct_agreement(runs[t], ref)
            k, _ = kappa_on_shared(runs[t], ref)
            ref_stats.append((t, ref_name, p, k, n))
            csv_rows.append({"metric": "pct_agreement", "rater_a": t, "rater_b": ref_name, "value": round(p, 4), "n": n})
            csv_rows.append({"metric": "cohen_kappa", "rater_a": t, "rater_b": ref_name, "value": round(k, 4), "n": n})
    p_sj, n_sj = pct_agreement(sync0, jun18)
    k_sj, _ = kappa_on_shared(sync0, jun18)

    # ---- flip count -------------------------------------------------------------- #
    csv_rows.append(
        {"metric": "n_flips", "rater_a": "run1+run2+run3", "rater_b": "", "value": len(flips), "n": len(t_ids)}
    )

    # ---- decision rule ----------------------------------------------------------- #
    favorable = mean_pairwise >= FAVORABLE_MIN_PAIRWISE and fleiss >= FAVORABLE_MIN_FLEISS
    verdict = "FAVORABLE" if favorable else "NOT FAVORABLE"

    # ---- write CSV ---------------------------------------------------------------- #
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)

    # ---- write MD ------------------------------------------------------------------ #
    metas = [run_meta(t) for t in RUN_TAGS]
    total_cost = sum(m["batch_cost_usd"] for m in metas)

    md: list[str] = [
        "# Judge self-consistency (test-retest, 3 re-runs)",
        "",
        f"_Generated {_now_iso()}_",
        "",
        "**Question.** The paper's Limitations state: \"the full labeling run is single-pass, so",
        "run-to-run judge variability is unmeasured.\" How stable are the production judge's labels",
        "across identical, independent runs?",
        "",
        "**Protocol.** The production judge (`gpt-5.2`, `reasoning_effort=medium`, `short2b` prompt,",
        "one pair per request, OpenAI Batch API — byte-identical request bodies, verified with `cmp`",
        "against the preserved June batch input) was re-run **3 independent times** on the 200-pair",
        "validation sample. No seed/temperature control exists for this model: cross-run variation is",
        "exactly the deployed judge's stochasticity.",
        "",
        "## Runs",
        "",
        "| run | batch id | submitted | rows | unparseable | batch cost |",
        "|---|---|---|---|---|---|",
    ]
    for m in metas:
        md.append(
            f"| {m['tag']} | `{m['batch_id']}` | {m['submitted_at']} | {m['n_lines']} "
            f"| {m['n_unparseable']} | ${m['batch_cost_usd']:.2f} |"
        )
    md += [
        f"\nTotal batch cost: **${total_cost:.2f}**.",
        "",
        "Provenance: a first third batch stalled in OpenAI's queue for ~12 h with 0/200 requests",
        "processed and was cancelled before producing any output; `run1b` is its byte-identical",
        "resubmission. No outputs from it",
        "were ever observed, so the trio above involves no selection on results.",
        "",
        "## Headline",
        "",
        f"- **Mean pairwise agreement across the 3 runs: {fmt_pct(mean_pairwise)}**"
        f" (pooled Wilson 95% CI {fmt_pct(lo_pool)}–{fmt_pct(hi_pool)}; n={pooled_n} pooled pair-comparisons)",
        f"- **Fleiss' κ (3 runs): {fmt_k(fleiss)}** (n={fleiss_n})",
        f"- **Unanimous on {fmt_pct(unanimity)}** of pairs ({len(unanimous)}/{len(t_ids)}); {len(flips)} pairs flip in at least one run",
        f"- Decision rule (pre-registered): favorable iff mean pairwise ≥ {FAVORABLE_MIN_PAIRWISE:.0%} and Fleiss' κ ≥ {FAVORABLE_MIN_FLEISS:.2f} → **{verdict}**",
        "",
        "## Pairwise agreement (the 3 re-runs)",
        "",
        "| pair | % agree | Cohen's κ | n |",
        "|---|---|---|---|",
    ]
    for a, b, p, k, n in pair_stats:
        md.append(f"| {a} × {b} | {fmt_pct(p)} | {fmt_k(k)} | {n} |")
    md += [
        "",
        "Per-pair Wilson 95% CIs (n≈199 each): "
        + "; ".join(
            f"{a}×{b} {fmt_pct(wilson_ci(round(p * n), n)[0])}–{fmt_pct(wilson_ci(round(p * n), n)[1])}"
            for a, b, p, _, n in pair_stats
        )
        + ". The three pairs share the same items, so the pooled CI understates uncertainty; treat per-pair CIs as primary.",
        "",
        "## Agreement with the human gold (200-pair sample, deduped join)",
        "",
        "| labeler | % agree | Cohen's κ | n |",
        "|---|---|---|---|",
    ]
    for name, p, k, n in gold_stats:
        md.append(f"| {name} | {fmt_pct(p)} | {fmt_k(k)} | {n} |")
    md += [
        "",
        "`majority-of-3` = per-pair majority vote of the three re-runs (3 votes each, no ties).",
        "If majority ≈ single runs, single-pass labeling loses nothing measurable vs. a 3-vote ensemble.",
        "",
        "## Reference: new runs vs the two pre-existing single runs",
        "",
        "| pair | % agree | Cohen's κ | n |",
        "|---|---|---|---|",
    ]
    for a, b, p, k, n in ref_stats:
        md.append(f"| {a} × {b} | {fmt_pct(p)} | {fmt_k(k)} | {n} |")
    md += [
        f"| sync0 × june-batch | {fmt_pct(p_sj)} | {fmt_k(k_sj)} | {n_sj} |",
        "",
        "`sync0` = the synchronous short2b run the paper's 96.0%/κ=0.914 is computed from",
        "(`data/judge_validation/short2b/gpt-5.2__medium.jsonl`). `june-batch` = the June 18",
        "batch-API smoke-test run (its `predictions.jsonl` has one corrupt line; tolerant parse ⇒",
        f"{jun18_lines} of 200 rows parsed — the file is preserved as-is).",
        "",
        "## Flips",
        "",
        f"Of {len(t_ids)} pairs judged in all three runs, {len(flips)} flip at least once:",
        "",
        "| id | event | relation | r1,r2,r3 | gold | paragraph (start) |",
        "|---|---|---|---|---|---|",
    ]
    for i in flips:
        rec = sample.get(i, {})
        rel = (rec.get("query_info") or {}).get("relation", "?")
        content = " ".join((rec.get("content") or "").split())[:60]
        votes = ",".join(str(runs[t][i]) for t in RUN_TAGS)
        md.append(
            f"| `{i[:8]}` | {rec.get('event_id', '?')} | {rel} | {votes} | {gold.get(i, '—')} "
            f"| {content}… |"
        )
    md += [
        "",
        "## Method notes",
        "",
        "- Join convention matches the paper (2026-07-14 decision): keep predictions in {0,1},",
        "  count **all 200 sampled pairs** (the earlier near-duplicate exclusion is retired),",
        "  intersect ids.",
        f"- Built-in self-check passed: sync0 vs gold reproduces {SELF_CHECK_AGREE}/{SELF_CHECK_N} = 96.0%, κ = {SELF_CHECK_KAPPA}.",
        "- Unparseable (`-1`) or missing predictions drop out of intersections; per-run row counts above.",
        "- Request bodies are byte-identical across runs and to the production/validation configuration;",
        "  only the batch job (and hence server-side sampling) differs.",
        "",
        f"CSV companion: `{OUT_CSV.relative_to(_PROJECT_ROOT)}`.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")

    # ---- stdout summary --------------------------------------------------------- #
    print(f"mean pairwise = {fmt_pct(mean_pairwise)}  (Wilson95 {fmt_pct(lo_pool)}–{fmt_pct(hi_pool)})")
    print(f"fleiss kappa  = {fmt_k(fleiss)}  (n={fleiss_n})")
    print(f"unanimity     = {fmt_pct(unanimity)}  ({len(unanimous)}/{len(t_ids)}); flips = {len(flips)}")
    for name, p, k, n in gold_stats:
        print(f"vs gold: {name:<22} {fmt_pct(p):>6}  kappa {fmt_k(k)}  (n={n})")
    print(f"verdict: {verdict}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
