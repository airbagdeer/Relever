"""Agreement metrics + human-readable report for the frontier-model cost estimate.

Reads the per-config ``.jsonl`` outputs (and the optional Claude ``opus-4.8.jsonl``
labeler) produced by ``estimate_frontier_cost.py``, then writes a cost comparison
(extrapolated to the full dataset) plus inter-labeler agreement tables.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    from . import frontier_cost_pricing as pricing_mod
except ImportError:  # run as a plain script
    import frontier_cost_pricing as pricing_mod


OPUS_LABELER = "opus-4.8"
MINIMAX_LABELER = "minimax"
# Human gold labels (manual). Dropped in as data/judge_validation/short/human__gold.jsonl,
# auto-discovered as the "human:gold" labeler; when present it adds a %agree_human column
# (every labeler scored against the human gold — the ideal reference).
HUMAN_LABELER = "human:gold"

# Frozen "gold" consensus for the %agree_consensus column. Candidate labelers
# (open-source models, decomposed-prompt runs, extra configs) are SCORED AGAINST
# this fixed set and are NEVER members of it — so adding labelers does not move
# the target. Experiment decision: the 4 frontier models @ xhigh + Opus-4.8.
CONSENSUS_LABELERS = [
    "gpt-5.1:xhigh",
    "gpt-5.2:xhigh",
    "gpt-5.4:xhigh",
    "gpt-5.5:xhigh",
    "opus-4.8",
]

# Suffix that marks an open-source (vLLM) labeler config, e.g. "glm-4.7:open_source".
OPEN_SOURCE_SUFFIX = ":open_source"

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:  # tolerate a partial trailing line if a writer is mid-append
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def config_key(model: str, effort: str) -> str:
    return f"{model}:{effort}"


def config_filename(model: str, effort: str) -> str:
    return f"{model}__{effort}.jsonl"


def load_predictions(
    raw_dir: Path, configs: list[tuple[str, str]], sample: list[dict]
) -> dict[str, dict[str, int]]:
    """Return {labeler -> {paragraph_id -> 0/1}} for every available labeler.

    Truncated / unparsed predictions (-1) are dropped so they don't pollute agreement.
    """
    labelers: dict[str, dict[str, int]] = {}

    # minimax (authoritative copy lives in the sample manifest)
    labelers[MINIMAX_LABELER] = {
        r["id"]: int(r["minimax_label"])
        for r in sample
        if str(r.get("minimax_label")) in ("0", "1")
    }

    # the GPT configs
    for model, effort in configs:
        recs = _read_jsonl(raw_dir / config_filename(model, effort))
        preds = {r["id"]: int(r["prediction"]) for r in recs if r.get("prediction") in (0, 1)}
        if preds:
            labelers[config_key(model, effort)] = preds

    # optional Claude Opus reference
    opus = _read_jsonl(raw_dir / f"{OPUS_LABELER}.jsonl")
    opus_preds = {r["id"]: int(r["prediction"]) for r in opus if r.get("prediction") in (0, 1)}
    if opus_preds:
        labelers[OPUS_LABELER] = opus_preds

    return labelers


# --------------------------------------------------------------------------- #
# Agreement metrics
# --------------------------------------------------------------------------- #
def pct_agreement(a: dict[str, int], b: dict[str, int]) -> tuple[float, int]:
    shared = set(a) & set(b)
    if not shared:
        return float("nan"), 0
    agree = sum(1 for i in shared if a[i] == b[i])
    return agree / len(shared), len(shared)


def pairwise_matrices(labelers: dict[str, dict[str, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(labelers)
    pct = pd.DataFrame(index=names, columns=names, dtype=float)
    kappa = pd.DataFrame(index=names, columns=names, dtype=float)
    for n in names:
        pct.loc[n, n] = 1.0
        kappa.loc[n, n] = 1.0
    for a, b in combinations(names, 2):
        shared = sorted(set(labelers[a]) & set(labelers[b]))
        if shared:
            ya = [labelers[a][i] for i in shared]
            yb = [labelers[b][i] for i in shared]
            p = sum(1 for x, y in zip(ya, yb) if x == y) / len(shared)
            try:
                k = float(cohen_kappa_score(ya, yb))
            except Exception:
                k = float("nan")
        else:
            p = k = float("nan")
        pct.loc[a, b] = pct.loc[b, a] = p
        kappa.loc[a, b] = kappa.loc[b, a] = k
    return pct.round(3), kappa.round(3)


def fleiss_kappa(label_matrix: np.ndarray) -> float:
    """Fleiss' kappa for an (n_items x n_raters) binary 0/1 matrix (no missing)."""
    n_items, n_raters = label_matrix.shape
    if n_items == 0 or n_raters < 2:
        return float("nan")
    n1 = label_matrix.sum(axis=1)
    n0 = n_raters - n1
    p_item = (n1 * (n1 - 1) + n0 * (n0 - 1)) / (n_raters * (n_raters - 1))
    p_bar = p_item.mean()
    p1 = n1.sum() / (n_items * n_raters)
    p_e = p1**2 + (1 - p1) ** 2
    if p_e >= 1.0:
        return 1.0
    return float((p_bar - p_e) / (1 - p_e))


def fleiss_over(labelers: dict[str, dict[str, int]], names: list[str]) -> tuple[float, int]:
    # Unique raters only: callers pass one name per (model, effort) row, so a model
    # showing up under both efforts would otherwise contribute the same judge twice
    # and inflate kappa (the published-elsewhere 0.749/0.752 vs the true 0.710/0.714).
    names = [n for n in dict.fromkeys(names) if n in labelers]
    if len(names) < 2:
        return float("nan"), 0
    ids = set.intersection(*[set(labelers[n]) for n in names]) if names else set()
    ids = sorted(ids)
    if not ids:
        return float("nan"), 0
    mat = np.array([[labelers[n][i] for n in names] for i in ids], dtype=int)
    return fleiss_kappa(mat), len(ids)


def majority_vote(labelers: dict[str, dict[str, int]], names: list[str]) -> dict[str, int]:
    names = [n for n in names if n in labelers]
    all_ids: set[str] = set()
    for n in names:
        all_ids |= set(labelers[n])
    vote: dict[str, int] = {}
    for i in all_ids:
        votes = [labelers[n][i] for n in names if i in labelers[n]]
        if not votes:
            continue
        ones = sum(votes)
        zeros = len(votes) - ones
        if ones == zeros:  # tie -> undecided, skip
            continue
        vote[i] = 1 if ones > zeros else 0
    return vote


# --------------------------------------------------------------------------- #
# Cost extrapolation
# --------------------------------------------------------------------------- #
def cost_rows(
    raw_dir: Path, configs: list[tuple[str, str]], population: dict
) -> pd.DataFrame:
    p_pos = population["p_pos"]
    n_dedup = population["n_full_dedup_est"]
    n_raw = population["n_full_raw_est"]
    rows = []
    for model, effort in configs:
        recs = _read_jsonl(raw_dir / config_filename(model, effort))
        if not recs:
            continue
        df = pd.DataFrame(recs)
        # Manual labelers (e.g. human:gold) carry only id/prediction — no token/cost
        # metering, so they don't belong in the cost table (they still count as a
        # labeler for agreement, loaded separately in load_predictions).
        if "cost_as_measured" not in df.columns:
            continue
        effort_used = df.get("effort_used", pd.Series([effort])).mode().iat[0]
        n_ok = int((df["prediction"].isin([0, 1])).sum())
        n_trunc = int((df.get("finish_reason") == "length").sum())

        def class_mean(col: str, label: int) -> float:
            sub = df[df["minimax_label"].astype(str) == str(label)][col]
            return float(sub.mean()) if len(sub) else float("nan")

        # population-reweighted per-paragraph cost (post-stratification)
        def reweighted(col: str) -> float:
            mp, mn = class_mean(col, 1), class_mean(col, 0)
            if np.isnan(mp):
                mp = float(df[col].mean())
            if np.isnan(mn):
                mn = float(df[col].mean())
            return p_pos * mp + (1 - p_pos) * mn

        per_para = reweighted("cost_as_measured")
        per_para_unc = reweighted("cost_uncached")
        disc = pricing_mod.batch_discount_for(model)  # 0.5 for OpenAI Batch API; 1.0 for DeepSeek (no batch API)
        rows.append(
            {
                "model": model,
                "effort": effort,
                "effort_used": effort_used,
                "n_ok": n_ok,
                "n_trunc": n_trunc,
                "mean_prompt_tok": round(float(df["prompt_tokens"].mean()), 1),
                "mean_cached_tok": round(float(df["cached_tokens"].mean()), 1),
                "mean_completion_tok": round(float(df["completion_tokens"].mean()), 1),
                "mean_reasoning_tok": round(float(df["reasoning_tokens"].mean()), 1),
                "usd_per_1k_para": round(per_para * 1000, 4),
                "full_dedup_sync_usd": round(per_para * n_dedup, 2),
                "full_dedup_batch_usd": round(per_para * n_dedup * disc, 2),
                "full_raw_sync_usd": round(per_para * n_raw, 2),
                "full_dedup_sync_uncached_usd": round(per_para_unc * n_dedup, 2),
                "smoke_test_usd": round(float(df["cost_as_measured"].sum()), 4),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("full_dedup_batch_usd").reset_index(drop=True) if len(out) else out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(
    raw_dir: Path,
    report_dir: Path,
    configs: list[tuple[str, str]],
    population: dict,
    sample: list[dict],
    consensus_labelers: list[str] | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    labelers = load_predictions(raw_dir, configs, sample)
    cost_df = cost_rows(raw_dir, configs, population)
    pct, kappa = pairwise_matrices(labelers)

    cost_df.to_csv(report_dir / "cost_metrics.csv", index=False)
    pct.to_csv(report_dir / "agreement_pairwise_pct.csv")
    kappa.to_csv(report_dir / "agreement_cohen_kappa.csv")

    ref = OPUS_LABELER if OPUS_LABELER in labelers else None
    # Frozen consensus: vote only over the fixed CONSENSUS_LABELERS that are present.
    # Candidates (OS / decomposed / other configs) are scored against it, not in it.
    consensus_names = [n for n in (consensus_labelers or CONSENSUS_LABELERS) if n in labelers]
    majority = majority_vote(labelers, consensus_names)

    has_human = HUMAN_LABELER in labelers

    # per-labeler agreement summary
    def _agreement_row(name: str, preds: dict[str, int]) -> dict:
        """Agreement of one prediction set vs human gold / Opus / minimax / consensus."""
        row: dict = {"labeler": name, "n_labeled": len(preds)}
        if has_human:  # human gold first — the primary reference when available
            ph, nh = pct_agreement(preds, labelers[HUMAN_LABELER])
            row["%agree_human"] = round(ph * 100, 1) if nh else None
        if ref:
            p, n = pct_agreement(preds, labelers[ref])
            row["%agree_opus"] = round(p * 100, 1) if n else None
        pm, nm = pct_agreement(preds, labelers[MINIMAX_LABELER]) if MINIMAX_LABELER in labelers else (float("nan"), 0)
        row["%agree_minimax"] = round(pm * 100, 1) if nm else None
        pmaj, nmaj = pct_agreement(preds, majority)
        row["%agree_consensus"] = round(pmaj * 100, 1) if nmaj else None
        return row

    summary_rows = [_agreement_row(name, labelers[name]) for name in labelers]

    # Experiment 3.2 — open-source ENSEMBLE: majority-vote the OS labelers and
    # score that committee against the frozen consensus (can a cheap OS committee
    # match a frontier model?).
    os_names = [n for n in labelers if n.endswith(OPEN_SOURCE_SUFFIX)]
    if len(os_names) >= 2:
        os_vote = majority_vote(labelers, os_names)
        summary_rows.append(_agreement_row(f"os-ensemble[{len(os_names)}]", os_vote))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_dir / "agreement_summary.csv", index=False)

    fleiss_all_med, n_med = fleiss_over(labelers, [config_key(m, "medium") for m, _ in configs])
    fleiss_all_xh, n_xh = fleiss_over(labelers, [config_key(m, "xhigh") for m, _ in configs])

    # medium-vs-xhigh within each model
    mve_rows = []
    for model in sorted({m for m, _ in configs}):
        a = labelers.get(config_key(model, "medium"))
        b = labelers.get(config_key(model, "xhigh"))
        if a and b:
            p, n = pct_agreement(a, b)
            mve_rows.append({"model": model, "medium_vs_xhigh_%agree": round(p * 100, 1), "n": n})

    md = _render_markdown(
        population, cost_df, summary_df, pct, kappa,
        fleiss_all_med, n_med, fleiss_all_xh, n_xh, mve_rows, labelers, ref,
    )
    md += _render_lineage(pct, labelers)
    md += _render_batching(report_dir)
    out = report_dir / "COST_COMPARISON.md"
    out.write_text(md)
    return out


def _render_batching(report_dir: Path) -> str:
    """Embed the same-query batching results if estimate_batching.py has run."""
    csv = report_dir / "batching_metrics.csv"
    if not csv.exists():
        return ""
    df = pd.read_csv(csv)
    return "\n".join([
        "\n## Multi-paragraph batching (gpt-5.2 @ medium, same-query)\n",
        "Packs N same-query paragraphs per call (system prompt + query framing written once → N "
        "labels back); cost is self-metered per batch and divided by N. Accuracy = agreement with "
        "the N=1 single-call verdicts. Sample: 5 queries × 20 paragraphs.\n",
        df.to_markdown(index=False),
        "",
        "- Larger N lowers cost/paragraph (shared prefix **and** less reasoning per paragraph) but "
        "perturbs a few % of labels vs single-call (`%agree_vs_N1`).",
        "- For comparison, **deepseek-v4-pro single-call** (see the cost table) is cheaper than "
        "batched gpt-5.2 with no batching accuracy tradeoff.",
    ])


def _render_lineage(pct: pd.DataFrame, labelers: dict[str, dict[str, int]]) -> str:
    """Experiment 3.3 — which frontier labeler does each OS model track most closely?

    Reads the already-computed pairwise %-agreement matrix; for each open-source
    labeler reports its agreement with every frontier labeler and the argmax.
    """
    os_names = [n for n in labelers if n.endswith(OPEN_SOURCE_SUFFIX)]
    frontier = [n for n in labelers if not n.endswith(OPEN_SOURCE_SUFFIX) and n != MINIMAX_LABELER]
    if not os_names or not frontier:
        return ""
    L = [
        "\n## Lineage pairing — which frontier model does each OS model track?\n",
        "Pairwise % agreement of each open-source labeler with each frontier labeler "
        "(from the matrix above). **best match** is the argmax — tests whether a model's "
        "judgments track its lineage (e.g. gpt-oss ↔ GPT-5.x).\n",
    ]
    L.append("| OS model | " + " | ".join(frontier) + " | best match |")
    L.append("|" + "---|" * (len(frontier) + 2))
    for osn in os_names:
        cells: list[str] = []
        best, best_v = None, -1.0
        for fn in frontier:
            v = float(pct.loc[osn, fn]) if (osn in pct.index and fn in pct.columns) else float("nan")
            cells.append(f"{v:.2f}" if v == v else "—")
            if v == v and v > best_v:
                best_v, best = v, fn
        best_str = f"**{best}** ({best_v:.2f})" if best else "—"
        L.append(f"| {osn} | " + " | ".join(cells) + f" | {best_str} |")
    return "\n".join(L)


def _render_markdown(
    population, cost_df, summary_df, pct, kappa,
    fleiss_med, n_med, fleiss_xh, n_xh, mve_rows, labelers, ref,
) -> str:
    L = []
    L.append("# Frontier-model reclassification — cost & agreement\n")
    L.append("## Population assumptions (extrapolation base)\n")
    if population.get("n_full_override"):
        L.append(
            f"- **Scenario: {population['n_full_override']:,} LLM invocations** (user-specified target).\n"
            f"- Positive rate p_pos = **{population['p_pos']*100:.2f}%** "
            f"(measured over {population['sample_files']} sampled events; reweights the balanced sample).\n"
            f"- Per-invocation cost assumes the measured **paragraph-length input profile** (~1.7K input "
            f"tok/call). If these invocations are shorter units (e.g. propositions or sentences), the "
            f"input portion shrinks (output/reasoning stay similar), so these figures are a conservative "
            f"upper bound on input cost.\n"
        )
    else:
        L.append(
            f"- Full dataset (deduped est.): **{population['n_full_dedup_est']:,}** paragraphs "
            f"(raw incl. splade/bm25 duplicates: {population['n_full_raw_est']:,}).\n"
            f"- Positive rate p_pos = **{population['p_pos']*100:.2f}%** "
            f"(from {population['sample_files']} sampled events; used to reweight balanced sample).\n"
            f"- Mean unique paragraphs/event ≈ {population['mean_unique_per_event']:.0f}.\n"
        )
    L.append("\n## Pricing used (USD / 1M tok, standard tier) — CONFIRM\n")
    L.append("| model | input | cached | output |\n|---|---|---|---|")
    for m, pr in pricing_mod.PRICING.items():
        L.append(f"| {m} | ${pr.input} | ${pr.cached_input} | ${pr.output} |")

    _n_full = population.get("n_full_override") or population["n_full_dedup_est"]
    L.append(f"\n## Cost per config (extrapolated to {_n_full:,} invocations)\n")
    if len(cost_df):
        cols = [
            "model", "effort", "effort_used", "mean_reasoning_tok", "usd_per_1k_para",
            "full_dedup_batch_usd", "full_dedup_sync_usd", "full_dedup_sync_uncached_usd",
            "n_ok", "n_trunc", "smoke_test_usd",
        ]
        L.append(cost_df[cols].to_markdown(index=False))
        L.append(
            f"\n- **Smoke-test actual spend (self-metered): "
            f"${cost_df['smoke_test_usd'].sum():.2f}** across all configs.\n"
            "- `_sync_` is real-time; `_uncached_` is the worst case if prompt caching never hits.\n"
            "- `full_dedup_batch_usd` applies the **OpenAI** Batch API (50% off) for gpt-* models. "
            "**DeepSeek has no Batch API, so its batch column == sync** (an off-peak window may cut "
            "~50%, but V4 off-peak rates are unconfirmed).\n"
        )
    else:
        L.append("_No config outputs found yet._\n")

    L.append("\n## Inter-labeler agreement\n")
    if ref:
        L.append(f"Reference labeler = **{ref}** (Claude Opus 4.8, max reasoning).\n")
    L.append(
        "`%agree_consensus` = agreement with the frozen majority vote of "
        f"**{', '.join(CONSENSUS_LABELERS)}** — candidates are scored against this fixed "
        "gold and are never members of it. `os-ensemble[k]` is the majority vote of the k "
        "open-source labelers, scored the same way.\n"
    )
    L.append(summary_df.to_markdown(index=False))
    L.append(
        f"\n- Fleiss' κ across the 4 GPT models @ medium = "
        f"**{fleiss_med:.3f}** (n={n_med}); @ xhigh = **{fleiss_xh:.3f}** (n={n_xh}).\n"
    )
    if mve_rows:
        L.append("\n**medium vs xhigh agreement within each model** (does costly xhigh change labels?):\n")
        L.append(pd.DataFrame(mve_rows).to_markdown(index=False))
    L.append("\n### Pairwise % agreement\n")
    L.append(pct.to_markdown())
    L.append("\n### Pairwise Cohen's κ\n")
    L.append(kappa.to_markdown())

    L.append("\n## Caveats\n")
    L.append(
        "- Dollar figures depend on the pricing table above — confirm against your account "
        "(token counts are exact).\n"
        "- p_pos and full-dataset size are extrapolated from a file sample.\n"
        "- Cost uses observed prompt-cache hits (warm-cache, matches a real high-throughput run); "
        "the `_uncached_` column bounds the no-cache case.\n"
        "- minimax is a biased labeler (conservative-negative / generous-positive), not gold.\n"
        "- DeepSeek: no Batch API (batch col == sync); and the $0.435/$0.87 V4-Pro rate was a "
        "promo (sources split on whether it is now permanent) — confirm the current rate.\n"
        "- Prompt policy defaults: completed-past = true (planned/terminated/exited = false); "
        "CE sub-parts ≠ CE but CE-by-constituents counts.\n"
    )

    if len(cost_df):
        L.append("\n## Batch cost by model & effort\n")
        L.append(cost_df[["model", "effort_used", "full_dedup_batch_usd"]].to_markdown(index=False))
    return "\n".join(L)
