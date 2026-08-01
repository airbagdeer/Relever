"""Generate the data-driven markdown reports + CSVs for the frontier-v2 eval.

These files carry the *numbers* (tables, correlations, significance) straight
from the computed frames so they are always reproducible. The flagship narrative
syntheses (``REPORT.md``, ``CORE_SUMMARY.md``) are authored separately on top of
these. Each analysis gets its own file, as requested.
"""

from __future__ import annotations

import pandas as pd

import eval_config as cfg
import reporting as rp

HEADLINE_COLS = ["ndcg@10", "ndcg@100", "precision@10", "recall@100", "recall@500",
                 "r_precision", "map", "mrr", "success@10"]


def _write(name: str, text: str) -> None:
    cfg.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.SUMMARY_DIR / name).write_text(text)


def _ranked(leaderboard: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    return leaderboard.sort_values(metric, ascending=False)


# --------------------------------------------------------------------------- #
def write_leaderboard_md(leaderboard: pd.DataFrame, meta: dict) -> None:
    lb = _ranked(leaderboard)
    head = rp.metadata_header(
        "Frontier-v2 Retrieval Leaderboard",
        [
            f"Task: relation-evidence retrieval (paragraph granularity), {meta['n_eval']} queries with R>0 "
            f"({meta['n_excluded']} of {meta['n_total']} excluded for R=0).",
            "Relevance = GPT-5.2 label == 1; R = positives in the merged label set: the depth-500 union "
            "pool of all 18 models plus, for the R>500 queries, delta labels covering each model's top-R "
            "band (TREC-style pooling — recall is relative to the pool, not all of Wikipedia).",
            f"R-precision is exact and averaged over the {meta['n_pr_exact']} queries whose top-R is fully "
            f"labeled for every model; {meta['n_pr_excluded']} high-R queries where the delta labelling did "
            "not converge are excluded from it (see `the paper's extended-pooling appendix`). All other metrics "
            f"use all {meta['n_eval']} R>0 queries.",
            f"Failed labels treated as non-relevant ({meta['n_failed']} across the whole dataset); "
            f"missing labels: {meta['n_missing']}.",
            "Each model retrieved top-500 per query. 🟩 marks the top-3 per column. Sorted by nDCG@10.",
        ],
    )
    headline = lb[HEADLINE_COLS].copy()
    headline.insert(0, "retriever", [cfg.MODEL_BY_ALIAS[a].retriever_type for a in headline.index])

    ndcg_cols = [f"ndcg@{k}" for k in (1, 10, 50, 100, 500)]
    prec_cols = [f"precision@{k}" for k in (1, 5, 10, 20, 50, 100)]
    rec_cols = [f"recall@{k}" for k in (10, 50, 100, 200, 500)]

    parts = [
        head,
        "\n## Headline metrics\n",
        rp.markdown_table(headline, highlight_cols=HEADLINE_COLS, index_name="model"),
        "\n\n## nDCG@k\n",
        rp.markdown_table(lb[ndcg_cols], highlight_cols=ndcg_cols, index_name="model"),
        "\n\n## Precision@k\n",
        rp.markdown_table(lb[prec_cols], highlight_cols=prec_cols, index_name="model"),
        "\n\n## Recall@k\n",
        rp.markdown_table(lb[rec_cols], highlight_cols=rec_cols, index_name="model"),
        "\n\n![nDCG@10](plots/leaderboard_ndcg10.png)\n",
        "\n![R-precision](plots/leaderboard_r_precision.png)\n",
        "\n![Metric heatmap](plots/metric_heatmap.png)\n",
    ]
    _write("LEADERBOARD.md", "\n".join(parts))


# --------------------------------------------------------------------------- #
def write_statistical_tests_md(sig: dict) -> None:
    # Per-metric effective n comes from the frames themselves: r_precision drops
    # the P@R-inexact queries (NaN rows), so its n is smaller than ndcg@10's.
    n_by_metric = {metric: int(sig[metric]["n"].iloc[0]) for metric in ("ndcg@10", "r_precision")}
    parts = [rp.metadata_header(
        "Statistical Significance — Pairwise Model Comparisons",
        [
            "Paired two-sided Wilcoxon signed-rank over the queries shared by all models: "
            + "; ".join(f"{m} over {n}" for m, n in n_by_metric.items())
            + " (r_precision excludes the high-R queries whose P@R is inexact — "
              "see `the paper's extended-pooling appendix`).",
            "Holm-Bonferroni correction across all 153 model pairs within each metric.",
            "Effect size = matched-pairs rank-biserial correlation (positive ⇒ row model better).",
            "Significance: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` else.",
        ],
    )]
    for metric in ("ndcg@10", "r_precision"):
        wins = sig[f"{metric}__wins"]
        parts.append(f"\n## {metric}: significant-win count (of 17 opponents, n={n_by_metric[metric]})\n")
        parts.append(rp.markdown_table(wins, highlight_cols=["sig_wins"], index_name="model"))
        pw = sig[metric]
        # Show the best model's (most sig wins) comparisons vs everyone — the most decision-relevant slice.
        best = wins.index[0]
        rows = pw[(pw["model_a"] == best) | (pw["model_b"] == best)].copy()
        rows["other"] = rows.apply(lambda r: r["model_b"] if r["model_a"] == best else r["model_a"], axis=1)
        rows["delta_vs_best"] = rows.apply(
            lambda r: (r["mean_a"] - r["mean_b"]) if r["model_a"] == best else (r["mean_b"] - r["mean_a"]), axis=1)
        # Reorient the effect size to the "best > other" direction so its sign matches delta_vs_best
        # (the raw rank_biserial is in model_a-minus-model_b order, and the best model may be model_b).
        rows["effect_best_gt_other"] = rows.apply(
            lambda r: r["rank_biserial"] if r["model_a"] == best else -r["rank_biserial"], axis=1)
        show = rows[["other", "delta_vs_best", "effect_best_gt_other", "p_holm", "sig"]].copy()
        show["p_holm"] = show["p_holm"].map(rp.fmt_p)
        show = show.sort_values("delta_vs_best", ascending=False).reset_index(drop=True)
        parts.append(f"\n### Best model (`{best}`) vs each other model — {metric}\n")
        parts.append(rp.markdown_table(show, highlight_cols=[], index_name=None))
    _write("STATISTICAL_TESTS.md", "\n".join(parts))


# --------------------------------------------------------------------------- #
def write_size_scaling_md(res: dict, n_eval: int) -> None:
    table = res["table"]
    v_ndcg = res["verdicts"]["ndcg@10"]
    obs = []
    for _, r in v_ndcg.iterrows():
        trend = "increases monotonically" if r["monotonic_increasing"] else "is **non-monotonic**"
        direction = "gain" if r["delta_small_to_large"] > 0 else "loss"
        obs.append(
            f"- **{r['family']}** ({r['sizes_b']} B): nDCG@10 {trend}; best at {r['best_size_b']:g} B; "
            f"net {direction} of {r['delta_small_to_large']:+.4f} from smallest to largest."
        )
    parts = [
        rp.metadata_header(
            "Model-Size Scaling",
            ["Does scaling parameters within a family improve relation-evidence retrieval?",
             f"Families with a genuine parameter ladder; metrics are means over {n_eval} R>0 queries."],
        ),
        "\n## Per-(family, size) metrics\n",
        rp.markdown_table(table.set_index("model"), highlight_cols=list(res["table"].columns[3:]), index_name="model"),
        "\n\n## Verdict (nDCG@10)\n",
        rp.markdown_table(v_ndcg.set_index("family"), highlight_cols=[], index_name="family"),
        "\n\n## Observations\n",
        "\n".join(obs),
        "\n\n![Size scaling](plots/size_scaling.png)\n",
    ]
    _write("SIZE_SCALING.md", "\n".join(parts))


# --------------------------------------------------------------------------- #
def write_difficulty_md(res: dict, n_eval: int) -> None:
    summ = res["summary"]
    best = res["best_model"]
    parts = [
        rp.metadata_header(
            "Query-Difficulty Analysis",
            [f"Metrics bucketed by R (number of positives, merged main+delta labels). {n_eval} R>0 queries.",
             f"Best model shown = `{best}`. `ceiling_recall@500 = mean(min(1, 500/R))`.",
             "`n_pr_exact` = queries in the bucket whose P@R is exact; the `r_precision` column averages "
             "only those (matters in the 500+ bucket, where 22 non-converged queries are excluded — "
             "see `the paper's extended-pooling appendix`). `mean_R` / the recall@500 ceiling use the merged R, "
             "so the 500+ bucket reflects the delta-grown positive counts.",
             "Two regimes: low-R queries are hard to *rank* (few needles); high-R queries are *recall-capped* by depth 500."],
        ),
        "\n## Per-bucket summary\n",
        rp.markdown_table(summ.set_index("R_bucket"), highlight_cols=[], index_name="R_bucket"),
        "\n\n## Per-model nDCG@10 by bucket\n",
        rp.markdown_table(res["matrix_ndcg10"], highlight_cols=list(res["matrix_ndcg10"].columns), index_name="model"),
        "\n\n![Difficulty](plots/difficulty_buckets.png)\n",
    ]
    _write("DIFFICULTY_ANALYSIS.md", "\n".join(parts))


# --------------------------------------------------------------------------- #
def write_score_separation_md(res: dict) -> None:
    s = res["spearman_auc_vs_ndcg10"]
    parts = [
        rp.metadata_header(
            "Score-Separation Diagnostics",
            ["Per-query ROC-AUC of (similarity score → pooled label), averaged per model (scale-free).",
             "Only queries whose retrieved list holds both a positive and a negative are scored."],
        ),
        "\n## Mean AUC vs ranking quality\n",
        rp.markdown_table(res["table"].set_index("model"), highlight_cols=["mean_auc", "ndcg@10"], index_name="model"),
        f"\n\n## Correlation\n\nSpearman ρ(mean AUC, nDCG@10) = **{s['rho']:+.3f}** "
        f"(p={rp.fmt_p(s['p'])}, n={s['n']}) — a {rp.corr_strength(s['rho'])} relationship: "
        "ranking quality is almost entirely explained by how well a model's raw scores separate "
        "pooled positives from negatives. Note all AUCs are modest (≈0.59–0.70), so even the best "
        "model's scores only weakly distinguish the exact relation from similar-but-wrong paragraphs.\n",
        "\n![Score separation](plots/score_separation.png)\n",
    ]
    _write("SCORE_SEPARATION.md", "\n".join(parts))


# --------------------------------------------------------------------------- #
def write_fusion_oracle_md(fusion_headline: pd.DataFrame, oracle: pd.DataFrame,
                           n_rp: int | None = None) -> None:
    rp_note = (f"r_precision on both sides is averaged over the same {n_rp} exact-P@R queries "
               "(the high-R non-converged queries are excluded, matching the leaderboard)."
               if n_rp else
               "r_precision on both sides is averaged over the same exact-P@R query set as the leaderboard.")
    parts = [
        rp.metadata_header(
            "Fusion & Oracle Headroom",
            [f"RRF fusion (k0={__import__('analysis_fusion_oracle').RRF_K0}) of all 18 models vs the best single model.",
             "Union-oracle recall@k = positives in the union of every model's top-k / R (upper bound for any combiner).",
             "Union-oracle recall@500 < 1 measures the positives that only the R>500 delta labels found "
             "beyond the depth-500 pool (it was 1.000 by construction before those labels were merged); "
             "k<500 remains the informative regime.",
             rp_note],
        ),
        "\n## RRF fusion vs best single model\n",
        rp.markdown_table(fusion_headline.set_index("metric"), highlight_cols=[], index_name="metric"),
        "\n\n## Union-oracle recall@k vs best single\n",
        rp.markdown_table(oracle.set_index("k"), highlight_cols=[], index_name="k"),
        "\n\n![Recall curves](plots/recall_curves.png)\n",
    ]
    _write("FUSION_ORACLE.md", "\n".join(parts))


# --------------------------------------------------------------------------- #
def _rel_phrase(rho: float) -> str:
    """Strength + sign description of a rank correlation (positive ⇒ benchmark predicts task)."""
    if abs(rho) < 0.1:
        return "no relationship"
    direction = "positive" if rho > 0 else "negative (inverse)"
    return f"{rp.corr_strength(rho)} {direction} relationship"


def write_benchmark_ranking_md(res: dict) -> None:
    parts = [rp.metadata_header(
        "Benchmark vs Task Ranking",
        ["Does a model's MTEB / RTEB standing predict its rank on relation-evidence retrieval?",
         "External ranks = each model's Borda rank on the MTEB(eng, v2) and RTEB(eng, beta) leaderboards in the "
         "**3 June 2026 snapshot** used in the paper (archived as `leaderboard_snapshot/`). Lower = better; "
         "`—` = outside the snapshot's captured top ranks (the dense models were selected from the top of these boards).",
         "Correlations: Spearman ρ / Kendall τ between leaderboard rank and within-subset task rank. "
         "Positive ⇒ the benchmark predicts task quality; ≤0 ⇒ it does not."],
    )]
    for metric in ("ndcg@10", "r_precision"):
        table = res["tables"][metric].copy()
        for col in ("mteb_rank", "rteb_rank"):
            table[col] = table[col].map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
        parts.append(f"\n## Task ranking by {metric}\n")
        parts.append(rp.markdown_table(table.set_index("model"), highlight_cols=[], index_name="model"))
        corr = res["correlations"][metric]
        parts.append("\n\n**Rank correlation with task:**\n")
        for bench in ("MTEB", "RTEB"):
            st = corr[bench]
            parts.append(
                f"- **{bench}** (n={st['n']}): Spearman ρ = **{st['spearman_rho']:+.3f}** (p={rp.fmt_p(st['spearman_p'])}); "
                f"Kendall τ = {st['kendall_tau']:+.3f} (p={rp.fmt_p(st['kendall_p'])}) — {_rel_phrase(st['spearman_rho'])}."
            )
    parts.append(
        "\n\n**Reading:** among the top-ranked models captured in the 3 June snapshot, neither MTEB nor RTEB rank "
        "shows a significant correlation with task quality (all p > 0.05). The two best task models overall — "
        "colbert-v2 and jina-embeddings-v4-late, both late-interaction — sit outside the snapshot's captured top "
        "ranks entirely. General-purpose embedding leaderboards do not predict relation-evidence retrieval quality.\n"
    )
    parts.append("\n![Benchmark vs task](plots/benchmark_vs_task_ndcg10.png)\n")
    _write("BENCHMARK_VS_TASK_RANKING.md", "\n".join(parts))
