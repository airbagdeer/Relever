"""Matplotlib (Agg) figures for the frontier-v2 evaluation report.

Every function takes already-computed frames/dicts and writes a PNG into
``summary/evaluation/plots/``. Late-interaction models are drawn in a distinct
colour throughout so the dense-vs-late-interaction split is visible at a glance.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import eval_config as cfg

DENSE_C = "#4C72B0"
LATE_C = "#C44E52"
ORACLE_C = "#55A868"
FUSION_C = "#8172B3"


def _color_for(alias: str) -> str:
    return LATE_C if cfg.MODEL_BY_ALIAS[alias].retriever_type == "late_interaction" else DENSE_C


def _save(fig, name: str) -> None:
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.PLOTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leaderboard_bar(leaderboard: pd.DataFrame, metric: str, name: str, title: str) -> None:
    s = leaderboard[metric].sort_values(ascending=True)
    colors = [_color_for(a) for a in s.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(s)), s.values, color=colors)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(title)
    for i, v in enumerate(s.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LATE_C), plt.Rectangle((0, 0), 1, 1, color=DENSE_C)]
    ax.legend(handles, ["late-interaction (baseline)", "dense (single-vector)"], loc="lower right", fontsize=9)
    ax.margins(x=0.12)
    _save(fig, name)


def plot_metric_heatmap(leaderboard: pd.DataFrame, metrics: list[str], name: str) -> None:
    lb = leaderboard.sort_values("ndcg@10", ascending=False)[metrics]
    # Per-column min-max normalisation for colour only; annotate raw values.
    norm = (lb - lb.min()) / (lb.max() - lb.min())
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(norm.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(lb.index)))
    ax.set_yticklabels(lb.index, fontsize=9)
    for i in range(lb.shape[0]):
        for j in range(lb.shape[1]):
            ax.text(j, i, f"{lb.values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="black" if norm.values[i, j] < 0.6 else "white")
    ax.set_title("Per-model metrics (colour = per-column min-max normalised)")
    fig.colorbar(im, ax=ax, fraction=0.025, label="normalised")
    _save(fig, name)


def plot_size_scaling(scaling_table: pd.DataFrame, name: str, metric: str = "ndcg@10") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for family, sub in scaling_table.groupby("family"):
        sub = sub.sort_values("params_b")
        ax.plot(sub["params_b"], sub[metric], marker="o", label=family)
        for _, r in sub.iterrows():
            ax.annotate(f"{r[metric]:.3f}", (r["params_b"], r[metric]), fontsize=8,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions, log scale)")
    ax.set_ylabel(metric)
    ax.set_title("Does scaling parameters help? (within-family size ladders)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, name)


def plot_recall_curves(oracle_frame: pd.DataFrame, fusion_row: dict, name: str) -> None:
    k = oracle_frame["k"].values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k, oracle_frame["union_oracle_recall"], marker="o", color=ORACLE_C, label="union oracle (any-of-18)")
    ax.plot(k, oracle_frame["best_single_recall"], marker="s", color=LATE_C, label="best single model")
    ax.plot(k, [fusion_row[f"recall@{kk}"] for kk in cfg.K_VALUES], marker="^", color=FUSION_C, label="RRF fusion (18)")
    ax.set_xscale("log")
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("recall@k")
    ax.set_title("Recall@k: best single vs RRF fusion vs union-oracle ceiling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, name)


def plot_score_separation(sep_table: pd.DataFrame, rho: float, name: str) -> None:
    t = sep_table.dropna(subset=["mean_auc"])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [_color_for(a) for a in t["model"]]
    ax.scatter(t["mean_auc"], t["ndcg@10"], c=colors, s=60)
    for _, r in t.iterrows():
        ax.annotate(r["model"], (r["mean_auc"], r["ndcg@10"]), fontsize=7,
                    textcoords="offset points", xytext=(4, 2))
    ax.set_xlabel("mean per-query ROC-AUC (score → label separation)")
    ax.set_ylabel("nDCG@10")
    ax.set_title(f"Score separation explains ranking quality (Spearman ρ = {rho:+.3f})")
    ax.grid(True, alpha=0.3)
    _save(fig, name)


def plot_difficulty(bucket_summary: pd.DataFrame, matrix: pd.DataFrame, best_model: str, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    buckets = bucket_summary["R_bucket"].tolist()
    x = range(len(buckets))

    # Left: nDCG@10 vs bucket for a few top models + all-model mean.
    top = [m for m in ["colbert-v2", "jina-embeddings-v4-late", "Qwen3-Embedding-8B", "jina-embeddings-v5-text-small"] if m in matrix.index]
    for mdl in top:
        axes[0].plot(x, [matrix.loc[mdl, b] for b in buckets], marker="o", label=mdl)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(buckets, rotation=30)
    axes[0].set_xlabel("R bucket (number of pooled positives)")
    axes[0].set_ylabel("nDCG@10")
    axes[0].set_title("Ranking quality vs query difficulty")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: recall@500 ceiling vs achieved (best model).
    axes[1].plot(x, bucket_summary["ceiling_recall@500"], marker="s", color="grey", linestyle="--", label="recall@500 ceiling min(1,500/R)")
    axes[1].plot(x, bucket_summary[f"{best_model}_recall@500"], marker="o", color=LATE_C, label=f"{best_model} recall@500")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(buckets, rotation=30)
    axes[1].set_xlabel("R bucket (number of pooled positives)")
    axes[1].set_ylabel("recall@500")
    axes[1].set_title("Recall ceiling vs achieved (depth-500 cap bites only at high R)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    _save(fig, name)


def plot_benchmark_scatter(table: pd.DataFrame, correlations: dict, metric: str, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, bench, rank_col in ((axes[0], "MTEB", "mteb_rank"), (axes[1], "RTEB", "rteb_rank")):
        sub = table.dropna(subset=[rank_col]).copy()
        # Full leaderboard ranks span 1..310; re-rank BOTH axes within the tested
        # subset so the diagonal (= perfect agreement) is meaningful. Spearman is
        # invariant to this, so the reported coefficient is unchanged.
        sub["ext_within"] = sub[rank_col].rank(method="min")
        sub["task_within"] = sub["task_rank"].rank(method="min")
        ax.scatter(sub["ext_within"], sub["task_within"], c=[_color_for(a) for a in sub["model"]], s=60)
        for _, r in sub.iterrows():
            ax.annotate(r["model"], (r["ext_within"], r["task_within"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        lims = [0, len(sub) + 1]
        ax.plot(lims, lims, "k--", alpha=0.4, label="perfect agreement")
        st = correlations[bench]
        ax.set_xlabel(f"{bench} rank within tested subset (lower = better)")
        ax.set_ylabel(f"task rank by {metric} (lower = better)")
        ax.set_title(f"{bench}: Spearman ρ={st['spearman_rho']:+.2f} (p={st['spearman_p']:.2f}), n={st['n']}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Do general embedding leaderboards predict task rank? ({metric})", y=1.02)
    _save(fig, name)
