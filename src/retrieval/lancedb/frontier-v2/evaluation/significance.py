"""Pairwise statistical significance between models on per-query metrics.

Paired **Wilcoxon signed-rank** test (two-sided) over the R>0 queries shared by
all models (1029; r_precision additionally drops the queries whose P@R is
inexact — NaN rows — leaving 1007; the per-pair ``n`` column records the
effective count), with **Holm-Bonferroni** correction applied across all
C(18,2)=153 model pairs within each metric family, and a matched-pairs
**rank-biserial** effect size. This
mirrors the convention already used in
``src/process_results/precision_at_r_significance.py`` (helpers reproduced here
so the evaluation package stays import-self-contained), upgraded to use
``scipy.stats.rankdata`` (average ranks) for the effect size so ties are handled
correctly.

Significance markers: ``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05, ``ns`` else.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

import eval_config as cfg


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment; returns adjusted p's in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = (m - rank) * p_values[i]
        a = max(prev, min(1.0, a))  # monotone non-decreasing, capped at 1
        adj[i] = a
        prev = a
    return adj


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for paired Wilcoxon, in [-1, 1].
    Positive => a tends to exceed b. Uses average ranks for ties."""
    d = a - b
    nz = d != 0
    if not nz.any():
        return 0.0
    ranks = rankdata(np.abs(d[nz]))  # average ranks
    pos = ranks[d[nz] > 0].sum()
    neg = ranks[d[nz] < 0].sum()
    total = pos + neg
    return float((pos - neg) / total) if total else 0.0


def fmt_sig(adj_p: float) -> str:
    if adj_p < 0.001:
        return "***"
    if adj_p < 0.01:
        return "**"
    if adj_p < 0.05:
        return "*"
    return "ns"


def metric_matrix(per_query: pd.DataFrame, metric: str) -> pd.DataFrame:
    """events (rows) x models (cols) matrix of the metric over R>0 queries."""
    sub = per_query[per_query["R"] > 0]
    pivot = sub.pivot(index="event_id", columns="model", values=metric)
    # Keep events with a value for every model. For r_precision this also drops
    # the P@R-inexact queries (NaN on all models) -> the exact 1007-query set.
    return pivot.dropna(axis=0)


def pairwise_tests(per_query: pd.DataFrame, metric: str) -> pd.DataFrame:
    """All 153 pairwise two-sided Wilcoxon tests for one metric, Holm-corrected."""
    mat = metric_matrix(per_query, metric)
    models = [m for m in cfg.MODEL_ALIASES if m in mat.columns]
    pairs = list(itertools.combinations(models, 2))

    recs = []
    p_raw: list[float] = []
    for ma, mb in pairs:
        a = mat[ma].to_numpy(dtype=float)
        b = mat[mb].to_numpy(dtype=float)
        if np.allclose(a, b):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            except ValueError:  # e.g. all differences zero
                stat, p = 0.0, 1.0
        rbc = rank_biserial(a, b)
        recs.append(
            {
                "model_a": ma,
                "model_b": mb,
                "n": int(len(a)),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "delta": float(a.mean() - b.mean()),
                "stat": float(stat),
                "p_raw": float(p),
                "rank_biserial": rbc,
            }
        )
        p_raw.append(float(p))

    p_holm = holm_bonferroni(p_raw)
    df = pd.DataFrame(recs)
    df["p_holm"] = p_holm
    df["sig"] = [fmt_sig(p) for p in p_holm]
    df["winner"] = np.where(
        df["p_holm"] < 0.05,
        np.where(df["delta"] > 0, df["model_a"], df["model_b"]),
        "tie",
    )
    return df


def wins_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Per model: how many of the other 17 it *significantly* beats / loses to."""
    models = sorted(set(pairwise["model_a"]) | set(pairwise["model_b"]))
    rows = []
    for mdl in models:
        sig = pairwise[(pairwise["p_holm"] < 0.05)]
        beats = int(((sig["winner"] == mdl)).sum())
        involved = sig[(sig["model_a"] == mdl) | (sig["model_b"] == mdl)]
        losses = int(len(involved) - (involved["winner"] == mdl).sum())
        rows.append({"model": mdl, "sig_wins": beats, "sig_losses": losses})
    return (
        pd.DataFrame(rows)
        .set_index("model")
        .reindex([m for m in cfg.MODEL_ALIASES if m in models])
        .sort_values("sig_wins", ascending=False)
    )


def run(per_query: pd.DataFrame, metrics: tuple[str, ...] = ("ndcg@10", "r_precision")) -> dict[str, pd.DataFrame]:
    """Compute + persist pairwise tests for the headline metrics. Returns the frames."""
    cfg.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        pw = pairwise_tests(per_query, metric)
        safe = metric.replace("@", "").replace("_", "")
        pw.to_csv(cfg.SUMMARY_DIR / f"significance_{safe}.csv", index=False)
        out[metric] = pw
        out[f"{metric}__wins"] = wins_summary(pw)
    return out


if __name__ == "__main__":
    pq = pd.read_csv(cfg.RESULTS_DIR / "all_models_per_query.csv")
    frames = run(pq)
    for metric in ("ndcg@10", "r_precision"):
        print(f"\n=== {metric}: significant-win count ===")
        print(frames[f"{metric}__wins"].to_string())
