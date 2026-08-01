"""Query-difficulty analysis: how do metrics degrade as the number of pooled
positives R changes, and how much of the loss is the 500-depth recall ceiling
rather than ranking failure?

R-buckets (closed integer ranges from eval_config.R_BUCKETS). For each bucket we
report the query count, the mean metric of the best single model and averaged
over all models, the theoretical recall@500 ceiling mean(min(1, 500/R)), and the
mean achieved recall@500. A per-model × bucket nDCG@10 matrix backs the plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import eval_config as cfg


def _bucket_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def assign_bucket(R: int) -> str | None:
    for lo, hi in cfg.R_BUCKETS:
        if R >= lo and (hi is None or R <= hi):
            return _bucket_label(lo, hi)
    return None


def _bucket_order() -> list[str]:
    return [_bucket_label(lo, hi) for lo, hi in cfg.R_BUCKETS]


def bucket_summary(per_query: pd.DataFrame, best_model: str = "colbert-v2") -> pd.DataFrame:
    """One row per R-bucket: counts, best/mean metrics, recall ceiling vs achieved."""
    df = per_query[per_query["R"] > 0].copy()
    df["bucket"] = df["R"].apply(assign_bucket)
    df["ceiling_recall@500"] = np.minimum(1.0, 500.0 / df["R"])

    # Unique-query stats (R is per-query; take one row per event for counts/ceiling).
    per_event = df.drop_duplicates("event_id")
    rows = []
    for b in _bucket_order():
        ev = per_event[per_event["bucket"] == b]
        if ev.empty:
            continue
        allm = df[df["bucket"] == b]
        best = allm[allm["model"] == best_model]
        rows.append(
            {
                "R_bucket": b,
                "n_queries": int(len(ev)),
                # Queries whose P@R is exact (r_precision means cover only these;
                # < n_queries in the 500+ bucket where 22 queries never converged).
                "n_pr_exact": int(ev["p_at_r_exact"].sum()),
                "mean_R": round(float(ev["R"].mean()), 1),
                f"{best_model}_ndcg@10": round(float(best["ndcg@10"].mean()), 4),
                f"{best_model}_r_precision": round(float(best["r_precision"].mean()), 4),
                "allmodels_ndcg@10": round(float(allm["ndcg@10"].mean()), 4),
                "ceiling_recall@500": round(float(ev["ceiling_recall@500"].mean()), 4),
                f"{best_model}_recall@500": round(float(best["recall@500"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def model_bucket_matrix(per_query: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    """models (rows) x R-bucket (cols) mean metric, for the heatmap/plot."""
    df = per_query[per_query["R"] > 0].copy()
    df["bucket"] = df["R"].apply(assign_bucket)
    mat = df.pivot_table(index="model", columns="bucket", values=metric, aggfunc="mean")
    cols = [b for b in _bucket_order() if b in mat.columns]
    return mat.reindex(index=[m for m in cfg.MODEL_ALIASES if m in mat.index], columns=cols).round(4)


def compute(per_query: pd.DataFrame, best_model: str = "colbert-v2") -> dict:
    return {
        "summary": bucket_summary(per_query, best_model),
        "matrix_ndcg10": model_bucket_matrix(per_query, "ndcg@10"),
        "best_model": best_model,
    }


if __name__ == "__main__":
    pq = pd.read_csv(cfg.RESULTS_DIR / "all_models_per_query.csv")
    res = compute(pq)
    print(res["summary"].to_string(index=False))
