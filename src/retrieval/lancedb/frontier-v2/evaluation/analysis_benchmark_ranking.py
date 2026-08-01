"""Benchmark-vs-task ranking: does a model's MTEB / RTEB standing predict its
quality on relation-evidence retrieval?

For each external benchmark we restrict to the models that appear on it, rank
them 1..n by the task metric (lower = better, matching the benchmark convention),
and correlate the two rankings with Spearman ρ and Kendall τ. A near-zero or
negative correlation is direct evidence that general-purpose embedding
leaderboards do not transfer to this task.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau, spearmanr

import eval_config as cfg


def _task_rank(leaderboard: pd.DataFrame, metric: str) -> pd.Series:
    """Dense task rank over all 18 models (1 = best on `metric`)."""
    return leaderboard[metric].rank(ascending=False, method="min").astype(int)


def ranking_table(leaderboard: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    """Per-model table: external ranks + task metric + task rank (all 18)."""
    task_rank = _task_rank(leaderboard, metric)
    rows = []
    for info in cfg.MODELS:
        rows.append(
            {
                "model": info.alias,
                "mteb_rank": info.mteb_rank,
                "rteb_rank": info.rteb_rank,
                f"task_{metric}": round(float(leaderboard.loc[info.alias, metric]), 4),
                "task_rank": int(task_rank.loc[info.alias]),
                "retriever": info.retriever_type,
            }
        )
    return pd.DataFrame(rows).sort_values("task_rank").reset_index(drop=True)


def _correlate(external: dict[str, int], leaderboard: pd.DataFrame, metric: str) -> dict:
    """Spearman ρ + Kendall τ between external rank and within-subset task rank.

    Spearman/Kendall are invariant to monotone transforms, so feeding the raw
    leaderboard rank and the within-subset task rank yields the same coefficient
    as ranking both within the tested subset."""
    aliases = [a for a in external if a in leaderboard.index]
    ext_rank = [external[a] for a in aliases]
    # Task rank within the subset (1 = best task metric among these models).
    metric_vals = pd.Series({a: leaderboard.loc[a, metric] for a in aliases})
    task_rank = metric_vals.rank(ascending=False, method="average")
    task_rank_list = [float(task_rank[a]) for a in aliases]

    rho, rho_p = spearmanr(ext_rank, task_rank_list)
    tau, tau_p = kendalltau(ext_rank, task_rank_list)
    return {
        "n": len(aliases),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "aliases": aliases,
    }


def compute(leaderboard: pd.DataFrame, metrics: tuple[str, ...] = ("ndcg@10", "r_precision")) -> dict:
    """Return tables + correlation stats for each headline metric × benchmark."""
    out: dict = {"tables": {}, "correlations": {}}
    for metric in metrics:
        out["tables"][metric] = ranking_table(leaderboard, metric)
        out["correlations"][metric] = {
            "MTEB": _correlate(cfg.MTEB_RANK, leaderboard, metric),
            "RTEB": _correlate(cfg.RTEB_RANK, leaderboard, metric),
        }
    return out


if __name__ == "__main__":
    lb = pd.read_csv(cfg.SUMMARY_DIR / "leaderboard.csv", index_col=0)
    res = compute(lb)
    print(res["tables"]["ndcg@10"].to_string(index=False))
    for metric, byb in res["correlations"].items():
        print(f"\n{metric}:")
        for bench, st in byb.items():
            print(
                f"  {bench} (n={st['n']}): Spearman ρ={st['spearman_rho']:+.3f} (p={st['spearman_p']:.3f}) | "
                f"Kendall τ={st['kendall_tau']:+.3f} (p={st['kendall_p']:.3f})"
            )
