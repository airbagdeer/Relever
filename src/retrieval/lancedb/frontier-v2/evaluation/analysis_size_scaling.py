"""Model-size scaling: within each family that ships a parameter ladder
(F2LLM-v2 4B/8B/14B, Octen 0.6B/4B/8B, Qwen3-Embedding 4B/8B), does scaling up
parameters improve relation-evidence retrieval?

Outputs a tidy per-(family,size) metric table plus a per-family verdict
(monotonic? best size? delta from smallest to largest).
"""

from __future__ import annotations

import pandas as pd

import eval_config as cfg

METRICS = ("ndcg@10", "r_precision", "recall@500", "map")


def scaling_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, aliases in cfg.SIZE_LADDERS.items():
        for alias in aliases:
            info = cfg.MODEL_BY_ALIAS[alias]
            rows.append(
                {
                    "family": family,
                    "model": alias,
                    "params_b": info.params_b,
                    **{m: round(float(leaderboard.loc[alias, m]), 4) for m in METRICS},
                }
            )
    return pd.DataFrame(rows)


def family_verdicts(table: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    """Per family: is `metric` monotonically increasing with size? best size? delta."""
    rows = []
    for family, sub in table.groupby("family"):
        sub = sub.sort_values("params_b")
        vals = sub[metric].tolist()
        sizes = sub["params_b"].tolist()
        best_idx = int(sub[metric].to_numpy().argmax())
        monotone = all(b >= a for a, b in zip(vals, vals[1:]))
        rows.append(
            {
                "family": family,
                "sizes_b": ", ".join(f"{s:g}" for s in sizes),
                f"{metric}_by_size": ", ".join(f"{v:.4f}" for v in vals),
                "best_size_b": sizes[best_idx],
                "monotonic_increasing": monotone,
                "delta_small_to_large": round(vals[-1] - vals[0], 4),
            }
        )
    return pd.DataFrame(rows)


def compute(leaderboard: pd.DataFrame) -> dict:
    table = scaling_table(leaderboard)
    return {
        "table": table,
        "verdicts": {m: family_verdicts(table, m) for m in ("ndcg@10", "r_precision")},
    }


if __name__ == "__main__":
    lb = pd.read_csv(cfg.SUMMARY_DIR / "leaderboard.csv", index_col=0)
    res = compute(lb)
    print(res["table"].to_string(index=False))
    print("\nVerdicts (nDCG@10):")
    print(res["verdicts"]["ndcg@10"].to_string(index=False))
