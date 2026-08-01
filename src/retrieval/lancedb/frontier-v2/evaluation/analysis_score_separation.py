"""Score-separation diagnostics: within each model's own retrieved list, how well
do the raw similarity scores separate the pooled positives from the negatives?

For every query we compute the ROC-AUC of (score -> label) over the model's
retrieved candidates (only queries whose retrieved list contains at least one
positive *and* one negative are defined), then average per model. AUC is
scale-free, so it is comparable across models with different score ranges
(cosine, max-sim, etc.). We then correlate mean AUC with nDCG@10 across models to
see how much of the ranking quality is explained by raw score separation.

Requires a raw pass over the candidate + label files.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import data_loading as dl
import eval_config as cfg


def compute(leaderboard: pd.DataFrame, verbose: bool = True) -> dict:
    auc_sums: dict[str, float] = defaultdict(float)
    auc_counts: dict[str, int] = defaultdict(int)

    eids = dl.event_ids()
    for n, eid in enumerate(eids, 1):
        ql = dl.load_query_labels(eid)
        if ql.R == 0:
            continue
        for info in cfg.MODELS:
            rl = dl.load_ranked_list(info.alias, eid, ql.labels)
            rel = np.asarray(rl.rel)
            if rel.sum() == 0 or rel.sum() == rel.size:
                continue  # AUC undefined without both classes present
            auc = roc_auc_score(rel, np.asarray(rl.scores))
            auc_sums[info.alias] += float(auc)
            auc_counts[info.alias] += 1
        if verbose and (n % 200 == 0 or n == len(eids)):
            print(f"  score-separation pass {n}/{len(eids)}", flush=True)

    rows = []
    for info in cfg.MODELS:
        c = auc_counts[info.alias]
        rows.append(
            {
                "model": info.alias,
                "mean_auc": round(auc_sums[info.alias] / c, 4) if c else float("nan"),
                "n_queries_auc": c,
                "ndcg@10": round(float(leaderboard.loc[info.alias, "ndcg@10"]), 4),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)

    valid = table.dropna(subset=["mean_auc"])
    rho, rho_p = spearmanr(valid["mean_auc"], valid["ndcg@10"])
    return {
        "table": table,
        "spearman_auc_vs_ndcg10": {"rho": float(rho), "p": float(rho_p), "n": int(len(valid))},
    }


if __name__ == "__main__":
    lb = pd.read_csv(cfg.SUMMARY_DIR / "leaderboard.csv", index_col=0)
    res = compute(lb)
    print(res["table"].to_string(index=False))
    s = res["spearman_auc_vs_ndcg10"]
    print(f"\nSpearman(mean_auc, ndcg@10) = {s['rho']:+.3f} (p={s['p']:.4f}, n={s['n']})")
