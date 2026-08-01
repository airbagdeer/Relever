"""Fusion & oracle headroom: how much is left on the table by using a single
model?

  * **RRF fusion** — Reciprocal Rank Fusion (k0=60) of all 18 models' top-500
    rankings, scored with the same metrics as a single model. Tests whether the
    models are complementary enough that combining them beats the best one.
  * **Union oracle recall@k** — the fraction of positives contained in the
    union of every model's top-k. This is the upper bound on recall@k reachable
    by *any* combination/selection of these 18 models at depth k.

Both require a raw pass over the candidate + label files (one read per
query×model).

r_precision comparability: the fused r_precision is averaged over the same
exact-P@R query set as the leaderboard (``pr_exclude`` drops the queries whose
per-model top-R is not fully labeled), so the fusion-vs-best-single comparison
is like-for-like. Note the fused run is RRF over depth-500 inputs: its P@R is
"P@R of the fused top-500 union", marginally understated where that union is
smaller than R (worst case qid 0427: pool 1054 < R 1065, ≈1%).
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

import data_loading as dl
import eval_config as cfg
import metrics as m

RRF_K0 = 60  # standard reciprocal-rank-fusion constant


def _rrf_fused_rel(ranked_lists: list[dl.RankedList], labels: dict[str, int]) -> list[int]:
    """Reciprocal-rank-fuse the models' rankings; return the fused relevance vector."""
    scores: dict[str, float] = defaultdict(float)
    for rl in ranked_lists:
        for rank, cid in enumerate(rl.ids, start=1):
            scores[cid] += 1.0 / (RRF_K0 + rank)
    fused_ids = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [1 if labels.get(c) == 1 else 0 for c in fused_ids]


def _union_positives_at_k(ranked_lists: list[dl.RankedList], labels: dict[str, int], k: int) -> int:
    union: set[str] = set()
    for rl in ranked_lists:
        union.update(rl.ids[:k])
    return sum(1 for c in union if labels.get(c) == 1)


def compute(leaderboard: pd.DataFrame, verbose: bool = True,
            pr_exclude: set[int] | None = None) -> dict:
    """``pr_exclude`` = event_ids whose P@R is inexact (NaN on the leaderboard);
    the fused r_precision skips them so both sides average the same query set."""
    pr_exclude = pr_exclude or set()
    fused_metric_sums: dict[str, float] = defaultdict(float)
    union_recall_sums: dict[int, float] = defaultdict(float)
    n_eval = 0
    rp_sum = 0.0
    n_rp = 0

    eids = dl.event_ids()
    for n, eid in enumerate(eids, 1):
        ql = dl.load_query_labels(eid)
        if ql.R == 0:
            continue
        n_eval += 1
        ranked = [dl.load_ranked_list(info.alias, eid, ql.labels) for info in cfg.MODELS]

        fused_rel = _rrf_fused_rel(ranked, ql.labels)
        fm = m.compute_query_metrics(fused_rel, ql.R, cfg.K_VALUES, cfg.SUCCESS_K_VALUES)
        for key, val in fm.items():
            if key == "r_precision":
                continue
            fused_metric_sums[key] += val
        if eid not in pr_exclude:
            rp_sum += fm["r_precision"]
            n_rp += 1

        for k in cfg.K_VALUES:
            union_recall_sums[k] += _union_positives_at_k(ranked, ql.labels, k) / ql.R

        if verbose and (n % 200 == 0 or n == len(eids)):
            print(f"  fusion/oracle pass {n}/{len(eids)}", flush=True)

    fusion_row = {k: v / n_eval for k, v in fused_metric_sums.items()}
    fusion_row["r_precision"] = rp_sum / n_rp if n_rp else float("nan")
    union_oracle = {k: union_recall_sums[k] / n_eval for k in cfg.K_VALUES}

    # Best single model per metric/k (column max of the leaderboard).
    best_single_recall = {k: float(leaderboard[f"recall@{k}"].max()) for k in cfg.K_VALUES}
    headline = ["ndcg@10", "r_precision", "recall@500", "map", "mrr"]
    best_single_headline = {met: float(leaderboard[met].max()) for met in headline}

    return {
        "n_eval": n_eval,
        "n_rp": n_rp,
        "fusion_row": fusion_row,
        "union_oracle_recall": union_oracle,
        "best_single_recall": best_single_recall,
        "best_single_headline": best_single_headline,
    }


def headline_frame(res: dict) -> pd.DataFrame:
    """RRF fusion vs best single model on the headline metrics."""
    mets = ["ndcg@10", "r_precision", "recall@500", "map", "mrr"]
    rows = []
    for met in mets:
        rows.append(
            {
                "metric": met,
                "best_single": round(res["best_single_headline"][met], 4),
                "rrf_fusion": round(res["fusion_row"][met], 4),
                "abs_gain": round(res["fusion_row"][met] - res["best_single_headline"][met], 4),
            }
        )
    return pd.DataFrame(rows)


def oracle_frame(res: dict) -> pd.DataFrame:
    """Best-single vs union-oracle recall@k (the complementarity headroom)."""
    rows = []
    for k in cfg.K_VALUES:
        bs = res["best_single_recall"][k]
        uo = res["union_oracle_recall"][k]
        rows.append(
            {
                "k": k,
                "best_single_recall": round(bs, 4),
                "union_oracle_recall": round(uo, 4),
                "headroom": round(uo - bs, 4),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    lb = pd.read_csv(cfg.SUMMARY_DIR / "leaderboard.csv", index_col=0)
    # NaN r_precision in the per-query CSV marks the P@R-inexact queries.
    pq = pd.read_csv(cfg.RESULTS_DIR / "all_models_per_query.csv")
    exclude = set(pq.loc[pq["r_precision"].isna(), "event_id"].astype(int))
    res = compute(lb, pr_exclude=exclude)
    print(f"\nn_eval={res['n_eval']} (r_precision over {res['n_rp']})")
    print("\nRRF fusion vs best single:")
    print(headline_frame(res).to_string(index=False))
    print("\nUnion-oracle recall@k vs best single:")
    print(oracle_frame(res).to_string(index=False))
