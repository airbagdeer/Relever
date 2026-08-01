"""Core evaluation pass: compute per-query IR metrics for all 18 models and
roll them up into a leaderboard.

Outputs (written by :func:`write_outputs`):
  * ``results/first_stage/<alias>/metrics.csv`` — one row per query (all 1100,
    including R==0 rows, flagged via the ``R`` column) with every metric.
  * ``results/first_stage/all_models_per_query.csv`` — the same data in long
    format (model, event_id, ...), the shared input for the significance and
    per-analysis modules.
  * ``summary/evaluation/leaderboard.csv`` — one row per model, mean of each
    metric over the queries with R>0, sorted by nDCG@10.

Aggregation rules:
  * Queries with R==0 (71 of 1100) are excluded from every leaderboard mean so
    all models are scored on the identical 1029-query set (required for the
    paired significance tests).
  * ``R`` comes from the merged main+delta labels (see ``data_loading``). The
    standard suite is computed at the top-500 design depth; ``r_precision``
    alone is recomputed from a deeper list (top-R) when R > 500, which the
    delta labels make fully judged for the converged queries.
  * ``r_precision`` is kept only where it is *exact* — every model's top-R
    fully labeled (``p_at_r_exact``). For the queries where the delta labelling
    did not converge (22; see the paper's extended-pooling appendix) it is set to
    NaN, so every r_precision mean and
    significance test runs on the identical exact set (1007 queries). All other
    metrics keep the full R>0 set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import data_loading as dl
import eval_config as cfg
import metrics as m


def metric_columns() -> list[str]:
    """Ordered metric column names (matches metrics.compute_query_metrics keys)."""
    cols: list[str] = []
    for k in cfg.K_VALUES:
        cols += [f"ndcg@{k}", f"precision@{k}", f"recall@{k}"]
    for k in cfg.SUCCESS_K_VALUES:
        cols.append(f"success@{k}")
    cols += ["mrr", "map", "r_precision"]
    return cols


@dataclass(frozen=True)
class EvalResults:
    per_query: pd.DataFrame      # long: model, event_id, R, R_pool, ..., p_at_r_exact, <metrics>
    leaderboard: pd.DataFrame    # index=model, columns=mean metrics (+ n_queries), R>0 only
    n_queries_total: int
    n_queries_eval: int          # R > 0
    n_queries_pr_exact: int      # R > 0 and every model's top-R fully labeled
    excluded_event_ids: list[int]     # R == 0
    pr_excluded_event_ids: list[int]  # R > 0 but P@R inexact (r_precision = NaN)
    total_missing_labels: int
    total_failed_labels: int


def evaluate_all(verbose: bool = True) -> EvalResults:
    metric_cols = metric_columns()
    rows: list[dict] = []
    excluded: list[int] = []
    pr_excluded: list[int] = []
    total_missing = 0
    total_failed = 0

    eids = dl.event_ids()
    for n, eid in enumerate(eids, 1):
        ql = dl.load_query_labels(eid)
        total_failed += ql.n_failed
        if ql.R == 0:
            excluded.append(eid)
        # Standard suite at the top-500 design depth; when R > 500 also load down
        # to top-R (the delta labels judge that band) so Precision@R is exact.
        depth = max(cfg.RETRIEVAL_DEPTH, min(ql.R, cfg.MAX_RANK))
        q_rows: list[dict] = []
        all_exact = True
        for info in cfg.MODELS:
            rlist = dl.load_ranked_list(info.alias, eid, ql.labels, depth=depth)
            total_missing += rlist.n_missing_labels
            rel_top = rlist.rel[: cfg.RETRIEVAL_DEPTH]
            qm = m.compute_query_metrics(
                rel_top, ql.R, cfg.K_VALUES, cfg.SUCCESS_K_VALUES
            )
            if ql.R > cfg.RETRIEVAL_DEPTH:
                qm["r_precision"] = m.r_precision(rlist.rel, ql.R)
            # Exact P@R needs the model's whole top-R materialised and labeled.
            exact = rlist.n_retrieved >= ql.R and not any(
                cid not in ql.labels for cid in rlist.ids[: ql.R]
            )
            all_exact &= exact
            q_rows.append(
                {
                    "model": info.alias,
                    "event_id": eid,
                    "R": ql.R,
                    "R_pool": ql.R_pool,
                    "n_retrieved": min(rlist.n_retrieved, cfg.RETRIEVAL_DEPTH),
                    "n_rel_retrieved": sum(rel_top),
                    **qm,
                }
            )
        # P@R exactness is a query property: one inexact model breaks the paired
        # design, so the query drops out of the r_precision aggregate entirely.
        q_exact = ql.R > 0 and all_exact
        if ql.R > 0 and not q_exact:
            pr_excluded.append(eid)
        for row in q_rows:
            row["p_at_r_exact"] = q_exact
            if ql.R > 0 and not q_exact:
                row["r_precision"] = float("nan")
        rows.extend(q_rows)
        if verbose and (n % 100 == 0 or n == len(eids)):
            print(f"  evaluated {n}/{len(eids)} queries", flush=True)

    ordered_cols = ["model", "event_id", "R", "R_pool", "n_retrieved",
                    "n_rel_retrieved", "p_at_r_exact", *metric_cols]
    per_query = pd.DataFrame(rows)[ordered_cols]

    # Leaderboard: mean over R>0 queries, per model. NaN r_precision rows (the
    # P@R-inexact queries) are skipped by mean() and counted separately.
    evalable = per_query[per_query["R"] > 0]
    lb = evalable.groupby("model")[metric_cols].mean()
    lb.insert(0, "n_queries", evalable.groupby("model").size())
    lb.insert(1, "n_queries_r_precision", evalable.groupby("model")["r_precision"].count())
    # Preserve registry order, then we sort by headline for display elsewhere.
    lb = lb.reindex(list(cfg.MODEL_ALIASES))

    n_eval = int((per_query["R"] > 0).sum() // max(len(cfg.MODELS), 1))
    return EvalResults(
        per_query=per_query,
        leaderboard=lb,
        n_queries_total=len(eids),
        n_queries_eval=n_eval,
        n_queries_pr_exact=n_eval - len(pr_excluded),
        excluded_event_ids=excluded,
        pr_excluded_event_ids=sorted(pr_excluded),
        total_missing_labels=total_missing,
        total_failed_labels=total_failed,
    )


def write_outputs(results: EvalResults) -> None:
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # Per-model per-query CSVs (machine-readable, under results/).
    for alias, sub in results.per_query.groupby("model"):
        out_dir = cfg.RESULTS_DIR / alias
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.drop(columns=["model"]).sort_values("event_id").to_csv(
            out_dir / "metrics.csv", index=False
        )

    # Combined long-format CSV (shared analysis input).
    results.per_query.sort_values(["model", "event_id"]).to_csv(
        cfg.RESULTS_DIR / "all_models_per_query.csv", index=False
    )

    # Leaderboard (sorted by headline nDCG@10 for readability).
    headline = f"ndcg@{cfg.HEADLINE_NDCG_K}"
    lb_sorted = results.leaderboard.sort_values(headline, ascending=False)
    lb_sorted.to_csv(cfg.SUMMARY_DIR / "leaderboard.csv")


if __name__ == "__main__":
    print("Running frontier-v2 evaluation over all queries x 18 models ...")
    res = evaluate_all()
    write_outputs(res)
    print(
        f"Done. queries_total={res.n_queries_total} "
        f"evaluated(R>0)={res.n_queries_eval} excluded(R=0)={len(res.excluded_event_ids)} "
        f"missing_labels={res.total_missing_labels} failed_labels={res.total_failed_labels}"
    )
    print(
        f"Exact P@R set: {res.n_queries_pr_exact} queries; "
        f"excluded from r_precision (inexact top-R): {len(res.pr_excluded_event_ids)}"
    )
    if res.pr_excluded_event_ids:
        print("  " + " ".join(cfg.qid(e) for e in res.pr_excluded_event_ids))
    headline = f"ndcg@{cfg.HEADLINE_NDCG_K}"
    print("\nTop 5 by", headline, ":")
    print(res.leaderboard.sort_values(headline, ascending=False)[[headline, "r_precision", "map", "mrr"]].head().round(4).to_string())
