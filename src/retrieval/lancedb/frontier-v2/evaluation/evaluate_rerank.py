"""Evaluate the (reranker x variant x retriever) grid over the frontier-v2 benchmark.

Reuses ``metrics.py`` and the aggregation rules of ``evaluate_frontier.py`` unchanged,
so a reranked row is directly comparable to the published first-stage leaderboard:

  * queries with R == 0 (71 of 1100) are excluded from every mean;
  * ``r_precision`` is recomputed from a deeper list when R > 500 and set to NaN for the
    22 queries where the delta labelling never converged;
  * ``p_at_r_exact`` is a *query* property and is unchanged by reranking (reranking
    permutes the top-500, so for R<=500 the top-R stays inside the fully-labeled pool,
    and for R>500 the id set of the top-R is identical).

Each query is loaded once and every system reuses it: the 18 retrievers' candidate
lists and the labels are hoisted out of the system loop, and each system's pooled score
file is read once per query rather than once per (query, retriever). Baseline
(first-stage) metrics are recomputed inline rather than read from
``results/first_stage/`` so a stale CSV can never silently define the delta.

Outputs
-------
    results/rerank/all_systems_per_query.csv      long format, the analysis input
    results/rerank/<system>/<retriever>/metrics.csv
    summary/rerank/leaderboard.csv                mean per (system, retriever)
    summary/rerank/delta_r_precision.csv          the 18 x N grid of dP@R

where ``<system>`` is ``<reranker>__<variant>@<rerank_depth>``.

Usage:
    uv run python evaluate_rerank.py                       # every system with a complete cache
    uv run python evaluate_rerank.py --rerankers identity-constant --limit-queries 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import _bootstrap  # noqa: F401  (sys.path: `config` + the rerank package's score_store)
import data_loading as dl
import eval_config as cfg
import metrics as m
import pandas as pd
import score_store
from evaluate_frontier import metric_columns
from reranker_registry import LISTWISE_ALIASES, spec_for, variants_for

BASELINE_SYSTEM = "first_stage"


@dataclass(frozen=True)
class System:
    """One column of the results grid: a reranker, its instruction variant, a depth."""

    reranker: str
    variant: str
    rerank_depth: int
    paradigm: str

    @property
    def name(self) -> str:
        return f"{self.reranker}__{self.variant}@{self.rerank_depth}"


def _pointwise_cache_complete(reranker: str, variant: str, event_ids: list[int]) -> bool:
    return all(score_store.scores_exist(reranker, variant, e) for e in event_ids)


def _listwise_cache_complete(reranker: str, event_ids: list[int]) -> bool:
    return all(
        score_store.listwise_exists(reranker, alias, e)
        for e in event_ids
        for alias in cfg.MODEL_ALIASES
    )


def _listwise_depth(reranker: str, event_ids: list[int]) -> int:
    """Read the rerank depth off disk instead of trusting a constant.

    ``rerank_listwise.py`` records the depth it actually used. If a shard were rerun with
    ``--depth 50`` (say, to dodge an OOM), the metrics would still be *computed* correctly
    — ``apply_listwise_permutation`` derives the prefix length from the permutation itself
    — but the results column would be labelled ``@100``. A mislabelled column in a paper
    is a wrong number. Take the depth from the data and refuse to proceed if it varies.
    """
    doc = score_store.read_listwise(reranker, cfg.MODEL_ALIASES[0], event_ids[0])
    depth = int(doc["depth"])
    if depth != len(doc["order"]):
        raise ValueError(
            f"[{reranker}] permutation for ({cfg.MODEL_ALIASES[0]}, q{event_ids[0]}) records "
            f"depth={depth} but holds {len(doc['order'])} ids."
        )
    if depth != cfg.LISTWISE_RERANK_DEPTH:
        print(f"  note: {reranker} was run at depth {depth}, not the default "
              f"{cfg.LISTWISE_RERANK_DEPTH}; reporting the on-disk depth.")
    return depth


def discover_systems(event_ids: list[int], requested: list[str] | None) -> list[System]:
    """Every (reranker, variant, depth) whose cache covers all ``event_ids``.

    Incomplete caches are skipped with a message rather than evaluated on a subset —
    a leaderboard row computed over a different query set is not comparable to the
    others and must never quietly appear next to them.
    """
    aliases = requested if requested else _all_known_aliases()
    systems: list[System] = []
    for alias in aliases:
        spec = spec_for(alias)
        if spec.paradigm == "listwise":
            if _listwise_cache_complete(alias, event_ids):
                systems.append(System(alias, "default", _listwise_depth(alias, event_ids), "listwise"))
            else:
                print(f"  skipping {alias}: listwise permutations incomplete")
            continue
        for variant in variants_for(alias):
            if not _pointwise_cache_complete(alias, variant, event_ids):
                print(f"  skipping {alias}/{variant}: pooled score cache incomplete")
                continue
            for depth in cfg.RERANK_DEPTHS:
                systems.append(System(alias, variant, depth, "pointwise"))
    return systems


def _all_known_aliases() -> list[str]:
    from reranker_registry import RERANKER_ALIASES  # noqa: PLC0415

    return list(RERANKER_ALIASES)


def _query_metrics(rlist, R: int) -> tuple[dict, list[int]]:
    """The standard suite at the design depth, with exact P@R when R > 500."""
    rel_top = rlist.rel[: cfg.RETRIEVAL_DEPTH]
    qm = m.compute_query_metrics(rel_top, R, cfg.K_VALUES, cfg.SUCCESS_K_VALUES)
    if R > cfg.RETRIEVAL_DEPTH:
        qm["r_precision"] = m.r_precision(rlist.rel, R)
    return qm, rel_top


def evaluate(systems: list[System], event_ids: list[int], verbose: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    pr_inexact: set[int] = set()

    for n, eid in enumerate(event_ids, 1):
        ql = dl.load_query_labels(eid)
        depth = max(cfg.RETRIEVAL_DEPTH, min(ql.R, cfg.MAX_RANK))

        # Hoisted out of the system loop: loaded once, reused by every system.
        bases = {a: dl.load_ranked_list(a, eid, ql.labels, depth=depth) for a in cfg.MODEL_ALIASES}

        # Exactness is a query property computed on the first-stage lists. Reranking
        # permutes the top-500 and cannot change it, so it is computed once here.
        all_exact = all(
            b.n_retrieved >= ql.R and not any(cid not in ql.labels for cid in b.ids[: ql.R])
            for b in bases.values()
        )
        q_exact = ql.R > 0 and all_exact
        if ql.R > 0 and not q_exact:
            pr_inexact.add(eid)

        def emit(system_name: str, reranker: str, variant: str, rerank_depth, retriever: str, rlist):
            qm, rel_top = _query_metrics(rlist, ql.R)
            if ql.R > 0 and not q_exact:
                qm["r_precision"] = float("nan")
            rows.append({
                "system": system_name, "reranker": reranker, "variant": variant,
                "rerank_depth": rerank_depth, "retriever": retriever, "event_id": eid,
                "R": ql.R, "R_pool": ql.R_pool,
                "n_retrieved": min(rlist.n_retrieved, cfg.RETRIEVAL_DEPTH),
                "n_rel_retrieved": sum(rel_top), "p_at_r_exact": q_exact, **qm,
            })

        for retriever, base in bases.items():
            emit(BASELINE_SYSTEM, BASELINE_SYSTEM, "-", 0, retriever, base)

        for system in systems:
            if system.paradigm == "pointwise":
                cached = score_store.read_scores(system.reranker, system.variant, eid)
                for retriever, base in bases.items():
                    rlist = dl.rerank_ranked_list(
                        base, cached, system.rerank_depth, ql.labels, reranker=system.reranker
                    )
                    emit(system.name, system.reranker, system.variant,
                         system.rerank_depth, retriever, rlist)
            else:
                for retriever, base in bases.items():
                    doc = score_store.read_listwise(system.reranker, retriever, eid)
                    # Every file, not just the probe in discover_systems: a single shard
                    # rerun at another depth would otherwise be labelled with the rest.
                    if int(doc["depth"]) != system.rerank_depth:
                        raise ValueError(
                            f"[{system.reranker}] ({retriever}, q{eid}) was reranked at depth "
                            f"{doc['depth']}, but this system reports depth "
                            f"{system.rerank_depth}. Rescore that shard, or evaluate the "
                            f"depths as separate systems."
                        )
                    rlist = dl.apply_listwise_permutation(
                        base, list(doc["order"]), ql.labels, reranker=system.reranker
                    )
                    emit(system.name, system.reranker, system.variant,
                         system.rerank_depth, retriever, rlist)

        if verbose and (n % 50 == 0 or n == len(event_ids)):
            print(f"  evaluated {n}/{len(event_ids)} queries", flush=True)

    if pr_inexact:
        print(f"  r_precision NaN'd on {len(pr_inexact)} queries (delta labelling never converged)")
    ordered = ["system", "reranker", "variant", "rerank_depth", "retriever", "event_id",
               "R", "R_pool", "n_retrieved", "n_rel_retrieved", "p_at_r_exact", *metric_columns()]
    return pd.DataFrame(rows)[ordered]


def build_leaderboard(per_query: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric over the R>0 queries, per (system, retriever)."""
    metric_cols = metric_columns()
    evalable = per_query[per_query["R"] > 0]
    lb = evalable.groupby(["system", "retriever"])[metric_cols].mean()
    lb.insert(0, "n_queries", evalable.groupby(["system", "retriever"]).size())
    lb.insert(1, "n_queries_r_precision",
              evalable.groupby(["system", "retriever"])["r_precision"].count())
    return lb.reset_index()


def build_delta_grid(leaderboard: pd.DataFrame, metric: str = "r_precision") -> pd.DataFrame:
    """The headline artifact: ``metric(reranked) - metric(first_stage)`` per cell.

    Rows are the 18 retrievers, columns the reranker systems. A column of near-zeros is
    the study's claim: the reranker did not rescue the retriever on this task.
    """
    pivot = leaderboard.pivot(index="retriever", columns="system", values=metric)
    if BASELINE_SYSTEM not in pivot.columns:
        raise KeyError(f"baseline column {BASELINE_SYSTEM!r} missing from the leaderboard.")
    baseline = pivot[BASELINE_SYSTEM]
    delta = pivot.drop(columns=[BASELINE_SYSTEM]).sub(baseline, axis=0)
    delta.insert(0, f"{metric}__first_stage", baseline)
    return delta.reindex(list(cfg.MODEL_ALIASES))


def write_outputs(per_query: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    cfg.RESULTS_RERANK_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SUMMARY_RERANK_DIR.mkdir(parents=True, exist_ok=True)

    per_query.sort_values(["system", "retriever", "event_id"]).to_csv(
        cfg.RESULTS_RERANK_DIR / "all_systems_per_query.csv", index=False
    )
    for (system, retriever), sub in per_query.groupby(["system", "retriever"]):
        out_dir = cfg.RESULTS_RERANK_DIR / system / retriever
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.drop(columns=["system", "retriever"]).sort_values("event_id").to_csv(
            out_dir / "metrics.csv", index=False
        )

    leaderboard.to_csv(cfg.SUMMARY_RERANK_DIR / "leaderboard.csv", index=False)
    for metric in ("r_precision", f"ndcg@{cfg.HEADLINE_NDCG_K}"):
        grid = build_delta_grid(leaderboard, metric)
        safe = metric.replace("@", "_at_")
        grid.to_csv(cfg.SUMMARY_RERANK_DIR / f"delta_{safe}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the reranker x retriever grid.")
    parser.add_argument("--rerankers", nargs="*", default=None,
                        help="Subset of aliases (default: every one with a complete cache).")
    parser.add_argument("--limit-queries", type=int, default=None, help="Smoke knob.")
    args = parser.parse_args()

    eids = dl.event_ids()
    if args.limit_queries:
        eids = eids[: args.limit_queries]

    print(f"Discovering systems over {len(eids)} queries ...")
    systems = discover_systems(eids, args.rerankers)
    if not systems:
        raise SystemExit("No reranker has a complete score cache. Run score_pool.py first.")
    print(f"  {len(systems)} system(s): {', '.join(s.name for s in systems)}")
    if any(s.reranker in LISTWISE_ALIASES for s in systems):
        print(f"  listwise systems are evaluated at depth {cfg.LISTWISE_RERANK_DEPTH} "
              f"(their published protocol); compare them to the @{cfg.LISTWISE_RERANK_DEPTH} "
              f"pointwise columns, not the @{cfg.RETRIEVAL_DEPTH} ones.")

    per_query = evaluate(systems, eids)
    leaderboard = build_leaderboard(per_query)
    write_outputs(per_query, leaderboard)

    print(f"\nWrote {cfg.RESULTS_RERANK_DIR} and {cfg.SUMMARY_RERANK_DIR}")
    grid = build_delta_grid(leaderboard, "r_precision")
    print("\nDelta P@R vs first stage (rows = retrievers):")
    print(grid.to_string(float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
