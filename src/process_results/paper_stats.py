"""Recompute every statistic quoted in the RELEVER paper from raw data.

Single source of truth for `paper/macros.tex`: run this before touching the
paper and paste the printed macro block. Provenance for each number is the
data path printed next to it. Reuses the frontier-v2 evaluation package
(data_loading / eval_config) so the label-merge semantics (main ∪ delta,
delta wins) are identical to the evaluation itself.

Run:
    uv run python src/process_results/paper_stats.py [--skip-corpus]

--skip-corpus skips the 12 GB splade_wikipedia_v2 scan (corpus size +
per-query candidate stats) and reuses nothing: the numbers are simply
omitted. Everything else takes seconds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "src" / "retrieval" / "lancedb" / "frontier-v2" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

import data_loading  # noqa: E402
import eval_config as cfg  # noqa: E402

SPLADE_V2_DIR = PROJECT_ROOT / "data" / "splade_wikipedia_v2"
MANIFEST_PATH = cfg.FRONTIER_CLASSIFIED_DIR / "_manifest.json"
DELTA_MANIFEST_PATH = cfg.FRONTIER_DELTA_DIR / "_manifest.json"
LEADERBOARD_CSV = cfg.SUMMARY_DIR / "leaderboard.csv"
RERANK_LEADERBOARD_CSV = cfg.SUMMARY_RERANK_DIR / "leaderboard.csv"
RERANK_SIG_CSV = (cfg.SUMMARY_RERANK_DIR
                  / "significance_rerank_vs_first_stage__rprecision.csv")
RERANK_SIG_INSTRUCT_CSV = (cfg.SUMMARY_RERANK_DIR
                           / "significance_rerank_instruct__rprecision.csv")

# Person-ish tokens for the Ethics scan (queries about people as a *type* are
# public-figure categories; the scan surfaces them for manual review).
PERSON_TYPE_TOKENS = (
    "people", "person", "players", "athletes", "actors", "actresses",
    "musicians", "singers", "authors", "writers", "politicians", "leaders",
    "ceos", "executives", "scientists", "researchers", "artists", "directors",
    "coaches", "founders", "entrepreneurs", "journalists", "celebrities",
    "wrestlers", "chefs", "philanthropists", "economists", "historians",
)


def fmt(n: float | int, comma: bool = True) -> str:
    if isinstance(n, float) and not n.is_integer():
        return f"{n:,.1f}" if comma else f"{n:.1f}"
    return f"{int(n):,}" if comma else str(int(n))


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def query_stats() -> dict:
    with cfg.V2_RELATIONS_PATH.open() as f:
        rows = json.load(f)
    n = len(rows)
    eids = sorted(int(r["event_id"]) for r in rows)
    relations = Counter(r["relation"] for r in rows)
    types = {r["abstract_entity_type"] for r in rows}
    constants = {r["constant_entity"] for r in rows}
    inst_counts = [len(r.get("abstract_entity_instances", [])) for r in rows]
    top_rel, top_rel_n = relations.most_common(1)[0]

    person_type_rows = [
        r for r in rows
        if any(tok in r["abstract_entity_type"].lower().split() for tok in PERSON_TYPE_TOKENS)
    ]
    return {
        "n_queries": n,
        "eid_min": eids[0],
        "eid_max": eids[-1],
        "eids_contiguous": eids == list(range(eids[0], eids[-1] + 1)),
        "n_relations": len(relations),
        "n_types": len(types),
        "n_constants": len(constants),
        "inst_min": min(inst_counts),
        "inst_max": max(inst_counts),
        "inst_mean": sum(inst_counts) / n,
        "top_relation": top_rel,
        "top_relation_share_pct": 100.0 * top_rel_n / n,
        "n_person_type_queries": len(person_type_rows),
        "person_type_examples": sorted({r["abstract_entity_type"] for r in person_type_rows})[:12],
    }


# --------------------------------------------------------------------------- #
# Corpus (12 GB scan)
# --------------------------------------------------------------------------- #
def corpus_stats(valid_eids: set[int]) -> dict:
    all_ids: set[str] = set()
    per_query_unique: list[int] = []
    n_files_used = 0
    stray_files: list[str] = []
    for p in sorted(SPLADE_V2_DIR.glob("*.json"), key=lambda q: int(q.stem)):
        eid = int(p.stem)
        if eid not in valid_eids:
            stray_files.append(p.name)
            continue
        with p.open() as f:
            doc = json.load(f)
        ids = {h["id"] for h in doc.get("bm25_results", [])}
        ids |= {h["id"] for h in doc.get("splade_results", [])}
        per_query_unique.append(len(ids))
        all_ids |= ids
        n_files_used += 1
    return {
        "corpus_size": len(all_ids),
        "n_files_used": n_files_used,
        "stray_files": stray_files,
        "cand_min": min(per_query_unique),
        "cand_median": statistics.median(per_query_unique),
        "cand_mean": sum(per_query_unique) / len(per_query_unique),
        "cand_max": max(per_query_unique),
    }


# --------------------------------------------------------------------------- #
# Labels (main + delta, via the evaluation package's own merge)
# --------------------------------------------------------------------------- #
def label_stats() -> dict:
    eids = data_loading.event_ids()
    per_query = [data_loading.load_query_labels(e) for e in eids]

    n_main = sum(q.n_pool - q.n_delta for q in per_query)
    n_delta = sum(q.n_delta for q in per_query)
    n_failed = sum(q.n_failed for q in per_query)
    pos_merged = sum(q.R for q in per_query)
    pos_main = sum(q.R_pool for q in per_query)

    r_values = [q.R for q in per_query]
    r_pos = [r for r in r_values if r > 0]
    n_r0 = sum(1 for r in r_values if r == 0)
    n_delta_queries = sum(1 for q in per_query if q.n_delta > 0)
    n_rpool_gt500 = sum(1 for q in per_query if q.R_pool > 500)
    n_rmerged_gt500 = sum(1 for r in r_values if r > 500)

    buckets: dict[str, int] = {}
    for lo, hi in cfg.R_BUCKETS:
        label = f"{lo}-{hi}" if hi is not None else f"{lo}+"
        buckets[label] = sum(1 for r in r_pos if r >= lo and (hi is None or r <= hi))
    n_r_lt50 = sum(1 for r in r_pos if r < 50)

    return {
        "n_label_queries": len(eids),
        "judged_main": n_main,
        "judged_delta": n_delta,
        "judged_total": n_main + n_delta,
        "n_failed": n_failed,
        "pos_main": pos_main,
        "pos_merged": pos_merged,
        "pos_rate_main_pct": 100.0 * pos_main / n_main,
        "pos_rate_merged_pct": 100.0 * pos_merged / (n_main + n_delta),
        "n_r0": n_r0,
        "n_eval": len(r_pos),
        "r_median": statistics.median(r_pos),
        "r_mean": sum(r_pos) / len(r_pos),
        "r_max": max(r_pos),
        "n_delta_queries": n_delta_queries,
        "n_rpool_gt500": n_rpool_gt500,
        "n_rmerged_gt500": n_rmerged_gt500,
        "r_buckets": buckets,
        "n_r_lt50": n_r_lt50,
        "pct_r_lt50": 100.0 * n_r_lt50 / len(r_pos),
    }


# --------------------------------------------------------------------------- #
# Labeling run: cost / tokens / batching, from the Batch API manifests
# --------------------------------------------------------------------------- #
def _manifest_totals(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        man = json.load(f)
    queries = man.get("queries", {})
    tot = {
        "n_queries": len(queries),
        "n_requests": 0,
        "n_predictions": 0,
        "cost_batch": 0.0,
        "cost_uncached": 0.0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }
    for q in queries.values():
        tot["n_requests"] += q.get("n_requests", 0)
        tot["n_predictions"] += q.get("n_predictions", 0)
        tot["cost_batch"] += q.get("cost_batch", 0.0)
        tot["cost_uncached"] += q.get("cost_uncached", 0.0)
        toks = q.get("tokens", {})
        for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"):
            tot[k] += toks.get(k, 0)
    tot["config"] = man.get("config", {})
    return tot


def labeling_run_stats() -> dict:
    main = _manifest_totals(MANIFEST_PATH)
    delta = _manifest_totals(DELTA_MANIFEST_PATH)
    out = {"main": main, "delta": delta}
    if main:
        out["batch_size_mean"] = main["n_predictions"] / max(main["n_requests"], 1)
    if main and delta:
        out["cost_batch_total"] = main["cost_batch"] + delta["cost_batch"]
    elif main:
        out["cost_batch_total"] = main["cost_batch"]
    return out


# --------------------------------------------------------------------------- #
# Cross-checks against the published evaluation CSV
# --------------------------------------------------------------------------- #
def leaderboard_crosscheck() -> dict:
    if not LEADERBOARD_CSV.exists():
        return {}
    import csv

    with LEADERBOARD_CSV.open() as f:
        rows = list(csv.DictReader(f))
    top = max(rows, key=lambda r: float(r["ndcg@10"]))
    return {
        "n_models": len(rows),
        "eval_n_queries": int(top["n_queries"]),
        "eval_n_queries_rprec": int(top["n_queries_r_precision"]),
        "best_model": top["model"],
        "best_ndcg10": float(top["ndcg@10"]),
        "best_rprec": float(top["r_precision"]),
        "best_mrr": float(top["mrr"]),
    }


# --------------------------------------------------------------------------- #
# Reranking study (summary/rerank, evaluate_rerank.py +
# significance_rerank.py). Macro means over the 18 retrievers, on the same
# 1,029 / 1,007 query sets as tab:main — guarded, not assumed.
# --------------------------------------------------------------------------- #
def _rerank_macro_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_system: dict[str, list[dict]] = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(r)
    means: dict[str, dict[str, float]] = {}
    for system, rs in by_system.items():
        if len(rs) != len(cfg.MODEL_ALIASES):
            raise ValueError(f"{system}: {len(rs)} retriever rows, expected 18")
        bad = {(r["n_queries"], r["n_queries_r_precision"]) for r in rs} - {("1029", "1007")}
        if bad:
            raise ValueError(f"{system}: unexpected query counts {bad}")
        means[system] = {c: sum(float(r[c]) for r in rs) / len(rs)
                         for c in ("r_precision", "ndcg@10")}
    return means


def rerank_stats() -> dict | None:
    if not RERANK_LEADERBOARD_CSV.exists():
        return None
    import csv

    with RERANK_LEADERBOARD_CSV.open() as f:
        rows = list(csv.DictReader(f))
    means = _rerank_macro_means(rows)
    fs = means["first_stage"]
    systems = [s for s in means if s != "first_stage"]
    rerankers = {s.split("__")[0] for s in systems}

    best500 = max((s for s in systems if s.endswith("@500")),
                  key=lambda s: means[s]["r_precision"])
    best_cell = max((r for r in rows if r["system"] != "first_stage"),
                    key=lambda r: float(r["r_precision"]))

    # Per-retriever best gain over its own first-stage P@R (the 18/18 claim).
    fs_by_ret = {r["retriever"]: float(r["r_precision"])
                 for r in rows if r["system"] == "first_stage"}
    best_gain = {
        ret: max(float(r["r_precision"]) for r in rows
                 if r["retriever"] == ret and r["system"] != "first_stage") - base
        for ret, base in fs_by_ret.items()
    }

    p100 = {s: means[s]["r_precision"] for s in systems
            if s.endswith("@100") and not s.startswith(("rank_zephyr", "jina-reranker"))}
    out = {
        "n_rerankers": len(rerankers),
        "n_systems": len(systems),
        "fs_pr": fs["r_precision"],
        "fs_ndcg": fs["ndcg@10"],
        "best_system": best500,
        "best_pr": means[best500]["r_precision"],
        "best_delta": means[best500]["r_precision"] - fs["r_precision"],
        "best_ndcg": means[best500]["ndcg@10"],
        "best_cell": (best_cell["system"], best_cell["retriever"],
                      float(best_cell["r_precision"])),
        "best_gain_min": min(best_gain.values()),
        "best_gain_max": max(best_gain.values()),
        "all_retrievers_improved": all(v > 0 for v in best_gain.values()),
        "qwen8_default_pr": means["Qwen3-Reranker-8B__default@500"]["r_precision"],
        "qwen8_instr_pr": means["Qwen3-Reranker-8B__task_instruct@500"]["r_precision"],
        "qwen8_default_ndcg": means["Qwen3-Reranker-8B__default@500"]["ndcg@10"],
        "qwen8_instr_ndcg": means["Qwen3-Reranker-8B__task_instruct@500"]["ndcg@10"],
        "qwen4_default_pr": means["Qwen3-Reranker-4B__default@500"]["r_precision"],
        "gemma_default_pr": means["bge-reranker-v2-gemma__default@500"]["r_precision"],
        "gemma_instr_pr": means["bge-reranker-v2-gemma__task_instruct@500"]["r_precision"],
        "zephyr_pr_100": means["rank_zephyr_7b_v1_full__default@100"]["r_precision"],
        "jina_pr_100": means["jina-reranker-v3__default@100"]["r_precision"],
        "pointwise_100_pr_range": (min(p100.values()), max(p100.values())),
    }

    # Cross-check: the study's first-stage rows must reproduce the published
    # first-stage leaderboard (identity regression at the aggregate level).
    if LEADERBOARD_CSV.exists():
        with LEADERBOARD_CSV.open() as f:
            pub = {r["model"]: r for r in csv.DictReader(f)}
        diffs = [
            abs(float(pub[r["retriever"]][c]) - float(r[c]))
            for r in rows if r["system"] == "first_stage"
            for c in ("r_precision", "ndcg@10")
        ]
        out["first_stage_max_abs_diff_vs_published"] = max(diffs)

    # Significance rollups (significance_rerank.py outputs).
    if RERANK_SIG_CSV.exists():
        with RERANK_SIG_CSV.open() as f:
            sig = list(csv.DictReader(f))
        improved = {}
        for r in sig:
            improved.setdefault(r["system"], 0)
            if float(r["p_holm"]) < 0.05 and float(r["delta"]) > 0:
                improved[r["system"]] += 1
        out["sig_improved_min"] = min(improved.values())
        out["sig_systems_18_of_18"] = sum(1 for v in improved.values() if v == 18)
        out["sig_n_systems"] = len(improved)
    if RERANK_SIG_INSTRUCT_CSV.exists():
        with RERANK_SIG_INSTRUCT_CSV.open() as f:
            sig = list(csv.DictReader(f))
        for system in ("Qwen3-Reranker-8B__task_instruct@500",
                       "bge-reranker-v2-gemma__task_instruct@500"):
            rs = [r for r in sig if r["system"] == system]
            out[f"instr[{system}]"] = {
                "sig_up": sum(1 for r in rs
                              if float(r["p_holm"]) < 0.05 and float(r["delta"]) > 0),
                "sig_down": sum(1 for r in rs
                                if float(r["p_holm"]) < 0.05 and float(r["delta"]) < 0),
            }
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-corpus", action="store_true", help="skip the 12GB corpus scan")
    args = ap.parse_args()

    q = query_stats()
    print("== Queries (data/queries/queries.json) ==")
    for k, v in q.items():
        print(f"  {k}: {v}")

    lab = label_stats()
    print("\n== Labels (data/labels, main ∪ delta via data_loading) ==")
    for k, v in lab.items():
        print(f"  {k}: {v}")

    run = labeling_run_stats()
    print("\n== Labeling run (Batch API manifests) ==")
    for part in ("main", "delta"):
        m = run.get(part)
        if m is None:
            print(f"  {part}: NO MANIFEST")
            continue
        print(
            f"  {part}: queries={m['n_queries']} requests={fmt(m['n_requests'])} "
            f"predictions={fmt(m['n_predictions'])} cost_batch=${m['cost_batch']:,.2f} "
            f"cost_uncached=${m['cost_uncached']:,.2f}"
        )
        print(
            f"         tokens: prompt={fmt(m['prompt_tokens'])} (cached {fmt(m['cached_tokens'])}) "
            f"completion={fmt(m['completion_tokens'])} (reasoning {fmt(m['reasoning_tokens'])})"
        )
        print(f"         config: {m['config']}")
    if "batch_size_mean" in run:
        # NOT a call-pack size: each request holds ONE paragraph; n_requests counts
        # only the final wave (seed-from-old + build --skip-done reuse earlier labels).
        print(f"  predictions per manifest request (resume artifact, main): "
              f"{run['batch_size_mean']:.1f}")
    if "cost_batch_total" in run:
        print(f"  TOTAL batch cost (main+delta): ${run['cost_batch_total']:,.2f}")

    lb = leaderboard_crosscheck()
    if lb:
        print("\n== Cross-check vs summary/evaluation/leaderboard.csv ==")
        for k, v in lb.items():
            print(f"  {k}: {v}")

    rr = rerank_stats()
    if rr is None:
        print("\n== Reranking study: summary/rerank/leaderboard.csv missing ==")
    else:
        print("\n== Reranking study (summary/rerank, macro over 18 retrievers) ==")
        for k, v in rr.items():
            print(f"  {k}: {v}")

    if not args.skip_corpus:
        valid = set(range(q["eid_min"], q["eid_max"] + 1))
        print("\n== Corpus (data/splade_wikipedia_v2, streaming 12GB...) ==")
        c = corpus_stats(valid)
        for k, v in c.items():
            print(f"  {k}: {v}")
    else:
        c = {}
        print("\n== Corpus scan skipped (--skip-corpus) ==")

    # ----------------------------------------------------------------- #
    print("\n== Suggested macros.tex values ==")
    lines = [
        ("statNQueries", fmt(q["n_queries"])),
        ("statNRelations", fmt(q["n_relations"])),
        ("statNTypes", fmt(q["n_types"])),
        ("statNConstants", fmt(q["n_constants"])),
        ("statInstancesMean", f"{q['inst_mean']:.1f}"),
        ("statNJudged", fmt(lab["judged_total"])),
        ("statNJudgedMain", fmt(lab["judged_main"])),
        ("statNJudgedDelta", fmt(lab["judged_delta"])),
        ("statNPositives", fmt(lab["pos_merged"])),
        ("statPosRate", f"{lab['pos_rate_merged_pct']:.1f}\\%"),
        ("statNEval", fmt(lab["n_eval"])),
        ("statNZeroR", fmt(lab["n_r0"])),
        ("statRMedian", fmt(lab["r_median"])),
        ("statRMean", fmt(lab["r_mean"])),
        ("statRMax", fmt(lab["r_max"])),
    ]
    if c:
        lines += [
            ("statNCorpus", fmt(c["corpus_size"])),
            ("statCandidatesMedian", fmt(c["cand_median"])),
        ]
    if "cost_batch_total" in run:
        lines.append(("statLabelCostUSD", f"{run['cost_batch_total']:,.0f}"))
    if rr:
        lines += [
            ("statNRerankers", fmt(rr["n_rerankers"])),
            ("statRerankFirstStagePR", f"{rr['fs_pr']:.3f}"),
            ("statRerankFirstStageNDCG", f"{rr['fs_ndcg']:.3f}"),
            ("statRerankBestPR", f"{rr['best_pr']:.3f}"),
            ("statRerankBestDelta", f"{rr['best_delta']:+.3f}"),
            ("statRerankBestNDCG", f"{rr['best_ndcg']:.3f}"),
            ("statRerankBestCellPR", f"{rr['best_cell'][2]:.3f}"),
            ("statRerankBestGainLo", f"{rr['best_gain_min']:+.3f}"),
            ("statRerankBestGainHi", f"{rr['best_gain_max']:+.3f}"),
        ]
    for name, val in lines:
        print(f"\\newcommand{{\\{name}}}{{{val}\\xspace}}")


if __name__ == "__main__":
    main()
