"""Extract the deduplicated false-positive population for the FP-taxonomy analysis.

For each selected system and each evaluated query (R_q > 0), take that system's
false-positive paragraphs (merged label 0, main ∪ delta — exactly as
``data_loading.load_query_labels`` constructs it, the paper's single ground truth),
under one of two depth rules:

  * ``topk_retrieved`` (default) — the negatives *within* the top-``k`` retrieved
    ranks. Matches the paper's headline nDCG@10 / P@10 depth; a query whose top-k is
    mostly positives contributes fewer than k negatives.
  * ``topk_negatives`` — the ``k`` highest-scored *negatives*, scanning past
    positives (below rank k when needed) so every query contributes exactly k
    (unless the ranking holds fewer than k negatives).

The union over the selected systems is deduplicated to unique (query, paragraph)
pairs; each pair is classified once by the taxonomy judge and the result maps back to
every selected system that retrieved it (with its rank).

Output (JSONL, one line per unique (event_id, chunk_id) pair):
  {event_id, id, systems: {alias: rank, ...}, n_systems, li, R_q, r_bucket}
  where ``li`` marks pairs retrieved by ≥1 late-interaction system and ``r_bucket``
  is the query's R_q bucket (eval_config.R_BUCKETS). A ``<stem>_stats.json`` sits
  beside it.

Read-only over candidates + labels; no API calls, no spend.

    # default: all 18 systems, top-10 retrieved negatives -> fp_population.jsonl
    uv run python src/retrieval/lancedb/frontier-v2/evaluation/extract_fp_population.py

    # two flagship systems, 10 highest-scored negatives -> a subset file
    uv run python .../extract_fp_population.py \\
        --systems colbert-v2 ICT-TIME-and-Querit-embedding-v1 \\
        --mode topk_negatives --top-k 10 \\
        --out data/fp_taxonomy/subset_2sys/subset_population.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import data_loading
import eval_config as cfg
from config import FP_TAXONOMY_DATA_PATH

TOP_K = 10  # matches the paper's headline nDCG@10 / P@10
SCAN_DEPTH = 3000  # topk_negatives: deepest rank materialised on disk (per-model files)
DEFAULT_OUT = FP_TAXONOMY_DATA_PATH / "fp_population.jsonl"


def r_bucket_name(r: int) -> str:
    """Human-readable eval_config.R_BUCKETS bucket for R_q (e.g. '10-49', '500+')."""
    for lo, hi in cfg.R_BUCKETS:
        if r >= lo and (hi is None or r <= hi):
            return f"{lo}+" if hi is None else f"{lo}-{hi}"
    raise ValueError(f"R_q={r} falls in no bucket")


def _select_negatives(
    ids: list[str], labels: dict[str, int], mode: str, top_k: int
) -> tuple[list[tuple[int, str]], Counter]:
    """Walk a ranked id list and return the (rank_idx, cid) negatives to keep, plus a
    label tally over the slots examined.

    ``topk_retrieved`` examines ranks 0..top_k-1 and keeps the negatives among them;
    ``topk_negatives`` scans until ``top_k`` negatives are collected (past positives).
    """
    kept: list[tuple[int, str]] = []
    tally: Counter = Counter()
    for rank_idx, cid in enumerate(ids):
        if mode == "topk_retrieved" and rank_idx >= top_k:
            break
        label = labels.get(cid)
        if label == 1:
            tally["positive"] += 1
        elif label is None:
            tally["unlabeled"] += 1
        elif label == -1:
            tally["failed"] += 1
        else:  # label == 0 — a judged negative
            tally["negative"] += 1
            kept.append((rank_idx, cid))
        if mode == "topk_negatives" and len(kept) >= top_k:
            break
    return kept, tally


def extract(systems: list[str], mode: str, top_k: int, out_path: Path,
            scan_depth: int = SCAN_DEPTH) -> dict:
    """Build + write the population for ``systems`` under ``mode``; return stats."""
    li_aliases = {m.alias for m in cfg.MODELS if m.retriever_type == "late_interaction"}
    depth = top_k if mode == "topk_retrieved" else scan_depth
    eids = data_loading.event_ids()

    pairs: dict[tuple[int, str], dict] = {}   # (eid, cid) -> {systems{alias:rank}, R_q}
    n_queries_eval = 0
    per_system_neg: Counter = Counter()
    totals: Counter = Counter()

    for eid in eids:
        ql = data_loading.load_query_labels(eid)
        if ql.R == 0:  # the 71 zero-R queries are excluded from all evaluation
            continue
        n_queries_eval += 1
        for alias in systems:
            rl = data_loading.load_ranked_list(alias, eid, ql.labels, depth=depth)
            kept, tally = _select_negatives(rl.ids, ql.labels, mode, top_k)
            totals.update(tally)
            per_system_neg[alias] += len(kept)
            for rank_idx, cid in kept:
                entry = pairs.setdefault((eid, cid), {"systems": {}, "R_q": ql.R})
                entry["systems"].setdefault(alias, rank_idx + 1)  # 1-based rank

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for (eid, cid), entry in sorted(pairs.items()):
            fh.write(json.dumps({
                "event_id": eid,
                "id": cid,
                "systems": entry["systems"],
                "n_systems": len(entry["systems"]),
                "li": any(a in li_aliases for a in entry["systems"]),
                "R_q": entry["R_q"],
                "r_bucket": r_bucket_name(entry["R_q"]),
            }) + "\n")

    n_neg = totals["negative"]
    stats = {
        "mode": mode,
        "top_k": top_k,
        "systems": list(systems),
        "n_systems": len(systems),
        "n_queries_evaluated": n_queries_eval,
        "n_slots_examined": sum(totals.values()),
        "n_positive": totals["positive"],
        "n_negative": n_neg,
        "n_unlabeled": totals["unlabeled"],
        "n_failed": totals["failed"],
        "n_unique_fp_pairs": len(pairs),
        "dedup_ratio": round(n_neg / len(pairs), 3) if pairs else None,
        "per_system_negatives": dict(sorted(per_system_neg.items())),
        "pairs_by_li": dict(Counter(
            "li" if e["systems"].keys() & li_aliases else "dense_only"
            for e in pairs.values())),
        "pairs_by_r_bucket": dict(sorted(Counter(
            r_bucket_name(e["R_q"]) for e in pairs.values()).items())),
        "n_systems_dist": dict(sorted(Counter(
            len(e["systems"]) for e in pairs.values()).items())),
    }
    stats_path = out_path.parent / f"{out_path.stem}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Mode {mode} · top-{top_k} · {len(systems)} system(s) · "
          f"{n_queries_eval} evaluated queries")
    print(f"Negatives kept: {n_neg:,} (pos {totals['positive']:,} · "
          f"unlabeled {totals['unlabeled']:,} · failed {totals['failed']:,} examined)")
    print(f"Per system: " + " · ".join(f"{a} {c:,}" for a, c in per_system_neg.items()))
    print(f"Unique (query, paragraph) pairs: {len(pairs):,} "
          f"(dedup {n_neg:,} -> {len(pairs):,}, {stats['dedup_ratio']}x sharing)")
    print(f"  by family: {stats['pairs_by_li']}")
    print(f"-> {out_path}")
    print(f"-> {stats_path}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=None,
                    help="system aliases to include (default: all 18)")
    ap.add_argument("--mode", choices=("topk_retrieved", "topk_negatives"),
                    default="topk_retrieved",
                    help="topk_retrieved = negatives within top-k ranks (paper depth); "
                         "topk_negatives = the k highest-scored negatives (scan past positives)")
    ap.add_argument("--top-k", dest="top_k", type=int, default=TOP_K)
    ap.add_argument("--scan-depth", dest="scan_depth", type=int, default=SCAN_DEPTH,
                    help="max ranks to scan in topk_negatives mode (default 3000)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output JSONL path (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    systems = args.systems or list(cfg.MODEL_ALIASES)
    unknown = [s for s in systems if s not in set(cfg.MODEL_ALIASES)]
    if unknown:
        raise SystemExit(f"Unknown system alias(es): {unknown}\n"
                         f"Known: {', '.join(cfg.MODEL_ALIASES)}")
    extract(systems, args.mode, args.top_k, Path(args.out), args.scan_depth)


if __name__ == "__main__":
    main()
