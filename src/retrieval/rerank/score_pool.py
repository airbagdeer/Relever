"""Score the frontier-v2 deduplicated candidate pool once per (query, reranker, variant).

This is the whole compute saving of the study, and it rests on one verified fact:
``data/candidates/<qid>/all_candidates_<qid>.json`` is *exactly* the union of
every retriever's top-500, and the GPT-5.2 labels cover *exactly* that pool. A pointwise
reranker's ``s(q, d)`` does not depend on the candidate set, so scoring the pool once
gives every one of the 18 retrievers its reranked top-500 by a plain sort of its own
ids — 2.18M pairs instead of 9.9M, and every depth <= 500 becomes free to re-evaluate
offline. ``tests/test_equivalence.py`` asserts this rather than assuming it.

Sharding: queries are split ``event_id % num_shards``, one process per GPU. Shards write
disjoint per-query files, so they never coordinate and a killed shard resumes by skipping
what exists. Resume is per-query, not per-model: a job that dies 900 queries in loses
nothing.

Usage (one shard):
    CUDA_VISIBLE_DEVICES=4 uv run python score_pool.py \\
        --reranker Qwen3-Reranker-8B --variant default --shard 0 --num-shards 4

Smoke:
    uv run python score_pool.py --reranker identity-constant --limit-queries 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import _rerank_bootstrap  # noqa: F401  (sets sys.path so `config` imports)
import score_store
from backends import build_reranker
from backends.base import ScoreHealth
from config import PROJECT_ROOT
from reranker_registry import (
    DEFAULT_VARIANT,
    RerankQuery,
    instruction_for,
    spec_for,
    variants_for,
)

logger = logging.getLogger(__name__)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Queries to score before the score-health gate fires. Enough for a stable estimate of
# distinct@10, cheap enough that a broken backend costs minutes rather than hours.
HEALTH_CHECK_AFTER = 8

FRONTIER_CANDIDATES_DIR = PROJECT_ROOT / "data" / "candidates"
V2_RELATIONS_PATH = PROJECT_ROOT / "data" / "queries" / "queries.json"


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


def load_queries(limit: int | None = None) -> list[RerankQuery]:
    """The v2 relational queries, in file order. Query text is the ``sentence`` field."""
    with open(V2_RELATIONS_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    if limit is not None:
        rows = rows[:limit]
    return [
        RerankQuery(
            event_id=int(r["event_id"]),
            sentence=str(r["sentence"]),
            relation=str(r.get("relation", "")),
            constant_entity=str(r.get("constant_entity", "")),
            abstract_entity_type=str(r.get("abstract_entity_type", "")),
        )
        for r in rows
    ]


def load_pool(event_id: int) -> tuple[list[str], list[str]]:
    """``(ids, texts)`` for one query's deduplicated candidate pool."""
    qid = score_store.qid_str(event_id)
    path = FRONTIER_CANDIDATES_DIR / qid / f"all_candidates_{qid}.json"
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    cands = doc.get("candidates", [])
    ids = [c["id"] for c in cands]
    texts = [c.get("text", "") for c in cands]
    n_empty = sum(1 for t in texts if not t)
    if n_empty:
        # A doc with no text would score as an empty string and rank arbitrarily.
        # aggregate_candidates.py writes "" when a id is missing from the corpus, so
        # this must be a hard error rather than a quietly mis-scored candidate.
        raise ValueError(f"[{qid}] {n_empty} pool candidate(s) have empty text in {path}.")
    return ids, texts


def score_one_query(reranker, spec, variant: str, query: RerankQuery, overwrite: bool,
                    health=None) -> bool:
    """Score one query's pool. Returns True if it was scored, False if skipped."""
    if not overwrite and score_store.scores_exist(spec.alias, variant, query.event_id):
        return False

    ids, texts = load_pool(query.event_id)
    instruction = instruction_for(spec, variant, query)
    scores = reranker.score(query.sentence, texts, instruction=instruction)

    if len(scores) != len(ids):
        raise ValueError(
            f"[{spec.alias}] query {query.event_id}: backend returned {len(scores)} scores "
            f"for {len(ids)} documents."
        )
    if health is not None:
        health.observe(scores)
    score_store.write_scores(spec.alias, variant, query.event_id, ids, scores)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the frontier-v2 candidate pool with one reranker.")
    parser.add_argument("--reranker", required=True, help="Alias from reranker_registry.")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        help="'default' (card instruction) or 'task_instruct' (ablation).")
    parser.add_argument("--shard", type=int, default=0, help="This shard's index (0-based).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards; queries split by event_id %% num_shards.")
    parser.add_argument("--limit-queries", type=int, default=None, help="Smoke knob: first N queries.")
    parser.add_argument("--overwrite", action="store_true", help="Rescore queries that already have a file.")
    parser.add_argument("--device", default=None, help="Override device (default: cuda -> mps -> cpu).")
    parser.add_argument("--attn-implementation", default=None,
                        help="e.g. flash_attention_2 for the decoder rerankers (needs flash-attn).")
    parser.add_argument("--ignore-score-health", action="store_true",
                        help="Do not abort when the top-10 scores are mostly tied. Only for "
                             "deliberate controls such as identity-constant.")
    parser.add_argument("--log-file", type=Path, default=None, help="'-' for stderr only.")
    args = parser.parse_args()

    # Validate before touching the filesystem: a fatal argument error must not leave a
    # log file behind suggesting a run happened.
    spec = spec_for(args.reranker)
    if spec.paradigm != "pointwise":
        raise SystemExit(
            f"{spec.alias!r} is {spec.paradigm}; its scores depend on the candidate set and "
            f"cannot use the pooled cache. Use rerank_listwise.py instead."
        )
    if args.variant not in variants_for(spec.alias):
        raise SystemExit(
            f"variant {args.variant!r} is undefined for {spec.alias!r} "
            f"(supported: {', '.join(variants_for(spec.alias))})."
        )

    if args.log_file is None:
        log_file: Path | None = (
            PROJECT_ROOT / "logs" / f"rerank_score_{args.reranker}_{args.variant}_s{args.shard}_{RUN_TIMESTAMP}.log"
        )
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    setup_logging(log_file)

    queries = load_queries(limit=args.limit_queries)
    shard = [q for q in queries if q.event_id % args.num_shards == args.shard]
    logger.info(
        "[%s/%s] shard %d/%d: %d of %d queries. Pool scoring at depth<=%d.",
        spec.alias, args.variant, args.shard, args.num_shards, len(shard), len(queries), spec.depth,
    )

    kwargs = {}
    if args.attn_implementation and spec.backend.startswith("hf_causal"):
        kwargs["attn_implementation"] = args.attn_implementation
    reranker = build_reranker(spec, device=args.device, **kwargs)

    # Resume is per-query and silent. This is what keeps it from being silent about the
    # thing that matters: a cache half-written on CPU/fp32 and half on GPU/bf16 would
    # interleave incomparable scores within one query set, and the ranking would be
    # subtly wrong with nothing to show for it.
    device_kind = str(getattr(reranker, "device", "unknown")).split(":")[0]
    score_store.check_or_write_manifest(spec.alias, args.variant, {
        "reranker": spec.alias,
        "variant": args.variant,
        "dtype": spec.dtype,
        "device_kind": device_kind,
        "tf32": bool(getattr(reranker, "tf32", False)),
        "max_length": spec.max_length,
        "padding_side": spec.padding_side,
    })

    health = ScoreHealth()
    n_scored = n_skipped = 0
    t0 = time.time()
    for i, query in enumerate(shard, 1):
        did_score = score_one_query(reranker, spec, args.variant, query, args.overwrite,
                                    health=health)
        n_scored += int(did_score)
        n_skipped += int(not did_score)

        # Fail after a handful of queries, not after 1100 wasted ones. A reranker whose
        # top-10 is mostly tied produces a column that silently duplicates the retriever.
        if health.n_queries == HEALTH_CHECK_AFTER and not spec.expects_tied_scores:
            problem = health.check(spec.alias)
            health.log(spec.alias)
            if problem and not args.ignore_score_health:
                raise SystemExit(
                    f"{problem}\n\nScored {n_scored} queries before aborting. Fix the "
                    f"backend, delete {score_store.scores_dir(spec.alias, args.variant)}, "
                    f"and rerun. Pass --ignore-score-health to override (and say so in "
                    f"the paper)."
                )

        if did_score and (n_scored % 25 == 0 or i == len(shard)):
            elapsed = time.time() - t0
            rate = n_scored / elapsed if elapsed else 0.0
            remaining = (len(shard) - i) / rate if rate else float("inf")
            logger.info(
                "[%s/%s] shard %d: %d/%d queries (%d skipped) | %.2f q/s | eta %.1f min",
                spec.alias, args.variant, args.shard, i, len(shard), n_skipped, rate, remaining / 60.0,
            )

    reranker.truncation.log(spec.alias)
    health.log(spec.alias)
    # No silent caps: a truncation rate above what the registry declares means the
    # numbers describe fragments of paragraphs, not paragraphs.
    if reranker.truncation.rate > max(spec.expected_truncation, 0.001) * 1.5:
        logger.warning(
            "[%s] truncation rate %.3f exceeds the expected %.3f. The scores describe "
            "truncated documents; check max_length before using these numbers.",
            spec.alias, reranker.truncation.rate, spec.expected_truncation,
        )

    logger.info(
        "[%s/%s] shard %d done: %d scored, %d skipped, %.1f min total.",
        spec.alias, args.variant, args.shard, n_scored, n_skipped, (time.time() - t0) / 60.0,
    )


if __name__ == "__main__":
    sys.exit(main())
