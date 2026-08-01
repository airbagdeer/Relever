"""Rerank each retriever's top-100 with a listwise model, once per (reranker, retriever).

Listwise scores depend on the candidate set, so the pooled cache that makes the pointwise
lane cheap does not apply here: this is 18 retrievers x 1100 queries of real work. Both
models are therefore run at the depth their papers publish (100), and the pointwise models
are additionally scored at depth 100 from the cache so the comparison is apples-to-apples.

Batching is across (retriever, query) pairs, not within a query: rank_zephyr's 9 sliding
windows are strictly sequential, so the GPU is only saturated by running many queries'
window-1 concurrently, then many queries' window-2, and so on. jina-v3 ignores the batch
(its ``rerank()`` takes one query) but costs one forward pass per item anyway.

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run python rerank_listwise.py \\
        --reranker jina-reranker-v3 --shard 0 --num-shards 4

    CUDA_VISIBLE_DEVICES=4,5,6,7 .venv-rankllm/bin/python rerank_listwise.py \\
        --reranker rank_zephyr_7b_v1_full --num-gpus 4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import _rerank_bootstrap  # noqa: F401  (sets sys.path so `config` imports)
import score_store
from backends import ListwiseItem, build_listwise_reranker
from config import PROJECT_ROOT
from reranker_registry import LISTWISE_DEPTH, spec_for
from score_pool import FRONTIER_CANDIDATES_DIR, load_queries, setup_logging

logger = logging.getLogger(__name__)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DEFAULT_BATCH_ITEMS = 64


def retriever_aliases() -> list[str]:
    """The 18 first-stage models, read from the evaluation registry (single source)."""
    import sys  # noqa: PLC0415

    eval_dir = PROJECT_ROOT / "src" / "retrieval" / "lancedb" / "frontier-v2" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.append(str(eval_dir))
    import eval_config  # noqa: PLC0415

    return list(eval_config.MODEL_ALIASES)


def load_pool_texts(event_id: int) -> dict[str, str]:
    """``{paragraph_id: text}`` for one query's deduplicated pool."""
    qid = score_store.qid_str(event_id)
    path = FRONTIER_CANDIDATES_DIR / qid / f"all_candidates_{qid}.json"
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return {c["id"]: c.get("text", "") for c in doc.get("candidates", [])}


def load_retriever_top_k(retriever: str, event_id: int, k: int) -> list[str]:
    """The retriever's top-``k`` paragraph ids, in retrieval rank order."""
    qid = score_store.qid_str(event_id)
    path = FRONTIER_CANDIDATES_DIR / qid / f"{retriever}_{qid}.json"
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    cands = sorted(doc.get("candidates", []), key=lambda c: c["rank"])[:k]
    return [c["id"] for c in cands]


def build_items(
    reranker: str, retrievers: list[str], queries, depth: int, overwrite: bool
) -> list[ListwiseItem]:
    """Every (retriever, query) job not already on disk."""
    items: list[ListwiseItem] = []
    for query in queries:
        pending = [
            r for r in retrievers
            if overwrite or not score_store.listwise_exists(reranker, r, query.event_id)
        ]
        if not pending:
            continue
        id_to_text = load_pool_texts(query.event_id)
        for retriever in pending:
            doc_ids = load_retriever_top_k(retriever, query.event_id, depth)
            missing = [i for i in doc_ids if i not in id_to_text]
            if missing:
                # The pool is the union of every retriever's top-500, so a top-100 id must
                # be in it. If not, the candidate dumps and the pool disagree.
                raise KeyError(
                    f"[{retriever}] q{query.event_id}: {len(missing)} of the top-{depth} ids "
                    f"are absent from the deduplicated pool (e.g. {missing[0]})."
                )
            items.append(
                ListwiseItem(
                    event_id=query.event_id,
                    retriever=retriever,
                    query=query.sentence,
                    doc_ids=tuple(doc_ids),
                    docs=tuple(id_to_text[i] for i in doc_ids),
                )
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Listwise-rerank each retriever's top-K.")
    parser.add_argument("--reranker", required=True, help="A listwise alias.")
    parser.add_argument("--depth", type=int, default=LISTWISE_DEPTH,
                        help=f"Rerank depth (default {LISTWISE_DEPTH}, each model's published protocol).")
    parser.add_argument("--retrievers", nargs="*", default=None, help="Subset of the 18 retrievers.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Queries split by event_id %% num_shards.")
    parser.add_argument("--num-gpus", type=int, default=1, help="rank_zephyr: vLLM tensor_parallel_size.")
    parser.add_argument("--batch-items", type=int, default=DEFAULT_BATCH_ITEMS,
                        help="(retriever, query) jobs handed to the model at once.")
    parser.add_argument("--limit-queries", type=int, default=None, help="Smoke knob.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-file", type=Path, default=None, help="'-' for stderr only.")
    args = parser.parse_args()

    if args.log_file is None:
        log_file: Path | None = (
            PROJECT_ROOT / "logs" / f"rerank_listwise_{args.reranker}_s{args.shard}_{RUN_TIMESTAMP}.log"
        )
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file

    spec = spec_for(args.reranker)
    if spec.paradigm != "listwise":
        raise SystemExit(
            f"{spec.alias!r} is pointwise; its scores are set-independent. Use score_pool.py, "
            f"which scores the deduplicated pool once and serves all 18 retrievers from it."
        )
    if spec.backend == "jina_v3":
        # Checked before a 0.6B model loads and 19,800 items are built, not after.
        from backends.listwise_jina_v3 import BLOCK_SIZE  # noqa: PLC0415

        if args.depth > BLOCK_SIZE:
            raise SystemExit(
                f"--depth {args.depth} exceeds jina-reranker-v3's block_size={BLOCK_SIZE}. "
                f"Beyond it, modeling.py splits documents into blocks that never attend to "
                f"each other and whose scores share only an averaged query vector — the "
                f"listwise premise would silently stop holding."
            )

    # Only now, once the arguments are known good, create the log file.
    setup_logging(log_file)

    retrievers = args.retrievers or retriever_aliases()
    queries = load_queries(limit=args.limit_queries)
    shard_queries = [q for q in queries if q.event_id % args.num_shards == args.shard]

    items = build_items(spec.alias, retrievers, shard_queries, args.depth, args.overwrite)
    n_total = len(shard_queries) * len(retrievers)
    logger.info(
        "[%s] shard %d/%d: %d of %d (retriever, query) jobs pending at depth %d "
        "(%d retrievers x %d queries).",
        spec.alias, args.shard, args.num_shards, len(items), n_total, args.depth,
        len(retrievers), len(shard_queries),
    )
    if not items:
        logger.info("[%s] nothing to do.", spec.alias)
        return

    kwargs = {}
    if spec.backend == "rank_zephyr":
        kwargs.update(num_gpus=args.num_gpus)
    reranker = build_listwise_reranker(spec, device=args.device, **kwargs)

    t0 = time.time()
    n_done = 0
    for start in range(0, len(items), args.batch_items):
        batch = items[start:start + args.batch_items]
        for res in reranker.rerank_many(batch):
            score_store.write_listwise(
                spec.alias, res.retriever, res.event_id, args.depth, res.order, res.scores
            )
            n_done += 1
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed else 0.0
        remaining = (len(items) - n_done) / rate if rate else float("inf")
        logger.info(
            "[%s] shard %d: %d/%d jobs | %.2f job/s | eta %.1f min",
            spec.alias, args.shard, n_done, len(items), rate, remaining / 60.0,
        )

    reranker.truncation.log(spec.alias)
    if spec.expected_truncation >= 1.0:
        logger.warning(
            "[%s] every passage is truncated by design (context_size=%s). The reported "
            "numbers describe reranking over passage fragments; state this in the paper.",
            spec.alias, 4096,
        )
    logger.info("[%s] shard %d done: %d jobs in %.1f min.",
                spec.alias, args.shard, n_done, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
