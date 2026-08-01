"""Aggregate per-model candidates into a deduplicated per-query union.

Run AFTER all per-model retrieval finished. For each query directory
``data/candidates/<qid>/`` it reads every ``<alias>_<qid>.json``
(the compact per-model files) and writes ``all_candidates_<qid>.json``: the
union of candidate ids across all models, deduplicated by paragraph id, with the
full paragraph text joined once and per-model provenance.

Each model contributes only its top ``--agg-per-model-k`` candidates (by rank,
default ``retrieve_core.AGG_PER_MODEL_K`` = 500) to this union, so the deduped
labeling pool stays bounded even though the per-model files store the full
retrieval depth (``retrieve_core.DEFAULT_K`` = 3000).

Text is joined from the cached v2 corpus (``load_v2_paragraph_corpus``) — the
same source the indexers embedded, so ids line up exactly. CPU-only; runs in the
main ``.venv``.

Output record (ordered by ``n_models`` desc, then ``best_rank`` asc)::

    {"event_id": 36, "query": "...", "n_models": 18, "n_candidates": 4123,
     "candidates": [
        {"id": "<uuid>", "text": "...", "n_models": 7, "best_rank": 2,
         "retrieved_by": {"Octen-Embedding-8B": {"rank": 2, "score": 0.9}, ...}},
        ...]}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from corpus_loader_v2 import load_v2_paragraph_corpus
from paths import SPLADE_WIKIPEDIA_V2_PATH, V2_CORPUS_CACHE_DIR

import retrieve_core

logger = logging.getLogger(__name__)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_id_to_text(use_cache: bool) -> dict[str, str]:
    """Map paragraph id -> text from the (cached) v2 corpus."""
    corpus = load_v2_paragraph_corpus(
        source_dir=SPLADE_WIKIPEDIA_V2_PATH,
        cache_dir=V2_CORPUS_CACHE_DIR,
        use_cache=use_cache,
    )
    logger.info("Loaded v2 corpus for text join: U=%d paragraphs.", len(corpus))
    return dict(zip(corpus.paragraph_ids, corpus.texts))


def _aggregate_one_query(
    event_id: int,
    query: str,
    id_to_text: dict[str, str],
    overwrite: bool,
    per_model_k: int | None,
) -> tuple[bool, int]:
    """Build all_candidates_<qid>.json for one query. Returns (written, n_missing_text).

    ``per_model_k`` caps how deep each model contributes to the deduped union: only
    a model's top-``per_model_k`` candidates (by rank) are unioned, even though the
    per-model files store the full retrieval depth (``retrieve_core.DEFAULT_K``).
    ``None`` means no cap (union everything in each file).
    """
    qdir = retrieve_core.candidates_dir_for(event_id)
    out_path = qdir / f"all_candidates_{retrieve_core.qid_str(event_id)}.json"
    if out_path.exists() and not overwrite:
        return False, 0
    if not qdir.is_dir():
        return False, 0

    # union by paragraph id; preserve per-model rank/score provenance.
    union: dict[str, dict] = {}
    models_seen: set[str] = set()
    for model_file in sorted(qdir.glob(f"*_{retrieve_core.qid_str(event_id)}.json")):
        if model_file.name.startswith("all_candidates_"):
            continue
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable %s: %s", model_file, exc)
            continue
        alias = doc.get("model") or model_file.name.rsplit("_", 1)[0]
        models_seen.add(alias)
        for cand in doc.get("candidates", []):
            cid = cand.get("id")
            if not cid:
                continue
            rank = int(cand.get("rank", 0))
            # Cap each model to its top-`per_model_k` (rank is 1-based; 1 = best).
            if per_model_k is not None and rank > per_model_k:
                continue
            score = float(cand.get("score", 0.0))
            rec = union.get(cid)
            if rec is None:
                rec = {"id": cid, "retrieved_by": {}}
                union[cid] = rec
            rec["retrieved_by"][alias] = {"rank": rank, "score": score}

    n_missing = 0
    candidates: list[dict] = []
    for cid, rec in union.items():
        text = id_to_text.get(cid)
        if text is None:
            n_missing += 1
            text = ""
        ranks = [v["rank"] for v in rec["retrieved_by"].values() if v.get("rank", 0) > 0]
        best_rank = min(ranks) if ranks else 0
        candidates.append({
            "id": cid,
            "text": text,
            "n_models": len(rec["retrieved_by"]),
            "best_rank": best_rank,
            "retrieved_by": rec["retrieved_by"],
        })

    # most models first; ties broken by best (lowest) rank achieved by any model.
    candidates.sort(key=lambda c: (-c["n_models"], c["best_rank"] if c["best_rank"] else 10**9))

    out_doc = {
        "event_id": int(event_id),
        "query": query,
        "n_models": len(models_seen),
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    tmp.replace(out_path)
    return True, n_missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-model candidates into deduped per-query unions.")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="Only aggregate the first N queries (smoke knob).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rewrite all_candidates_<qid>.json even if present.")
    parser.add_argument("--agg-per-model-k", type=int, default=retrieve_core.AGG_PER_MODEL_K,
                        help="Cap each model to its top-K (by rank) when building the deduped "
                             "union; the per-model files keep their full depth. "
                             f"<=0 disables the cap. Default {retrieve_core.AGG_PER_MODEL_K}.")
    parser.add_argument("--no-cache", action="store_true", help="Skip the v2 corpus pickle cache.")
    parser.add_argument("--log-file", type=Path, default=None)
    # parse_known_args so the shared orchestrator can forward retriever-only flags
    # (e.g. --k, --chunk, --gpus, --table-suffix) without crashing the aggregator.
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"aggregate_candidates(frontier-v2): ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file: Path | None = Path("logs") / f"v2_aggregate_candidates_{RUN_TIMESTAMP}.log"
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = retrieve_core.setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

    event_ids, queries = retrieve_core.load_v2_queries(limit=args.limit_queries)
    logger.info("Aggregating %d queries.", len(event_ids))

    id_to_text = _build_id_to_text(use_cache=not args.no_cache)

    per_model_k = args.agg_per_model_k if args.agg_per_model_k > 0 else None
    logger.info("Per-model union cap: %s (per-model files keep their full depth).",
                per_model_k if per_model_k is not None else "disabled")

    written = 0
    skipped = 0
    total_missing = 0
    from tqdm import tqdm
    for eid, query in tqdm(list(zip(event_ids, queries)), desc="aggregate"):
        did_write, n_missing = _aggregate_one_query(
            eid, query, id_to_text, overwrite=args.overwrite, per_model_k=per_model_k)
        total_missing += n_missing
        if did_write:
            written += 1
        else:
            skipped += 1

    logger.info("Aggregation done: wrote %d, skipped %d (already present or no dir).", written, skipped)
    if total_missing:
        logger.warning("%d candidate id(s) had no text in the v2 corpus (wrote empty text).", total_missing)


if __name__ == "__main__":
    main()
