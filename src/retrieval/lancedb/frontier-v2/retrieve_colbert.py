"""frontier-v2 ColBERTv2 (pylate) retriever — late-interaction MaxSim.

Query-side mirror of ``index_colbert.py``: documents were encoded with
``is_query=False``; here we encode the v2 queries with ``is_query=True`` and
brute-force MaxSim them against the stored ``token_vectors`` table to get the
top-K paragraphs per query. The MaxSim + Arrow-scan + GPU top-k heap come from
``retrieve_core`` (which reuses ``evaluate_minimax``'s primitives).

Table root = durable home (colbert is the one model the indexer kept on home;
``root_for('colbert-v2')`` resolves it). Runs in ``.venv-colbert``.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from model_registry import resolve_model_id

import retrieve_core

logger = logging.getLogger(__name__)
ALIAS = "colbert-v2"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-K v2 candidates (ColBERTv2 / pylate, MaxSim).")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--k", type=int, default=retrieve_core.DEFAULT_K)
    parser.add_argument("--lancedb-root", choices=("home", "scratch", "auto"), default="auto")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--table-suffix", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encode-batch", type=int, default=64, help="Query encode batch size.")
    parser.add_argument("--chunk", type=int, default=2048, help="Corpus rows per GPU MaxSim scan batch.")
    parser.add_argument("--q-batch-size", type=int, default=64,
                        help="Queries per MaxSim inner mini-batch (bounds peak GPU memory).")
    parser.add_argument("--max-scan-batches", type=int, default=None,
                        help="Smoke knob: stop the corpus scan after N batches (PARTIAL results).")
    parser.add_argument("--log-file", type=Path, default=None)
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"retrieve_colbert(frontier-v2): ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file: Path | None = Path("logs") / f"v2_retrieve_colbert_{RUN_TIMESTAMP}.log"
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = retrieve_core.setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

    event_ids, queries = retrieve_core.load_v2_queries(limit=args.limit_queries)
    logger.info("Loaded %d v2 queries (event_id %d..%d).", len(event_ids), event_ids[0], event_ids[-1])

    if not args.overwrite and retrieve_core.model_already_done(ALIAS, event_ids):
        logger.info("[%s] all %d query files already exist; skipping (use --overwrite).", ALIAS, len(event_ids))
        return

    # Open the table first so a missing index fails fast before model load.
    table, table_name, root = retrieve_core.open_paragraph_table(ALIAS, args.lancedb_root, args.table_suffix)
    logger.info("[%s] table=%s root=%s rows=%d", ALIAS, table_name, root, table.count_rows())

    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")

    try:
        from pylate import models as pylate_models  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pylate is not installed in the current environment. Run with .venv-colbert/bin/python:\n"
            "    CUDA_VISIBLE_DEVICES=<gpu> .venv-colbert/bin/python "
            "src/retrieval/lancedb/frontier-v2/retrieve_colbert.py ..."
        ) from exc

    logger.info("[%s] loading pylate ColBERTv2 on %s", ALIAS, device)
    model = pylate_models.ColBERT(model_name_or_path=resolve_model_id(ALIAS), device=str(device))
    try:
        logger.info("[%s] encoding %d queries (is_query=True)", ALIAS, len(queries))
        q_embs_raw = model.encode(
            queries, batch_size=args.encode_batch, is_query=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        q_token_lists = [emb.tolist() for emb in q_embs_raw]
        q_dim = len(q_token_lists[0][0]) if q_token_lists and q_token_lists[0] else 128
        q_docs, q_mask = retrieve_core.pad_query_token_vectors(q_token_lists, q_dim, device)
        logger.info("[%s] query tensor %s (mask sum=%d)", ALIAS, tuple(q_docs.shape), int(q_mask.sum().item()))

        retrieve_core.maxsim_retrieve_and_write(
            alias=ALIAS, table=table, q_docs=q_docs, q_mask=q_mask, q_dim=q_dim,
            device=device, event_ids=event_ids, queries=queries, k=args.k,
            chunk=args.chunk, q_batch_size=args.q_batch_size,
            max_scan_batches=args.max_scan_batches,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("[%s] retrieval complete.", ALIAS)


if __name__ == "__main__":
    main()
