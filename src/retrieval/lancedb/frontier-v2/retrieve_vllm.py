"""frontier-v2 vLLM dense retriever (Qwen3-Embedding-4B / 8B).

Query-side mirror of ``index_vllm.py``. The indexer embedded DOCUMENTS as plain
text (Qwen3-Embedding applies an instruction to *queries only*), so here we
prepend the Qwen3-Embedding query instruction to each query before embedding,
then pull top-K from the model's LanceDB table.

Reuses ``index_vllm``'s engine loader + embedding extractor verbatim (same
weights, pooling, and L2-normalization as the documents were written with), so
query and document vectors live in the same space.

Runs in a dedicated vLLM venv (vllm 0.19.0). That env also needs lancedb
+ pyarrow + pylance (the orchestrator ensures this), same as for indexing.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from index_vllm import _embed_chunk, _load_vllm, _setup_logging
from model_registry import VLLM_ALIASES, resolve_model_id, spec_for

import retrieve_core

logger = logging.getLogger(__name__)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Official Qwen3-Embedding query instruction (documents get none). Matches the
# model card's get_detailed_instruct(task, query) -> "Instruct: {task}\nQuery:{q}".
QWEN3_QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _format_query(query: str) -> str:
    return f"Instruct: {QWEN3_QUERY_TASK}\nQuery:{query}"


def retrieve_one_vllm(
    *,
    alias: str,
    model_id: str,
    event_ids: list[int],
    queries: list[str],
    k: int,
    lancedb_root: str,
    table_suffix: str,
    overwrite: bool,
    search_chunk: int,
) -> None:
    if not overwrite and retrieve_core.model_already_done(alias, event_ids):
        logger.info("[%s] all %d query files already exist; skipping (use --overwrite to redo).",
                    alias, len(event_ids))
        return

    table, table_name, root = retrieve_core.open_paragraph_table(alias, lancedb_root, table_suffix)
    logger.info("[%s] table=%s root=%s rows=%d", alias, table_name, root, table.count_rows())

    # Encode queries with vLLM, then FREE the engine (it reserves ~85% of the GPU)
    # so the GPU dense search has headroom.
    spec = spec_for(alias)
    llm = None
    try:
        llm = _load_vllm(model_id, dtype=spec.dtype or "bfloat16")
        logger.info("[%s] encoding %d queries (with Qwen3 query instruction)", alias, len(queries))
        # _embed_chunk char-truncates, embeds, and L2-normalizes each vector —
        # identical to the document path, just on the instruction-prefixed queries.
        q_emb = _embed_chunk(llm, [_format_query(q) for q in queries])
    finally:
        del llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retrieve_core.search_and_write_dense_gpu(
        alias=alias, table=table, query_embeddings=q_emb,
        event_ids=event_ids, queries=queries, device=device, k=k, chunk=search_chunk,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-K v2 candidates (vLLM, Qwen3-Embedding).")
    parser.add_argument("--models", nargs="+", default=VLLM_ALIASES,
                        help="vLLM-backed aliases (default: Qwen3-Embedding-8B, -4B).")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="Accepted for symmetry; vLLM uses the GPU from CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--k", type=int, default=retrieve_core.DEFAULT_K)
    parser.add_argument("--lancedb-root", choices=("home", "scratch", "auto"), default="auto")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--table-suffix", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--search-chunk", type=int, default=50_000,
                        help="Rows per GPU dense-search scan batch (peak GPU memory knob).")
    parser.add_argument("--log-file", type=Path, default=None)
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"retrieve_vllm(frontier-v2): ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file: Path | None = Path("logs") / f"v2_retrieve_vllm_{RUN_TIMESTAMP}.log"
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = _setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

    event_ids, queries = retrieve_core.load_v2_queries(limit=args.limit_queries)
    logger.info("Loaded %d v2 queries (event_id %d..%d).", len(event_ids), event_ids[0], event_ids[-1])

    failures: list[str] = []
    for alias in args.models:
        try:
            model_id = resolve_model_id(alias)
        except KeyError as exc:
            logger.error("[%s] %s — skipping.", alias, exc)
            failures.append(alias)
            continue
        try:
            retrieve_one_vllm(
                alias=alias, model_id=model_id, event_ids=event_ids, queries=queries,
                k=args.k, lancedb_root=args.lancedb_root, table_suffix=args.table_suffix,
                overwrite=args.overwrite, search_chunk=args.search_chunk,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] FAILED (continuing): %r", alias, exc)
            failures.append(alias)

    if failures:
        logger.error("vLLM retrieval finished with failures: %s", failures)
        sys.exit(3)
    logger.info("vLLM retrieval complete for: %s", args.models)


if __name__ == "__main__":
    main()
