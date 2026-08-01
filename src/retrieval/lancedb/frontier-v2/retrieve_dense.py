"""frontier-v2 dense retriever CLI (sentence-transformers backend).

Query-side mirror of ``index_dense.py``: for each dense ST model, encode the
1050 v2 queries and pull the top-K paragraphs from the model's existing LanceDB
table, writing a compact per-(model, query) candidate file.

The only thing that differs from indexing is the *query* encoding. Indexing
encoded DOCUMENTS, auto-selecting the model's ``document``/``passage``/``doc``
prompt (``_dense_core`` lines 124-130). Retrieval is the exact symmetric mirror:
auto-select the model's ``query`` prompt when the spec doesn't pin one — that is
how the decoder-LLM embedders (Qwen3 family, Octen, F2LLM, QZhou, MoD, ICT,
jina-v5) carry their query instruction. Models without a ``query`` prompt are
encoded as plain normalized text, matching their plain document side.

Run from the project root, e.g.::

    CUDA_VISIBLE_DEVICES=6 uv run python \
        src/retrieval/lancedb/frontier-v2/retrieve_dense.py \
        --models F2LLM-v2-8B Octen-Embedding-4B --gpus 0 --k 500
"""

from __future__ import annotations

# Must precede any torch import (incl. transitive via _dense_core).
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
from _dense_core import _build_model, _mmx
from model_registry import ST_DENSE_ALIASES, resolve_model_id, spec_for

import retrieve_core

logger = logging.getLogger(__name__)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _query_encode_kwargs(model, spec, model_id: str) -> dict:
    """Query-side encode kwargs, mirroring the indexer's document-side contract.

    Starts from the spec's ``encode_kwargs`` (usually empty); if no prompt/task
    is pinned, auto-selects the model's own ``query`` prompt (symmetric to the
    document auto-detect in ``_dense_core``). Always L2-normalizes.
    """
    kw: dict = dict(spec.encode_kwargs)
    kw.setdefault("batch_size", 128)
    kw["show_progress_bar"] = True
    kw["convert_to_numpy"] = True
    kw["normalize_embeddings"] = True

    # Query-ONLY prompt pins (from the spec) take precedence over auto-detect: they cover models
    # whose query instruction the generic 'query'-key auto-detect can't reach — a non-standard
    # prompt key (query_prompt_name, e.g. jasper's 's2p_query') or a literal instruction string
    # for repos that ship no prompts config (query_prompt, e.g. LGAI).
    if spec.query_prompt_name is not None:
        kw["prompt_name"] = spec.query_prompt_name
        logger.info("[%s] pinned query prompt_name=%r", spec.alias, spec.query_prompt_name)
    elif spec.query_prompt is not None:
        kw["prompt"] = spec.query_prompt
        logger.info("[%s] pinned literal query prompt (len=%d).", spec.alias, len(spec.query_prompt))

    if not ({"prompt_name", "prompt", "task"} & kw.keys()):
        prompts = getattr(model, "prompts", None) or {}
        if "query" in prompts:
            kw["prompt_name"] = "query"
            logger.info("[%s] auto-selected query prompt_name='query'", spec.alias)
        else:
            logger.info("[%s] no 'query' prompt; encoding plain normalized text.", spec.alias)

    # jina v3 parity with the indexer's task-kwarg contract (task, not prompt_name).
    if "jina-embeddings-v3" in model_id:
        kw.pop("prompt_name", None)
        kw["task"] = "retrieval.query"
    return kw


def retrieve_one_dense(
    *,
    alias: str,
    model_id: str,
    spec,
    device: str,
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

    # Encode the queries, then free the model so the GPU dense search has headroom.
    model = _build_model(model_id, spec, device)
    try:
        kw = _query_encode_kwargs(model, spec, model_id)
        logger.info("[%s] encoding %d queries (encode kwargs=%s)",
                    alias, len(queries),
                    {kk: vv for kk, vv in kw.items() if kk != "show_progress_bar"})
        q_emb = model.encode(queries, **kw)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    retrieve_core.search_and_write_dense_gpu(
        alias=alias, table=table, query_embeddings=q_emb,
        event_ids=event_ids, queries=queries, device=device, k=k, chunk=search_chunk,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-K v2 candidates (dense / sentence-transformers).")
    parser.add_argument("--models", nargs="+", default=ST_DENSE_ALIASES,
                        help="Aliases of dense ST models (default: all 14).")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="Post-CUDA_VISIBLE_DEVICES GPU index for encoding (default: [0]).")
    parser.add_argument("--k", type=int, default=retrieve_core.DEFAULT_K,
                        help="Candidates per query (default: 500).")
    parser.add_argument("--lancedb-root", choices=("home", "scratch", "auto"), default="auto",
                        help="Per-model LanceDB root. 'auto' consults the model spec's disk_target.")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="Smoke knob: only retrieve for the first N queries.")
    parser.add_argument("--table-suffix", type=str, default="",
                        help="Suffix on LanceDB table names (e.g. '_smoke').")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-retrieve even if a model's query files already exist.")
    parser.add_argument("--search-chunk", type=int, default=50_000,
                        help="Rows per GPU dense-search scan batch (peak GPU memory knob).")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Log file (in addition to stderr). '-' disables file logging.")
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"retrieve_dense(frontier-v2): ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file: Path | None = Path("logs") / f"v2_retrieve_dense_{RUN_TIMESTAMP}.log"
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = _mmx._setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

    event_ids, queries = retrieve_core.load_v2_queries(limit=args.limit_queries)
    logger.info("Loaded %d v2 queries (event_id %d..%d).", len(event_ids), event_ids[0], event_ids[-1])

    device = f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s ; lancedb-root mode: %s ; k=%d", device, args.lancedb_root, args.k)

    failures: list[str] = []
    for alias in args.models:
        spec = spec_for(alias)
        if spec.backend in ("pylate", "jina_v4_hf"):
            logger.warning("[%s] backend=%s needs its own retriever; skipping here.", alias, spec.backend)
            continue
        try:
            model_id = resolve_model_id(alias)
        except KeyError as exc:
            logger.error("[%s] %s — skipping.", alias, exc)
            failures.append(alias)
            continue
        try:
            retrieve_one_dense(
                alias=alias, model_id=model_id, spec=spec, device=device,
                event_ids=event_ids, queries=queries, k=args.k,
                lancedb_root=args.lancedb_root, table_suffix=args.table_suffix,
                overwrite=args.overwrite, search_chunk=args.search_chunk,
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-model failures
            logger.error("[%s] FAILED (continuing with next model): %r", alias, exc)
            failures.append(alias)

    if failures:
        logger.error("Dense retrieval finished with %d failed model(s): %s", len(failures), failures)
        sys.exit(3)
    logger.info("Dense retrieval complete for: %s", args.models)


if __name__ == "__main__":
    main()
