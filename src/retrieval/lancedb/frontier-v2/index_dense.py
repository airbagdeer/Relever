"""frontier-v2 dense indexer CLI (sentence-transformers backend).

Indexes the v2 SPLADE/BM25 paragraph corpus with the sentence-transformers
dense models in the frontier-v2 set. Per-model behaviour (dtype, padding,
doc prompt, task/prompt_name, single-vs-multi GPU) comes from
``model_registry.MODEL_SPECS``; the encode loop is ``_dense_core``.

Key differences from the minimax dense indexer:
  - corpus comes from ``corpus_loader_v2`` (the nested v2 JSON shape),
  - alias resolution is the union of all three config model dicts,
  - LanceDB root is chosen PER MODEL (``--lancedb-root {home,scratch,auto}``),
  - a per-alias try/except isolates failures so one bad model (e.g. a
    low-adoption repo that won't load under transformers 5.x) never aborts the
    rest of the slot's model list.

Run from the project root, e.g.::

    CUDA_VISIBLE_DEVICES=6 uv run python \
        src/retrieval/lancedb/frontier-v2/index_dense.py \
        --models F2LLM-v2-8B QZhou-Embedding --gpus 0 --lancedb-root auto
"""

from __future__ import annotations

# Must precede any torch import (incl. transitive via _dense_core).
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import lancedb

import _bootstrap  # noqa: F401
from checkpoint import CheckpointStore, default_state_dir, drop_target_tables
from config import LANCEDB_PATH, get_classified_collection_name
from corpus_loader_v2 import load_v2_paragraph_corpus
from model_registry import ST_DENSE_ALIASES, resolve_model_id, root_for, spec_for
from paths import SPLADE_WIKIPEDIA_V2_PATH, V2_CORPUS_CACHE_DIR
from _dense_core import _mmx, index_one_model_v2

logger = logging.getLogger(__name__)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index v2 paragraphs to LanceDB (dense / sentence-transformers).")
    parser.add_argument("--models", nargs="+", default=ST_DENSE_ALIASES,
                        help="Aliases of dense ST models to index (default: all 14).")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="Post-CUDA_VISIBLE_DEVICES GPU indices for the pool (default: [0]).")
    parser.add_argument("--lancedb-root", choices=("home", "scratch", "auto"), default="auto",
                        help="Per-model LanceDB root. 'auto' consults the model spec's disk_target.")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Smoke-test knob: read only the first N v2 JSON files.")
    parser.add_argument("--table-suffix", type=str, default="",
                        help="Suffix appended to LanceDB table names (e.g. '_smoke').")
    parser.add_argument("--reset-tables", action="store_true",
                        help="Drop each model's target table (on its own root) before indexing.")
    parser.add_argument("--force-rerun", nargs="*", default=[],
                        help="Aliases to reindex even if the checkpoint says done.")
    parser.add_argument("--initial-batch", type=int, default=None,
                        help="Override the per-model initial batch size.")
    parser.add_argument("--no-cache", action="store_true", help="Skip the v2 corpus pickle cache.")
    parser.add_argument("--n-shards", type=int, default=_mmx.N_LENGTH_SHARDS_TARGET,
                        help="Target number of length-bucketed shards.")
    parser.add_argument("--source-format", choices=("splade_v2",), default="splade_v2",
                        help="Corpus format (only splade_v2 is supported here).")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Log file (in addition to stderr). '-' disables file logging.")
    # parse_known_args so the shared orchestrator can forward flags meant for
    # the colbert / jina / vllm indexers without crashing here.
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"index_dense(frontier-v2): ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file: Path | None = Path("logs") / f"v2_index_dense_{RUN_TIMESTAMP}.log"
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = _mmx._setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

    # Load the v2 corpus once; all models share it.
    corpus = load_v2_paragraph_corpus(
        source_dir=SPLADE_WIKIPEDIA_V2_PATH,
        limit_files=args.limit_files,
        cache_dir=V2_CORPUS_CACHE_DIR,
        use_cache=not args.no_cache,
    )
    if len(corpus) == 0:
        logger.error("Empty v2 corpus loaded from %s. Aborting.", SPLADE_WIKIPEDIA_V2_PATH)
        sys.exit(1)
    logger.info("Loaded v2 corpus: U=%d unique paragraphs.", len(corpus))

    # Checkpoint always lives on home (single source of truth across all roots).
    checkpoint = CheckpointStore(default_state_dir(LANCEDB_PATH))
    pool_devices = _mmx._build_pool_devices(args.gpus)
    logger.info("Pool devices: %s ; lancedb-root mode: %s", pool_devices, args.lancedb_root)

    force_set = set(args.force_rerun)
    failures: list[str] = []
    for alias in args.models:
        spec = spec_for(alias)
        # pylate / jina_v4_hf need their dedicated indexers. vllm-backed aliases
        # (Qwen3-Embedding) ARE valid sentence-transformers models, so index_dense
        # runs them when explicitly listed — this is the orchestrator's ST
        # fallback path when vLLM is unavailable.
        if spec.backend in ("pylate", "jina_v4_hf"):
            logger.warning("[%s] backend=%s needs its own indexer; skipping here.",
                           alias, spec.backend)
            continue
        try:
            model_id = resolve_model_id(alias)
        except KeyError as exc:
            logger.error("[%s] %s — skipping.", alias, exc)
            failures.append(alias)
            continue

        root = root_for(alias, args.lancedb_root)
        root.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(root)
        logger.info("[%s] LanceDB root: %s", alias, root)

        if args.reset_tables:
            table_name = get_classified_collection_name(alias, "paragraph") + args.table_suffix
            dropped = drop_target_tables(db, [table_name])
            if dropped:
                logger.info("[%s] reset dropped: %s", alias, dropped)

        initial_batch = args.initial_batch if args.initial_batch is not None else spec.init_batch
        try:
            index_one_model_v2(
                alias=alias,
                model_id=model_id,
                spec=spec,
                corpus=corpus,
                db=db,
                table_suffix=args.table_suffix,
                pool_devices=pool_devices,
                initial_batch=initial_batch,
                checkpoint=checkpoint,
                force_rerun=alias in force_set,
                n_shards_target=args.n_shards,
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-model failures
            logger.error("[%s] FAILED (continuing with next model): %r", alias, exc)
            failures.append(alias)

    if failures:
        logger.error("Dense indexing finished with %d failed model(s): %s",
                     len(failures), failures)
        # Non-zero exit so the orchestrator surfaces partial failure, but only
        # after every model was attempted.
        sys.exit(3)
    logger.info("Dense indexing complete for: %s", args.models)


if __name__ == "__main__":
    main()
