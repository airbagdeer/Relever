"""frontier-v2 jina-embeddings-v4 late-interaction wrapper — v2 corpus, home root.

Thin CLI around ``minimax.index_jina_v4_late.index_jina_v4_late`` (loaded by
explicit path). Only the corpus source (v2) and LanceDB root (durable home,
this is the largest + slowest-to-recompute multi-vector table) differ; the
HF AutoModel multi-vector encoder, schema, and checkpoint logic are reused.

Runs in ``.venv-jina-v4`` (transformers <5, required by jina-v4's vendored
custom modeling).
"""

from __future__ import annotations

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
import torch

import _bootstrap
from checkpoint import CheckpointStore, default_state_dir, drop_target_tables
from config import LANCEDB_PATH, get_classified_collection_name
from corpus_loader_v2 import load_v2_paragraph_corpus
from model_registry import root_for, spec_for
from paths import SPLADE_WIKIPEDIA_V2_PATH, V2_CORPUS_CACHE_DIR

_mmx = _bootstrap.load_minimax("index_jina_v4_late.py", "mmx_jina_v4_late")

logger = logging.getLogger(__name__)
ALIAS = "jina-embeddings-v4-late"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index v2 paragraphs to LanceDB (jina-embeddings-v4 multi-vector / late-interaction)."
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--lancedb-root", choices=("home", "scratch", "auto"), default="auto")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--table-suffix", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-forward-pass batch (default: model spec init_batch).")
    parser.add_argument("--encode-chunk", type=int, default=_mmx.DEFAULT_ENCODE_CHUNK)
    parser.add_argument("--reset-tables", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--source-format", choices=("splade_v2",), default="splade_v2")
    parser.add_argument("--log-file", type=Path, default=None)
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"index_jina_v4_late(frontier-v2): ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file: Path | None = Path("logs") / f"v2_index_jina_v4_late_{RUN_TIMESTAMP}.log"
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = _mmx._setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

    device = f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu"

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

    root = root_for(ALIAS, args.lancedb_root)
    root.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(root)
    logger.info("[%s] LanceDB root: %s", ALIAS, root)
    checkpoint = CheckpointStore(default_state_dir(LANCEDB_PATH))

    table_name = get_classified_collection_name(ALIAS, "paragraph") + args.table_suffix
    if args.reset_tables:
        drop_target_tables(db, [table_name])

    batch_size = args.batch_size if args.batch_size is not None else spec_for(ALIAS).init_batch
    _mmx.index_jina_v4_late(
        corpus=corpus,
        db=db,
        table_name=table_name,
        device=device,
        batch_size=batch_size,
        encode_chunk=args.encode_chunk,
        checkpoint=checkpoint,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
