"""ColBERTv2 multi-vector indexer using pylate.

Indexes one granularity at a time (paragraph / sentence / proposition).
Pylate's `models.ColBERT(...)` exposes raw token vectors directly (shape
`(num_tokens, 128)` per document), so we can write our own LanceDB schema
without going through PLAID/FAISS. Embeddings are equivalent to the reference
`colbert-ir/colbertv2.0` checkpoint (validated by the parity smoke test in
`tests/test_colbert_parity.py`).

Schema:
    paragraph_id : string  (row id: paragraph_id, sentence id "pid_i", or proposition id "pid_i")
    text         : string
    metadata     : string  (JSON, same shape as dense indexer)
    token_vectors: list<list<float32, 128>>   (variable outer, fixed inner)
"""

from __future__ import annotations

# IMPORTANT: set torch alloc-conf before torch is imported (and before pylate
# imports it transitively). These also propagate to spawned subprocesses.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from config import (  # noqa: E402
    CLASSIFIED_PARAGRAPHS_PATHS,
    CLASSIFIED_PROPOSITIONS_MINIMAX_PATH,
    CLASSIFIED_SENTENCES_MINIMAX_PATH,
    LANCEDB_PATH,
    MODEL_IDS,
    get_classified_collection_name,
    lancedb_path_for,
)

# Minimax paths everywhere: the runner script (run_index_minimax.sh) indexes the
# minimax corpus, so all three granularities must source from the minimax
# directories. Previously sentence/proposition pointed at the gpt_nano dirs,
# which caused index runs to silently embed the wrong corpus.
GRANULARITY_DEFAULT_SOURCE: dict[str, Path] = {
    "paragraph": CLASSIFIED_PARAGRAPHS_PATHS["minimax"],
    "sentence": CLASSIFIED_SENTENCES_MINIMAX_PATH,
    "proposition": CLASSIFIED_PROPOSITIONS_MINIMAX_PATH,
}
from checkpoint import (  # noqa: E402
    CheckpointStore,
    _db_tables,
    default_state_dir,
)
from corpus_loader import (  # noqa: E402
    ParagraphCorpus,
    default_cache_dir,
    load_paragraph_corpus,
)

logger = logging.getLogger(__name__)

COLBERT_ALIAS = "colbert-v2"
TOKEN_DIM = 128
WRITE_CHUNK_SIZE = 5_000


DEFAULT_LOG_DIR = Path("logs")
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging(log_file: Path | None) -> Path | None:
    formatter = LOG_FORMATTER
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    return log_file


def _default_log_path(stem: str = "index_colbert") -> Path:
    return DEFAULT_LOG_DIR / f"{stem}_{RUN_TIMESTAMP}.log"


def _per_model_log_path(alias: str) -> Path:
    safe = alias.replace("/", "-").replace(" ", "_")
    return DEFAULT_LOG_DIR / f"model_{safe}_{RUN_TIMESTAMP}.log"


def _attach_per_model_handler(alias: str) -> tuple[logging.FileHandler, Path]:
    path = _per_model_log_path(alias)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(LOG_FORMATTER)
    logging.getLogger().addHandler(handler)
    return handler, path


def _detach_per_model_handler(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    try:
        handler.close()
    except (OSError, ValueError):
        pass


def _build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("paragraph_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field(
                "token_vectors",
                pa.list_(pa.list_(pa.float32(), TOKEN_DIM)),
            ),
        ]
    )


def _to_token_lists(emb) -> list[list[float]]:
    """Convert a pylate token-vector tensor/array to LanceDB-friendly list-of-list[float]."""
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != TOKEN_DIM:
        raise ValueError(f"Unexpected token-vector shape {arr.shape}; expected (L, {TOKEN_DIM}).")
    return arr.tolist()


def _encode_batch(model, texts: list[str], batch_size: int):
    """Encode a batch of paragraphs with pylate; return list of (L, 128) arrays/tensors."""
    return model.encode(
        texts,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def index_colbert(
    corpus: ParagraphCorpus,
    db: lancedb.DBConnection,
    table_name: str,
    device: str,
    batch_size: int,
    encode_chunk: int,
    checkpoint: CheckpointStore,
    alias: str,
    force_rerun: bool,
    document_length: int | None = None,
) -> None:
    try:
        from pylate import models as pylate_models  # imported lazily so the dense indexer doesn't pay this cost
    except ImportError as exc:
        raise SystemExit(
            "pylate is not installed in the current environment. Pylate 1.4.0 caps "
            "transformers at <=4.56.2 which conflicts with the project's main pin "
            "(>=4.57.3). Install it into a dedicated venv:\n"
            "    uv venv .venv-colbert --python 3.12.7\n"
            "    uv pip install --python .venv-colbert/bin/python "
            "pylate>=1.4.0 lancedb>=0.29.2 pyarrow>=23.0.1 tqdm numpy torch\n"
            "and re-run via run_index_minimax.sh, which uses .venv-colbert for this step."
        ) from exc

    U = len(corpus)
    if not force_rerun and checkpoint.is_done(table_name, U, db, table_name):
        logger.info("[%s] Already done (rows=%d). Skipping.", alias, U)
        return

    per_model_handler, per_model_log_path = _attach_per_model_handler(alias)
    logger.info("[%s] Per-model log: %s", alias, per_model_log_path.resolve())

    if table_name in _db_tables(db):
        logger.info("[%s] Dropping existing table %s before reindex.", alias, table_name)
        db.drop_table(table_name)

    checkpoint.mark_running(table_name, U)
    started = time.time()

    try:
        # document_length=None keeps pylate's default (180 tokens) for the existing minimax
        # callers; frontier-v2 passes 512 so full Wikipedia paragraphs (~473 tok avg) aren't
        # truncated to the first ~38% of their tokens.
        colbert_kwargs: dict = {
            "model_name_or_path": MODEL_IDS[COLBERT_ALIAS],
            "device": device,
        }
        if document_length is not None:
            colbert_kwargs["document_length"] = document_length
        logger.info("[%s] Loading pylate ColBERTv2 on %s (document_length=%s) ...",
                    alias, device, document_length if document_length is not None else "default")
        model = pylate_models.ColBERT(**colbert_kwargs)

        schema = _build_schema()
        tbl = None
        rows_written = 0
        n_chunks = math.ceil(U / encode_chunk)

        for chunk_idx, start in enumerate(
            tqdm(range(0, U, encode_chunk), total=n_chunks, desc=f"[{alias}] encode+write")
        ):
            stop = min(start + encode_chunk, U)
            texts_chunk = corpus.texts[start:stop]

            embeddings_list = _encode_batch(model, texts_chunk, batch_size=batch_size)

            rows: list[dict] = []
            for i, emb in enumerate(embeddings_list):
                tokens = _to_token_lists(emb)
                rows.append(
                    {
                        "paragraph_id": corpus.paragraph_ids[start + i],
                        "text": corpus.texts[start + i],
                        "metadata": corpus.metadata_jsons[start + i],
                        "token_vectors": tokens,
                    }
                )

            for write_start in range(0, len(rows), WRITE_CHUNK_SIZE):
                write_stop = min(write_start + WRITE_CHUNK_SIZE, len(rows))
                batch_rows = rows[write_start:write_stop]
                if tbl is None:
                    tbl = db.create_table(table_name, data=batch_rows, schema=schema, mode="overwrite")
                else:
                    tbl.add(batch_rows)

            rows_written = stop

        if tbl is None:
            raise RuntimeError("ColBERT indexer produced no rows.")

        try:
            tbl.cleanup_old_versions()
        except (RuntimeError, OSError) as exc:
            logger.warning("cleanup_old_versions failed for %s: %s", table_name, exc)

        checkpoint.mark_done(table_name, rows_written=rows_written, U=U)
        logger.info(
            "[%s] Done: %d rows in %s (%.1f min).",
            alias,
            rows_written,
            table_name,
            (time.time() - started) / 60.0,
        )
    except Exception as exc:  # noqa: BLE001
        checkpoint.mark_failed(table_name, error=repr(exc), U=U)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _detach_per_model_handler(per_model_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index minimax paragraphs to LanceDB (ColBERTv2 / pylate).")
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="Post-CUDA_VISIBLE_DEVICES GPU indices. ColBERT runs on the first index; "
        "additional indices are reserved for future multi-process sharding.",
    )
    parser.add_argument(
        "--granularity",
        choices=("paragraph", "sentence", "proposition"),
        default="paragraph",
        help=(
            "Which granularity to index. Picks the matching default source dir "
            "(see GRANULARITY_DEFAULT_SOURCE) unless --source-dir is also set."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Override the default source directory for the chosen granularity.",
    )
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--table-suffix", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--encode-chunk",
        type=int,
        default=20_000,
        help="Encode + write this many paragraphs per pass (controls peak RAM).",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Path to write logs to (in addition to stderr). Defaults to "
            "logs/index_colbert_<timestamp>.log. Pass '-' to disable file "
            "logging."
        ),
    )
    # parse_known_args so the runner can forward shared flags to both scripts
    # without crashing on the dense-only ones (e.g. --max-attn-gib, --models,
    # --reset-tables, --initial-batch, --n-shards).
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"index_colbert: ignoring unknown args (likely dense-only): {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file = _default_log_path("index_colbert")
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = _setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

    device = f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu"

    source_dir: Path = (
        args.source_dir if args.source_dir is not None else GRANULARITY_DEFAULT_SOURCE[args.granularity]
    )
    logger.info("Source for granularity=%s: %s", args.granularity, source_dir)
    cache_dir = default_cache_dir(LANCEDB_PATH)
    corpus = load_paragraph_corpus(
        source_dir=source_dir,
        granularity_level=args.granularity,
        limit_files=args.limit_files,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
    )
    if len(corpus) == 0:
        logger.error("Empty corpus loaded from %s. Aborting.", source_dir)
        sys.exit(1)

    # Per-granularity LanceDB root (sentence/proposition route to local-disk
    # scratch via LANCEDB_SCRATCH_PATH). Checkpoint stays under LANCEDB_PATH.
    lancedb_root = lancedb_path_for(args.granularity)
    lancedb_root.mkdir(parents=True, exist_ok=True)
    logger.info("LanceDB root for granularity=%s: %s", args.granularity, lancedb_root)
    db = lancedb.connect(lancedb_root)
    checkpoint = CheckpointStore(default_state_dir(LANCEDB_PATH))

    table_name = get_classified_collection_name(COLBERT_ALIAS, args.granularity) + args.table_suffix

    index_colbert(
        corpus=corpus,
        db=db,
        table_name=table_name,
        device=device,
        batch_size=args.batch_size,
        encode_chunk=args.encode_chunk,
        checkpoint=checkpoint,
        alias=COLBERT_ALIAS,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
