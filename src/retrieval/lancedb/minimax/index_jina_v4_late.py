"""jina-embeddings-v4 multi-vector (late-interaction) indexer.

Indexes one granularity at a time (paragraph / sentence / proposition) using
jina-v4's late-interaction mode: per-token vectors projected to 128 dims,
identical on-disk schema to the ColBERTv2 indexer so the existing MaxSim eval
and BM25 ensemble path in ``evaluate_minimax.py`` can read it unchanged.

Why HF transformers and not sentence-transformers:
    ``sentence-transformers`` silently aggregates jina-v4 to a single 2048-d
    vector even when ``task="retrieval"`` is passed. Multi-vector mode is only
    exposed through the model's HF ``custom_modeling.py`` via
    ``model.encode_text(..., return_multivector=True)``.

Schema (matches ``index_colbert.py``):
    paragraph_id : string  (row id: paragraph_id for paragraph,
                            "pid_i" for sentence/proposition)
    text         : string
    metadata     : string  (JSON, same shape as the dense indexer)
    token_vectors: list<list<float32, 128>>   (variable outer L, fixed inner 128)
"""

from __future__ import annotations

# IMPORTANT: set torch alloc-conf before torch is imported.
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

ALIAS = "jina-embeddings-v4-late"
TOKEN_DIM = 128
WRITE_CHUNK_SIZE = 5_000
DEFAULT_BATCH_SIZE = 8           # bf16, A100-80GB; long Wikipedia paragraphs
DEFAULT_ENCODE_CHUNK = 20_000

GRANULARITY_DEFAULT_SOURCE: dict[str, Path] = {
    "paragraph": CLASSIFIED_PARAGRAPHS_PATHS["minimax"],
    "sentence": CLASSIFIED_SENTENCES_MINIMAX_PATH,
    "proposition": CLASSIFIED_PROPOSITIONS_MINIMAX_PATH,
}

DEFAULT_LOG_DIR = Path("logs")
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging(log_file: Path | None) -> Path | None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(LOG_FORMATTER)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    return log_file


def _default_log_path(stem: str = "index_jina_v4_late") -> Path:
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
    """Convert a (L, 128) tensor/array to LanceDB-friendly list-of-list[float]."""
    if hasattr(emb, "detach"):
        emb = emb.detach().to(torch.float32).cpu().numpy()
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != TOKEN_DIM:
        raise ValueError(f"Unexpected token-vector shape {arr.shape}; expected (L, {TOKEN_DIM}).")
    # L2-normalize per token so MaxSim is cosine-equivalent (matches ColBERT path).
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    arr = arr / norms
    return arr.tolist()


def _load_model(device: str):
    """Load jina-embeddings-v4 in HF multi-vector mode."""
    from transformers import AutoModel

    model_id = MODEL_IDS[ALIAS]
    logger.info("[%s] Loading %s (HF AutoModel, bf16) on %s ...", ALIAS, model_id, device)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return model


def _encode_batch(model, texts: list[str], batch_size: int):
    """Encode passages and return a list of per-document (L, 128) tensors."""
    # jina-v4 custom_modeling exposes encode_text with batched dispatch and
    # the prompt_name / task / return_multivector knobs we need.
    with torch.no_grad():
        embeddings = model.encode_text(
            texts=texts,
            task="retrieval",
            prompt_name="passage",
            return_multivector=True,
            batch_size=batch_size,
        )
    # encode_text returns a list of (L_i, 128) tensors (one per input text).
    return embeddings


def index_jina_v4_late(
    corpus: ParagraphCorpus,
    db: lancedb.DBConnection,
    table_name: str,
    device: str,
    batch_size: int,
    encode_chunk: int,
    checkpoint: CheckpointStore,
    force_rerun: bool,
) -> None:
    U = len(corpus)
    if not force_rerun and checkpoint.is_done(table_name, U, db, table_name):
        logger.info("[%s] Already done (rows=%d). Skipping.", ALIAS, U)
        return

    per_model_handler, per_model_log = _attach_per_model_handler(ALIAS)
    logger.info("[%s] Per-model log: %s", ALIAS, per_model_log.resolve())

    if table_name in _db_tables(db):
        logger.info("[%s] Dropping existing table %s before reindex.", ALIAS, table_name)
        db.drop_table(table_name)

    checkpoint.mark_running(table_name, U)
    started = time.time()

    try:
        model = _load_model(device)

        schema = _build_schema()
        tbl = None
        rows_written = 0
        n_chunks = math.ceil(U / encode_chunk)

        for start in tqdm(range(0, U, encode_chunk), total=n_chunks, desc=f"[{ALIAS}] encode+write"):
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
            raise RuntimeError("jina-v4-late indexer produced no rows.")

        try:
            tbl.cleanup_old_versions()
        except (RuntimeError, OSError) as exc:
            logger.warning("cleanup_old_versions failed for %s: %s", table_name, exc)

        checkpoint.mark_done(table_name, rows_written=rows_written, U=U)
        logger.info(
            "[%s] Done: %d rows in %s (%.1f min).",
            ALIAS,
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
    parser = argparse.ArgumentParser(
        description="Index minimax corpus to LanceDB (jina-embeddings-v4 multi-vector / late-interaction)."
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="Post-CUDA_VISIBLE_DEVICES GPU indices. Indexing runs on the first index.",
    )
    parser.add_argument(
        "--granularity",
        choices=("paragraph", "sentence", "proposition"),
        default="paragraph",
    )
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--table-suffix", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--encode-chunk",
        type=int,
        default=DEFAULT_ENCODE_CHUNK,
        help="Encode + write this many texts per pass (controls peak RAM).",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to write logs to. Defaults to logs/index_jina_v4_late_<ts>.log; '-' disables file logging.",
    )
    # parse_known_args so the shared runner can forward dense-only flags without crashing.
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"index_jina_v4_late: ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file = _default_log_path()
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

    lancedb_root = lancedb_path_for(args.granularity)
    lancedb_root.mkdir(parents=True, exist_ok=True)
    logger.info("LanceDB root for granularity=%s: %s", args.granularity, lancedb_root)
    db = lancedb.connect(lancedb_root)
    checkpoint = CheckpointStore(default_state_dir(LANCEDB_PATH))

    table_name = get_classified_collection_name(ALIAS, args.granularity) + args.table_suffix

    index_jina_v4_late(
        corpus=corpus,
        db=db,
        table_name=table_name,
        device=device,
        batch_size=args.batch_size,
        encode_chunk=args.encode_chunk,
        checkpoint=checkpoint,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
