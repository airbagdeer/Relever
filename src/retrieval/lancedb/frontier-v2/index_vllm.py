"""frontier-v2 vLLM dense indexer (Qwen3-Embedding-4B / 8B).

vLLM has official, low-risk embedding/pooling support for the Qwen3-Embedding
family, and is markedly faster than sentence-transformers for these decoder-LLM
embedders. This indexer writes the SAME LanceDB schema as ``_dense_core`` /
``minimax.index_dense`` (``id`` / ``vector`` / ``text`` / ``metadata``) so the
downstream evaluator reads vLLM-produced tables identically to ST ones.

Runs in a dedicated vLLM venv (vllm 0.19.0, torch 2.10+cu128).
That env must also have ``lancedb`` + ``pyarrow`` installed (the orchestrator
ensures this). Documents are encoded as PLAIN TEXT (Qwen3-Embedding applies an
instruction to queries only); outputs are L2-normalized before writing.

Single GPU per process (select via ``CUDA_VISIBLE_DEVICES``); ``--gpus`` is
accepted for CLI symmetry but vLLM uses tensor_parallel_size=1 here.

If vLLM cannot load these models in the smoke test, fall back to the ST path by
moving the two aliases into an ``index_dense.py`` ``--models`` list (they are
valid ST aliases too — just flip the call site; no code change needed).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# vLLM is chatty on import; keep its logs at INFO but avoid the usage stats ping.
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import lancedb
import numpy as np

import _bootstrap  # noqa: F401
from checkpoint import CheckpointStore, _db_tables, default_state_dir, drop_target_tables
from config import LANCEDB_PATH, get_classified_collection_name
from corpus_loader_v2 import load_v2_paragraph_corpus
from model_registry import VLLM_ALIASES, resolve_model_id, root_for, spec_for
from paths import SPLADE_WIKIPEDIA_V2_PATH, V2_CORPUS_CACHE_DIR

logger = logging.getLogger(__name__)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Bound the model context so KV/activation memory stays modest and batches stay
# large. Paragraphs almost never exceed this; longer ones are char-truncated.
MAX_MODEL_LEN = 8192
CHAR_TRUNCATE = MAX_MODEL_LEN * 4  # ~4 chars/token safety margin before tokenization
ENCODE_CHUNK = 50_000              # embed + write this many docs per pass
WRITE_CHUNK_SIZE = 50_000


def _setup_logging(log_file: Path | None) -> Path | None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    for h in handlers:
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for h in handlers:
        root.addHandler(h)
    return log_file


def _load_vllm(model_id: str, dtype: str):
    """Construct a vLLM offline embedding engine, tolerant of API churn.

    vllm renamed the embedding selector from ``task="embed"`` to
    ``runner="pooling"`` across releases; try the newer form first, fall back.
    """
    from vllm import LLM

    common = dict(
        model=model_id,
        tensor_parallel_size=1,
        dtype=dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=False,
    )
    last_exc: Exception | None = None
    for selector in ({"runner": "pooling"}, {"task": "embed"}):
        try:
            llm = LLM(**common, **selector)
            logger.info("vLLM loaded %s with %s", model_id, selector)
            return llm
        except TypeError as exc:
            last_exc = exc
            logger.warning("vLLM ctor rejected %s (%s); trying next form.", selector, exc)
    raise RuntimeError(f"Could not construct vLLM embedding engine for {model_id}: {last_exc}")


def _extract_embedding(out) -> np.ndarray:
    """Pull the embedding vector out of a vLLM pooling output, L2-normalized."""
    data = getattr(out.outputs, "embedding", None)
    if data is None:
        data = getattr(out.outputs, "data", None)
    if data is None:
        raise AttributeError("vLLM pooling output exposed neither .embedding nor .data")
    if hasattr(data, "detach"):
        data = data.detach().to("cpu").float().numpy()
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


def _embed_chunk(llm, texts: list[str]) -> list[np.ndarray]:
    prepared = [t[:CHAR_TRUNCATE] for t in texts]
    outputs = llm.embed(prepared)
    return [_extract_embedding(o) for o in outputs]


def index_one_vllm(alias: str, model_id: str, corpus, db, table_suffix: str,
                   checkpoint: CheckpointStore, force_rerun: bool) -> None:
    table_name = get_classified_collection_name(alias, "paragraph") + table_suffix
    U = len(corpus)
    if not force_rerun and checkpoint.is_done(table_name, U, db, table_name):
        logger.info("[%s] Already done (rows=%d). Skipping.", alias, U)
        return

    checkpoint.mark_running(table_name, U)
    started = time.time()
    llm = None
    tbl = None
    try:
        if table_name in _db_tables(db):
            logger.info("[%s] Dropping existing table %s before reindex.", alias, table_name)
            db.drop_table(table_name)

        spec = spec_for(alias)
        llm = _load_vllm(model_id, dtype=spec.dtype or "bfloat16")

        rows_written = 0
        for start in range(0, U, ENCODE_CHUNK):
            stop = min(start + ENCODE_CHUNK, U)
            vecs = _embed_chunk(llm, corpus.texts[start:stop])
            for w0 in range(0, len(vecs), WRITE_CHUNK_SIZE):
                w1 = min(w0 + WRITE_CHUNK_SIZE, len(vecs))
                rows = [
                    {
                        "id": corpus.paragraph_ids[start + k],
                        "vector": vecs[k],
                        "text": corpus.texts[start + k],
                        "metadata": corpus.metadata_jsons[start + k],
                    }
                    for k in range(w0, w1)
                ]
                if tbl is None:
                    tbl = db.create_table(table_name, data=rows, mode="overwrite")
                else:
                    tbl.add(rows)
                rows_written += len(rows)
            logger.info("[%s] %d/%d embedded+written", alias, stop, U)

        if tbl is None:
            raise RuntimeError(f"[{alias}] vLLM produced no rows.")
        try:
            tbl.cleanup_old_versions()
        except (RuntimeError, OSError) as exc:
            logger.warning("cleanup_old_versions failed for %s: %s", table_name, exc)

        checkpoint.mark_done(table_name, rows_written=rows_written, U=U)
        logger.info("[%s] Done: %d rows in %s (%.1f min).",
                    alias, rows_written, table_name, (time.time() - started) / 60.0)
    except Exception as exc:  # noqa: BLE001
        checkpoint.mark_failed(table_name, error=repr(exc), U=U)
        raise
    finally:
        del llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Index v2 paragraphs to LanceDB (vLLM, Qwen3-Embedding).")
    parser.add_argument("--models", nargs="+", default=VLLM_ALIASES,
                        help="vLLM-backed aliases to index (default: Qwen3-Embedding-8B, -4B).")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="Accepted for symmetry; vLLM uses the GPU exposed by CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--lancedb-root", choices=("home", "scratch", "auto"), default="auto")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--table-suffix", type=str, default="")
    parser.add_argument("--reset-tables", action="store_true")
    parser.add_argument("--force-rerun", nargs="*", default=[])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--source-format", choices=("splade_v2",), default="splade_v2")
    parser.add_argument("--log-file", type=Path, default=None)
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"index_vllm(frontier-v2): ignoring unknown args: {_unknown}", file=sys.stderr)

    if args.log_file is None:
        log_file: Path | None = Path("logs") / f"v2_index_vllm_{RUN_TIMESTAMP}.log"
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = _setup_logging(log_file)
    if actual_log is not None:
        logger.info("Log file: %s", actual_log.resolve())

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

    checkpoint = CheckpointStore(default_state_dir(LANCEDB_PATH))
    force_set = set(args.force_rerun)
    failures: list[str] = []
    for alias in args.models:
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
            tname = get_classified_collection_name(alias, "paragraph") + args.table_suffix
            drop_target_tables(db, [tname])
        try:
            index_one_vllm(alias, model_id, corpus, db, args.table_suffix,
                           checkpoint, force_rerun=alias in force_set)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] FAILED (continuing): %r", alias, exc)
            failures.append(alias)

    if failures:
        logger.error("vLLM indexing finished with failures: %s", failures)
        sys.exit(3)
    logger.info("vLLM indexing complete for: %s", args.models)


if __name__ == "__main__":
    main()
