"""Dense (single-vector) embedding indexer for the minimax corpus.

Indexes one granularity at a time (paragraph / sentence / proposition). Loads
the corpus once via `corpus_loader.load_paragraph_corpus`, then iterates over
the requested models. Each model is encoded with
`SentenceTransformer.encode_multi_process` across the GPU indices passed via
`--gpus` (which are post-`CUDA_VISIBLE_DEVICES` indices). Models are processed
sequentially (one at a time) so each model gets all GPUs in the lane.

Writes to LanceDB table `thesis-qdrant-classified-{alias}-{granularity}`.
"""

from __future__ import annotations

# IMPORTANT: these env vars must be set before torch is imported so that the
# CUDA caching allocator picks them up. They also propagate to spawned worker
# processes via `os.environ`.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import lancedb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Allow importing src/retrieval/config.py and sibling modules without requiring
# the project to be installed as a package.
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
GRANULARITY_DEFAULT_SOURCE: dict[str, "Path"] = {
    "paragraph": CLASSIFIED_PARAGRAPHS_PATHS["minimax"],
    "sentence": CLASSIFIED_SENTENCES_MINIMAX_PATH,
    "proposition": CLASSIFIED_PROPOSITIONS_MINIMAX_PATH,
}
from checkpoint import (  # noqa: E402
    CheckpointStore,
    _db_tables,
    default_state_dir,
    drop_target_tables,
)
from corpus_loader import (  # noqa: E402
    ParagraphCorpus,
    default_cache_dir,
    load_paragraph_corpus,
)

logger = logging.getLogger(__name__)

DEFAULT_DENSE_MODELS = [
    "gte-small",
    "all-MiniLM-L6-v2",
    "bge-base-en-v1.5",
    "jina-embeddings-v3",
    "jina-embeddings-v4",
    "Qwen3-Embedding-0.6B",
    "bge-m3",
]
COLBERT_ALIAS = "colbert-v2"

PER_MODEL_BATCH: dict[str, int] = {
    # Small models (<=110M params): batch is essentially memory-free, headroom
    # is limited only by HBM activations.
    "gte-small": 16384,
    "all-MiniLM-L6-v2": 16384,
    "bge-base-en-v1.5": 8192,
    # Medium (560M-600M, 1024-dim) at long-tail Wikipedia paragraphs (up to
    # 8K tokens for bge-m3). 2048 is the empirically OOM-safe ceiling on A100
    # 80GB; the OOM-fallback halves further if a particular shard happens to
    # exceed it.
    "bge-m3": 2048,
    "Qwen3-Embedding-0.6B": 1024,
    "jina-embeddings-v3": 2048,
    # jina-embeddings-v4 (3.8B params) attention is O(batch * heads * seq^2);
    # at batch 1024 / seq 512 a single attention matrix is ~32 GiB, which
    # OOMs even on an 80GB A100 once the model + grad-free activations are
    # loaded (~57 GB resident). 256 is the conservative starting point.
    "jina-embeddings-v4": 256,
}
DEFAULT_BATCH_SIZE = 1024

# Models whose internal modules (e.g. torch.nn.utils.parametrize for LoRA
# adapters) cannot be pickled into spawned worker processes — they raise
# `RuntimeError: Serialization of parametrized modules is only supported
# through state_dict()` at `start_multi_process_pool`. For these we fall
# back to single-GPU encoding on the first device of the requested pool.
MULTIPROCESS_INCOMPATIBLE: set[str] = {
    "jina-embeddings-v3",
    "jina-embeddings-v4",
}

WRITE_CHUNK_SIZE = 50_000

# Length-bucketed adaptive batching.
#
# Wikipedia paragraphs vary ~50x in length. Attention memory is O(seq^2), so
# a batch sized for the long-tail wastes ~10x throughput on short paragraphs.
# We sort the corpus by length, split into N shards, and pick a per-shard
# batch size that scales with 1/seq^2 relative to PER_MODEL_BATCH.
#
# REF_SEQ_LEN_FOR_BASE_BATCH is the seq length at which PER_MODEL_BATCH was
# tuned to be safe; CHARS_PER_TOKEN_ESTIMATE is a quick proxy that avoids
# tokenizing the corpus before length-bucketing (the encoder will tokenize
# anyway). MAX_BATCH_GROWTH caps the upscale factor so we don't blow past
# kernel saturation on truly tiny paragraphs.
N_LENGTH_SHARDS_TARGET = 8
REF_SEQ_LEN_FOR_BASE_BATCH = 512
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_BATCH_GROWTH = 8
MIN_DOCS_PER_WORKER_PER_SHARD = 25_000
MAX_MP_CHUNK_SIZE = 50_000

# Hard ceiling: cap the per-forward-pass batch so the attention matrix
# (batch * heads * seq^2 * 4 bytes for fp32) stays under this byte budget.
# 12 GiB leaves room for model weights (up to ~16 GiB for jina-v4),
# intermediate activations, and the caching allocator's headroom on a
# single 80 GiB A100. Tunable at runtime via --max-attn-gib.
ATTENTION_BYTES_BUDGET = 12 * (1024**3)
DEFAULT_HEADS_GUESS = 16


DEFAULT_LOG_DIR = Path("logs")
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging(log_file: Path | None) -> Path | None:
    """Wire root logging to stderr + an optional file. Returns the file path."""
    formatter = LOG_FORMATTER
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    actual_log: Path | None = log_file
    if actual_log is not None:
        actual_log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(actual_log, mode="a", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Replace any handlers basicConfig may have set so re-imports don't duplicate.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    return actual_log


def _default_log_path(stem: str = "index_dense") -> Path:
    return DEFAULT_LOG_DIR / f"{stem}_{RUN_TIMESTAMP}.log"


def _per_model_log_path(alias: str) -> Path:
    safe = alias.replace("/", "-").replace(" ", "_")
    return DEFAULT_LOG_DIR / f"model_{safe}_{RUN_TIMESTAMP}.log"


def _attach_per_model_handler(alias: str) -> tuple[logging.FileHandler, Path]:
    """Attach a FileHandler to the root logger that captures everything for
    `alias` until detached. Returns the handler + its path so the caller can
    close + remove it cleanly.
    """
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


def _resolve_model_id(alias: str) -> str:
    if alias in MODEL_IDS:
        return MODEL_IDS[alias]
    raise KeyError(f"Unknown model alias: {alias!r}. Add it to MODEL_IDS in src/retrieval/config.py.")


def _make_target_table_names(aliases: list[str], granularity: str, suffix: str) -> list[str]:
    return [get_classified_collection_name(alias, granularity) + suffix for alias in aliases]


def _build_pool_devices(gpu_ids: list[int]) -> list[str]:
    return [f"cuda:{i}" for i in gpu_ids]


def _is_parametrize_pickle_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "parametrized modules" in msg or "parametrize" in msg


def _encode_shard(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    pool: dict | None,
    chunk_size: int,
    extra_kwargs: dict,
) -> "torch.Tensor":
    """Encode one length-bucketed shard. `pool` is a started pool or None
    (meaning encode on the model's currently-bound device, single process).
    """
    if pool is None:
        kw = {
            "batch_size": batch_size,
            "show_progress_bar": True,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        kw.update(extra_kwargs)
        return model.encode(texts, **kw)

    kw = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "normalize_embeddings": True,
        "chunk_size": chunk_size,
    }
    kw.update(extra_kwargs)
    return model.encode_multi_process(texts, pool, **kw)


def _memory_capped_batch(seq_len: int, heads: int = DEFAULT_HEADS_GUESS) -> int:
    """Hard ceiling so attention bytes stay under ATTENTION_BYTES_BUDGET."""
    if ATTENTION_BYTES_BUDGET <= 0 or seq_len <= 0:
        return 10**9  # effectively no cap
    seq_len = max(1, int(seq_len))
    bytes_per_doc = heads * seq_len * seq_len * 4
    return max(1, ATTENTION_BYTES_BUDGET // bytes_per_doc)


def _adaptive_batch(base_batch: int, max_chars_in_shard: int) -> int:
    """Pick a per-forward-pass batch size based on the longest doc in a shard.

    Memory ≈ batch * heads * seq^2, so for a fixed memory budget batch must
    scale as 1/seq^2. We treat the configured PER_MODEL_BATCH as "safe at
    REF_SEQ_LEN_FOR_BASE_BATCH tokens" and rescale from there. We then floor
    the result by a hard memory cap (independent of model alias) so that an
    accidentally aggressive heuristic still can't OOM.
    """
    if max_chars_in_shard <= 0:
        return min(base_batch, _memory_capped_batch(1))
    seq_len = max(1.0, max_chars_in_shard / CHARS_PER_TOKEN_ESTIMATE)
    factor = (REF_SEQ_LEN_FOR_BASE_BATCH / seq_len) ** 2
    bs = max(1, int(base_batch * factor))
    bs = min(bs, base_batch * MAX_BATCH_GROWTH)
    bs = min(bs, _memory_capped_batch(int(seq_len)))
    return max(1, bs)


def _length_bucketed_shards(
    texts: list[str],
    n_workers: int,
    n_shards_target: int = N_LENGTH_SHARDS_TARGET,
) -> list[list[int]]:
    """Sort indices by char length and split into roughly-equal-count shards.

    Each shard contains docs of similar length, so within a shard the encoder
    sees uniform sequences and we can pick a fat batch for short shards and
    a small batch for long shards.

    Shard size is also bounded below so each worker gets enough work to
    amortize multi-process IPC: with N GPUs we want at least
    MIN_DOCS_PER_WORKER_PER_SHARD docs per worker per shard.
    """
    n = len(texts)
    if n == 0:
        return []
    workers = max(1, n_workers)
    min_shard_size = MIN_DOCS_PER_WORKER_PER_SHARD * workers
    n_shards = max(1, min(n_shards_target, n // min_shard_size))
    if n_shards == 0:
        n_shards = 1
    sorted_indices = sorted(range(n), key=lambda i: len(texts[i]))
    shard_size = (n + n_shards - 1) // n_shards
    return [sorted_indices[i : i + shard_size] for i in range(0, n, shard_size)]


_OOM_SIGNATURES = (
    "out of memory",
    "cuda error: out of memory",
    "cudaerror: out of memory",
    "cuda oom",
    "cudnn_status_alloc_failed",
)


def _looks_like_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(sig in msg for sig in _OOM_SIGNATURES):
        return True
    # encode_multi_process kills its workers on CUDA OOM; the parent then
    # surfaces a BrokenPipeError / ConnectionResetError / EOFError /
    # multiprocessing.queues.Empty / RuntimeError("worker died") rather
    # than the original OOM. Treat any of those as OOM-equivalent so we
    # can halve the batch and retry instead of bubbling out.
    cls_name = type(exc).__name__.lower()
    return any(
        sig in cls_name
        for sig in ("brokenpipe", "connectionreset", "eoferror", "brokenprocesspool")
    ) or "worker" in msg and ("died" in msg or "exit" in msg)


def _encode_shard_with_oom_fallback(
    model: SentenceTransformer,
    texts: list[str],
    initial_batch: int,
    pool: dict | None,
    chunk_size: int,
    extra_kwargs: dict,
):
    batch = initial_batch
    while batch >= 1:
        try:
            return _encode_shard(
                model=model,
                texts=texts,
                batch_size=batch,
                pool=pool,
                chunk_size=chunk_size,
                extra_kwargs=extra_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - explicit broad OOM detection
            if not _looks_like_oom(exc):
                raise
            logger.warning(
                "OOM (or worker died) at batch_size=%d (n=%d): %s. Halving.",
                batch,
                len(texts),
                exc,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch = batch // 2
    raise RuntimeError("Could not encode shard at any batch size")


def index_one_model(
    alias: str,
    model_id: str,
    corpus: ParagraphCorpus,
    db: lancedb.DBConnection,
    granularity: str,
    table_suffix: str,
    pool_devices: list[str],
    initial_batch: int,
    checkpoint: CheckpointStore,
    force_rerun: bool,
    n_shards_target: int = N_LENGTH_SHARDS_TARGET,
) -> None:
    """Encode `corpus` with `model_id` and write to LanceDB.

    Strategy:
      1. Sort the corpus by char length and split into N length-bucketed shards.
      2. For each shard, pick a per-forward-pass batch size that scales with
         1/seq^2 relative to PER_MODEL_BATCH (short shards → big batch, long
         shards → safe batch). OOM fallback halves on failure.
      3. Reuse one multi-process pool for all shards (one model load per
         worker) and pass `chunk_size = shard_size / num_workers` so workers
         do less IPC.
      4. Stream-write each shard's embeddings to LanceDB before encoding the
         next shard, so peak host RAM is one shard, not the full corpus.
    """
    table_name = get_classified_collection_name(alias, granularity) + table_suffix
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

    model: SentenceTransformer | None = None
    pool: dict | None = None
    tbl = None
    try:
        logger.info("[%s] Loading model %s onto %s (warmup)", alias, model_id, pool_devices[0])
        model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=pool_devices[0],
        )

        # GritLM DynamicCache fix (kept for parity with legacy script).
        if "GritLM" in model_id and hasattr(model[0], "auto_model"):
            try:
                model[0].auto_model.config.use_cache = False
            except (AttributeError, RuntimeError) as exc:
                logger.warning("[%s] Could not disable use_cache: %s", alias, exc)

        # Jina v3 and v4 disagree on the task API:
        #   v3: task="retrieval.passage" / "retrieval.query"
        #   v4: task="retrieval" (no .passage suffix), passage/query is selected
        #       via prompt_name from the loaded prompts dict (keys: query, passage)
        # Calling v4 with "retrieval.passage" raises ValueError: Invalid task.
        extra_kwargs: dict = {}
        if "jina-embeddings-v3" in model_id:
            logger.info("[%s] Setting task='retrieval.passage' (jina v3).", alias)
            extra_kwargs["task"] = "retrieval.passage"
        elif "jina-embeddings-v4" in model_id:
            logger.info("[%s] Setting task='retrieval', prompt_name='passage' (jina v4).", alias)
            extra_kwargs["task"] = "retrieval"
            extra_kwargs["prompt_name"] = "passage"
            # jina-v4's stock custom_st.py treats any text starting with
            # "http" as an image URL and tries to GET it. Sentences split
            # from Wikipedia paragraphs occasionally start with a bare URL,
            # which then fails as PIL.UnidentifiedImageError. Force the
            # tokenizer to always process inputs as text.
            try:
                _module = model[0]
                _processor = _module.processor
                _max_seq_len = getattr(_module, "max_seq_length", 8192) or 8192

                def _tokenize_text_only(_self, texts, padding=True, **_kwargs):
                    text_list = list(texts)
                    text_indices = list(range(len(text_list)))
                    feats = _processor.process_texts(text_list, max_length=_max_seq_len)
                    encoding = {f"text_{k}": v for k, v in feats.items()}
                    encoding["text_indices"] = text_indices
                    return encoding

                import types as _types

                _module.tokenize = _types.MethodType(_tokenize_text_only, _module)
                logger.info("[%s] Patched tokenize() to skip URL-as-image detection.", alias)
            except (AttributeError, RuntimeError) as exc:
                logger.warning(
                    "[%s] Could not patch jina-v4 tokenize (%s); URL-shaped texts may crash.",
                    alias, exc,
                )

        effective_pool = pool_devices
        if alias in MULTIPROCESS_INCOMPATIBLE and len(pool_devices) > 1:
            logger.info(
                "[%s] Pinning to single GPU %s (model is in MULTIPROCESS_INCOMPATIBLE).",
                alias,
                pool_devices[0],
            )
            effective_pool = pool_devices[:1]

        if len(effective_pool) > 1:
            try:
                pool = model.start_multi_process_pool(target_devices=effective_pool)
                logger.info("[%s] Multi-process pool started on %s", alias, effective_pool)
            except RuntimeError as exc:
                if not _is_parametrize_pickle_error(exc):
                    raise
                logger.warning(
                    "[%s] Pool failed (parametrize-pickle): %s. Falling back to single-GPU on %s.",
                    alias,
                    exc,
                    effective_pool[0],
                )
                pool = None
                effective_pool = effective_pool[:1]

        n_workers = len(effective_pool)
        shards = _length_bucketed_shards(
            corpus.texts, n_workers=n_workers, n_shards_target=n_shards_target
        )
        logger.info(
            "[%s] U=%d split into %d length-bucketed shards on %d worker(s).",
            alias,
            U,
            len(shards),
            n_workers,
        )

        rows_written = 0
        for shard_idx, shard_indices in enumerate(shards, start=1):
            shard_texts = [corpus.texts[i] for i in shard_indices]
            max_chars = max(len(t) for t in shard_texts)
            shard_batch = _adaptive_batch(initial_batch, max_chars)
            mp_chunk_size = max(1, len(shard_indices) // max(n_workers, 1))
            mp_chunk_size = min(mp_chunk_size, MAX_MP_CHUNK_SIZE)
            logger.info(
                "[%s] shard %d/%d size=%d max_chars=%d -> batch=%d mp_chunk=%d",
                alias,
                shard_idx,
                len(shards),
                len(shard_indices),
                max_chars,
                shard_batch,
                mp_chunk_size,
            )

            embeddings = _encode_shard_with_oom_fallback(
                model=model,
                texts=shard_texts,
                initial_batch=shard_batch,
                pool=pool,
                chunk_size=mp_chunk_size,
                extra_kwargs=extra_kwargs,
            )

            # Stream rows to LanceDB in WRITE_CHUNK_SIZE-row sub-batches so
            # peak per-shard memory is bounded by the embedding array itself
            # rather than embeddings + duplicated row dicts.
            shard_n = len(shard_indices)
            for sub_start in range(0, shard_n, WRITE_CHUNK_SIZE):
                sub_stop = min(sub_start + WRITE_CHUNK_SIZE, shard_n)
                rows = [
                    {
                        "id": corpus.paragraph_ids[shard_indices[k]],
                        "vector": embeddings[k],
                        "text": corpus.texts[shard_indices[k]],
                        "metadata": corpus.metadata_jsons[shard_indices[k]],
                    }
                    for k in range(sub_start, sub_stop)
                ]
                if tbl is None:
                    tbl = db.create_table(table_name, data=rows, mode="overwrite")
                else:
                    tbl.add(rows)
                rows_written += len(rows)
                del rows
            del embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if tbl is None:
            raise RuntimeError(f"[{alias}] No shards produced; corpus was empty?")

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
    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        checkpoint.mark_failed(table_name, error=repr(exc), U=U)
        raise
    finally:
        if pool is not None and model is not None:
            try:
                model.stop_multi_process_pool(pool)
            except (RuntimeError, OSError) as exc:
                logger.warning("[%s] stop_multi_process_pool: %s", alias, exc)
        try:
            del model  # type: ignore[name-defined]
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _detach_per_model_handler(per_model_handler)


def main() -> None:
    global ATTENTION_BYTES_BUDGET  # noqa: PLW0603 - tunable via CLI flag below

    parser = argparse.ArgumentParser(description="Index minimax paragraphs to LanceDB (dense models).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_DENSE_MODELS,
        help="Aliases (from MODEL_IDS) of dense models to index.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Post-CUDA_VISIBLE_DEVICES GPU indices to use as a multi-process pool.",
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
        help=(
            "Override the default source directory for the chosen granularity. "
            "Useful for ad-hoc reruns against a custom corpus."
        ),
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Smoke-test knob: only read first N JSON files.",
    )
    parser.add_argument(
        "--table-suffix",
        type=str,
        default="",
        help="Suffix appended to the LanceDB table name (use '_smoke' for tests).",
    )
    parser.add_argument(
        "--reset-tables",
        action="store_true",
        help="Drop the target tables for all listed models before indexing.",
    )
    parser.add_argument(
        "--force-rerun",
        nargs="*",
        default=[],
        help="Aliases to forcibly reindex even if checkpoint says done.",
    )
    parser.add_argument(
        "--initial-batch",
        type=int,
        default=None,
        help="Override per-model initial batch size.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip corpus pickle cache.",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=N_LENGTH_SHARDS_TARGET,
        help=(
            "Target number of length-bucketed shards. More shards = finer "
            "adaptive batching (better speed) but more dispatch overhead. "
            "Default %(default)d."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Path to write logs to (in addition to stderr). Defaults to "
            "logs/index_dense_<timestamp>.log. Pass '-' to disable file "
            "logging."
        ),
    )
    parser.add_argument(
        "--max-attn-gib",
        type=float,
        default=ATTENTION_BYTES_BUDGET / (1024**3),
        help=(
            "Hard ceiling on per-forward-pass attention memory (GiB). "
            "Used as a backstop to the adaptive batch heuristic. Default "
            "%(default).1f GiB. Lower this if you keep OOMing; raise it "
            "for max throughput on small models with lots of free HBM."
        ),
    )
    # parse_known_args (instead of parse_args) so the runner can forward a
    # single set of flags to both index_dense.py and index_colbert.py without
    # crashing on the other side's flags. We log unknowns to stderr.
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"index_dense: ignoring unknown args (likely colbert-only): {_unknown}", file=sys.stderr)

    ATTENTION_BYTES_BUDGET = int(args.max_attn_gib * (1024**3))

    if args.log_file is None:
        log_file = _default_log_path("index_dense")
    elif str(args.log_file) == "-":
        log_file = None
    else:
        log_file = args.log_file
    actual_log = _setup_logging(log_file)
    if actual_log is not None:
        # First message both to stderr and to file: confirms the path.
        logger.info("Log file: %s", actual_log.resolve())

    if COLBERT_ALIAS in args.models:
        logger.error(
            "%s is multi-vector and is handled by index_colbert.py — drop it from --models.",
            COLBERT_ALIAS,
        )
        sys.exit(2)

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
    logger.info("Loaded corpus: U=%d unique %s rows.", len(corpus), args.granularity)

    # Per-granularity LanceDB root. Sentence/proposition tables are large
    # (35-200 GB); they live on local-disk scratch via LANCEDB_SCRATCH_PATH.
    # Paragraph stays at LANCEDB_PATH alongside the checkpoint store.
    lancedb_root = lancedb_path_for(args.granularity)
    lancedb_root.mkdir(parents=True, exist_ok=True)
    logger.info("LanceDB root for granularity=%s: %s", args.granularity, lancedb_root)
    db = lancedb.connect(lancedb_root)

    # Checkpoint always lives under LANCEDB_PATH (single source of truth
    # across granularities; otherwise the resume logic would have to scan
    # multiple roots).
    checkpoint = CheckpointStore(default_state_dir(LANCEDB_PATH))

    if args.reset_tables:
        target_table_names = _make_target_table_names(args.models, args.granularity, args.table_suffix)
        dropped = drop_target_tables(db, target_table_names)
        logger.info("Reset complete; dropped %d tables: %s", len(dropped), dropped)

    pool_devices = _build_pool_devices(args.gpus)
    logger.info("Pool devices: %s", pool_devices)

    force_set = set(args.force_rerun)
    for alias in args.models:
        model_id = _resolve_model_id(alias)
        initial_batch = (
            args.initial_batch
            if args.initial_batch is not None
            else PER_MODEL_BATCH.get(alias, DEFAULT_BATCH_SIZE)
        )
        index_one_model(
            alias=alias,
            model_id=model_id,
            corpus=corpus,
            db=db,
            granularity=args.granularity,
            table_suffix=args.table_suffix,
            pool_devices=pool_devices,
            initial_batch=initial_batch,
            checkpoint=checkpoint,
            force_rerun=alias in force_set,
            n_shards_target=args.n_shards,
        )


if __name__ == "__main__":
    main()
