"""Spec-driven dense (single-vector) encode loop for frontier-v2.

This is a thin adaptation of ``minimax/index_dense.py``'s ``index_one_model``.
It reuses minimax's proven, stateless machinery verbatim (length-bucketed
shards, the 1/seq^2 adaptive batch heuristic, the OOM-halving fallback, the
per-model log handler) and only changes the two things the original hardcodes:

  1. Model construction is driven by a :class:`model_registry.ModelSpec`
     (dtype / model_kwargs / config_kwargs / tokenizer_kwargs / trust_remote_code),
     so 8-14B models load in bf16, last-token embedders get left padding, the
     vision-language model loads text-only, etc.
  2. Document-side encode knobs come from the spec: an optional ``doc_prompt``
     prepended to every passage, and ``encode_kwargs`` (e.g. jina v5
     ``task="retrieval", prompt_name="passage"``).

Everything else — table schema (``id``/``vector``/``text``/``metadata``),
streaming writes, checkpoint marking — matches the minimax dense indexer so the
downstream evaluator reads these tables identically.
"""

from __future__ import annotations

import gc
import logging
import time

import lancedb
import torch
from sentence_transformers import SentenceTransformer

import _bootstrap
from checkpoint import CheckpointStore, _db_tables
from config import get_classified_collection_name
from corpus_loader import ParagraphCorpus
from model_registry import ModelSpec

# Reuse the stateless dense machinery from the minimax indexer (loaded by
# explicit path to avoid the index_dense.py name collision).
_mmx = _bootstrap.load_minimax("index_dense.py", "mmx_index_dense")

logger = logging.getLogger(__name__)

# Map spec dtype strings to real torch.dtype objects. A torch.dtype object is
# unambiguously accepted by both transformers <5 (torch_dtype=) and 5.x; a bare
# string relies on an internal convenience that isn't guaranteed across custom
# from_pretrained overrides. Unknown strings pass through unchanged.
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _build_model(model_id: str, spec: ModelSpec, device: str) -> SentenceTransformer:
    """Construct a SentenceTransformer per the model spec."""
    model_kwargs = dict(spec.model_kwargs)
    if spec.dtype:
        model_kwargs.setdefault("torch_dtype", _DTYPE_MAP.get(spec.dtype, spec.dtype))

    st_kwargs: dict = {"trust_remote_code": spec.trust_remote_code, "device": device}
    if model_kwargs:
        st_kwargs["model_kwargs"] = model_kwargs
    if spec.config_kwargs:
        st_kwargs["config_kwargs"] = dict(spec.config_kwargs)
    if spec.tokenizer_kwargs:
        st_kwargs["tokenizer_kwargs"] = dict(spec.tokenizer_kwargs)

    logger.info("[%s] Loading %s on %s (kwargs=%s)", spec.alias, model_id, device,
                {k: v for k, v in st_kwargs.items() if k != "device"})
    return SentenceTransformer(model_id, **st_kwargs)


def index_one_model_v2(
    alias: str,
    model_id: str,
    spec: ModelSpec,
    corpus: ParagraphCorpus,
    db: lancedb.DBConnection,
    table_suffix: str,
    pool_devices: list[str],
    initial_batch: int,
    checkpoint: CheckpointStore,
    force_rerun: bool,
    n_shards_target: int = _mmx.N_LENGTH_SHARDS_TARGET,
) -> None:
    """Encode ``corpus`` with ``model_id`` per ``spec`` and write to LanceDB.

    Mirrors ``minimax.index_one_model`` (granularity fixed to ``paragraph``).
    """
    table_name = get_classified_collection_name(alias, "paragraph") + table_suffix
    U = len(corpus)

    if not force_rerun and checkpoint.is_done(table_name, U, db, table_name):
        logger.info("[%s] Already done (rows=%d). Skipping.", alias, U)
        return

    per_model_handler, per_model_log_path = _mmx._attach_per_model_handler(alias)
    logger.info("[%s] Per-model log: %s", alias, per_model_log_path.resolve())

    checkpoint.mark_running(table_name, U)
    started = time.time()

    model: SentenceTransformer | None = None
    pool: dict | None = None
    tbl = None
    try:
        # Inside the try so a drop failure is recorded as mark_failed rather
        # than leaving the checkpoint stuck in "running".
        if table_name in _db_tables(db):
            logger.info("[%s] Dropping existing table %s before reindex.", alias, table_name)
            db.drop_table(table_name)

        model = _build_model(model_id, spec, pool_devices[0])

        # Document-side encode knobs. If the spec doesn't pin a prompt/task,
        # auto-select the model's own document-side prompt: many ST embedders
        # (Qwen3-Embedding and its derivatives — Octen/F2LLM/QZhou/MoD/ICT)
        # ship a 'document'/'passage' prompt, and using it matches the model's
        # intended asymmetric encoding rather than feeding bare text. When that
        # prompt is empty it is a harmless no-op; when non-empty it is the
        # correct document instruction. The slots run one GPU per model
        # (pool=None -> model.encode), which accepts prompt_name directly.
        extra_kwargs: dict = dict(spec.encode_kwargs)
        if not ({"prompt_name", "prompt", "task"} & extra_kwargs.keys()):
            prompts = getattr(model, "prompts", None) or {}
            for _cand in ("document", "passage", "doc"):
                if _cand in prompts:
                    extra_kwargs["prompt_name"] = _cand
                    logger.info("[%s] auto-selected document prompt_name=%r", alias, _cand)
                    break
        doc_prompt = spec.doc_prompt or ""

        # Single-GPU pin for models whose internals can't be pickled into
        # spawned workers (LoRA / parametrize): mirrors minimax behaviour but
        # driven by the spec flag plus the legacy alias set as a backstop.
        effective_pool = pool_devices
        force_single = (not spec.multiprocess_ok) or (alias in _mmx.MULTIPROCESS_INCOMPATIBLE)
        if force_single and len(pool_devices) > 1:
            logger.info("[%s] Pinning to single GPU %s (multiprocess disabled).",
                        alias, pool_devices[0])
            effective_pool = pool_devices[:1]

        if len(effective_pool) > 1:
            try:
                pool = model.start_multi_process_pool(target_devices=effective_pool)
                logger.info("[%s] Multi-process pool started on %s", alias, effective_pool)
            except RuntimeError as exc:
                if not _mmx._is_parametrize_pickle_error(exc):
                    raise
                logger.warning("[%s] Pool failed (parametrize-pickle): %s. Single-GPU on %s.",
                               alias, exc, effective_pool[0])
                pool = None
                effective_pool = effective_pool[:1]

        n_workers = len(effective_pool)
        shards = _mmx._length_bucketed_shards(
            corpus.texts, n_workers=n_workers, n_shards_target=n_shards_target
        )
        logger.info("[%s] U=%d split into %d length-bucketed shards on %d worker(s).",
                    alias, U, len(shards), n_workers)

        rows_written = 0
        for shard_idx, shard_indices in enumerate(shards, start=1):
            shard_texts = [corpus.texts[i] for i in shard_indices]
            if doc_prompt:
                shard_texts = [doc_prompt + t for t in shard_texts]
            max_chars = max(len(t) for t in shard_texts)
            shard_batch = _mmx._adaptive_batch(initial_batch, max_chars)
            mp_chunk_size = max(1, len(shard_indices) // max(n_workers, 1))
            mp_chunk_size = min(mp_chunk_size, _mmx.MAX_MP_CHUNK_SIZE)
            logger.info("[%s] shard %d/%d size=%d max_chars=%d -> batch=%d mp_chunk=%d",
                        alias, shard_idx, len(shards), len(shard_indices), max_chars,
                        shard_batch, mp_chunk_size)

            embeddings = _mmx._encode_shard_with_oom_fallback(
                model=model,
                texts=shard_texts,
                initial_batch=shard_batch,
                pool=pool,
                chunk_size=mp_chunk_size,
                extra_kwargs=extra_kwargs,
            )

            shard_n = len(shard_indices)
            for sub_start in range(0, shard_n, _mmx.WRITE_CHUNK_SIZE):
                sub_stop = min(sub_start + _mmx.WRITE_CHUNK_SIZE, shard_n)
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
        logger.info("[%s] Done: %d rows in %s (%.1f min).",
                    alias, rows_written, table_name, (time.time() - started) / 60.0)
    except Exception as exc:  # noqa: BLE001 - record any failure, re-raise for caller isolation
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
        _mmx._detach_per_model_handler(per_model_handler)
