"""On-disk layout for cached reranker scores. Single source of truth.

Both the scorer (``score_pool.py``, run on the cluster) and the evaluator
(``frontier-v2/evaluation/data_loading.py``, run anywhere) import this module, so the
path convention and the parquet schema are defined exactly once.

Pointwise (the union-pool cache)::

    data/frontier_rerank_scores/<reranker>/<variant>/<qid>.parquet
        id:    string   -- paragraph id from all_candidates_<qid>.json
        score: float32  -- higher is more relevant; scale is model-specific

Listwise (per-retriever permutations; no reusable score)::

    data/frontier_rerank_scores/<reranker>/listwise/<retriever>/<qid>.json
        {"event_id", "reranker", "retriever", "depth", "order": [id, ...],
         "scores": [float, ...] | null}

``float32`` is lossless here: every backend casts its logits to fp32 before
extraction, so the stored value is exactly the value that was computed. Ties are
therefore genuine ties, and the evaluator's stable sort breaks them by original
retriever rank rather than arbitrarily.

Writes are atomic (tmp file + ``replace``) because the sweep is resumable by
skipping existing files: a half-written parquet from a killed job would otherwise be
silently accepted on the next run.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import _rerank_bootstrap  # noqa: F401  (sets sys.path so `config` imports)
from config import FRONTIER_RERANK_SCORES_PATH

LISTWISE_DIRNAME = "listwise"


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
# The scorer runs one process per GPU. Per-query outputs never collide (queries are
# sharded by event_id % num_shards), but the run manifest is a single file that every
# shard writes. A fixed `<name>.tmp` staging path therefore races: shard A creates it,
# shard B truncates it, A renames it away, and B's rename fails with FileNotFoundError.
#
# `mkstemp` gives each writer its own staging file in the destination directory, so
# `os.replace` is atomic *and* collision-free. Concurrent writers all succeed and the
# last one wins — which is correct here, because they write identical content.
def _staging_path(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return tmp


def _atomic_write_text(path: Path, payload: str) -> None:
    tmp = _staging_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def qid_str(event_id: int) -> str:
    """Zero-padded 4-digit query id, e.g. 36 -> '0036'. Matches frontier-v2."""
    return str(int(event_id)).zfill(4)


# --------------------------------------------------------------------------- #
# Pointwise score cache
# --------------------------------------------------------------------------- #
def scores_dir(reranker: str, variant: str) -> Path:
    return FRONTIER_RERANK_SCORES_PATH / reranker / variant


def scores_path(reranker: str, variant: str, event_id: int) -> Path:
    return scores_dir(reranker, variant) / f"{qid_str(event_id)}.parquet"


def write_scores(
    reranker: str, variant: str, event_id: int, ids: Sequence[str], scores: Sequence[float]
) -> Path:
    """Atomically persist one query's pooled scores."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    if len(ids) != len(scores):
        raise ValueError(
            f"[{reranker}/{variant}/{qid_str(event_id)}] {len(ids)} ids vs {len(scores)} scores."
        )
    path = scores_path(reranker, variant, event_id)
    table = pa.table(
        {
            "id": pa.array(list(ids), type=pa.string()),
            "score": pa.array(list(scores), type=pa.float32()),
        }
    )
    tmp = _staging_path(path)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def read_scores(reranker: str, variant: str, event_id: int) -> dict[str, float]:
    """``{paragraph_id: score}`` for one query. Raises if the file is absent."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    path = scores_path(reranker, variant, event_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached scores at {path}. Run score_pool.py for "
            f"reranker={reranker!r} variant={variant!r} first."
        )
    table = pq.read_table(path)
    return dict(zip(table.column("id").to_pylist(), table.column("score").to_pylist()))


def scores_exist(reranker: str, variant: str, event_id: int) -> bool:
    return scores_path(reranker, variant, event_id).exists()


# --------------------------------------------------------------------------- #
# Run manifest: guards resume against a silently different scoring configuration
# --------------------------------------------------------------------------- #
MANIFEST_NAME = "_manifest.json"
# Fields that change the numbers. Resuming a cache with any of these different would
# interleave, say, CPU/fp32 scores with GPU/bf16 ones inside a single query set — the
# ranking would be subtly wrong and nothing would say so.
#
# `device_kind` matters because resolve_dtype downgrades half precision to fp32 on CPU,
# so (dtype, device_kind) — not dtype alone — determines the arithmetic that ran.
# `tf32` matters because TF32 matmul rounds fp32 inputs to a 10-bit mantissa: the outputs
# stay fp32 but are not bit-identical to strict fp32, so the two must never be mixed.
MANIFEST_KEYS = ("reranker", "variant", "dtype", "device_kind", "tf32", "max_length",
                 "padding_side")


def manifest_path(reranker: str, variant: str) -> Path:
    return scores_dir(reranker, variant) / MANIFEST_NAME


def read_manifest(reranker: str, variant: str) -> dict | None:
    path = manifest_path(reranker, variant)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_or_write_manifest(reranker: str, variant: str, config: dict) -> None:
    """Record the scoring configuration, or fail if it contradicts an existing cache.

    Resume is per-query and silent by design; this is what stops it from being silent
    about the thing that matters. Delete the cache directory to rescore from scratch.

    Every shard calls this concurrently at startup, and they all write the same content.
    Two shards can both observe "no manifest" and both write — that is fine, because the
    write is atomic and the payloads are identical. What is *not* fine is sharing a staging
    path; see ``_staging_path``.
    """
    existing = read_manifest(reranker, variant)
    current = {k: config.get(k) for k in MANIFEST_KEYS}
    if existing is not None:
        prior = {k: existing.get(k) for k in MANIFEST_KEYS}
        if prior != current:
            diff = {k: (prior[k], current[k]) for k in MANIFEST_KEYS if prior[k] != current[k]}
            raise ValueError(
                f"[{reranker}/{variant}] existing score cache was produced with a different "
                f"configuration: {diff} (prior -> current). Resuming would mix incomparable "
                f"scores inside one query set. Delete {scores_dir(reranker, variant)} to rescore."
            )
        return
    _atomic_write_text(
        manifest_path(reranker, variant), json.dumps(current, indent=2, sort_keys=True)
    )


# --------------------------------------------------------------------------- #
# Listwise permutations
# --------------------------------------------------------------------------- #
def listwise_path(reranker: str, retriever: str, event_id: int) -> Path:
    return (
        FRONTIER_RERANK_SCORES_PATH
        / reranker
        / LISTWISE_DIRNAME
        / retriever
        / f"{qid_str(event_id)}.json"
    )


def write_listwise(
    reranker: str,
    retriever: str,
    event_id: int,
    depth: int,
    order: Sequence[str],
    scores: Sequence[float] | None = None,
) -> Path:
    """Atomically persist one (reranker, retriever, query) permutation.

    ``order`` is the reranked prefix — a permutation of the retriever's top-``depth``
    ids. ``scores`` is present only for models that emit one (jina-v3); rank_zephyr
    generates a permutation string and has no score to record.
    """
    path = listwise_path(reranker, retriever, event_id)
    doc = {
        "event_id": int(event_id),
        "reranker": reranker,
        "retriever": retriever,
        "depth": int(depth),
        "order": list(order),
        "scores": list(scores) if scores is not None else None,
    }
    _atomic_write_text(path, json.dumps(doc, ensure_ascii=False))
    return path


def read_listwise(reranker: str, retriever: str, event_id: int) -> Mapping:
    path = listwise_path(reranker, retriever, event_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No listwise permutation at {path}. Run rerank_listwise.py for "
            f"reranker={reranker!r} retriever={retriever!r} first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def listwise_exists(reranker: str, retriever: str, event_id: int) -> bool:
    return listwise_path(reranker, retriever, event_id).exists()
