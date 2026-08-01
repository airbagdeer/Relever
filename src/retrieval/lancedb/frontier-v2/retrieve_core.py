"""Shared machinery for frontier-v2 candidate retrieval.

Retrieval is the query-side mirror of the frontier-v2 indexers: open each
model's existing LanceDB paragraph table (at the root the indexer wrote it to,
via ``model_registry.root_for``) and pull the top-K candidates per v2 query.

The heavy compute is reused verbatim from the proven minimax evaluator
(``evaluate_minimax``): for the two multi-vector models we reuse its brute-force
GPU MaxSim (``_maxsim``) + Arrow token-vector padding (``_pad_arrow_token_vectors``
/ ``_pad_token_vectors``) and the streaming-scan + GPU top-k-heap pattern. This
module only adds the candidate-dump I/O the eval doesn't need.

The minimax evaluator is loaded *lazily* (only inside the MaxSim path), so the
dense + vLLM backends — which import this module from venvs that may lack the
evaluator's full dependency chain — pay nothing for it.

Output (per the task spec)::

    data/candidates/<qid>/<alias>_<qid>.json

where ``qid = str(event_id).zfill(4)``. Each per-model file is compact
(``id`` + ``score`` + ``rank``); full paragraph text is joined once, afterwards,
by ``aggregate_candidates.py`` into ``all_candidates_<qid>.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import _bootstrap  # noqa: F401  (sets sys.path so config / minimax modules import)
from config import get_classified_collection_name
from model_registry import root_for
from paths import FRONTIER_CANDIDATES_DIR, V2_RELATIONS_PATH

logger = logging.getLogger(__name__)

# Per-model retrieval depth: each model's <alias>_<qid>.json keeps its top-DEFAULT_K
# candidates (rank 1..DEFAULT_K). evaluation/data_loading.py reads these deep ranked
# lists for the metric suite.
DEFAULT_K = 3000

# Shallower per-model cap applied ONLY when aggregate_candidates.py builds the
# deduped all_candidates_<qid>.json union: each model contributes just its top
# AGG_PER_MODEL_K (by rank), so the cross-model labeling pool stays bounded even
# though the per-model files store DEFAULT_K.
AGG_PER_MODEL_K = 500


# --------------------------------- logging ---------------------------------


def setup_logging(log_file):
    """Minimal root-logger setup (stderr + optional file). Dependency-light so
    every backend venv can call it without importing the dense/eval stack."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
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


# --------------------------------- queries ---------------------------------


def load_v2_queries(limit: int | None = None) -> tuple[list[int], list[str]]:
    """Return ``(event_ids, sentences)`` for the v2 query set, in file order.

    The query text is each object's ``sentence`` field; the id is ``event_id``
    (an int, 0..1049). ``limit`` keeps only the first N queries (smoke knob).
    """
    with open(V2_RELATIONS_PATH, "r", encoding="utf-8") as f:
        relations = json.load(f)
    if limit is not None:
        relations = relations[:limit]
    event_ids = [int(r["event_id"]) for r in relations]
    sentences = [str(r["sentence"]) for r in relations]
    return event_ids, sentences


# --------------------------------- paths -----------------------------------


def qid_str(event_id: int) -> str:
    """Zero-padded 4-digit query id, e.g. 36 -> '0036'."""
    return str(int(event_id)).zfill(4)


def candidates_dir_for(event_id: int) -> Path:
    return FRONTIER_CANDIDATES_DIR / qid_str(event_id)


def model_candidate_path(alias: str, event_id: int) -> Path:
    """e.g. data/candidates/0036/Octen-Embedding-8B_0036.json"""
    return candidates_dir_for(event_id) / f"{alias}_{qid_str(event_id)}.json"


def model_already_done(alias: str, event_ids: Sequence[int]) -> bool:
    """True iff every per-(model, query) file already exists (model-level resume)."""
    return all(model_candidate_path(alias, eid).exists() for eid in event_ids)


# --------------------------------- output ----------------------------------


def write_model_candidates(
    alias: str,
    event_id: int,
    query: str,
    candidates: list[dict],
    k: int,
) -> None:
    """Write one compact per-(model, query) JSON file (atomic via temp + rename).

    ``candidates`` is an ordered list of ``{"id": str, "rank": int, "score": float}``
    (rank 1 = best; score is higher-is-better).
    """
    out_path = model_candidate_path(alias, event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "event_id": int(event_id),
        "query": query,
        "model": alias,
        "k": int(k),
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    tmp.replace(out_path)


# ------------------------------ table access -------------------------------


def open_paragraph_table(alias: str, lancedb_root_mode: str = "auto", table_suffix: str = ""):
    """Open the model's paragraph table at the root the indexer wrote it to.

    Returns ``(table, table_name, root)``. Raises ``FileNotFoundError`` /
    ``ValueError`` if the table is missing (caller isolates per-model).
    """
    import lancedb

    root = root_for(alias, lancedb_root_mode)
    db = lancedb.connect(root)
    table_name = get_classified_collection_name(alias, "paragraph") + table_suffix
    table = db.open_table(table_name)
    return table, table_name, root


# ------------------------------- dense path --------------------------------


def search_and_write_dense(
    *,
    alias: str,
    table,
    query_embeddings: Any,        # indexable by [i] -> 1-D vector (np.ndarray or list)
    event_ids: list[int],
    queries: list[str],
    k: int = DEFAULT_K,
) -> None:
    """Per-query LanceDB ANN search -> compact top-k candidate files.

    Mirrors ``evaluate_minimax._evaluate_dense``'s search step: LanceDB returns
    rows distance-ascending (best first); we keep ``score = -distance`` (higher =
    better) and rank by arrival order, deduped on ``id`` defensively.
    """
    from tqdm import tqdm

    for i in tqdm(range(len(event_ids)), desc=f"{alias} dense-search"):
        eid = event_ids[i]
        query = queries[i]
        try:
            # Select only id + _distance: the per-model file stores id+score+rank,
            # so we avoid transferring the vector/text/metadata for k*Q rows.
            # Naming _distance explicitly also silences lancedb's autoprojection
            # deprecation warning (which would otherwise log once per query).
            raw = table.search(query_embeddings[i]).select(["id", "_distance"]).limit(k).to_list()
        except (RuntimeError, ValueError, KeyError) as exc:
            logger.error("[%s] search failed for event %s: %s", alias, eid, exc)
            raw = []

        seen: set[str] = set()
        candidates: list[dict] = []
        for row in raw:
            cid = row.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            distance = float(row.get("_distance", 0.0))
            candidates.append(
                {"id": cid, "rank": len(candidates) + 1, "score": float(-distance)}
            )
            if len(candidates) >= k:
                break
        write_model_candidates(alias, eid, query, candidates, k)


def search_and_write_dense_gpu(
    *,
    alias: str,
    table,
    query_embeddings: Any,        # (Q, dim) array/tensor of query vectors
    event_ids: list[int],
    queries: list[str],
    device,
    k: int = DEFAULT_K,
    chunk: int = 50_000,
) -> None:
    """Exact GPU top-k dense search — one streaming pass over the ``vector`` column.

    Per-query LanceDB flat search re-scans the whole column once *per query*, which
    is minutes-to-hours for 1050 queries over ~1.04M high-dim vectors. Instead we
    stream the column once, move each batch to the GPU, score ALL queries at once
    with a matmul (cosine; both sides L2-normalized), and keep a per-query GPU
    top-k heap — the exact same pattern as ``maxsim_retrieve_and_write``, just
    single-vector. Results are exact (full dot products), typically ~1-2 min/model.

    The caller should free its encoder (ST model / vLLM engine) BEFORE calling this
    so the streamed batches have GPU headroom.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    q = torch.as_tensor(np.asarray(query_embeddings, dtype=np.float32), device=device)
    if q.ndim != 2:
        q = q.reshape(len(event_ids), -1)
    q = F.normalize(q, dim=1)
    Q = int(q.shape[0])

    heap_capacity = k
    best_scores = torch.full((Q, heap_capacity), float("-inf"), dtype=torch.float32, device=device)
    best_idx = torch.full((Q, heap_capacity), -1, dtype=torch.int64, device=device)
    cpu_ids: list[str] = []

    lance_ds = table.to_lance()
    scanner = lance_ds.scanner(columns=["id", "vector"], batch_size=chunk)

    n_seen = 0
    for arrow_batch in tqdm(scanner.to_batches(), desc=f"{alias} gpu-search"):
        ids = arrow_batch.column("id").to_pylist()
        M = len(ids)
        if M == 0:
            continue
        # FixedSizeList<float32, dim> -> (M, dim) via the flat child buffer.
        flat = np.asarray(arrow_batch.column("vector").values, dtype=np.float32)
        d = torch.from_numpy(flat.reshape(M, -1)).to(device, non_blocking=True)
        d = F.normalize(d, dim=1)
        with torch.no_grad():
            scores = q @ d.T                                              # (Q, M) cosine
            batch_idx = (
                torch.arange(n_seen, n_seen + M, dtype=torch.int64, device=device)
                .unsqueeze(0)
                .expand(Q, -1)
            )
            combined_scores = torch.cat([best_scores, scores], dim=1)
            combined_idx = torch.cat([best_idx, batch_idx], dim=1)
            kk = min(heap_capacity, combined_scores.shape[1])
            top_vals, top_pos = combined_scores.topk(kk, dim=1)
            best_scores = top_vals
            best_idx = torch.gather(combined_idx, dim=1, index=top_pos)
        cpu_ids.extend(ids)
        n_seen += M
        del d, scores, combined_scores, combined_idx, batch_idx

    logger.info("[%s] scanned %d rows (gpu dense); writing top-%d for %d queries.",
                alias, n_seen, k, Q)
    best_scores_cpu = best_scores.detach().cpu().numpy()
    best_idx_cpu = best_idx.detach().cpu().numpy()
    for qi in range(Q):
        scores_qi = best_scores_cpu[qi]
        idx_qi = best_idx_cpu[qi]
        order = np.argsort(-scores_qi)
        seen: set[str] = set()
        candidates: list[dict] = []
        for j in order:
            s = float(scores_qi[j])
            row_int = int(idx_qi[j])
            if not np.isfinite(s) or row_int < 0:
                continue
            cid = cpu_ids[row_int]
            if not cid or cid in seen:
                continue
            seen.add(cid)
            candidates.append({"id": cid, "rank": len(candidates) + 1, "score": s})
            if len(candidates) >= k:
                break
        write_model_candidates(alias, event_ids[qi], queries[qi], candidates, k)


# ---------------------------- multi-vector path ----------------------------


def pad_query_token_vectors(token_lists, dim: int, device):
    """Pad a list of ``(L_i, dim)`` query token-vector arrays to ``(Q, L_max, dim)``.

    Thin wrapper over the evaluator's ``_pad_token_vectors`` (lazy-loaded), so the
    multi-vector retrievers build query tensors with the exact same code the eval
    used. ``load_minimax`` caches the module, so this is a cheap call.
    """
    _eval = _bootstrap.load_minimax("evaluate_minimax.py", "mmx_evaluate")
    return _eval._pad_token_vectors(token_lists, dim=dim, device=device)


def maxsim_retrieve_and_write(
    *,
    alias: str,
    table,
    q_docs,            # (Q, Lq, H) float32 tensor on `device`
    q_mask,            # (Q, Lq) bool tensor on `device`
    q_dim: int,
    device,
    event_ids: list[int],
    queries: list[str],
    k: int = DEFAULT_K,
    chunk: int = 2048,
    q_batch_size: int = 64,
    max_scan_batches: int | None = None,
) -> None:
    """Brute-force GPU MaxSim over the multi-vector table -> top-k candidate files.

    Streams the corpus in Arrow batches and keeps a per-query GPU top-k heap,
    reusing the exact primitives from ``evaluate_minimax`` (``_maxsim``,
    ``_pad_arrow_token_vectors``). Paragraph granularity: each row is one unique
    paragraph (``paragraph_id`` has no underscore), so the heap holds top-k
    paragraphs directly.

    ``max_scan_batches`` (smoke knob): stop after this many corpus batches instead
    of scanning the whole table. Produces *partial* results — only for validating
    the full code path quickly; never use it for a real retrieval run.
    """
    import numpy as np
    import torch
    from tqdm import tqdm

    # Lazy: only the two multi-vector backends (.venv-colbert / .venv-jina-v4)
    # ever reach here, and both have the evaluator's deps. Keeps dense/vLLM light.
    _eval = _bootstrap.load_minimax("evaluate_minimax.py", "mmx_evaluate")
    _maxsim = _eval._maxsim
    _pad_arrow_token_vectors = _eval._pad_arrow_token_vectors

    Q = int(q_docs.shape[0])
    heap_capacity = k  # 1 row == 1 paragraph at paragraph granularity
    best_scores = torch.full((Q, heap_capacity), float("-inf"), dtype=torch.float32, device=device)
    best_idx = torch.full((Q, heap_capacity), -1, dtype=torch.int64, device=device)
    cpu_ids: list[str] = []

    lance_ds = table.to_lance()
    scanner = lance_ds.scanner(columns=["paragraph_id", "token_vectors"], batch_size=chunk)

    n_seen = 0
    for batch_no, arrow_batch in enumerate(tqdm(scanner.to_batches(), desc=f"{alias} maxsim-scan")):
        if max_scan_batches is not None and batch_no >= max_scan_batches:
            logger.warning("[%s] stopping scan early after %d batches (--max-scan-batches; PARTIAL results).",
                           alias, max_scan_batches)
            break
        pids = arrow_batch.column("paragraph_id").to_pylist()
        tokens_col = arrow_batch.column("token_vectors")
        M = len(pids)
        if M == 0:
            continue
        d_docs, d_mask = _pad_arrow_token_vectors(tokens_col, dim=q_dim, device=device)
        with torch.no_grad():
            scores = _maxsim(q_docs, q_mask, d_docs, d_mask, q_batch_size=q_batch_size)  # (Q, M)
            batch_idx = (
                torch.arange(n_seen, n_seen + M, dtype=torch.int64, device=device)
                .unsqueeze(0)
                .expand(Q, -1)
            )
            combined_scores = torch.cat([best_scores, scores], dim=1)
            combined_idx = torch.cat([best_idx, batch_idx], dim=1)
            kk = min(heap_capacity, combined_scores.shape[1])
            top_vals, top_pos = combined_scores.topk(kk, dim=1)
            best_scores = top_vals
            best_idx = torch.gather(combined_idx, dim=1, index=top_pos)
        cpu_ids.extend(pids)
        n_seen += M
        del d_docs, d_mask, scores, combined_scores, combined_idx, batch_idx

    logger.info("[%s] scanned %d rows; writing top-%d candidates for %d queries.",
                alias, n_seen, k, Q)

    best_scores_cpu = best_scores.detach().cpu().numpy()
    best_idx_cpu = best_idx.detach().cpu().numpy()
    for qi in range(Q):
        scores_qi = best_scores_cpu[qi]
        idx_qi = best_idx_cpu[qi]
        order = np.argsort(-scores_qi)
        seen: set[str] = set()
        candidates: list[dict] = []
        for j in order:
            s = float(scores_qi[j])
            row_int = int(idx_qi[j])
            if not np.isfinite(s) or row_int < 0:
                continue
            cid = cpu_ids[row_int]
            if not cid or cid in seen:
                continue
            seen.add(cid)
            candidates.append({"id": cid, "rank": len(candidates) + 1, "score": s})
            if len(candidates) >= k:
                break
        write_model_candidates(alias, event_ids[qi], queries[qi], candidates, k)
