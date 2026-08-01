"""Evaluator for minimax-indexed LanceDB tables (dense + ColBERTv2).

Sibling of `src/retrieval/lancedb/evaluate_lancedb.py`. Reuses its
ground-truth loaders and per-K averaging helper; replaces the
2K-column-per-query CSV with per-query JSONL plus an aggregated
``summary.json`` under ``results/minimax/<alias>-lancedb/``.

Dense path: SentenceTransformer query encoding + LanceDB ANN search,
with the same jina-v3 / jina-v4 task kwargs as the indexer's
passage-side call.

ColBERT path: brute-force GPU MaxSim over the multi-vector table. We
roll our own MaxSim (10 lines, no pylate-version risk) and stream the
corpus in Arrow batches so peak GPU memory stays bounded. Pylate is
imported lazily, so the dense path runs from the main ``.venv`` and the
ColBERT path runs from ``.venv-colbert``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import lancedb
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                # for siblings (no current users)
sys.path.insert(0, str(_HERE.parents[0]))     # src/retrieval/lancedb  -> evaluate_lancedb
sys.path.insert(0, str(_HERE.parents[1]))     # src/retrieval          -> config

from config import (  # noqa: E402
    CLASSIFIED_PARAGRAPHS_PATHS,
    LANCEDB_PATH,
    MODEL_IDS,
    RELATIONS_PATH,
    RESULTS_PATH,
    get_classified_collection_name,
    lancedb_path_for,
)
from evaluate_lancedb import (  # noqa: E402  - reuse, don't fork
    compute_avg_precision_recall_at_k,
    load_ground_truth_counts,
    load_ground_truth_labels,
)

logger = logging.getLogger(__name__)

COLBERT_ALIAS = "colbert-v2"
JINA_V4_LATE_ALIAS = "jina-embeddings-v4-late"
DEFAULT_K_VALUES = (10, 20, 30, 50, 100, 500, 1000)
SEARCH_BUFFER_MULT = 10           # pull k*10 raw hits and dedup-to-paragraph
COLBERT_HEAP_BUFFER_MULT = 10     # keep k*10 row-level hits per query during the scan


# ----------------------------- shared utilities -----------------------------


@dataclass(frozen=True)
class Hit:
    rank: int
    paragraph_id: str
    row_id: str
    label: int | None
    score: float


def _resolve_alias_and_model_id(model_arg: str) -> tuple[str, str]:
    """Mirror the alias/model_id resolution in evaluate_lancedb.py."""
    if model_arg in MODEL_IDS:
        return model_arg, MODEL_IDS[model_arg]
    for a, mid in MODEL_IDS.items():
        if mid == model_arg:
            return a, mid
    return model_arg, model_arg


def _load_relations() -> list[dict]:
    with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_metadata(raw) -> dict:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _label_for(gt_labels: dict[str, dict[str, int]], event_id: str, paragraph_id: str) -> int | None:
    per_event = gt_labels.get(event_id)
    if per_event is None:
        return None
    return per_event.get(str(paragraph_id))


def _cumulative_positives(hits: list[Hit], k: int) -> list[int]:
    """Cumulative count of positives in the first i ranks, for i in 1..k.

    If the hit list is shorter than k, we extend with the last value (the
    same convention as compute_avg_precision_recall_at_k uses on the read
    side via ``cumulative[-1]``).
    """
    out: list[int] = []
    running = 0
    for h in hits[:k]:
        if h.label == 1:
            running += 1
        out.append(running)
    while len(out) < k:
        out.append(running)
    return out


def _cumulative_negatives(hits: list[Hit], k: int) -> list[int]:
    out: list[int] = []
    running = 0
    for h in hits[:k]:
        if h.label != 1:
            running += 1
        out.append(running)
    while len(out) < k:
        out.append(running)
    return out


def _per_query_metrics(hits: list[Hit], R: int) -> dict[str, float]:
    """Precision@R / Recall@R, computed against the deduped hit list.

    Matches evaluate_lancedb.py: precision_at_R == recall_at_R, both equal
    (positives in top R) / R.
    """
    if R <= 0:
        return {"precision_at_R": 0.0, "recall_at_R": 0.0}
    positives_in_top_R = sum(1 for h in hits[:R] if h.label == 1)
    val = positives_in_top_R / R
    return {"precision_at_R": float(val), "recall_at_R": float(val)}


def _ensemble_paragraph_hits(
    *,
    dense_pid_scores: dict[str, tuple[float, str]],
    bm25_pid_scores: dict[str, tuple[float, str]],
    gt_labels: dict[str, dict[str, int]],
    event_id: str,
    k: int,
    weight_dense: float = 0.5,
    weight_bm25: float = 0.5,
) -> list[Hit]:
    """Score-level ensemble of dense + BM25 at paragraph granularity.

    Each input dict maps ``paragraph_id -> (score_higher_is_better, row_id)``.
    The dense side uses ``-distance`` as similarity so higher = better; the
    BM25 side passes its raw score (already higher = better).

    For each paragraph in the union, scores are min-max normalized
    *within their own side*, then combined as
    ``weight_dense * dense_norm + weight_bm25 * bm25_norm``. Paragraphs
    that appear in only one side get 0 on the missing side (treated as
    "below worst returned").
    """
    all_pids = list(dense_pid_scores.keys() | bm25_pid_scores.keys())
    if not all_pids:
        return []
    n = len(all_pids)
    dense_raw = np.zeros(n, dtype=np.float32)
    bm25_raw = np.zeros(n, dtype=np.float32)
    has_dense = np.zeros(n, dtype=bool)
    has_bm25 = np.zeros(n, dtype=bool)
    for j, pid in enumerate(all_pids):
        if pid in dense_pid_scores:
            dense_raw[j] = dense_pid_scores[pid][0]
            has_dense[j] = True
        if pid in bm25_pid_scores:
            bm25_raw[j] = bm25_pid_scores[pid][0]
            has_bm25[j] = True

    # Lazy import keeps fast --help / non-ensemble code path light.
    from bm25_index import minmax_normalize

    dense_norm = np.zeros(n, dtype=np.float32)
    bm25_norm = np.zeros(n, dtype=np.float32)
    if has_dense.any():
        dense_norm[has_dense] = minmax_normalize(dense_raw[has_dense])
    if has_bm25.any():
        bm25_norm[has_bm25] = minmax_normalize(bm25_raw[has_bm25])
    combined = weight_dense * dense_norm + weight_bm25 * bm25_norm
    order = np.argsort(-combined)

    hits: list[Hit] = []
    for j in order[:k]:
        pid = all_pids[j]
        row_id = (
            (dense_pid_scores.get(pid) or (None, None))[1]
            or (bm25_pid_scores.get(pid) or (None, None))[1]
            or pid
        )
        label = _label_for(gt_labels, event_id, pid)
        hits.append(
            Hit(
                rank=len(hits) + 1,
                paragraph_id=pid,
                row_id=row_id,
                label=label,
                score=float(combined[j]),
            )
        )
    return hits


def _summarise(
    queries: list[dict],
    k: int,
    k_values: Iterable[int],
) -> dict:
    """Aggregate Precision@K / Recall@K / avg_negatives @K across queries."""
    cumulative_positives = [_cumulative_positives([Hit(**h) for h in q["hits"]], k) for q in queries]
    cumulative_negatives = [_cumulative_negatives([Hit(**h) for h in q["hits"]], k) for q in queries]
    totals_R = [int(q["R"]) for q in queries]

    metrics: dict = {}
    for kk in k_values:
        if kk > k:
            continue
        p, r = compute_avg_precision_recall_at_k(cumulative_positives, totals_R, kk)
        neg_at_k = [
            (cn[kk - 1] if len(cn) >= kk else (cn[-1] if cn else 0))
            for cn in cumulative_negatives
        ]
        metrics[str(kk)] = {
            "precision": p,
            "recall": r,
            "avg_negatives": float(np.mean(neg_at_k)) if neg_at_k else 0.0,
        }

    if queries:
        metrics["at_R"] = {
            "precision": float(np.mean([q["metrics"]["precision_at_R"] for q in queries])),
            "recall": float(np.mean([q["metrics"]["recall_at_R"] for q in queries])),
        }
    return metrics


def _write_outputs(
    out_dir: Path,
    granularity: str,
    summary_header: dict,
    queries: list[dict],
    k: int,
    k_values: Iterable[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{granularity}.jsonl"
    summary_path = out_dir / f"{granularity}.summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    summary = {
        **summary_header,
        "granularity": granularity,
        "k": k,
        "n_queries": len(queries),
        "problematic_event_ids": [q["event_id"] for q in queries if q.get("_problematic")],
        "metrics_at_k": _summarise(queries, k=k, k_values=k_values),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %s (%d queries) and %s", jsonl_path, len(queries), summary_path)


# ------------------------------- dense path --------------------------------


def _dense_encode_kwargs_for_query(model_id: str) -> dict:
    """Mirror the indexer's passage-side prompts, but for query side."""
    kw: dict = {
        "batch_size": 128,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if "jina-embeddings-v3" in model_id:
        kw["task"] = "retrieval.query"
    elif "jina-embeddings-v4" in model_id:
        kw["task"] = "retrieval"
        kw["prompt_name"] = "query"
    return kw


def _dense_hits_from_pid_scores(
    *,
    dense_pid_scores: dict[str, tuple[float, str]],
    gt_labels: dict[str, dict[str, int]],
    event_id: str,
    k: int,
) -> list[Hit]:
    """Top-K paragraph-deduped Hits sorted by dense similarity (higher first).

    ``dense_pid_scores`` stores (similarity_higher_is_better, row_id). The
    emitted Hit's ``score`` is the original distance (i.e. ``-similarity``)
    so JSONL output stays compatible with prior dense-only runs.
    """
    ranked = sorted(dense_pid_scores.items(), key=lambda kv: -kv[1][0])[:k]
    hits: list[Hit] = []
    for pid, (sim, row_id) in ranked:
        hits.append(
            Hit(
                rank=len(hits) + 1,
                paragraph_id=pid,
                row_id=row_id,
                label=_label_for(gt_labels, event_id, pid),
                score=float(-sim),  # back to distance (lower=better) for the dense-only score field
            )
        )
    return hits


def _evaluate_dense(
    alias: str,
    model_id: str,
    granularities: list[str],
    k: int,
    ground_truth_source: str,
    table_suffix: str,
    out_dir: Path,
    k_values: Iterable[int],
    ensemble_with_bm25: bool = False,
    bm25_top_n: int = 10000,
    ensemble_out_dir: Optional[Path] = None,
) -> None:
    from sentence_transformers import SentenceTransformer  # local import for fast --help

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading dense model %s on %s", model_id, device)
    model = SentenceTransformer(model_id, trust_remote_code=True, device=device)

    if "GritLM" in model_id and hasattr(model[0], "auto_model"):
        try:
            model[0].auto_model.config.use_cache = False
        except (AttributeError, RuntimeError) as exc:
            logger.warning("Could not disable use_cache for GritLM: %s", exc)

    relations = _load_relations()
    queries_text = [r["sentence"] for r in relations]
    logger.info("Encoding %d queries ...", len(queries_text))
    query_embeddings = model.encode(queries_text, **_dense_encode_kwargs_for_query(model_id))

    gt_labels = load_ground_truth_labels(source=ground_truth_source)

    # Lazy imports so the non-ensemble code path stays cheap.
    if ensemble_with_bm25:
        from config import CLASSIFIED_SOURCE_PATHS as _CLASSIFIED_SOURCE_PATHS
        from bm25_index import (
            build_or_load_bm25,
            build_or_load_query_hits,
            default_bm25_cache_dir,
        )

    for granularity in granularities:
        gt_counts = load_ground_truth_counts(granularity=granularity, source=ground_truth_source)
        # Per-granularity LanceDB root: paragraph in home, sentence/proposition
        # on local-disk scratch (configurable via LANCEDB_SCRATCH_PATH).
        lancedb_root = lancedb_path_for(granularity)
        if not lancedb_root.exists():
            logger.warning("LanceDB root %s for granularity=%s not found; skipping.", lancedb_root, granularity)
            continue
        db = lancedb.connect(lancedb_root)
        table_name = get_classified_collection_name(alias, granularity) + table_suffix
        try:
            table = db.open_table(table_name)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Table %s not found at %s (%s); skipping granularity %s.", table_name, lancedb_root, exc, granularity)
            continue

        logger.info("Evaluating %s on %s (table=%s, root=%s, ensemble_bm25=%s)",
                    alias, granularity, table_name, lancedb_root, ensemble_with_bm25)

        # Lazy: build/load BM25 + precompute top-N per query (cached on disk so
        # subsequent model evals on the same granularity reuse the work).
        bm25_bundle = None
        bm25_query_hits = None
        if ensemble_with_bm25:
            source_dir = _CLASSIFIED_SOURCE_PATHS[ground_truth_source][granularity]
            cache_dir = default_bm25_cache_dir(LANCEDB_PATH)
            bm25_bundle = build_or_load_bm25(
                source_dir=source_dir,
                granularity_level=granularity,
                cache_dir=cache_dir,
            )
            bm25_query_hits = build_or_load_query_hits(
                bundle=bm25_bundle,
                granularity_level=granularity,
                queries=queries_text,
                n_per_query=bm25_top_n,
                cache_dir=cache_dir,
            )

        # Unified per-query loop: compute the dense per-paragraph score dict
        # once, then derive (a) the dense-only top-K and (b) if requested, the
        # dense+BM25 ensemble top-K from the SAME dict. This avoids the
        # historical need to run the dense model twice when the user wants
        # both result variants.
        dense_queries_out: list[dict] = []
        ensemble_queries_out: list[dict] | None = [] if ensemble_with_bm25 else None
        search_limit = k * SEARCH_BUFFER_MULT

        from tqdm import tqdm
        for i, relation in enumerate(tqdm(relations, desc=f"{alias}/{granularity}")):
            event_id = str(relation["event_id"])
            R = gt_counts.get(event_id)
            R_val = int(R) if R is not None else 0
            dense_pid_scores: dict[str, tuple[float, str]] = {}

            if R is None:
                logger.warning("No ground-truth count for event %s; emitting empty hits.", event_id)
            else:
                try:
                    raw = table.search(query_embeddings[i]).limit(search_limit).to_list()
                except (RuntimeError, ValueError) as exc:
                    logger.error("Search failed for event %s: %s", event_id, exc)
                    raw = []

                # Build the per-paragraph best dense score dict in one pass.
                # raw is distance-ascending so the FIRST row we see for a given
                # paragraph is its best (= lowest distance = highest similarity).
                for row in raw:
                    meta = _parse_metadata(row.get("metadata"))
                    pid = meta.get("paragraph_id")
                    if not pid:
                        # minimax indexer always sets paragraph_id; fall back only as a safety net.
                        raw_id = str(row.get("id", "") or meta.get("row_id", ""))
                        pid = raw_id.split("_")[0] if "_" in raw_id else raw_id
                    if not pid or pid in dense_pid_scores:
                        continue
                    row_id = meta.get("row_id") or str(row.get("id", "")) or pid
                    distance = float(row.get("_distance", 0.0))
                    # Store similarity (higher=better) so it composes cleanly
                    # with the min-max-normalize step in the ensemble.
                    dense_pid_scores[pid] = (-distance, row_id)

            # --- dense-only output (always written when this function runs) ---
            dense_hits = _dense_hits_from_pid_scores(
                dense_pid_scores=dense_pid_scores, gt_labels=gt_labels, event_id=event_id, k=k,
            )
            dense_problematic = (R is None) or (not dense_hits)
            dense_query_doc = {
                "event_id": event_id,
                "event": relation["sentence"],
                "R": R_val,
                "hits": [h.__dict__ for h in dense_hits],
                "metrics": _per_query_metrics(dense_hits, R_val),
            }
            if dense_problematic:
                dense_query_doc["_problematic"] = True
            dense_queries_out.append(dense_query_doc)

            # --- ensemble output (optional, same dense dict reused) ---
            if ensemble_with_bm25:
                assert bm25_bundle is not None and bm25_query_hits is not None
                assert ensemble_queries_out is not None
                bm25_pid_scores: dict[str, tuple[float, str]] = {}
                bm25_idx_i = bm25_query_hits.indices[i]
                bm25_score_i = bm25_query_hits.scores[i]
                parent_ids = bm25_bundle.parent_paragraph_ids
                row_ids_bm25 = bm25_bundle.row_ids
                for idx, score in zip(bm25_idx_i, bm25_score_i):
                    pid = parent_ids[int(idx)]
                    if pid in bm25_pid_scores:
                        continue
                    bm25_pid_scores[pid] = (float(score), row_ids_bm25[int(idx)])

                ensemble_hits = _ensemble_paragraph_hits(
                    dense_pid_scores=dense_pid_scores,
                    bm25_pid_scores=bm25_pid_scores,
                    gt_labels=gt_labels,
                    event_id=event_id,
                    k=k,
                )
                ensemble_problematic = (R is None) or (not ensemble_hits)
                ensemble_query_doc = {
                    "event_id": event_id,
                    "event": relation["sentence"],
                    "R": R_val,
                    "hits": [h.__dict__ for h in ensemble_hits],
                    "metrics": _per_query_metrics(ensemble_hits, R_val),
                }
                if ensemble_problematic:
                    ensemble_query_doc["_problematic"] = True
                ensemble_queries_out.append(ensemble_query_doc)

        # --- write outputs (dense-only always; ensemble if requested) ---
        dense_summary_header = {
            "alias": alias,
            "model_id": model_id,
            "ground_truth_source": ground_truth_source,
            "score_type": "distance",
            "retriever": "dense",
            "table_name": table_name,
        }
        _write_outputs(out_dir, granularity, dense_summary_header, dense_queries_out, k=k, k_values=k_values)

        if ensemble_with_bm25:
            assert ensemble_queries_out is not None
            assert ensemble_out_dir is not None, "ensemble_out_dir is required when ensemble_with_bm25=True"
            ensemble_summary_header = {
                "alias": alias,
                "model_id": model_id,
                "ground_truth_source": ground_truth_source,
                "score_type": "ensemble_norm_05_05",
                "retriever": "dense+bm25",
                "table_name": table_name,
                "bm25_top_n": bm25_top_n,
            }
            _write_outputs(ensemble_out_dir, granularity, ensemble_summary_header, ensemble_queries_out, k=k, k_values=k_values)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------------- BM25-only path --------------------------


def _evaluate_bm25_only(
    granularities: list[str],
    k: int,
    ground_truth_source: str,
    out_dir: Path,
    k_values: Iterable[int],
    bm25_top_n: int = 10000,
) -> None:
    """Pure-BM25 evaluation over the minimax corpus, no dense/ColBERT side.

    Uses the same BM25 index + per-query top-N caches the ensemble path
    builds, so running ``--bm25-only`` first warms the cache for subsequent
    ensemble runs (or vice versa).
    """
    from config import CLASSIFIED_SOURCE_PATHS as _CLASSIFIED_SOURCE_PATHS
    from bm25_index import build_or_load_bm25, build_or_load_query_hits, default_bm25_cache_dir

    relations = _load_relations()
    queries_text = [r["sentence"] for r in relations]
    gt_labels = load_ground_truth_labels(source=ground_truth_source)

    for granularity in granularities:
        gt_counts = load_ground_truth_counts(granularity=granularity, source=ground_truth_source)
        source_dir = _CLASSIFIED_SOURCE_PATHS[ground_truth_source][granularity]
        cache_dir = default_bm25_cache_dir(LANCEDB_PATH)
        bundle = build_or_load_bm25(
            source_dir=source_dir,
            granularity_level=granularity,
            cache_dir=cache_dir,
        )
        query_hits = build_or_load_query_hits(
            bundle=bundle,
            granularity_level=granularity,
            queries=queries_text,
            n_per_query=bm25_top_n,
            cache_dir=cache_dir,
        )

        logger.info("Evaluating bm25-only on granularity=%s (corpus=%d rows, top_n=%d)",
                    granularity, len(bundle.row_ids), bm25_top_n)

        queries_out: list[dict] = []
        parent_ids = bundle.parent_paragraph_ids
        row_ids = bundle.row_ids

        from tqdm import tqdm
        for qi, relation in enumerate(tqdm(relations, desc=f"bm25/{granularity}")):
            event_id = str(relation["event_id"])
            R = gt_counts.get(event_id)
            R_val = int(R) if R is not None else 0
            problematic = R is None

            # Walk BM25 top-N (score-desc) and dedup to top-K parent paragraphs.
            idx_q = query_hits.indices[qi]
            score_q = query_hits.scores[qi]
            seen: set[str] = set()
            hits: list[Hit] = []
            for idx, score in zip(idx_q, score_q):
                idx_i = int(idx)
                pid = parent_ids[idx_i]
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                label = _label_for(gt_labels, event_id, pid)
                hits.append(
                    Hit(
                        rank=len(hits) + 1,
                        paragraph_id=pid,
                        row_id=row_ids[idx_i],
                        label=label,
                        score=float(score),
                    )
                )
                if len(hits) >= k:
                    break

            if not hits:
                problematic = True

            query_doc = {
                "event_id": event_id,
                "event": relation["sentence"],
                "R": R_val,
                "hits": [h.__dict__ for h in hits],
                "metrics": _per_query_metrics(hits, R_val),
            }
            if problematic:
                query_doc["_problematic"] = True
            queries_out.append(query_doc)

        summary_header = {
            "alias": "bm25",
            "model_id": "bm25s/lucene",
            "ground_truth_source": ground_truth_source,
            "score_type": "bm25",
            "retriever": "bm25-only",
            "table_name": f"bm25_index({granularity})",
            "bm25_top_n": bm25_top_n,
        }
        _write_outputs(out_dir, granularity, summary_header, queries_out, k=k, k_values=k_values)


# ------------------------------ ColBERT path -------------------------------


def _pad_token_vectors(
    token_lists: list[list[list[float]]],
    dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of ``(L_i, dim)`` token-vector arrays to ``(M, L_max, dim)``.

    Used for the query side (Q lists, max length ~32). For per-batch document
    padding inside the scan loop we use `_pad_arrow_token_vectors` which goes
    through arrow's flat float buffer (10-20x faster than this nested-list path).

    Returns (docs, mask) with mask True at valid positions.
    """
    if not token_lists:
        return (
            torch.zeros((0, 0, dim), dtype=torch.float32, device=device),
            torch.zeros((0, 0), dtype=torch.bool, device=device),
        )
    lengths = [len(t) for t in token_lists]
    L_max = max(lengths) if lengths else 0
    M = len(token_lists)
    docs = torch.zeros((M, L_max, dim), dtype=torch.float32, device=device)
    mask = torch.zeros((M, L_max), dtype=torch.bool, device=device)
    for i, toks in enumerate(token_lists):
        L = len(toks)
        if L == 0:
            continue
        docs[i, :L, :] = torch.tensor(toks, dtype=torch.float32, device=device)
        mask[i, :L] = True
    return docs, mask


def _pad_arrow_token_vectors(
    tokens_col,                    # pyarrow ListArray<FixedSizeList<float32, dim>>
    dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad an arrow ListArray<FixedSizeList<float32, dim>> column to
    ``(M, L_max, dim)`` via numpy slicing of the flat float buffer.

    Avoids the 10-20x slowdown of building ``torch.tensor(nested_python_list)``
    per row. Returns (docs, mask) on `device`.
    """
    M = len(tokens_col)
    if M == 0:
        return (
            torch.zeros((0, 0, dim), dtype=torch.float32, device=device),
            torch.zeros((0, 0), dtype=torch.bool, device=device),
        )
    outer_offsets = np.asarray(tokens_col.offsets, dtype=np.int64)        # (M+1,)
    inner_fixed = tokens_col.values                                       # FixedSizeList<float32, dim>
    flat = np.asarray(inner_fixed.values, dtype=np.float32)               # (sum(L_i) * dim,)
    all_tokens = flat.reshape(-1, dim)                                    # (sum(L_i), dim)

    lengths = outer_offsets[1:] - outer_offsets[:-1]                      # (M,)
    L_max = int(lengths.max()) if M > 0 else 0
    docs_np = np.zeros((M, L_max, dim), dtype=np.float32)
    mask_np = np.zeros((M, L_max), dtype=bool)
    for i in range(M):
        L = int(lengths[i])
        if L == 0:
            continue
        s = int(outer_offsets[i])
        docs_np[i, :L] = all_tokens[s : s + L]
        mask_np[i, :L] = True
    docs = torch.from_numpy(docs_np).to(device, non_blocking=True)
    mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
    return docs, mask


def _maxsim(
    queries: torch.Tensor,    # (Q, Lq, H)
    q_mask: torch.Tensor,     # (Q, Lq) bool
    docs: torch.Tensor,       # (M, Ld, H)
    d_mask: torch.Tensor,     # (M, Ld) bool
    q_batch_size: int = 100,
) -> torch.Tensor:
    """Return (Q, M) MaxSim scores: sum over q-tokens of max over d-tokens.

    Invalid doc tokens contribute ``-inf`` to the inner max so they're never
    selected; invalid q-tokens contribute 0 to the outer sum.

    The intermediate einsum tensor has shape ``(Q, M, Lq, Ld)`` which for
    realistic eval sizes (Q ≥ 1000, M ≈ 4096, Lq=32, Ld up to 500) blows past
    typical GPU memory. We mini-batch over queries to bound peak memory:
    each slice handles ``q_batch_size`` queries (default 100, matching the
    historical Q for which the single-pass implementation was sized).
    """
    Q = queries.shape[0]
    M = docs.shape[0]
    if M == 0:
        return torch.zeros((Q, 0), dtype=queries.dtype, device=queries.device)

    out = torch.empty((Q, M), dtype=queries.dtype, device=queries.device)
    for qs in range(0, Q, q_batch_size):
        qe = min(qs + q_batch_size, Q)
        qb = queries[qs:qe]                                          # (qb, Lq, H)
        qm = q_mask[qs:qe]                                           # (qb, Lq)
        sims = torch.einsum("qlh,mkh->qmlk", qb, docs)               # (qb, M, Lq, Ld)
        sims = sims.masked_fill(~d_mask[None, :, None, :], float("-inf"))
        per_token_max = sims.max(dim=-1).values                       # (qb, M, Lq)
        per_token_max = per_token_max.masked_fill(~qm[:, None, :], 0.0)
        out[qs:qe] = per_token_max.sum(dim=-1)
        del sims, per_token_max
    return out                                                       # (Q, M)


def _run_maxsim_eval(
    *,
    alias: str,
    relations: list[dict],
    queries_text: list[str],
    q_docs: torch.Tensor,
    q_mask: torch.Tensor,
    q_dim: int,
    device: torch.device,
    granularities: list[str],
    k: int,
    ground_truth_source: str,
    table_suffix: str,
    out_dir: Path,
    chunk: int,
    k_values: Iterable[int],
    ensemble_with_bm25: bool,
    bm25_top_n: int,
    ensemble_out_dir: Optional[Path],
    retriever_label: str,
    ensemble_retriever_label: str,
) -> None:
    """Shared MaxSim eval loop for any multi-vector retriever (ColBERT, jina-v4-late).

    Caller is responsible for loading the encoder, encoding ``queries_text`` to
    ``q_docs``/``q_mask`` (already padded + on ``device``), and disposing of the
    encoder model. This helper handles per-granularity table scan, GPU top-k
    heap, optional BM25 ensemble fusion, and JSONL output.
    """
    if ensemble_with_bm25:
        from config import CLASSIFIED_SOURCE_PATHS as _CLASSIFIED_SOURCE_PATHS
        from bm25_index import (
            build_or_load_bm25,
            build_or_load_query_hits,
            default_bm25_cache_dir,
        )

    gt_labels = load_ground_truth_labels(source=ground_truth_source)

    Q = q_docs.shape[0]
    heap_capacity = k * COLBERT_HEAP_BUFFER_MULT

    for granularity in granularities:
        gt_counts = load_ground_truth_counts(granularity=granularity, source=ground_truth_source)
        # Per-granularity LanceDB root (sentence/proposition on local-disk
        # scratch, paragraph in home).
        lancedb_root = lancedb_path_for(granularity)
        if not lancedb_root.exists():
            logger.warning("LanceDB root %s for granularity=%s not found; skipping.", lancedb_root, granularity)
            continue
        db = lancedb.connect(lancedb_root)
        table_name = get_classified_collection_name(alias, granularity) + table_suffix
        try:
            table = db.open_table(table_name)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Table %s not found at %s (%s); skipping granularity %s.", table_name, lancedb_root, exc, granularity)
            continue

        logger.info("Evaluating %s on %s (table=%s, chunk=%d, ensemble_bm25=%s)",
                    alias, granularity, table_name, chunk, ensemble_with_bm25)

        # Lazy: build/load BM25 + precompute top-N per query (cached on disk).
        bm25_bundle = None
        bm25_query_hits = None
        if ensemble_with_bm25:
            source_dir = _CLASSIFIED_SOURCE_PATHS[ground_truth_source][granularity]
            cache_dir = default_bm25_cache_dir(LANCEDB_PATH)
            bm25_bundle = build_or_load_bm25(
                source_dir=source_dir,
                granularity_level=granularity,
                cache_dir=cache_dir,
            )
            bm25_query_hits = build_or_load_query_hits(
                bundle=bm25_bundle,
                granularity_level=granularity,
                queries=queries_text,
                n_per_query=bm25_top_n,
                cache_dir=cache_dir,
            )

        # Per-query top-heap kept entirely on GPU: scores + integer indices
        # into the CPU-side `cpu_pids` / `cpu_row_ids` lists. We never move
        # pid/row_id strings batch-by-batch — they grow once on the CPU and
        # we look them up at the end.
        best_scores = torch.full((Q, heap_capacity), float("-inf"), dtype=torch.float32, device=device)
        best_idx = torch.full((Q, heap_capacity), -1, dtype=torch.int64, device=device)
        cpu_pids: list[str] = []
        cpu_row_ids: list[str] = []

        lance_ds = table.to_lance()
        scanner = lance_ds.scanner(
            columns=["paragraph_id", "token_vectors"],   # metadata not needed: parent id derivable from row id
            batch_size=chunk,
        )

        n_seen = 0
        from tqdm import tqdm
        for arrow_batch in tqdm(scanner.to_batches(), desc=f"{alias}/{granularity}"):
            pids_arrow = arrow_batch.column("paragraph_id").to_pylist()
            tokens_col = arrow_batch.column("token_vectors")
            M = len(pids_arrow)
            if M == 0:
                continue

            # Parent paragraph id: for paragraph granularity row_id == parent
            # (no underscore); for sentence/proposition row_id is "pid_i".
            # Splitting on the first underscore is correct and ~100x faster
            # than parsing JSON metadata for the same field.
            parent_pids_batch = [p.split("_", 1)[0] for p in pids_arrow]

            d_docs, d_mask = _pad_arrow_token_vectors(tokens_col, dim=q_dim, device=device)
            with torch.no_grad():
                scores = _maxsim(q_docs, q_mask, d_docs, d_mask)             # (Q, M)

                # GPU top-k merge: concat (best, fresh) along the candidate
                # axis, then topk back to heap_capacity. Track which source
                # row each kept score came from via a parallel integer index.
                batch_idx = torch.arange(
                    n_seen, n_seen + M, dtype=torch.int64, device=device
                ).unsqueeze(0).expand(Q, -1)                                   # (Q, M)
                combined_scores = torch.cat([best_scores, scores], dim=1)      # (Q, heap+M)
                combined_idx = torch.cat([best_idx, batch_idx], dim=1)
                kk = min(heap_capacity, combined_scores.shape[1])
                top_vals, top_pos = combined_scores.topk(kk, dim=1)
                best_scores = top_vals
                best_idx = torch.gather(combined_idx, dim=1, index=top_pos)

            cpu_pids.extend(parent_pids_batch)
            cpu_row_ids.extend(pids_arrow)
            n_seen += M
            del d_docs, d_mask, scores, combined_scores, combined_idx, batch_idx

        logger.info("Scanned %d rows for granularity=%s", n_seen, granularity)

        # Materialize per-query JSONL. Dual-output: ColBERT-only is always
        # written; ColBERT+BM25 ensemble is added when --ensemble-with-bm25 is set.
        # Both are derived from the same heap, so we never re-scan the corpus.
        best_scores_cpu = best_scores.detach().cpu().numpy()
        best_idx_cpu = best_idx.detach().cpu().numpy()
        colbert_queries_out: list[dict] = []
        ensemble_queries_out: list[dict] | None = [] if ensemble_with_bm25 else None
        for qi, relation in enumerate(relations):
            event_id = str(relation["event_id"])
            R = gt_counts.get(event_id)
            R_val = int(R) if R is not None else 0

            scores_qi = best_scores_cpu[qi]
            idx_qi = best_idx_cpu[qi]
            order = np.argsort(-scores_qi)

            # Build per-paragraph best MaxSim score dict (single pass, no early exit).
            colbert_pid_scores: dict[str, tuple[float, str]] = {}
            for j in order:
                s = float(scores_qi[j])
                row_int = int(idx_qi[j])
                if not np.isfinite(s) or row_int < 0:
                    continue
                pid = cpu_pids[row_int]
                if not pid or pid in colbert_pid_scores:
                    continue
                colbert_pid_scores[pid] = (s, cpu_row_ids[row_int])

            # --- ColBERT-only output (always written) ---
            ranked = sorted(colbert_pid_scores.items(), key=lambda kv: -kv[1][0])[:k]
            colbert_hits: list[Hit] = []
            for pid, (s, row_id) in ranked:
                colbert_hits.append(
                    Hit(
                        rank=len(colbert_hits) + 1,
                        paragraph_id=pid,
                        row_id=row_id,
                        label=_label_for(gt_labels, event_id, pid),
                        score=s,  # MaxSim, higher=better; keep as-is for compat with prior runs
                    )
                )
            colbert_problematic = (R is None) or (not colbert_hits)
            colbert_query_doc = {
                "event_id": event_id,
                "event": relation["sentence"],
                "R": R_val,
                "hits": [h.__dict__ for h in colbert_hits],
                "metrics": _per_query_metrics(colbert_hits, R_val),
            }
            if colbert_problematic:
                colbert_query_doc["_problematic"] = True
            colbert_queries_out.append(colbert_query_doc)

            # --- ensemble output (optional) ---
            if ensemble_with_bm25:
                assert bm25_bundle is not None and bm25_query_hits is not None
                assert ensemble_queries_out is not None
                bm25_pid_scores: dict[str, tuple[float, str]] = {}
                bm25_idx_i = bm25_query_hits.indices[qi]
                bm25_score_i = bm25_query_hits.scores[qi]
                parent_ids = bm25_bundle.parent_paragraph_ids
                row_ids_bm25 = bm25_bundle.row_ids
                for idx, score in zip(bm25_idx_i, bm25_score_i):
                    pid = parent_ids[int(idx)]
                    if pid in bm25_pid_scores:
                        continue
                    bm25_pid_scores[pid] = (float(score), row_ids_bm25[int(idx)])

                ensemble_hits = _ensemble_paragraph_hits(
                    dense_pid_scores=colbert_pid_scores,
                    bm25_pid_scores=bm25_pid_scores,
                    gt_labels=gt_labels,
                    event_id=event_id,
                    k=k,
                )
                ensemble_problematic = (R is None) or (not ensemble_hits)
                ensemble_query_doc = {
                    "event_id": event_id,
                    "event": relation["sentence"],
                    "R": R_val,
                    "hits": [h.__dict__ for h in ensemble_hits],
                    "metrics": _per_query_metrics(ensemble_hits, R_val),
                }
                if ensemble_problematic:
                    ensemble_query_doc["_problematic"] = True
                ensemble_queries_out.append(ensemble_query_doc)

        # --- write outputs ---
        dense_summary_header = {
            "alias": alias,
            "model_id": MODEL_IDS.get(alias, alias),
            "ground_truth_source": ground_truth_source,
            "score_type": "maxsim",
            "retriever": retriever_label,
            "table_name": table_name,
            "n_rows_scanned": n_seen,
        }
        _write_outputs(out_dir, granularity, dense_summary_header, colbert_queries_out, k=k, k_values=k_values)

        if ensemble_with_bm25:
            assert ensemble_queries_out is not None
            assert ensemble_out_dir is not None, "ensemble_out_dir is required when ensemble_with_bm25=True"
            ensemble_summary_header = {
                "alias": alias,
                "model_id": MODEL_IDS.get(alias, alias),
                "ground_truth_source": ground_truth_source,
                "score_type": "ensemble_norm_05_05",
                "retriever": ensemble_retriever_label,
                "table_name": table_name,
                "n_rows_scanned": n_seen,
                "bm25_top_n": bm25_top_n,
            }
            _write_outputs(ensemble_out_dir, granularity, ensemble_summary_header, ensemble_queries_out, k=k, k_values=k_values)


def _evaluate_colbert(
    granularities: list[str],
    k: int,
    ground_truth_source: str,
    table_suffix: str,
    out_dir: Path,
    chunk: int,
    gpu_idx: int,
    k_values: Iterable[int],
    ensemble_with_bm25: bool = False,
    bm25_top_n: int = 10000,
    ensemble_out_dir: Optional[Path] = None,
) -> None:
    """Thin wrapper: load pylate ColBERTv2, encode queries, dispatch to MaxSim helper."""
    try:
        from pylate import models as pylate_models  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pylate is not installed in the current environment. Rerun with .venv-colbert/bin/python:\n"
            "    CUDA_VISIBLE_DEVICES=<gpu> .venv-colbert/bin/python "
            "src/retrieval/lancedb/minimax/evaluate_minimax.py --model colbert-v2 ..."
        ) from exc

    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    logger.info("Loading pylate ColBERTv2 on %s", device)
    model = pylate_models.ColBERT(
        model_name_or_path=MODEL_IDS[COLBERT_ALIAS],
        device=str(device),
    )

    relations = _load_relations()
    queries_text = [r["sentence"] for r in relations]
    logger.info("Encoding %d queries (ColBERT) ...", len(queries_text))
    q_embs_raw = model.encode(
        queries_text,
        batch_size=64,
        is_query=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    q_token_lists = [emb.tolist() for emb in q_embs_raw]
    q_dim = len(q_token_lists[0][0]) if q_token_lists and q_token_lists[0] else 128
    q_docs, q_mask = _pad_token_vectors(q_token_lists, dim=q_dim, device=device)
    logger.info("Query tensor: %s (mask sum=%d)", tuple(q_docs.shape), int(q_mask.sum().item()))

    try:
        _run_maxsim_eval(
            alias=COLBERT_ALIAS,
            relations=relations,
            queries_text=queries_text,
            q_docs=q_docs,
            q_mask=q_mask,
            q_dim=q_dim,
            device=device,
            granularities=granularities,
            k=k,
            ground_truth_source=ground_truth_source,
            table_suffix=table_suffix,
            out_dir=out_dir,
            chunk=chunk,
            k_values=k_values,
            ensemble_with_bm25=ensemble_with_bm25,
            bm25_top_n=bm25_top_n,
            ensemble_out_dir=ensemble_out_dir,
            retriever_label="colbert",
            ensemble_retriever_label="colbert+bm25",
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluate_jina_v4_late(
    granularities: list[str],
    k: int,
    ground_truth_source: str,
    table_suffix: str,
    out_dir: Path,
    chunk: int,
    gpu_idx: int,
    k_values: Iterable[int],
    ensemble_with_bm25: bool = False,
    bm25_top_n: int = 10000,
    ensemble_out_dir: Optional[Path] = None,
) -> None:
    """jina-embeddings-v4 in multi-vector (late-interaction) mode.

    Uses the HF interface directly — ``sentence-transformers`` silently
    aggregates jina-v4 to a single 2048-d vector. The multi-vector path is
    only exposed via ``model.encode_text(..., return_multivector=True)``.
    """
    from transformers import AutoModel

    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    model_id = MODEL_IDS[JINA_V4_LATE_ALIAS]
    logger.info("Loading %s (HF AutoModel, bf16) on %s", model_id, device)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    relations = _load_relations()
    queries_text = [r["sentence"] for r in relations]
    logger.info("Encoding %d queries (jina-v4 late-interaction) ...", len(queries_text))
    with torch.no_grad():
        q_embs_raw = model.encode_text(
            texts=queries_text,
            task="retrieval",
            prompt_name="query",
            return_multivector=True,
            batch_size=32,
        )

    # Convert per-query (Lq, 128) tensors to L2-normalized python list[list[float]]
    # so we can reuse the existing _pad_token_vectors path.
    q_token_lists: list[list[list[float]]] = []
    for emb in q_embs_raw:
        arr = emb.detach().to(torch.float32).cpu().numpy()
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        arr = arr / norms
        q_token_lists.append(arr.tolist())

    q_dim = len(q_token_lists[0][0]) if q_token_lists and q_token_lists[0] else 128
    q_docs, q_mask = _pad_token_vectors(q_token_lists, dim=q_dim, device=device)
    logger.info("Query tensor: %s (mask sum=%d)", tuple(q_docs.shape), int(q_mask.sum().item()))

    try:
        _run_maxsim_eval(
            alias=JINA_V4_LATE_ALIAS,
            relations=relations,
            queries_text=queries_text,
            q_docs=q_docs,
            q_mask=q_mask,
            q_dim=q_dim,
            device=device,
            granularities=granularities,
            k=k,
            ground_truth_source=ground_truth_source,
            table_suffix=table_suffix,
            out_dir=out_dir,
            chunk=chunk,
            k_values=k_values,
            ensemble_with_bm25=ensemble_with_bm25,
            bm25_top_n=bm25_top_n,
            ensemble_out_dir=ensemble_out_dir,
            retriever_label="jina-v4-late",
            ensemble_retriever_label="jina-v4-late+bm25",
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------- CLI ------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate minimax-indexed LanceDB tables (dense + ColBERT + BM25).")
    parser.add_argument("--model", type=str, required=False, default=None,
                        help="Model alias (from MODEL_IDS) or HF id. Required unless --bm25-only is set.")
    parser.add_argument(
        "--granularities",
        nargs="+",
        default=["paragraph", "sentence", "proposition"],
        choices=("paragraph", "sentence", "proposition"),
    )
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument(
        "--ground-truth",
        choices=sorted(CLASSIFIED_PARAGRAPHS_PATHS.keys()),
        default="minimax",
    )
    parser.add_argument("--table-suffix", type=str, default="")
    parser.add_argument(
        "--colbert-chunk",
        type=int,
        default=4096,
        help="ColBERT-only: corpus rows per GPU scan batch.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for ColBERT encode + scan.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=RESULTS_PATH / "minimax",
        help="Root output dir; each (alias, granularity) lands under <out-root>/<alias>-lancedb/.",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help=(
            "Pure-BM25 evaluation, no dense / ColBERT side. --model is ignored. "
            "Output dir: <out-root>/bm25-lancedb/<granularity>.{jsonl,summary.json}."
        ),
    )
    parser.add_argument(
        "--ensemble-with-bm25",
        action="store_true",
        help=(
            "Ensemble each model's score with BM25 over the same minimax corpus. "
            "Per query, retrieve top-N from each side, min-max normalize, then 0.5/0.5 "
            "weighted sum at paragraph granularity. Output is written to "
            "<out-root>/<alias>-bm25-ensemble-lancedb/ to keep the dense-only baseline."
        ),
    )
    parser.add_argument(
        "--bm25-top-n",
        type=int,
        default=10000,
        help="Per-query BM25 top-N retrieved when --ensemble-with-bm25. Default 10000.",
    )
    args = parser.parse_args()

    if args.bm25_only:
        out_dir = args.out_root / "bm25-lancedb"
        logger.info("BM25-only eval. k=%d granularities=%s gt=%s top_n=%d",
                    args.k, args.granularities, args.ground_truth, args.bm25_top_n)
        logger.info("Output dir: %s", out_dir)
        _evaluate_bm25_only(
            granularities=args.granularities,
            k=args.k,
            ground_truth_source=args.ground_truth,
            out_dir=out_dir,
            k_values=DEFAULT_K_VALUES,
            bm25_top_n=args.bm25_top_n,
        )
        logger.info("Done.")
        return

    if not args.model:
        parser.error("--model is required unless --bm25-only is set.")

    alias, model_id = _resolve_alias_and_model_id(args.model)
    # Dense-only output always lands at <alias>-lancedb/. When --ensemble-with-bm25
    # is set, the SAME invocation also writes the ensemble variant to
    # <alias>-bm25-ensemble-lancedb/ — saves a second expensive dense pass.
    out_dir = args.out_root / f"{alias}-lancedb"
    ensemble_out_dir = (
        args.out_root / f"{alias}-bm25-ensemble-lancedb" if args.ensemble_with_bm25 else None
    )

    logger.info("alias=%s  model_id=%s  k=%d  granularities=%s  gt=%s  ensemble_bm25=%s",
                alias, model_id, args.k, args.granularities, args.ground_truth, args.ensemble_with_bm25)
    logger.info("Output dir (dense): %s", out_dir)
    if ensemble_out_dir is not None:
        logger.info("Output dir (ensemble): %s", ensemble_out_dir)

    if alias == COLBERT_ALIAS:
        _evaluate_colbert(
            granularities=args.granularities,
            k=args.k,
            ground_truth_source=args.ground_truth,
            table_suffix=args.table_suffix,
            out_dir=out_dir,
            chunk=args.colbert_chunk,
            gpu_idx=args.gpu,
            k_values=DEFAULT_K_VALUES,
            ensemble_with_bm25=args.ensemble_with_bm25,
            bm25_top_n=args.bm25_top_n,
            ensemble_out_dir=ensemble_out_dir,
        )
    elif alias == JINA_V4_LATE_ALIAS:
        _evaluate_jina_v4_late(
            granularities=args.granularities,
            k=args.k,
            ground_truth_source=args.ground_truth,
            table_suffix=args.table_suffix,
            out_dir=out_dir,
            chunk=args.colbert_chunk,
            gpu_idx=args.gpu,
            k_values=DEFAULT_K_VALUES,
            ensemble_with_bm25=args.ensemble_with_bm25,
            bm25_top_n=args.bm25_top_n,
            ensemble_out_dir=ensemble_out_dir,
        )
    else:
        _evaluate_dense(
            alias=alias,
            model_id=model_id,
            granularities=args.granularities,
            k=args.k,
            ground_truth_source=args.ground_truth,
            table_suffix=args.table_suffix,
            out_dir=out_dir,
            k_values=DEFAULT_K_VALUES,
            ensemble_with_bm25=args.ensemble_with_bm25,
            bm25_top_n=args.bm25_top_n,
            ensemble_out_dir=ensemble_out_dir,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
