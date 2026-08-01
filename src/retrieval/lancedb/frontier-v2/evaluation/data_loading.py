"""Load ranked candidates + GPT-5.2 labels and join them into per-query,
per-model binary relevance vectors.

Data shapes (confirmed on disk over all 1100 queries):
  * candidates:  ``data/candidates/<qid>/<alias>_<qid>.json`` ->
                 {"event_id", "query", "model", "k", "n_candidates",
                  "candidates": [{"id", "rank", "score"}, ...]}  (already ranked,
                 down to rank ``eval_config.MAX_RANK`` = 3000)
  * labels:      ``data/labels/<qid>/predictions_<qid>.jsonl``
                 -> one JSON object per line {"id", "event_id", "prediction"}
                 with prediction in {1 (positive), 0 (negative), -1 (failed)}.
  * delta labels: ``data/labels/extended_pooling_delta/<qid>/
                 predictions_<qid>.jsonl`` — same schema/setup, produced for the
                 61 R>500 queries by labelling beyond the depth-500 pool so that
                 exact Precision@R is computable (see
                 the paper's extended-pooling appendix).
  * queries:     ``data/queries/queries.json`` -> list of objects
                 with "event_id", "sentence", "relation", "constant_entity",
                 "abstract_entity_type".

Relevance model: the ground truth per query is the **union** of the pooled and
delta labels (delta wins on the intersection, which is empty on disk). A chunk
is relevant iff its label == 1; ``R`` is the number of label==1 chunks in that
union (identical for all 18 models on a query — TREC-style pooling, depth 500
everywhere except the R>500 queries whose pools extend to their top-R bands).
Failed labels (-1; zero in the current data) and any candidate id missing from
the label files are treated as non-relevant (rel = 0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import _bootstrap  # noqa: F401  (sys.path: `config` + the rerank package's score_store)
import eval_config as cfg
import score_store


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueryMeta:
    event_id: int
    sentence: str
    relation: str
    constant_entity: str
    abstract_entity_type: str


@dataclass(frozen=True)
class QueryLabels:
    event_id: int
    labels: dict[str, int]   # chunk_id -> prediction (1 / 0 / -1), main ∪ delta
    R: int                   # positives (label == 1) in the merged label set
    R_pool: int              # positives in the depth-500 pool alone (pre-delta R)
    n_pool: int              # total classified chunks (main ∪ delta)
    n_delta: int             # delta labels merged in (0 when no delta file exists)
    n_failed: int            # number of failed labels (-1)


@dataclass(frozen=True)
class RankedList:
    alias: str
    event_id: int
    ids: list[str]           # chunk ids in rank order
    rel: list[int]           # 1 if positive else 0, in rank order
    scores: list[float]      # retrieval score in rank order
    n_retrieved: int
    n_missing_labels: int    # candidate ids absent from the label file (expected 0)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _candidate_path(alias: str, event_id: int):
    q = cfg.qid(event_id)
    return cfg.FRONTIER_CANDIDATES_DIR / q / f"{alias}_{q}.json"


def _predictions_path(event_id: int):
    q = cfg.qid(event_id)
    return cfg.FRONTIER_CLASSIFIED_DIR / q / f"predictions_{q}.jsonl"


def _delta_predictions_path(event_id: int):
    q = cfg.qid(event_id)
    return cfg.FRONTIER_DELTA_DIR / q / f"predictions_{q}.jsonl"


def event_ids() -> list[int]:
    """Sorted event ids that have both a candidate dir and a predictions file."""
    ids: list[int] = []
    for d in sorted(p.name for p in cfg.FRONTIER_CLASSIFIED_DIR.iterdir() if p.is_dir()):
        if not d.isdigit():
            continue
        eid = int(d)
        if _predictions_path(eid).exists():
            ids.append(eid)
    return ids


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_queries() -> dict[int, QueryMeta]:
    """event_id -> QueryMeta for all 1100 relational queries."""
    with cfg.V2_RELATIONS_PATH.open() as f:
        rows = json.load(f)
    out: dict[int, QueryMeta] = {}
    for r in rows:
        eid = int(r["event_id"])
        out[eid] = QueryMeta(
            event_id=eid,
            sentence=r.get("sentence", ""),
            relation=r.get("relation", ""),
            constant_entity=r.get("constant_entity", ""),
            abstract_entity_type=r.get("abstract_entity_type", ""),
        )
    return out


def _read_predictions(path) -> dict[str, int]:
    """{chunk_id: prediction} from one predictions JSONL file."""
    out: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["id"]] = int(d["prediction"])
    return out


def load_query_labels(event_id: int) -> QueryLabels:
    """Merged (main ∪ delta) labels for a query, with R derived from the union.

    Mirrors ``union_labels`` in the delta labeler
    (src/create_dataset/classifying_data/batch_api/label_frontier_delta_batch.py):
    the delta file, when present, is merged over the pooled labels (delta wins on
    the — empty in practice — intersection). ``R_pool`` preserves the pre-delta R
    so the depth-500-pool view stays recomputable.
    """
    labels = _read_predictions(_predictions_path(event_id))
    R_pool = sum(1 for v in labels.values() if v == 1)
    n_delta = 0
    delta_path = _delta_predictions_path(event_id)
    if delta_path.exists():
        delta = _read_predictions(delta_path)
        n_delta = len(delta)
        labels.update(delta)
    R = sum(1 for v in labels.values() if v == 1)
    return QueryLabels(
        event_id=event_id,
        labels=labels,
        R=R,
        R_pool=R_pool,
        n_pool=len(labels),
        n_delta=n_delta,
        n_failed=sum(1 for v in labels.values() if v == -1),
    )


def load_ranked_list(
    alias: str, event_id: int, labels: dict[str, int], depth: int | None = None
) -> RankedList:
    """Read one model's ranked candidates and join to labels -> relevance vector.

    ``depth`` defaults to ``cfg.RETRIEVAL_DEPTH`` (resolved at call time so tests
    can monkeypatch the config). The standard metric suite is computed at the
    top-500 design depth; the evaluator passes a deeper ``depth`` (up to top-R)
    only to compute exact Precision@R for R>500 queries, whose top-R bands are
    covered by the delta labels. ``n_missing_labels`` always counts unlabeled
    candidates within the first ``cfg.RETRIEVAL_DEPTH`` ranks only, so its
    meaning is depth-independent (expected 0: the pool is the top-500 union).
    """
    if depth is None:
        depth = cfg.RETRIEVAL_DEPTH
    with _candidate_path(alias, event_id).open() as f:
        doc = json.load(f)
    cands = sorted(doc.get("candidates", []), key=lambda c: c["rank"])[:depth]
    ids: list[str] = []
    rel: list[int] = []
    scores: list[float] = []
    n_missing = 0
    for i, c in enumerate(cands):
        cid = c["id"]
        ids.append(cid)
        scores.append(float(c.get("score", 0.0)))
        if cid in labels:
            rel.append(1 if labels[cid] == 1 else 0)
        else:
            rel.append(0)
            if i < cfg.RETRIEVAL_DEPTH:
                n_missing += 1
    return RankedList(
        alias=alias,
        event_id=event_id,
        ids=ids,
        rel=rel,
        scores=scores,
        n_retrieved=len(ids),
        n_missing_labels=n_missing,
    )


# --------------------------------------------------------------------------- #
# Reranked lists
# --------------------------------------------------------------------------- #
def _rebuild(alias: str, event_id: int, ids: list[str], scores: list[float],
             labels: dict[str, int]) -> RankedList:
    """Re-derive the relevance vector for a reordered id list."""
    rel = [1 if labels.get(cid) == 1 else 0 for cid in ids]
    n_missing = sum(1 for cid in ids[: cfg.RETRIEVAL_DEPTH] if cid not in labels)
    return RankedList(
        alias=alias,
        event_id=event_id,
        ids=ids,
        rel=rel,
        scores=scores,
        n_retrieved=len(ids),
        n_missing_labels=n_missing,
    )


def rerank_ranked_list(
    base: RankedList,
    cached_scores: dict[str, float],
    rerank_depth: int,
    labels: dict[str, int],
    reranker: str = "?",
) -> RankedList:
    """Reorder ``base``'s top-``rerank_depth`` by ``cached_scores``. Pure; no I/O.

    Standard rerank-top-K protocol: only the first ``rerank_depth`` candidates are
    reordered; ranks beyond it keep the retriever's original order. Ranks 501+ have no
    cached reranker score anyway — the labeled pool is the union of the models' top-500.

    **Ties break by original retriever rank.** ``sorted`` is stable and ``base.ids`` is
    already in rank order, so a reranker returning constant scores reproduces the
    first-stage list exactly. That property is what makes the ``identity-constant``
    regression against the published leaderboard meaningful.

    A candidate inside ``rerank_depth`` with no cached score is a hard error, never a
    silent ``rel=0``: it would mean the score cache and the candidate dump disagree, and
    the ranking would be quietly wrong rather than obviously broken.

    ``p_at_r_exact`` is unaffected by reranking. For R<=500 the top-R is drawn from the
    (permuted) top-500, which is fully labeled; for R>500 the id *set* of the top-R is
    unchanged. Reranked metrics therefore stay on the same exact-P@R query set as the
    published first-stage leaderboard, and the two are directly comparable.
    """
    if rerank_depth <= 0:
        raise ValueError(f"rerank_depth must be positive, got {rerank_depth}.")

    prefix, tail = base.ids[:rerank_depth], base.ids[rerank_depth:]
    missing = [cid for cid in prefix if cid not in cached_scores]
    if missing:
        raise KeyError(
            f"[{reranker}] {len(missing)} of {base.alias}'s top-{rerank_depth} candidates "
            f"for query {base.event_id} have no cached score (e.g. {missing[0]}). "
            f"The score cache and the candidate dump disagree."
        )

    order = sorted(range(len(prefix)), key=lambda i: -cached_scores[prefix[i]])  # stable
    new_ids = [prefix[i] for i in order] + tail
    new_scores = [float(cached_scores[prefix[i]]) for i in order] + base.scores[rerank_depth:]
    return _rebuild(base.alias, base.event_id, new_ids, new_scores, labels)


def apply_listwise_permutation(
    base: RankedList,
    order: list[str],
    labels: dict[str, int],
    reranker: str = "?",
) -> RankedList:
    """Apply a listwise model's permutation of ``base``'s top-``len(order)``. Pure.

    Listwise rerankers score set-dependently, so there is no reusable per-document score
    to cache; ``rerank_listwise.py`` writes the permutation directly. It must be exactly
    a permutation of the retriever's prefix — a dropped or hallucinated id means the
    model's output was mis-parsed, so this raises rather than scoring a corrupted list.

    ``scores`` is synthesised as descending ranks purely to keep ``RankedList`` populated.
    It carries no meaning and must never be compared across models.
    """
    k = len(order)
    prefix = base.ids[:k]
    if sorted(order) != sorted(prefix):
        raise ValueError(
            f"[{reranker}] listwise permutation for ({base.alias}, q{base.event_id}) is not a "
            f"permutation of the retriever's top-{k}: {len(set(order) - set(prefix))} unexpected "
            f"id(s), {len(set(prefix) - set(order))} missing."
        )
    new_ids = list(order) + base.ids[k:]
    new_scores = [float(k - i) for i in range(k)] + base.scores[k:]
    return _rebuild(base.alias, base.event_id, new_ids, new_scores, labels)


def load_reranked_list(
    reranker: str,
    variant: str,
    retriever: str,
    event_id: int,
    labels: dict[str, int],
    rerank_depth: int,
    depth: int | None = None,
) -> RankedList:
    """I/O wrapper: load the retriever's list + the cached scores, then rerank.

    The evaluator does not use this — it hoists both loads out of its inner loop — but
    it is the convenient entry point for tests and ad-hoc inspection.
    """
    base = load_ranked_list(retriever, event_id, labels, depth=depth)
    cached = score_store.read_scores(reranker, variant, event_id)
    return rerank_ranked_list(base, cached, rerank_depth, labels, reranker=reranker)


def load_listwise_reranked_list(
    reranker: str,
    retriever: str,
    event_id: int,
    labels: dict[str, int],
    depth: int | None = None,
) -> RankedList:
    """I/O wrapper for the listwise permutation files."""
    base = load_ranked_list(retriever, event_id, labels, depth=depth)
    doc = score_store.read_listwise(reranker, retriever, event_id)
    return apply_listwise_permutation(base, list(doc["order"]), labels, reranker=reranker)
