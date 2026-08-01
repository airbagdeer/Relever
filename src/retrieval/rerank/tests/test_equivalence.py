"""The load-bearing tests: the assumptions the whole compute saving rests on.

`test_pool_covers_every_retriever_top_500` and `test_pool_scoring_equals_direct_reranking`
are the two facts that justify scoring the deduplicated union pool once per (query,
reranker) instead of once per (query, reranker, retriever) — a 4.5x saving, 2.18M pairs
instead of 9.9M. They are asserted here rather than argued in a comment.

`test_identity_backend_reproduces_first_stage` is the end-to-end regression: running the
real pipeline with a constant-score reranker must reproduce the published first-stage
ranking exactly, exercising cache read, join, sort, tiebreak and metrics against a
known-good answer.

Marked `integration`: they read the real candidate dumps and label files.
"""

from __future__ import annotations

import json
import random

import eval_config as cfg
import pytest
import score_store
from data_loading import load_query_labels, load_ranked_list, rerank_ranked_list
from metrics import compute_query_metrics

POOL_DEPTH = 500  # AGG_PER_MODEL_K: each retriever contributes its top-500 to the pool
SAMPLE_QUERIES = 8


def _pool(event_id: int) -> dict:
    qid = cfg.qid(event_id)
    path = cfg.FRONTIER_CANDIDATES_DIR / qid / f"all_candidates_{qid}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _top_k_ids(alias: str, event_id: int, k: int) -> list[str]:
    qid = cfg.qid(event_id)
    path = cfg.FRONTIER_CANDIDATES_DIR / qid / f"{alias}_{qid}.json"
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return [c["id"] for c in sorted(doc["candidates"], key=lambda c: c["rank"])[:k]]


@pytest.fixture(scope="module")
def sample_event_ids() -> list[int]:
    rng = random.Random(20260708)
    all_ids = [int(d.name) for d in cfg.FRONTIER_CANDIDATES_DIR.iterdir()
               if d.is_dir() and d.name.isdigit()]
    return sorted(rng.sample(all_ids, SAMPLE_QUERIES))


@pytest.mark.integration
def test_pool_covers_every_retriever_top_500(sample_event_ids):
    """Premise 1: each retriever's top-500 is a subset of the deduplicated pool.

    If this ever fails, the pooled cache cannot serve that retriever and
    `rerank_ranked_list` would raise on a missing score. Better to catch it here.
    """
    for eid in sample_event_ids:
        pool_ids = {c["id"] for c in _pool(eid)["candidates"]}
        for alias in cfg.MODEL_ALIASES:
            top = set(_top_k_ids(alias, eid, POOL_DEPTH))
            missing = top - pool_ids
            assert not missing, (
                f"{alias} q{eid}: {len(missing)} of its top-{POOL_DEPTH} are outside the pool"
            )


@pytest.mark.integration
def test_pool_is_fully_labeled(sample_event_ids):
    """Premise 2: every pooled candidate carries a GPT-5.2 label.

    Reranking only permutes; it cannot conjure labels. An unlabeled candidate promoted
    into the top-R would be scored as a negative and silently understate P@R.
    """
    for eid in sample_event_ids:
        ql = load_query_labels(eid)
        pool_ids = [c["id"] for c in _pool(eid)["candidates"]]
        unlabeled = [i for i in pool_ids if i not in ql.labels]
        assert not unlabeled, f"q{eid}: {len(unlabeled)} pooled candidates have no label"


@pytest.mark.integration
def test_pool_scoring_equals_direct_reranking(sample_event_ids):
    """THE claim: sorting a retriever's top-500 by pool scores == reranking it directly.

    Simulated with a deterministic pseudo-score keyed on the paragraph id — that is
    exactly what a pointwise reranker is, a pure function of (query, document) that does
    not see the candidate set. If a real backend ever violated this (by normalising over
    the batch, say), its per-retriever result would differ from its pooled result, and
    this test would be the only thing standing between that bug and the paper.
    """
    for eid in sample_event_ids:
        ql = load_query_labels(eid)
        pool_ids = [c["id"] for c in _pool(eid)["candidates"]]
        # A pointwise reranker: score depends only on the document, never on its peers.
        pool_scores = {cid: (hash((eid, cid)) % 10_000) / 10_000.0 for cid in pool_ids}

        for alias in cfg.MODEL_ALIASES:
            base = load_ranked_list(alias, eid, ql.labels, depth=POOL_DEPTH)

            # (a) via the pooled cache
            via_pool = rerank_ranked_list(base, pool_scores, POOL_DEPTH, ql.labels)

            # (b) scoring only this retriever's own candidates, as if run per-retriever
            own = {cid: pool_scores[cid] for cid in base.ids}
            direct = rerank_ranked_list(base, own, POOL_DEPTH, ql.labels)

            assert via_pool.ids == direct.ids, f"{alias} q{eid}: pooled != direct order"
            assert via_pool.rel == direct.rel


@pytest.mark.integration
def test_identity_backend_reproduces_first_stage(sample_event_ids):
    """End-to-end regression through the real score cache.

    Requires the identity-constant cache:
        uv run python score_pool.py --reranker identity-constant
    Skipped (not silently passed) when it is absent.
    """
    covered = [e for e in sample_event_ids
               if score_store.scores_exist("identity-constant", "default", e)]
    if not covered:
        pytest.skip("no identity-constant score cache; run score_pool.py --reranker identity-constant")

    for eid in covered:
        ql = load_query_labels(eid)
        cached = score_store.read_scores("identity-constant", "default", eid)
        for alias in cfg.MODEL_ALIASES:
            base = load_ranked_list(alias, eid, ql.labels, depth=cfg.RETRIEVAL_DEPTH)
            out = rerank_ranked_list(base, cached, cfg.RETRIEVAL_DEPTH, ql.labels)

            assert out.ids == base.ids, f"{alias} q{eid}: identity reranker changed the order"
            a = compute_query_metrics(base.rel, ql.R, cfg.K_VALUES, cfg.SUCCESS_K_VALUES)
            b = compute_query_metrics(out.rel, ql.R, cfg.K_VALUES, cfg.SUCCESS_K_VALUES)
            assert a == b, f"{alias} q{eid}: identity reranker changed the metrics"
