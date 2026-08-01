"""Unit tests for the delta-label merge and the exact-P@R logic.

Covers ``data_loading.load_query_labels`` (main ∪ delta union, R vs R_pool),
``data_loading.load_ranked_list`` (call-time depth default, deep loading,
top-500-scoped ``n_missing_labels``) and the ``evaluate_frontier.evaluate_all``
exactness rule (deep r_precision when R > depth; NaN + ``p_at_r_exact=False``
when any model's top-R is unlabeled or shorter than R) — all on synthetic data
in ``tmp_path``, with ``eval_config`` monkeypatched (small depths, a fake
2-model registry). Reference semantics for the union match ``union_labels`` in
src/create_dataset/classifying_data/batch_api/label_frontier_delta_batch.py.

Run: uv run pytest src/retrieval/lancedb/frontier-v2/evaluation/test_data_loading_delta.py -v
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

import data_loading as dl
import eval_config as cfg
import evaluate_frontier as ev
from eval_config import ModelInfo

# --------------------------------------------------------------------------- #
# Synthetic-data helpers
# --------------------------------------------------------------------------- #
FAKE_MODELS = (
    ModelInfo("mA", "fam", None, "dense", mteb_rank=None, rteb_rank=None),
    ModelInfo("mB", "fam", None, "dense", mteb_rank=None, rteb_rank=None),
)


def _write_labels(root, eid: int, labels: dict[str, int]) -> None:
    q = str(eid).zfill(4)
    d = root / q
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"id": cid, "event_id": eid, "prediction": pred})
             for cid, pred in labels.items()]
    (d / f"predictions_{q}.jsonl").write_text("\n".join(lines) + "\n")


def _write_candidates(root, alias: str, eid: int, ids: list[str]) -> None:
    q = str(eid).zfill(4)
    d = root / q
    d.mkdir(parents=True, exist_ok=True)
    doc = {
        "event_id": eid,
        "query": f"query {eid}",
        "model": alias,
        "k": len(ids),
        "n_candidates": len(ids),
        "candidates": [{"id": cid, "rank": r, "score": 1.0 / r}
                       for r, cid in enumerate(ids, start=1)],
    }
    (d / f"{alias}_{q}.json").write_text(json.dumps(doc))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Point the whole evaluation at a tmp tree with small depths + 2 fake models."""
    classified = tmp_path / "classified"
    delta = classified / "extended_pooling_delta"
    candidates = tmp_path / "candidates"
    classified.mkdir()
    candidates.mkdir()
    monkeypatch.setattr(cfg, "FRONTIER_CLASSIFIED_DIR", classified)
    monkeypatch.setattr(cfg, "FRONTIER_DELTA_DIR", delta)
    monkeypatch.setattr(cfg, "FRONTIER_CANDIDATES_DIR", candidates)
    monkeypatch.setattr(cfg, "RETRIEVAL_DEPTH", 3)
    monkeypatch.setattr(cfg, "MAX_RANK", 6)
    monkeypatch.setattr(cfg, "K_VALUES", (1, 3))
    monkeypatch.setattr(cfg, "SUCCESS_K_VALUES", (1,))
    monkeypatch.setattr(cfg, "HEADLINE_NDCG_K", 3)
    monkeypatch.setattr(cfg, "MODELS", FAKE_MODELS)
    monkeypatch.setattr(cfg, "MODEL_ALIASES", tuple(m.alias for m in FAKE_MODELS))
    return {"classified": classified, "delta": delta, "candidates": candidates}


# --------------------------------------------------------------------------- #
# load_query_labels: main ∪ delta
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_labels_without_delta_backward_compat(env):
    _write_labels(env["classified"], 7, {"a": 1, "b": 0, "c": 1})
    ql = dl.load_query_labels(7)
    assert (ql.R, ql.R_pool, ql.n_delta, ql.n_pool) == (2, 2, 0, 3)
    assert ql.n_failed == 0


@pytest.mark.unit
def test_labels_merge_delta_grows_R(env):
    _write_labels(env["classified"], 7, {"a": 1, "b": 0})
    _write_labels(env["delta"], 7, {"c": 1, "d": 1, "e": 0})
    ql = dl.load_query_labels(7)
    assert (ql.R, ql.R_pool, ql.n_delta, ql.n_pool) == (3, 1, 3, 5)


@pytest.mark.unit
def test_labels_delta_wins_on_overlap(env):
    # Empty in practice (verified on disk), but the merge order must match
    # label_frontier_delta_batch.union_labels: delta overwrites main.
    _write_labels(env["classified"], 7, {"a": 0, "b": 0})
    _write_labels(env["delta"], 7, {"a": 1})
    ql = dl.load_query_labels(7)
    assert ql.labels["a"] == 1
    assert (ql.R, ql.n_pool) == (1, 2)


@pytest.mark.unit
def test_labels_count_failed_over_union(env):
    _write_labels(env["classified"], 7, {"a": -1, "b": 1})
    _write_labels(env["delta"], 7, {"c": -1})
    ql = dl.load_query_labels(7)
    assert ql.n_failed == 2


# --------------------------------------------------------------------------- #
# load_ranked_list: depth + n_missing scoping
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ranked_list_default_depth_is_calltime_config(env):
    _write_candidates(env["candidates"], "mA", 7, ["a", "b", "c", "d", "e", "f"])
    labels = {c: 1 for c in "abcdef"}
    rl = dl.load_ranked_list("mA", 7, labels)
    assert rl.n_retrieved == 3  # cfg.RETRIEVAL_DEPTH monkeypatched to 3
    deep = dl.load_ranked_list("mA", 7, labels, depth=5)
    assert deep.n_retrieved == 5
    assert deep.ids == ["a", "b", "c", "d", "e"]


@pytest.mark.unit
def test_ranked_list_missing_counted_only_in_top_depth(env):
    # Unlabeled at ranks 2, 4, 5 — only the rank-2 one is within RETRIEVAL_DEPTH=3.
    _write_candidates(env["candidates"], "mA", 7, ["a", "u1", "b", "u2", "u3"])
    labels = {"a": 1, "b": 0}
    rl = dl.load_ranked_list("mA", 7, labels, depth=5)
    assert rl.n_missing_labels == 1
    assert rl.rel == [1, 0, 0, 0, 0]  # unlabeled -> non-relevant


# --------------------------------------------------------------------------- #
# evaluate_all: exactness rule + NaN propagation
# --------------------------------------------------------------------------- #
def _build_eval_dataset(env):
    """5 queries × 2 models over RETRIEVAL_DEPTH=3, MAX_RANK=6.

    eid 1: R=0                              -> excluded from aggregates.
    eid 2: R=2 <= depth                     -> exact, standard path.
    eid 3: R=4 > depth via delta, converged -> deep exact r_precision.
    eid 4: R=4 > depth, mB unlabeled in top-R      -> NaN, pr-excluded.
    eid 5: R=5 > depth, mA list shorter than R     -> NaN, pr-excluded.
    """
    _write_labels(env["classified"], 1, {"n1": 0})
    _write_candidates(env["candidates"], "mA", 1, ["n1"])
    _write_candidates(env["candidates"], "mB", 1, ["n1"])

    _write_labels(env["classified"], 2, {"p1": 1, "p2": 1, "n1": 0, "n2": 0})
    _write_candidates(env["candidates"], "mA", 2, ["p1", "n1", "p2", "n2"])
    _write_candidates(env["candidates"], "mB", 2, ["n1", "p1", "p2", "n2"])

    _write_labels(env["classified"], 3, {"q1": 1, "q2": 1, "q3": 1, "m1": 0})
    _write_labels(env["delta"], 3, {"q4": 1, "m2": 0})  # union R=4 > depth 3
    _write_candidates(env["candidates"], "mA", 3, ["q1", "m1", "q2", "q4", "q3", "m2"])
    _write_candidates(env["candidates"], "mB", 3, ["q4", "q1", "q2", "q3", "m1", "m2"])

    _write_labels(env["classified"], 4, {"r1": 1, "r2": 1, "r3": 1, "r4": 1})
    _write_candidates(env["candidates"], "mA", 4, ["r1", "r2", "r3", "r4", "x1"])
    _write_candidates(env["candidates"], "mB", 4, ["r1", "u1", "r2", "r3", "r4"])  # u1 unlabeled at rank 2 <= R

    _write_labels(env["classified"], 5, {"s1": 1, "s2": 1, "s3": 1, "s4": 1, "s5": 1})
    _write_candidates(env["candidates"], "mA", 5, ["s1", "s2", "s3", "s4"])  # 4 < R=5
    _write_candidates(env["candidates"], "mB", 5, ["s1", "s2", "s3", "s4", "s5"])


@pytest.fixture()
def eval_results(env):
    _build_eval_dataset(env)
    return ev.evaluate_all(verbose=False)


@pytest.mark.unit
def test_evaluate_query_sets(eval_results):
    res = eval_results
    assert res.n_queries_total == 5
    assert res.n_queries_eval == 4
    assert res.excluded_event_ids == [1]
    assert res.pr_excluded_event_ids == [4, 5]
    assert res.n_queries_pr_exact == 2


@pytest.mark.unit
def test_evaluate_deep_exact_r_precision(eval_results):
    pq = eval_results.per_query.set_index(["model", "event_id"])
    # eid 3, union R=4: mA deep rel = [1,0,1,1] -> 3/4; mB = [1,1,1,1] -> 1.0.
    assert pq.loc[("mA", 3), "r_precision"] == pytest.approx(3 / 4)
    assert pq.loc[("mB", 3), "r_precision"] == pytest.approx(1.0)
    assert bool(pq.loc[("mA", 3), "p_at_r_exact"]) is True
    assert int(pq.loc[("mA", 3), "R"]) == 4
    assert int(pq.loc[("mA", 3), "R_pool"]) == 3
    # Standard suite stays at depth 3 with the union R as denominator:
    # mA rel_top = [1,0,1] -> recall@3 = 2/4.
    assert pq.loc[("mA", 3), "recall@3"] == pytest.approx(2 / 4)
    # n_retrieved / n_rel_retrieved keep top-500(=3) semantics despite deep loading.
    assert int(pq.loc[("mA", 3), "n_retrieved"]) == 3
    assert int(pq.loc[("mA", 3), "n_rel_retrieved"]) == 2


@pytest.mark.unit
def test_evaluate_shallow_query_unchanged_path(eval_results):
    pq = eval_results.per_query.set_index(["model", "event_id"])
    # eid 2, R=2 <= depth: r_precision from the top-3 vector, no NaN.
    assert pq.loc[("mA", 2), "r_precision"] == pytest.approx(1 / 2)
    assert bool(pq.loc[("mA", 2), "p_at_r_exact"]) is True


@pytest.mark.unit
def test_evaluate_inexact_queries_nan_on_all_models(eval_results):
    pq = eval_results.per_query.set_index(["model", "event_id"])
    for eid in (4, 5):
        for alias in ("mA", "mB"):
            assert math.isnan(pq.loc[(alias, eid), "r_precision"])
            assert bool(pq.loc[(alias, eid), "p_at_r_exact"]) is False
    # The NaN is surgical: every other metric on those rows is finite.
    for col in ("ndcg@3", "precision@3", "recall@3", "mrr", "map", "success@1"):
        assert not math.isnan(pq.loc[("mA", 4), col])
    # eid 4: mA itself is fully labeled to R — excluded only because mB is not.
    assert pq.loc[("mA", 4), "recall@3"] == pytest.approx(3 / 4)


@pytest.mark.unit
def test_evaluate_r0_query_flags(eval_results):
    pq = eval_results.per_query.set_index(["model", "event_id"])
    assert bool(pq.loc[("mA", 1), "p_at_r_exact"]) is False
    assert pq.loc[("mA", 1), "r_precision"] == 0.0  # metrics convention for R=0


@pytest.mark.unit
def test_leaderboard_nan_skipping_and_counts(eval_results):
    lb = eval_results.leaderboard
    assert (lb["n_queries"] == 4).all()
    assert (lb["n_queries_r_precision"] == 2).all()
    # Means over the exact set only: eids 2 and 3.
    assert lb.loc["mA", "r_precision"] == pytest.approx((1 / 2 + 3 / 4) / 2)
    assert lb.loc["mB", "r_precision"] == pytest.approx((1 / 2 + 1.0) / 2)


@pytest.mark.unit
def test_csv_round_trip(eval_results, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cfg, "SUMMARY_DIR", tmp_path / "summary")
    ev.write_outputs(eval_results)
    pq = pd.read_csv(cfg.RESULTS_DIR / "all_models_per_query.csv")
    assert pq["p_at_r_exact"].dtype == bool
    nan_eids = sorted(pq.loc[pq["r_precision"].isna(), "event_id"].unique())
    assert nan_eids == [4, 5]
    lb = pd.read_csv(cfg.SUMMARY_DIR / "leaderboard.csv", index_col=0)
    assert (lb["n_queries_r_precision"] == 2).all()
