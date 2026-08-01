"""Tests for the 2026-07 judge-report fixes.

Covers:
- fleiss_over must ignore duplicate rater names (the 0.749-vs-0.710 bug);
- the gold-pair dedup exclusion is gone: loaders keep every labeled pair;
- prompt_variant_comparison's tune/held split is first-N-by-file-order;
- McNemar exact p matches the closed forms behind the paper's 0.0117 / 0.0075.
"""

import json

import pytest

from src.create_dataset.classifying_data import agreement_significance as sig
from src.create_dataset.classifying_data import frontier_cost_report as fcr
from src.create_dataset.classifying_data import prompt_variant_comparison as pvc

FORMERLY_EXCLUDED_ID = "9d7c6b97-eb64-4d1e-98a6-0d3c479c366e"


def test_fleiss_over_ignores_duplicate_rater_names():
    labelers = {
        "a": {f"i{k}": v for k, v in enumerate([1, 1, 0, 0, 1, 0])},
        "b": {f"i{k}": v for k, v in enumerate([1, 0, 0, 1, 1, 0])},
    }
    k_unique, n_unique = fcr.fleiss_over(labelers, ["a", "b"])
    k_dup, n_dup = fcr.fleiss_over(labelers, ["a", "b", "a"])
    assert n_dup == n_unique == 6
    assert k_dup == pytest.approx(k_unique)


def test_load_predictions_keeps_all_labeled_pairs(tmp_path):
    rows = [
        {"id": FORMERLY_EXCLUDED_ID, "prediction": 1},
        {"id": "other-id", "prediction": 0},
    ]
    (tmp_path / "m__e.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    labelers = fcr.load_predictions(tmp_path, [("m", "e")], sample=[])
    assert set(labelers["m:e"]) == {FORMERLY_EXCLUDED_ID, "other-id"}


def test_split_gold_is_first_n_by_file_order():
    gold = {f"id{k}": k % 2 for k in range(6)}  # insertion order id0..id5
    tune, held = pvc.split_gold(gold, 3)
    assert tune == {"id0", "id1", "id2"}
    assert held == {"id3", "id4", "id5"}


def test_mcnemar_exact_p_closed_forms():
    # tune half: 10 fixes / 1 regression -> p = 24/2048
    assert sig.mcnemar_exact_p(1, 10) == pytest.approx(24 / 2048, abs=1e-6)
    # pooled: 15 fixes / 3 regressions -> p = 1976/262144
    assert sig.mcnemar_exact_p(3, 15) == pytest.approx(1976 / 262144, abs=1e-6)
