"""Unit tests for prepare_relations_v2 (drop zero-positive + renumber).

Verifies the zero-positive detection (positive == classification 1), the
loud-failure guards (missing classification file, expected-count mismatch),
and that renumbering produces a contiguous 0..N-1 range without mutating the
input relations.
"""

import json
from pathlib import Path

import pytest

from prepare_relations_v2 import (
    find_zero_positive_event_ids,
    main,
    positive_count,
    renumber,
)


def _write_classified(dir_: Path, event_id: int, labels):
    """Write a classified file for `event_id` with the given classification labels."""
    payload = [
        {"id": f"{event_id}-{i}", "content": "x", "classification": c}
        for i, c in enumerate(labels)
    ]
    (dir_ / f"{event_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _relations(event_ids):
    return [
        {"event_id": eid, "sentence": f"sentence {eid}", "relation": "r",
         "constant_entity": "c", "abstract_entity_type": "t",
         "abstract_entity_instances": ["i"], "instantiated_sentences": ["s"]}
        for eid in event_ids
    ]


def test_positive_count_counts_only_ones(tmp_path):
    _write_classified(tmp_path, 7, [1, 0, 1, -1, 0])
    assert positive_count(tmp_path / "7.json") == 2


def test_find_zero_positive_event_ids(tmp_path):
    relations = _relations([10, 11, 12, 13])
    _write_classified(tmp_path, 10, [0, 1, 0])   # has a positive
    _write_classified(tmp_path, 11, [0, 0, -1])  # zero positive
    _write_classified(tmp_path, 12, [1])         # has a positive
    _write_classified(tmp_path, 13, [0, -1])     # zero positive

    assert find_zero_positive_event_ids(relations, tmp_path) == [11, 13]


def test_missing_classification_file_raises(tmp_path):
    relations = _relations([10, 11])
    _write_classified(tmp_path, 10, [1])
    # No file for event 11.
    with pytest.raises(FileNotFoundError):
        find_zero_positive_event_ids(relations, tmp_path)


def test_renumber_is_contiguous_and_nonmutating():
    kept = _relations([10, 12, 50])
    renumbered, id_map = renumber(kept)

    assert [it["event_id"] for it in renumbered] == [0, 1, 2]
    assert id_map == {10: 0, 12: 1, 50: 2}
    # Payload other than event_id is carried through.
    assert renumbered[2]["sentence"] == "sentence 50"
    # Original input list is not mutated.
    assert [it["event_id"] for it in kept] == [10, 12, 50]


def test_main_drops_renumbers_and_writes(tmp_path):
    relations = _relations([0, 1, 2, 3])
    source = tmp_path / "queries.json"
    source.write_text(json.dumps(relations), encoding="utf-8")

    classified = tmp_path / "classified"
    classified.mkdir()
    _write_classified(classified, 0, [1])         # keep
    _write_classified(classified, 1, [0, 0])      # drop
    _write_classified(classified, 2, [0, 1])      # keep
    _write_classified(classified, 3, [0])         # drop

    out = tmp_path / "v2" / "queries_seed.json"
    id_map = tmp_path / "v2" / "seed_id_map.json"

    main(source, classified, out, id_map, expected_zero=2)

    seed = json.loads(out.read_text())
    assert [it["event_id"] for it in seed] == [0, 1]
    assert {it["sentence"] for it in seed} == {"sentence 0", "sentence 2"}
    # Map records old->new only for survivors.
    assert json.loads(id_map.read_text()) == {"0": 0, "2": 1}


def test_main_aborts_on_expected_count_mismatch(tmp_path):
    relations = _relations([0, 1])
    source = tmp_path / "queries.json"
    source.write_text(json.dumps(relations), encoding="utf-8")

    classified = tmp_path / "classified"
    classified.mkdir()
    _write_classified(classified, 0, [1])     # keep
    _write_classified(classified, 1, [0])     # 1 zero-positive, but we expect 5

    out = tmp_path / "v2" / "queries_seed.json"
    id_map = tmp_path / "v2" / "seed_id_map.json"

    with pytest.raises(SystemExit):
        main(source, classified, out, id_map, expected_zero=5)
    # Nothing written on abort.
    assert not out.exists()
