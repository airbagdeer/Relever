"""Unit + smoke tests for corpus_loader.

The synthetic test verifies dedup + occurrence aggregation in isolation. The
real-data smoke test reads 5 files from the cluster's minimax dir; gated by
MINIMAX_REAL_DIR so it only runs where the data exists.
"""

import json
import os
from pathlib import Path

import pytest

from corpus_loader import load_paragraph_corpus


def _write(dir_: Path, name: str, payload):
    (dir_ / name).write_text(json.dumps(payload), encoding="utf-8")


def test_dedup_aggregates_occurrences_across_files(tmp_path: Path) -> None:
    pid = "pid-shared"
    common_text = "Wikipedia paragraph about a topic."
    _write(tmp_path, "0.json", [
        {"id": pid, "content": common_text, "source_query": "q0", "classification": 0},
        {"id": "pid-only-0", "content": "other", "source_query": "q0", "classification": 1},
    ])
    _write(tmp_path, "1.json", [
        {"id": pid, "content": common_text, "source_query": "q1", "classification": 1},
    ])
    _write(tmp_path, "2.json", [
        {"id": pid, "content": common_text, "source_query": "q2", "classification": -1},
    ])

    corpus = load_paragraph_corpus(source_dir=tmp_path, use_cache=False)

    assert len(corpus) == 2
    idx = corpus.paragraph_ids.index(pid)
    meta = json.loads(corpus.metadata_jsons[idx])
    assert meta["paragraph_id"] == pid
    assert meta["granularity_level"] == "paragraph"
    assert len(meta["occurrences"]) == 3
    assert set(meta["event_ids"]) == {"0", "1", "2"}
    assert meta["positive_event_ids"] == ["1"]
    assert meta["any_positive"] is True

    # The other paragraph appears only in file 0 with classification=1.
    other_idx = corpus.paragraph_ids.index("pid-only-0")
    other_meta = json.loads(corpus.metadata_jsons[other_idx])
    assert other_meta["event_ids"] == ["0"]
    assert other_meta["positive_event_ids"] == ["0"]
    assert other_meta["any_positive"] is True


def test_skips_entries_without_id_or_content(tmp_path: Path) -> None:
    _write(tmp_path, "0.json", [
        {"id": "", "content": "no id", "source_query": "q", "classification": 1},
        {"id": "pid-no-content", "content": "", "source_query": "q", "classification": 1},
        {"id": "pid-ok", "content": "kept", "source_query": "q", "classification": 0},
    ])
    corpus = load_paragraph_corpus(source_dir=tmp_path, use_cache=False)
    assert corpus.paragraph_ids == ["pid-ok"]


def test_text_mismatch_keeps_first_and_logs(tmp_path: Path, caplog) -> None:
    pid = "pid-mismatch"
    _write(tmp_path, "0.json", [
        {"id": pid, "content": "first text", "source_query": "q0", "classification": 1},
    ])
    _write(tmp_path, "1.json", [
        {"id": pid, "content": "different text", "source_query": "q1", "classification": 0},
    ])
    with caplog.at_level("WARNING"):
        corpus = load_paragraph_corpus(source_dir=tmp_path, use_cache=False)
    assert len(corpus) == 1
    assert corpus.texts[0] == "first text"
    assert any("differing content" in rec.message for rec in caplog.records)


def test_limit_files(tmp_path: Path) -> None:
    for i in range(5):
        _write(tmp_path, f"{i}.json", [
            {"id": f"pid-{i}", "content": "text", "source_query": "q", "classification": 1},
        ])
    corpus = load_paragraph_corpus(source_dir=tmp_path, limit_files=2, use_cache=False)
    assert {pid for pid in corpus.paragraph_ids} == {"pid-0", "pid-1"}


def test_sentence_corpus_explodes_and_dedups(tmp_path: Path) -> None:
    """Sentences across event files share (paragraph_id, idx) and aggregate."""
    pid = "pid-A"
    sents = ["First sentence.", "Second sentence."]
    _write(tmp_path, "0.json", [
        {"id": pid, "sentences": sents, "source_query": "q0", "classification": 1},
    ])
    _write(tmp_path, "1.json", [
        {"id": pid, "sentences": sents, "source_query": "q1", "classification": 0},
    ])
    corpus = load_paragraph_corpus(source_dir=tmp_path, granularity_level="sentence", use_cache=False)
    assert len(corpus) == 2  # two sentences, deduped across events

    # Row ids are paragraph_id_<i>
    assert set(corpus.paragraph_ids) == {f"{pid}_0", f"{pid}_1"}

    idx0 = corpus.paragraph_ids.index(f"{pid}_0")
    meta0 = json.loads(corpus.metadata_jsons[idx0])
    assert meta0["paragraph_id"] == pid
    assert meta0["granularity_level"] == "sentence"
    assert meta0["row_id"] == f"{pid}_0"
    assert len(meta0["occurrences"]) == 2
    assert set(meta0["event_ids"]) == {"0", "1"}
    assert meta0["positive_event_ids"] == ["0"]
    assert corpus.texts[idx0] == sents[0]


def test_proposition_corpus_uses_propositions_field(tmp_path: Path) -> None:
    pid = "pid-P"
    props = ["AWS supports healthcare.", "AWS is a cloud provider."]
    _write(tmp_path, "0.json", [
        {"id": pid, "propositions": props, "source_query": "q0", "classification": 1},
    ])
    corpus = load_paragraph_corpus(source_dir=tmp_path, granularity_level="proposition", use_cache=False)
    assert len(corpus) == 2
    assert set(corpus.paragraph_ids) == {f"{pid}_0", f"{pid}_1"}
    for i, prop_text in enumerate(props):
        idx = corpus.paragraph_ids.index(f"{pid}_{i}")
        assert corpus.texts[idx] == prop_text
        meta = json.loads(corpus.metadata_jsons[idx])
        assert meta["granularity_level"] == "proposition"
        assert meta["paragraph_id"] == pid


def test_unknown_granularity_raises(tmp_path: Path) -> None:
    _write(tmp_path, "0.json", [{"id": "x", "content": "y"}])
    with pytest.raises(ValueError, match="Unsupported granularity_level"):
        load_paragraph_corpus(source_dir=tmp_path, granularity_level="bogus", use_cache=False)


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_REAL_DIR"),
    reason="Set MINIMAX_REAL_DIR to the path of data/classified_splade_paragraphs_minimax to run.",
)
def test_real_minimax_5_files() -> None:
    src = Path(os.environ["MINIMAX_REAL_DIR"]).resolve()
    corpus = load_paragraph_corpus(source_dir=src, limit_files=5, use_cache=False)
    assert len(corpus) > 0, "Expected at least one paragraph from 5 real minimax files."
    sample_meta = json.loads(corpus.metadata_jsons[0])
    assert sample_meta["paragraph_id"] == corpus.paragraph_ids[0]
    assert sample_meta["granularity_level"] == "paragraph"
    assert sample_meta["occurrences"], "Each entry should have at least one occurrence."


# ---------------------------------------------------------------------------
# Sentence / proposition granularities
# ---------------------------------------------------------------------------

def test_sentence_flattens_nested_list(tmp_path: Path) -> None:
    pid = "pid-A"
    _write(tmp_path, "0.json", [
        {
            "id": pid,
            "sentences": ["First sentence.", "Second.", ""],
            "event_id": 0,
            "source_query": "q0",
            "classification": 1,
        },
        {
            "id": "pid-B",
            "sentences": ["Lonely sentence."],
            "event_id": 0,
            "source_query": "q0",
            "classification": 0,
        },
    ])
    corpus = load_paragraph_corpus(
        source_dir=tmp_path, granularity_level="sentence", use_cache=False
    )

    # 3 rows: pid-A_0, pid-A_1 (skipping the empty third), pid-B_0
    assert sorted(corpus.paragraph_ids) == ["pid-A_0", "pid-A_1", "pid-B_0"]

    idx = corpus.paragraph_ids.index("pid-A_0")
    assert corpus.texts[idx] == "First sentence."
    meta = json.loads(corpus.metadata_jsons[idx])
    assert meta["paragraph_id"] == pid
    assert meta["row_id"] == "pid-A_0"
    assert meta["granularity_level"] == "sentence"
    assert meta["positive_event_ids"] == ["0"]
    assert meta["any_positive"] is True


def test_sentence_aggregates_across_events(tmp_path: Path) -> None:
    pid = "pid-shared"
    sents = ["S0", "S1"]
    _write(tmp_path, "0.json", [{
        "id": pid, "sentences": sents,
        "event_id": 0, "source_query": "q0", "classification": 1,
    }])
    _write(tmp_path, "1.json", [{
        "id": pid, "sentences": sents,
        "event_id": 1, "source_query": "q1", "classification": 0,
    }])

    corpus = load_paragraph_corpus(
        source_dir=tmp_path, granularity_level="sentence", use_cache=False
    )
    assert sorted(corpus.paragraph_ids) == ["pid-shared_0", "pid-shared_1"]

    idx = corpus.paragraph_ids.index("pid-shared_0")
    meta = json.loads(corpus.metadata_jsons[idx])
    assert set(meta["event_ids"]) == {"0", "1"}
    assert meta["positive_event_ids"] == ["0"]
    assert len(meta["occurrences"]) == 2


def test_proposition_flattens_nested_list(tmp_path: Path) -> None:
    pid = "pid-P"
    _write(tmp_path, "0.json", [{
        "id": pid,
        "propositions": ["Prop one.", "Prop two."],
        "event_id": 7,
        "source_query": "q",
        "classification": 1,
    }])
    corpus = load_paragraph_corpus(
        source_dir=tmp_path, granularity_level="proposition", use_cache=False
    )
    assert sorted(corpus.paragraph_ids) == ["pid-P_0", "pid-P_1"]

    meta = json.loads(corpus.metadata_jsons[corpus.paragraph_ids.index("pid-P_1")])
    assert meta["paragraph_id"] == pid
    assert meta["granularity_level"] == "proposition"
    assert meta["any_positive"] is True


def test_cache_isolation_by_granularity(tmp_path: Path) -> None:
    """Same source_dir loaded with paragraph vs sentence must use different cache files."""
    pid = "pid-X"
    # File has BOTH a content and a sentences field so the same source_dir
    # is parseable under both granularities (defensive — not the typical
    # production layout).
    _write(tmp_path, "0.json", [{
        "id": pid,
        "content": "Hello world.",
        "sentences": ["Hello world.", "Second sentence."],
        "event_id": 0,
        "source_query": "q",
        "classification": 1,
    }])
    cache = tmp_path / "_cache"

    para = load_paragraph_corpus(
        source_dir=tmp_path,
        granularity_level="paragraph",
        cache_dir=cache,
        use_cache=True,
    )
    sent = load_paragraph_corpus(
        source_dir=tmp_path,
        granularity_level="sentence",
        cache_dir=cache,
        use_cache=True,
    )
    assert len(para) == 1
    assert len(sent) == 2

    pkl_files = sorted(p.name for p in cache.glob("corpus_*.pkl"))
    # Two distinct cache pickles → granularity made it into the signature.
    assert len(pkl_files) == 2, f"expected 2 cache files, got {pkl_files}"
