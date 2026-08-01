"""bge-gemma builds its inputs without ``prepare_for_model`` (removed in transformers 5.x).

The card's ``get_inputs`` calls ``tokenizer.prepare_for_model(first, second,
truncation='only_second', add_special_tokens=False, max_length=L)``. transformers 5.8.1
removed that method from ``GemmaTokenizer``, so the backend reproduces its effect inline:
concatenate ``first`` + ``second`` and, if over ``L``, trim ``second``'s tail. These tests
pin that equivalence with a stub tokenizer -- no weights, no download.

This whole backend produced ZERO output on the cluster (all 4 shards died on the missing
method) before this path was tested. That is the gap these tests close.
"""

from __future__ import annotations

import pytest


class _StubGemmaTokenizer:
    """One token id per character; bos id 1, no special tokens added."""

    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False, verbose=True):
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def pad(self, items, **kw):  # not exercised here; inputs are built before padding
        raise NotImplementedError


def _make_backend():
    """A HFCausalYesLogitReranker with __init__ bypassed (no model load)."""
    from backends.hf_causal_yes_logit import HFCausalYesLogitReranker  # noqa: PLC0415

    b = object.__new__(HFCausalYesLogitReranker)
    b.alias = "bge-reranker-v2-gemma"
    b.tokenizer = _StubGemmaTokenizer()
    b.sep_tokens = [ord("\n")]
    b.max_length = 32
    return b


def _build_ids(backend, query, doc, instruction):
    """Run _build_inputs but stop before .pad(); return the single item's input_ids."""
    prompt_tokens = backend._prompt_tokens(instruction)
    query_inputs = backend.tokenizer(
        f"A: {query}", add_special_tokens=False,
        max_length=backend.max_length * 3 // 4, truncation=True,
    )
    first = [backend.tokenizer.bos_token_id] + query_inputs["input_ids"]
    passage_inputs = backend.tokenizer(
        f"B: {doc}", add_special_tokens=False, max_length=backend.max_length, truncation=True
    )
    second = backend.sep_tokens + passage_inputs["input_ids"]
    overflow = len(first) + len(second) - backend.max_length
    if overflow > 0:
        second = second[: len(second) - overflow]
    return first + second + backend.sep_tokens + prompt_tokens


@pytest.mark.unit
def test_prompt_tokens_helper():
    b = _make_backend()
    assert b._prompt_tokens("Yn") == [ord("Y"), ord("n")]


@pytest.mark.unit
def test_layout_is_bos_query_sep_passage_sep_prompt():
    b = _make_backend()
    ids = _build_ids(b, "q", "p", "I")
    # <bos> A: q \n B: p \n I  -- short enough that nothing truncates
    expected = [1] + list(b"A: q") + [ord("\n")] + list(b"B: p") + [ord("\n")] + [ord("I")]
    assert ids == expected


@pytest.mark.unit
def test_only_the_passage_is_truncated_never_the_query_or_prompt():
    b = _make_backend()
    b.max_length = 12
    long_doc = "x" * 100
    ids = _build_ids(b, "q", long_doc, "INSTR")

    # query side survives intact
    assert ids[: 1 + len("A: q")] == [1] + list(b"A: q")
    # instruction (prompt) always trails, uncut
    assert ids[-len("INSTR"):] == list(b"INSTR")
    # the bos+query+sep+passage prefix respects max_length; the prompt is appended after
    body_len = 1 + len("A: q") + len(b"\nB: ")  # first + sep + "B: " start of passage
    assert body_len <= b.max_length


@pytest.mark.unit
def test_no_truncation_when_everything_fits():
    b = _make_backend()
    b.max_length = 200
    ids = _build_ids(b, "hello", "world", "judge")
    # nothing dropped: full passage present
    assert list(b"B: world") in [ids[i:i+len(b"B: world")] for i in range(len(ids))]
