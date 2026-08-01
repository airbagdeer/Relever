"""Every backend must actually measure truncation, not assume it.

`base.TruncationCounter`'s whole purpose is that "a model that silently truncates is a
model whose numbers mean something different from what the table claims". A backend that
reports a hardcoded 0 defeats that for itself while looking compliant — which is precisely
what `listwise_jina_v3` did before this test existed.

Stub tokenizers, so no weights are downloaded.
"""

from __future__ import annotations

import pytest


class _StubTokenizer:
    """One token per character, so token length == len(text)."""

    def __call__(self, texts, add_special_tokens=False, verbose=True, **kw):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[0] * len(t) for t in texts]}


@pytest.mark.unit
def test_jina_v3_counts_truncated_documents():
    """It used to `observe(n, 0)` unconditionally, so its truncation rate was a fiction."""
    from backends.listwise_jina_v3 import MAX_DOC_LENGTH, JinaV3Reranker  # noqa: PLC0415

    rr = object.__new__(JinaV3Reranker)   # skip __init__: no model, no weights
    rr.tokenizer = _StubTokenizer()

    docs = ["a" * (MAX_DOC_LENGTH - 1), "b" * MAX_DOC_LENGTH, "c" * (MAX_DOC_LENGTH + 1)]
    # strictly greater than the cap counts as truncated
    assert rr._count_truncated(docs) == 1
    assert rr._count_truncated([]) == 0
    assert rr._count_truncated(["z" * (MAX_DOC_LENGTH + 500)] * 4) == 4


@pytest.mark.unit
def test_jina_v3_declares_no_truncation_and_must_therefore_measure_it():
    """The registry claims jina-v3 truncates nothing at a 1024-token document budget.
    That claim is only meaningful because the backend now checks it every run."""
    import reranker_registry as rr  # noqa: PLC0415
    from backends.listwise_jina_v3 import MAX_DOC_LENGTH  # noqa: PLC0415

    assert rr.spec_for("jina-reranker-v3").expected_truncation == 0.0
    assert MAX_DOC_LENGTH == rr.DOC_BUDGET_TOKENS


@pytest.mark.unit
def test_truncation_counter_rate_and_accumulation():
    from backends.base import TruncationCounter  # noqa: PLC0415

    c = TruncationCounter()
    assert c.rate == 0.0          # no division by zero on an empty run
    c.observe(100, 10)
    c.observe(100, 0)
    assert c.n_docs == 200
    assert c.n_truncated == 10
    assert c.rate == pytest.approx(0.05)
