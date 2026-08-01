"""Length bucketing, OOM backoff, and scatter-back alignment.

Every pointwise backend sorts documents by length before batching (so a batch pads to its
own longest member rather than the global maximum), then scatters the scores back into the
caller's order. A bug anywhere in that path silently attaches scores to the wrong
documents — every metric would still compute, and every number would be wrong.

Randomized rather than example-based, because the failure depends on the interaction of
budget, batch cap, and length distribution.
"""

from __future__ import annotations

import random

import pytest
from backends.base import length_bucketed_batches, score_with_oom_backoff

TRIALS = 500


@pytest.mark.unit
def test_bucketing_yields_every_document_exactly_once():
    rng = random.Random(20260708)
    for _ in range(TRIALS):
        n = rng.randint(0, 40)
        docs = ["x" * rng.randint(1, 4000) for _ in range(n)]
        budget = rng.choice([50, 500, 5000, 100_000])
        max_batch = rng.choice([1, 3, 8, 512])
        # The decoder backends pass their prompt/template size here. Exercised because a
        # non-zero overhead is exactly the case that went untested when it was dead code.
        overhead = rng.choice([0, 4, 84, 150])

        seen: list[int] = []
        for idx, batch_docs in length_bucketed_batches(docs, budget, max_batch, overhead):
            assert len(idx) == len(batch_docs)
            assert len(idx) <= max_batch
            for j, i in enumerate(idx):
                assert batch_docs[j] is docs[i], "batch_docs[j] must be docs[idx[j]]"
            seen.extend(idx)

        assert sorted(seen) == list(range(n)), "a document was dropped or duplicated"


@pytest.mark.unit
def test_per_doc_overhead_shrinks_batches():
    """The prompt scaffold rides along with every document, so counting it must make
    batches smaller — not be silently ignored, as it was before."""
    docs = ["x" * 370] * 20  # ~100 estimated tokens each

    without = list(length_bucketed_batches(docs, token_budget=1000, max_batch_size=512))
    with_ = list(length_bucketed_batches(docs, token_budget=1000, max_batch_size=512,
                                         per_doc_overhead=100))

    assert max(len(i) for i, _ in without) > max(len(i) for i, _ in with_)
    # and neither drops anything
    assert sorted(i for b, _ in without for i in b) == list(range(20))
    assert sorted(i for b, _ in with_ for i in b) == list(range(20))


@pytest.mark.unit
def test_a_single_oversized_document_is_still_emitted():
    """One doc larger than the whole token budget must be yielded alone, not dropped
    and not looped on forever."""
    batches = list(length_bucketed_batches(["y" * 100_000], token_budget=10, max_batch_size=8))
    assert len(batches) == 1
    assert batches[0][0] == [0]


@pytest.mark.unit
def test_empty_input_yields_nothing():
    assert list(length_bucketed_batches([], 1000, 8)) == []


@pytest.mark.unit
def test_oom_backoff_output_aligns_with_its_docs_argument():
    """score_with_oom_backoff re-chunks the whole bucket on OOM. Whatever chunk size it
    settles on, the returned list must line up with `docs` positionally."""
    docs = [f"d{i}" for i in range(37)]
    truth = {d: float(i) for i, d in enumerate(docs)}

    for chunk in (1, 5, 37, 100):
        out = score_with_oom_backoff(lambda ds: [truth[d] for d in ds], docs, chunk, "t")
        assert out == [float(i) for i in range(37)], f"misaligned at chunk={chunk}"


@pytest.mark.unit
def test_bucketing_then_scatter_reconstructs_input_order():
    """The exact pattern every backend uses. A pointwise score is a pure function of the
    document, so the reconstructed list must equal the unbatched one."""
    rng = random.Random(11)
    docs = [f"d{i}" + "z" * rng.randint(1, 500) for i in range(37)]
    truth = {d: float(i) for i, d in enumerate(docs)}

    scores: list[float | None] = [None] * len(docs)
    for idx, batch_docs in length_bucketed_batches(docs, 2000, 8):
        got = score_with_oom_backoff(
            lambda ds: [truth[d] for d in ds], batch_docs, len(batch_docs), "t"
        )
        for i, s in zip(idx, got):
            scores[i] = s

    assert scores == [float(i) for i in range(37)]


@pytest.mark.unit
def test_non_oom_errors_propagate():
    """Only CUDA OOM triggers a retry. A real bug must not be retried into oblivion."""
    def boom(_ds):
        raise RuntimeError("something else went wrong")

    with pytest.raises(RuntimeError, match="something else"):
        score_with_oom_backoff(boom, ["a", "b"], 2, "t")
