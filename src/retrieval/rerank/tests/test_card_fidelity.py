"""Do the backends reproduce their own model cards? And are they batch-invariant?

Marked ``gpu``: these download real weights. Deselected by default
(``-m "not gpu"``); run explicitly with ``-m gpu``.

Two kinds of test, and the second is the one that earns its keep:

1. **Card fidelity.** Each backend must reproduce the example output printed on its own
   model card. Catches a mangled prompt template, a wrong token id, or a
   sigmoid applied where the card returns a raw logit.

2. **Batch invariance.** For any *pointwise* reranker, ``score(q, [d1..dn])`` must equal
   ``[score(q, [di]) for i]``: the score is a function of (query, document) alone.
   Scoring a document next to different neighbours cannot change it.

   Batch invariance is how a wrong ``padding_side`` gets caught. The decoder rerankers
   read ``logits[:, -1, :]``; with right padding that position is a pad token, so a
   batched call silently returns noise while a batch of one still looks fine. No golden
   value is needed and the test generalises to every backend — including any added later.
"""

from __future__ import annotations

import math

import pytest
from backends import build_reranker
from reranker_registry import DEFAULT_VARIANT, POINTWISE_ALIASES, spec_for

pytestmark = pytest.mark.gpu

# Documents of deliberately different lengths: with left/right padding confusion, an
# equal-length batch can accidentally pass. The padding must actually be exercised.
_DOCS = [
    "hi",
    "The giant panda is a bear species endemic to China. It is characterised by its "
    "bold black-and-white coat and rounded body, and it lives in a few mountain ranges "
    "in central China, mainly in Sichuan, but also in neighbouring Shaanxi and Gansu.",
    "Berlin is well known for its museums.",
]
_QUERY = "what is panda?"


def _instruction(spec):
    from reranker_registry import RerankQuery, instruction_for  # noqa: PLC0415

    q = RerankQuery(0, _QUERY, "is_a", "bear species", "Pandas")
    return instruction_for(spec, DEFAULT_VARIANT, q)


@pytest.mark.parametrize("alias", [a for a in POINTWISE_ALIASES if a != "colbertv2.0"])
def test_batch_invariance(alias):
    """A pointwise score must not depend on which documents share its batch.

    This is the padding-side canary. It is also the definition of "pointwise", which is
    the property the pooled score cache relies on.
    """
    spec = spec_for(alias)
    reranker = build_reranker(spec, device="cpu")
    instruction = _instruction(spec)

    batched = reranker.score(_QUERY, _DOCS, instruction=instruction)
    singles = [reranker.score(_QUERY, [d], instruction=instruction)[0] for d in _DOCS]

    for i, (b, s) in enumerate(zip(batched, singles)):
        assert math.isclose(b, s, rel_tol=1e-3, abs_tol=1e-3), (
            f"[{alias}] doc {i}: batched={b:.6f} vs single={s:.6f}. The score depends on the "
            f"batch — check tokenizer.padding_side (expected {spec.padding_side!r})."
        )


def test_ms_marco_reproduces_card_example():
    """Card: predict([...]) -> [9.218911, -4.0780287]. Raw logits, Identity activation."""
    reranker = build_reranker(spec_for("ms-marco-MiniLM-L12-v2"), device="cpu")
    query = "How many people live in Berlin?"
    docs = [
        "Berlin had a population of 3,520,031 registered inhabitants in an area of "
        "891.82 square kilometers.",
        "Berlin is well known for its museums.",
    ]
    scores = reranker.score(query, docs)

    assert scores[0] == pytest.approx(9.218911, abs=0.05)
    assert scores[1] == pytest.approx(-4.0780287, abs=0.05)
    assert scores[1] < 0, "a raw logit must be able to go negative; a sigmoid crept in"


def test_bge_m3_reproduces_card_example():
    """Card: ``compute_score(['query', 'passage'])`` -> ``-5.65234375``.

    Note the pair is the *literal* strings 'query' and 'passage' — the card's other
    snippet (``[['what is panda?', 'hi'], ...]``) prints no expected value. Cross-check:
    the card's ``normalize=True`` output ``0.003497010252573502`` is exactly
    ``sigmoid(-5.65234375)``, which pins this as a raw, unnormalised logit.
    """
    reranker = build_reranker(spec_for("bge-reranker-v2-m3"), device="cpu")
    scores = reranker.score("query", ["passage"])

    assert scores[0] == pytest.approx(-5.65234375, abs=0.05)
    assert 1.0 / (1.0 + math.exp(-scores[0])) == pytest.approx(0.003497010252573502, abs=1e-4)


def test_bge_m3_ranks_the_relevant_passage_higher():
    reranker = build_reranker(spec_for("bge-reranker-v2-m3"), device="cpu")
    scores = reranker.score("what is panda?", ["hi", _DOCS[1]])
    assert scores[1] > scores[0], "the panda passage must outrank 'hi'"


def test_qwen3_returns_unbounded_log_odds_not_a_saturating_probability():
    """We store log-odds `logit(yes) - logit(no)`, not the card's `P(yes)`.

    P(yes) = sigmoid(log-odds) is a strictly monotone transform, so the RANKING is
    identical -- and every metric here is rank-only. But P(yes) rounds to exactly 1.0 once
    the gap exceeds ~17, tying every confidently-relevant document; the stable sort would
    then hand the ranking back to the retriever. Verified against Qwen3-Reranker-0.6B:
    argsort(P(yes)) == argsort(log-odds), and the log-odds resolve differences the card's
    bf16 logits quantise away (max deviation 0.0606 = bf16 spacing at |logit| ~ 9).
    """
    spec = spec_for("Qwen3-Reranker-4B")
    reranker = build_reranker(spec, device="cpu")
    scores = reranker.score("what is panda?", ["hi", _DOCS[1]], instruction=_instruction(spec))

    assert scores[1] > scores[0], "the panda passage must outrank 'hi'"
    assert not all(0.0 <= s <= 1.0 for s in scores), (
        "scores look like probabilities; a sigmoid/softmax crept back in and will saturate"
    )


def test_bge_gemma_returns_an_unbounded_logit():
    """bge-gemma emits a RAW 'Yes' logit — no softmax, no yes/no contrast.

    If someone "helpfully" adds a sigmoid, this fails: the scale would collapse to [0,1]
    and stop matching what the registry documents.
    """
    spec = spec_for("bge-reranker-v2-gemma")
    reranker = build_reranker(spec, device="cpu")
    scores = reranker.score("what is panda?", ["hi", _DOCS[1]], instruction=_instruction(spec))

    assert scores[1] > scores[0]
    assert not all(0.0 <= s <= 1.0 for s in scores), (
        "scores look like probabilities; bge-gemma must return an unbounded raw logit"
    )


def test_instruction_required_where_the_slot_exists():
    """Instruction-aware models must refuse to score without one, rather than
    silently substituting an empty string and quietly changing the prompt."""
    spec = spec_for("Qwen3-Reranker-4B")
    reranker = build_reranker(spec, device="cpu")
    with pytest.raises(ValueError, match="instruction is required"):
        reranker.score("q", ["d"], instruction=None)
