"""Registry self-consistency. No models, no disk — pure invariants.

These catch a mis-declared spec before any GPU time is spent on it. The
``padding_side`` invariant is the important one: every backend that reads
``logits[:, -1, :]`` must declare left padding, and getting it wrong produces
plausible-looking noise rather than an error.
"""

from __future__ import annotations

import pytest
import reranker_registry as rr

# Backends whose score is read at the final sequence position. For these, right
# padding puts a pad token there and silently corrupts every score.
LAST_POSITION_BACKENDS = {"hf_causal_yesno", "hf_causal_yes_logit"}


@pytest.mark.unit
def test_expected_set_of_rerankers():
    """10 evaluated models: 8 pointwise + 2 listwise. All open-weight (no Cohere API)."""
    assert len(rr.RERANKER_ALIASES) == 10
    assert len(rr.POINTWISE_ALIASES) == 8
    assert len(rr.LISTWISE_ALIASES) == 2
    assert "cohere-rerank-v3.5" not in rr.RERANKER_ALIASES
    assert "GTE-ModernColBERT-v1" in rr.POINTWISE_ALIASES


@pytest.mark.unit
@pytest.mark.parametrize("alias", rr.RERANKER_ALIASES)
def test_hf_id_resolves(alias):
    assert rr.spec_for(alias).hf_id


@pytest.mark.unit
@pytest.mark.parametrize("alias", rr.RERANKER_ALIASES)
def test_last_position_backends_declare_left_padding(alias):
    spec = rr.spec_for(alias)
    if spec.backend in LAST_POSITION_BACKENDS:
        assert spec.padding_side == "left", (
            f"{alias} reads logits[:, -1, :] but declares padding_side="
            f"{spec.padding_side!r}; right padding would read a pad position."
        )


@pytest.mark.unit
def test_mxbai_declares_left_padding_despite_the_encoder_sounding_name():
    """mxbai-rerank-large-v2 is a Qwen2 causal LM, not an encoder cross-encoder.

    The original config.py comment grouped it with the encoders. Its score is
    logit('1') - logit('0') at the last position, so left padding is mandatory.
    """
    spec = rr.spec_for("mxbai-rerank-large-v2")
    assert spec.padding_side == "left"


@pytest.mark.unit
@pytest.mark.parametrize("alias", rr.RERANKER_ALIASES)
def test_instruction_awareness_is_coherent(alias):
    spec = rr.spec_for(alias)
    if spec.instruction_aware:
        assert spec.default_instruction, f"{alias} is instruction-aware but has no default"
        assert rr.variants_for(alias) == (rr.DEFAULT_VARIANT, rr.TASK_INSTRUCT_VARIANT)
    else:
        assert spec.default_instruction is None
        assert rr.variants_for(alias) == (rr.DEFAULT_VARIANT,)


@pytest.mark.unit
def test_only_the_three_documented_models_are_instruction_aware():
    assert set(rr.INSTRUCTION_AWARE_ALIASES) == {
        "Qwen3-Reranker-4B", "Qwen3-Reranker-8B", "bge-reranker-v2-gemma",
    }


@pytest.mark.unit
@pytest.mark.parametrize("alias", rr.POINTWISE_ALIASES)
def test_pointwise_models_declare_a_max_length(alias):
    """Nothing may rely on a tokenizer's own default: mxbai's card sets none, and
    bge-m3 would otherwise inherit model_max_length=8192."""
    assert rr.spec_for(alias).max_length is not None


@pytest.mark.unit
@pytest.mark.parametrize("alias", rr.RERANKER_ALIASES)
def test_depth_matches_paradigm(alias):
    spec = rr.spec_for(alias)
    expected = rr.LISTWISE_DEPTH if spec.paradigm == "listwise" else rr.POINTWISE_DEPTH
    assert spec.depth == expected


@pytest.mark.unit
def test_isolated_venvs_are_declared_for_the_conflicting_backends():
    """pylate pins transformers<=4.56.2; rank_llm needs vLLM + py3.11. Neither can
    share the main venv, and the backend must say so before it fails to import."""
    assert rr.spec_for("colbertv2.0").venv == ".venv-colbert"
    assert rr.spec_for("GTE-ModernColBERT-v1").venv == ".venv-colbert"  # same pylate stack
    assert rr.spec_for("rank_zephyr_7b_v1_full").venv == ".venv-rankllm"
    pylate = {"colbertv2.0", "GTE-ModernColBERT-v1"}
    for alias in set(rr.RERANKER_ALIASES) - pylate - {"rank_zephyr_7b_v1_full"}:
        assert rr.spec_for(alias).venv == ".venv"


@pytest.mark.unit
def test_template_overhead_is_declared_for_the_chat_templated_model():
    """mxbai wraps every pair in an 84-token chat template (measured via
    apply_chat_template on an empty pair). Using the BERT-ish default of 4 would make the
    truncation diagnostic silently UNDER-report, since `available` would be too large.

    ms-marco wraps the pair in 3 BERT special tokens, so the default 4 is a conservative
    +1 overestimate — the safe direction.
    """
    assert rr.spec_for("mxbai-rerank-large-v2").template_overhead == 84
    assert rr.spec_for("ms-marco-MiniLM-L12-v2").template_overhead == 4
    # XLM-R: <s> q </s></s> d </s>
    assert rr.spec_for("bge-reranker-v2-m3").template_overhead == 4


@pytest.mark.unit
def test_listwise_depth_has_a_single_definition():
    """eval_config imports LISTWISE_DEPTH rather than redeclaring it, so the two cannot
    drift and mislabel a results column."""
    import eval_config  # noqa: PLC0415

    assert eval_config.LISTWISE_RERANK_DEPTH is rr.LISTWISE_DEPTH


@pytest.mark.unit
def test_models_that_must_truncate_declare_it():
    """No silent caps. A model architecturally forced to cut documents says so, and
    score_pool warns when the observed rate exceeds the declaration."""
    assert rr.spec_for("ms-marco-MiniLM-L12-v2").expected_truncation > 0  # hard 512 cap
    assert rr.spec_for("colbertv2.0").expected_truncation > 0             # hard 512 cap
    assert rr.spec_for("rank_zephyr_7b_v1_full").expected_truncation == 1.0  # 4096 context
    # The long-context models must NOT be truncating at a 1024-token doc budget.
    # GTE-ModernColBERT (ModernBERT 8K) is here despite being late-interaction, unlike
    # colbertv2.0 whose BERT backbone forces a 512 cap.
    for alias in ("bge-reranker-v2-m3", "mxbai-rerank-large-v2",
                  "Qwen3-Reranker-4B", "Qwen3-Reranker-8B", "bge-reranker-v2-gemma",
                  "GTE-ModernColBERT-v1"):
        assert rr.spec_for(alias).expected_truncation == 0.0


@pytest.mark.unit
def test_task_instruction_names_all_three_query_components():
    """The ablation's whole point is that the reranker is told exactly what to look for."""
    q = rr.RerankQuery(0, "Universities participate in the Fulbright Program.",
                       "participates_in", "the Fulbright Program", "Universities")
    text = rr.task_instruction(q)
    assert q.abstract_entity_type in text
    assert q.relation in text
    assert q.constant_entity in text


@pytest.mark.unit
def test_task_instruct_variant_is_rejected_for_models_without_a_slot():
    q = rr.RerankQuery(0, "s", "r", "c", "a")
    with pytest.raises(ValueError, match="no instruction slot"):
        rr.instruction_for(rr.spec_for("bge-reranker-v2-m3"), rr.TASK_INSTRUCT_VARIANT, q)


@pytest.mark.unit
def test_debug_spec_is_not_in_the_evaluated_set():
    """identity-constant must be constructible but never appear in a results table."""
    assert rr.spec_for("identity-constant").backend == "identity"
    assert "identity-constant" not in rr.RERANKER_ALIASES
    with pytest.raises(KeyError, match="load no model"):
        _ = rr.spec_for("identity-constant").hf_id


# --------------------------------------------------------------------------- #
# Cache manifest: resume must not silently mix scoring configurations
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_manifest_rejects_a_changed_scoring_config(tmp_path, monkeypatch):
    """Resuming a CPU/fp32 cache on GPU/bf16 must fail loudly, not interleave scores."""
    import score_store  # noqa: PLC0415

    monkeypatch.setattr(score_store, "FRONTIER_RERANK_SCORES_PATH", tmp_path)
    cfg = {"reranker": "m", "variant": "default", "dtype": "float16",
           "device_kind": "cpu", "tf32": False, "max_length": 1088, "padding_side": "right"}

    score_store.check_or_write_manifest("m", "default", cfg)   # first write
    score_store.check_or_write_manifest("m", "default", cfg)   # identical -> no-op

    with pytest.raises(ValueError, match="different .*configuration"):
        score_store.check_or_write_manifest("m", "default", {**cfg, "device_kind": "cuda"})

    with pytest.raises(ValueError, match="different .*configuration"):
        score_store.check_or_write_manifest("m", "default", {**cfg, "max_length": 512})

    # TF32 rounds fp32 matmul inputs, so its scores are not bit-identical to strict fp32
    with pytest.raises(ValueError, match="different .*configuration"):
        score_store.check_or_write_manifest("m", "default", {**cfg, "tf32": True})
