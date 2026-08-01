"""Single source of truth for the reranker set: scoring contract per model.

Every field below was verified against the model's own HuggingFace card and
``config.json`` on 2026-07-08, not from memory. Where a card is silent (mxbai's
``max_length``) or where its example value is merely illustrative rather than a
requirement (bge-m3's ``max_length=512``), the deviation is called out in the
per-model comment.

Two paradigms, and the distinction drives the whole harness:

  * ``pointwise``  — ``s(q, d)`` is independent of the candidate set, so the
    deduplicated union pool (``all_candidates_<qid>.json``) can be scored ONCE
    per (query, reranker) and every retriever's reranked top-500 falls out as a
    sort of its own ids. 2.18M pairs instead of 9.9M.
  * ``listwise``   — ``s(q, d)`` depends on the other candidates, so the cache
    is unusable and the model must be run once per (reranker, retriever) at its
    native depth-100 protocol.

Score scales are mutually incomparable (P(yes) in [0,1]; a raw logit; a log-odds
difference). Rank-only metrics such as Precision@R are invariant to this, but
score fusion / thresholding / cross-model averaging are NOT valid and must never
be done on these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import _rerank_bootstrap  # noqa: F401  (sets sys.path so `config` imports)
from config import TOP_RERANKER_MODEL_IDS

# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #
# Document budget, in tokens. The v2 paragraphs are mean 475 / p90 659 / p99 778
# / max ~1006 tokens, so 1024 leaves >99.9% untruncated. Each model's
# ``max_length`` is this budget plus its own template + query overhead; queries
# are at most 28 tokens.
DOC_BUDGET_TOKENS = 1024
QUERY_BUDGET_TOKENS = 32

# Depth at which each paradigm reranks. Pointwise matches the labeled pool depth
# exactly (so P@R is exact for the 968 queries with R<=500). Listwise uses the
# depth its papers publish; rerank@100 for the pointwise models is derived from
# the same cache for an apples-to-apples comparison.
POINTWISE_DEPTH = 500
LISTWISE_DEPTH = 100

Paradigm = str   # "pointwise" | "listwise"
Backend = str    # see backends/ for the dispatch table
Variant = str    # "default" | "task_instruct"

DEFAULT_VARIANT: Variant = "default"
TASK_INSTRUCT_VARIANT: Variant = "task_instruct"


@dataclass(frozen=True)
class RerankQuery:
    """A v2 relational query plus the structure the task-instruct ablation names."""

    event_id: int
    sentence: str
    relation: str
    constant_entity: str
    abstract_entity_type: str


@dataclass(frozen=True)
class RerankerSpec:
    alias: str
    backend: Backend
    paradigm: Paradigm = "pointwise"
    # Total sequence budget (document + query + template). ``None`` means the
    # backend derives it (listwise models manage their own context).
    max_length: int | None = None
    doc_budget: int = DOC_BUDGET_TOKENS
    dtype: str = "bfloat16"
    # Tokenizer padding side. "left" is MANDATORY wherever the score is read
    # from ``logits[:, -1, :]`` — right padding reads a pad position and
    # silently corrupts every score without raising.
    padding_side: str = "right"
    venv: str = ".venv"
    # Supports an instruction slot -> eligible for the task_instruct ablation.
    instruction_aware: bool = False
    # Card-documented default instruction (None where the model has no slot).
    default_instruction: str | None = None
    # Non-document tokens the model wraps around each pair (special tokens, chat
    # template, instruction scaffold). Used ONLY to compute the truncation diagnostic,
    # never to build inputs. Over-estimating is safe (it over-reports truncation);
    # under-estimating silently hides it, so err high.
    template_overhead: int = 4
    # True only for deliberate controls whose scores are meant to be identical (the
    # identity backend). Everything else must produce a top-10 that it actually orders;
    # score_pool aborts otherwise, because a tied top-10 means the stable sort -- not the
    # reranker -- is deciding the ranking.
    expects_tied_scores: bool = False
    # Alarm ceiling for the fraction of paragraphs this model truncates, NOT a claim
    # about the observed rate. score_pool.py logs the true rate every run and warns when
    # it exceeds this. A non-zero value must correspond to a *known* architectural limit
    # (a hard context window), never to a mis-set max_length.
    expected_truncation: float = 0.0
    notes: str = ""

    @property
    def hf_id(self) -> str:
        try:
            return TOP_RERANKER_MODEL_IDS[self.alias]
        except KeyError:
            raise KeyError(
                f"{self.alias!r} has no HuggingFace id in config.TOP_RERANKER_MODEL_IDS. "
                f"Debug backends (e.g. 'identity-constant') load no model and must not "
                f"access .hf_id."
            ) from None

    @property
    def depth(self) -> int:
        return LISTWISE_DEPTH if self.paradigm == "listwise" else POINTWISE_DEPTH


# --------------------------------------------------------------------------- #
# The evaluated set: 8 pointwise + 2 listwise.
# All models have open weights. The backend interface stays pluggable so an API
# reranker (e.g. Cohere) could be added later with one adapter class.
# --------------------------------------------------------------------------- #
_QWEN3_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_BGE_GEMMA_DEFAULT_INSTRUCTION = (
    "Given a query A and a passage B, determine whether the passage contains an "
    "answer to the query by providing a prediction of either 'Yes' or 'No'."
)

RERANKER_SPECS: tuple[RerankerSpec, ...] = (
    # --- encoder cross-encoders -------------------------------------------------
    RerankerSpec(
        alias="ms-marco-MiniLM-L12-v2",
        backend="st_cross_encoder",
        max_length=512,          # HARD cap: BertForSequenceClassification, max_position_embeddings=512
        dtype="float32",         # 33M params; fp32 costs nothing and avoids bf16 score ties
        padding_side="right",
        # 512 total - query - specials leaves ~495 tokens for the document, so the longest
        # paragraphs lose their tail: 5.7% measured on a 3-query sample (406 / 7129 docs).
        # 0.15 is the alarm ceiling, not the claim; the true rate is logged every run.
        # A property of the model's position embeddings, reported as such — not an
        # artifact of this evaluation.
        expected_truncation=0.15,
        notes="raw logit; config.json pins sbert_ce_default_activation_function=Identity",
    ),
    RerankerSpec(
        alias="bge-reranker-v2-m3",
        backend="hf_seq_cls",
        max_length=1088,         # 1024 doc + 32 query + specials
        # float32, not float16. The score is a single logit off a Linear(1024 -> 1) head;
        # in fp16 it lands on a coarse grid and, measured on the real pool, only 82.6% of
        # the top-500 scores were distinct (distinct@10 was fine at 0.988). Ties are broken
        # by first-stage rank, which is biased TOWARD the retriever rather than neutral --
        # it makes reranking look less effective than it is. 568M in fp32 is 2.3 GB and
        # roughly doubles a 34-minute job. Cheap insurance for the headline metric.
        dtype="float32",
        padding_side="right",
        # The card's example uses max_length=512, but that is illustrative: the
        # bge-m3 backbone supports 8192 and long context is its headline feature.
        # 1088 is the faithful use for 475-token paragraphs.
        notes="raw logit; XLMRobertaForSequenceClassification, 568M",
    ),
    # --- decoder-LLM pointwise rerankers ----------------------------------------
    RerankerSpec(
        alias="mxbai-rerank-large-v2",
        backend="st_cross_encoder",
        max_length=1152,         # 1024 doc + 32 query + 84 chat-template overhead + slack
        # float32, NOT bfloat16. The score is a log-odds read straight off the model's
        # output, and bf16 carries ~8 mantissa bits: the card's own example prints
        # [12., 0.5, 11.125, 0.8125, ...] -- a ~0.06-0.125 grid. Over a top-500 whose
        # log-odds span ~[0, 12] that is ~150 bins for 500 documents, so hundreds of
        # documents tie and the stable sort silently returns first-stage order.
        # ms-marco (33M) is fp32 for the same reason and shows distinct@500 = 1.000.
        # Costs ~2-3x the bf16 runtime on a 1.5B model; worth it for a ranking metric.
        dtype="float32",
        padding_side="left",     # MANDATORY: score is read at the last position
        # Measured, not estimated: apply_chat_template on an empty (query, document) pair
        # tokenizes to exactly 84 tokens. Assuming the BERT-ish default of 4 here would
        # make the truncation diagnostic silently UNDER-report. On the current corpus this
        # changes nothing (longest paragraph = 917 tokens vs 1054 available, 137 headroom),
        # but the diagnostic must be right *before* a longer paragraph arrives, not after.
        template_overhead=84,
        # The card states no max_length ("long-context support"); sentence_bert_config.json
        # has no max_seq_length key, so nothing is truncated by default. We set it.
        notes="Qwen2ForCausalLM 1.54B (ProRank-1.5B); score = logit('1') - logit('0'), "
              "token ids 16/15 per 1_LogitScore/config.json. Needs sentence-transformers>=5.4.0",
    ),
    RerankerSpec(
        alias="Qwen3-Reranker-4B",
        backend="hf_causal_yesno",
        max_length=1152,         # 1024 doc + 32 query + prefix/suffix/instruct overhead
        dtype="bfloat16",
        padding_side="left",     # MANDATORY
        instruction_aware=True,
        default_instruction=_QWEN3_DEFAULT_INSTRUCTION,
        notes="Qwen3ForCausalLM 4.0B; score = log-odds logit(yes)-logit(no), computed in fp32 "
              "from the last hidden state (h . (w_yes - w_no)). Strictly monotone in the "
              "card's P(yes), so rankings match, but without its saturation at 0.0/1.0 or "
              "the bf16 logit grid. Score with transformers, NOT vLLM (vllm#21681)",
    ),
    RerankerSpec(
        alias="Qwen3-Reranker-8B",
        backend="hf_causal_yesno",
        max_length=1152,
        dtype="bfloat16",
        padding_side="left",     # MANDATORY
        instruction_aware=True,
        default_instruction=_QWEN3_DEFAULT_INSTRUCTION,
        notes="Qwen3ForCausalLM 8.2B; same contract as the 4B. The scale ceiling of the study",
    ),
    RerankerSpec(
        alias="bge-reranker-v2-gemma",
        backend="hf_causal_yes_logit",
        max_length=1088,         # pair budget; the prompt suffix is appended after truncation
        dtype="bfloat16",
        padding_side="left",     # GemmaTokenizer class default; asserted, never overridden
        instruction_aware=True,
        default_instruction=_BGE_GEMMA_DEFAULT_INSTRUCTION,
        notes="GemmaForCausalLM 2.5B; score = RAW 'Yes' logit (capital Y), no softmax, computed "
              "in fp32 as h . w_yes rather than read off the bf16 logits. Instruction is a "
              "SUFFIX after the passage. Query capped at 3/4*L, doc truncated",
    ),
    # --- late interaction --------------------------------------------------------
    RerankerSpec(
        alias="colbertv2.0",
        # Used as pylate's `document_length`. 512 is ColBERT's BERT-backbone ceiling, so
        # min(ceiling, 1024) = 512 — and it matches index_colbert.py's --document-length,
        # keeping colbert's reranker arm consistent with its own retrieval arm.
        max_length=512,
        backend="pylate_colbert",
        doc_budget=DOC_BUDGET_TOKENS,
        dtype="float32",
        venv=".venv-colbert",    # pylate pins transformers<=4.56.2; cannot share .venv
        expected_truncation=0.15,  # same 512-token ceiling as ms-marco-MiniLM
        notes="110M ColBERT MaxSim. Set-independent => shares the pooled score cache",
    ),
    RerankerSpec(
        alias="GTE-ModernColBERT-v1",
        # pylate `document_length`. ModernBERT's backbone handles 8K context, so unlike
        # colbertv2.0's hard 512 cap this sees the full paragraph at the 1024 doc budget --
        # hence expected_truncation=0.0. Late interaction over ModernBERT rather than BERT.
        max_length=DOC_BUDGET_TOKENS,   # 1024
        backend="pylate_colbert",
        doc_budget=DOC_BUDGET_TOKENS,
        dtype="float32",
        venv=".venv-colbert",    # same pylate stack as colbertv2.0 (transformers<=4.56.2 has ModernBERT)
        expected_truncation=0.0,
        notes="LightOn GTE-ModernColBERT (128-dim ColBERT MaxSim via pylate); ModernBERT 8K "
              "context => no forced truncation. Set-independent => shares the pooled cache",
    ),
    # --- listwise ----------------------------------------------------------------
    RerankerSpec(
        alias="jina-reranker-v3",
        backend="jina_v3",
        paradigm="listwise",
        max_length=None,         # 131k context; rerank() takes max_doc_length instead
        dtype="bfloat16",
        notes="0.6B 'last-but-not-late'. block_size=125, so depth-100 is ONE forward pass "
              "(no cross-block score gluing). .rerank() only — compute_score() does not exist. "
              "Causal attention => scores depend on doc order; we feed retriever rank order",
    ),
    RerankerSpec(
        alias="rank_zephyr_7b_v1_full",
        backend="rank_zephyr",
        paradigm="listwise",
        max_length=None,         # rank_llm owns the 4096 context_size
        dtype="bfloat16",
        venv=".venv-rankllm",    # needs vLLM + py3.11; rank_llm@main has no HF-generate fallback
        # context_size=4096 vs 20 passages x ~475 tokens: rank_llm shrinks passages to
        # ~190 tokens, so the model sees roughly 40% of each paragraph. This is a
        # property of the model and is REPORTED, not hidden.
        expected_truncation=1.0,
        notes="7B; sliding window 20 / stride 10 => 9 sequential windows at depth 100. "
              "Emits a permutation, no scores. num_passes=1 (rho=3 buys +0.004 nDCG for 3x cost)",
    ),
)

# Not a model. `identity-constant` scores every document identically, so the stable
# sort in `load_reranked_list` must leave the retriever's order untouched and the whole
# pipeline must reproduce the published first-stage leaderboard. Kept out of
# RERANKER_ALIASES so it never lands in a results table, but constructible by alias.
DEBUG_SPECS: tuple[RerankerSpec, ...] = (
    RerankerSpec(
        alias="identity-constant",
        backend="identity",
        dtype="float32",
        expects_tied_scores=True,  # every score is 0.0 on purpose; exempt from the gate
        notes="regression harness for the join/sort/tiebreak path; loads no model",
    ),
)

SPEC_BY_ALIAS: dict[str, RerankerSpec] = {s.alias: s for s in (*RERANKER_SPECS, *DEBUG_SPECS)}
RERANKER_ALIASES: tuple[str, ...] = tuple(s.alias for s in RERANKER_SPECS)
POINTWISE_ALIASES: tuple[str, ...] = tuple(
    s.alias for s in RERANKER_SPECS if s.paradigm == "pointwise"
)
LISTWISE_ALIASES: tuple[str, ...] = tuple(
    s.alias for s in RERANKER_SPECS if s.paradigm == "listwise"
)
# Eligible for the task-tailored instruction ablation.
INSTRUCTION_AWARE_ALIASES: tuple[str, ...] = tuple(
    s.alias for s in RERANKER_SPECS if s.instruction_aware
)


def spec_for(alias: str) -> RerankerSpec:
    try:
        return SPEC_BY_ALIAS[alias]
    except KeyError:
        raise KeyError(
            f"Unknown reranker alias {alias!r}. Known: {', '.join(RERANKER_ALIASES)}"
        ) from None


# --------------------------------------------------------------------------- #
# Instructions
# --------------------------------------------------------------------------- #
def task_instruction(query: RerankQuery) -> str:
    """The per-query instruction for the ``task_instruct`` ablation.

    Names the three components a relational query decomposes into, filled in from
    ``queries.json``. This is deliberately the *strongest* form of the
    reviewer's objection: the reranker is told the abstract entity type, the exact
    relation, and the constant entity, rather than being left to infer them from
    the query sentence. If P@R still does not move, "the reranker didn't know what
    to look for" is no longer available as an explanation.
    """
    return (
        f"Retrieve passages in which an instance of the entity type "
        f"'{query.abstract_entity_type}' stands in exactly the relation "
        f"'{query.relation}' to '{query.constant_entity}'. A passage that names a "
        f"similar but different relation, or a different entity, is not relevant."
    )


def instruction_for(spec: RerankerSpec, variant: Variant, query: RerankQuery) -> str | None:
    """Resolve the instruction string for a (model, variant, query).

    Returns ``None`` for models with no instruction slot. The ``task_instruct``
    variant is only defined for instruction-aware models; requesting it for any
    other model is a programming error, not a silent fallback.
    """
    if not spec.instruction_aware:
        if variant != DEFAULT_VARIANT:
            raise ValueError(
                f"{spec.alias!r} has no instruction slot; variant {variant!r} is undefined for it."
            )
        return None
    if variant == DEFAULT_VARIANT:
        return spec.default_instruction
    if variant == TASK_INSTRUCT_VARIANT:
        return task_instruction(query)
    raise ValueError(f"Unknown variant {variant!r} for {spec.alias!r}.")


def variants_for(alias: str) -> tuple[Variant, ...]:
    """Variants that are meaningful for this reranker."""
    spec = spec_for(alias)
    if spec.instruction_aware:
        return (DEFAULT_VARIANT, TASK_INSTRUCT_VARIANT)
    return (DEFAULT_VARIANT,)
