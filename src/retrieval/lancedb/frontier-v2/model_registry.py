"""Single source of truth for the frontier-v2 model set.

Provides:
  - ``resolve_model_id(alias)``: union lookup across the three model dicts in
    ``config.py`` (MODEL_IDS, TOP_ENGLISH_MTEB_MODEL_IDS, TOP_ENGLISH_RTEB_MODEL_IDS).
    Most frontier-v2 aliases live ONLY in the two TOP_ENGLISH_* dicts, so the
    minimax ``_resolve_model_id`` (which only consults MODEL_IDS) cannot resolve
    them — this union does.
  - ``ModelSpec`` + ``MODEL_SPECS``: per-model inference + disk-routing config.
  - ``spec_for(alias)`` / ``root_for(alias, override)``: lookups used by the
    indexers.
  - Curated alias lists for the orchestrator (the deduplicated 18-model set
    split by backend).

Design facts (from the per-model research, 2026-06-04):
  - Nearly all dense models are decoder-LLM embedders (Qwen3 / Qwen2.5 / Mistral)
    with last-token / EOS / mean pooling + L2 norm. DOCUMENTS are encoded as
    PLAIN TEXT; any instruction applies only to queries (handled at eval time).
  - sentence-transformers reads each model's pooling from its own config, so a
    single generic ST indexer covers them with only per-model knobs (dtype,
    left padding for last-token pooling, optional doc prefix / task prompt).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import _bootstrap  # noqa: F401  (sets sys.path so `config` imports)
from config import (
    LANCEDB_PATH,
    LANCEDB_SCRATCH_PATH,
    MODEL_IDS,
    TOP_ENGLISH_MTEB_MODEL_IDS,
    TOP_ENGLISH_RTEB_MODEL_IDS,
)

# ---------------------------------------------------------------------------
# Alias -> HuggingFace id resolution (union, first-wins).
# ---------------------------------------------------------------------------
_ALIAS_SOURCES = (MODEL_IDS, TOP_ENGLISH_MTEB_MODEL_IDS, TOP_ENGLISH_RTEB_MODEL_IDS)


def resolve_model_id(alias: str) -> str:
    for d in _ALIAS_SOURCES:
        if alias in d:
            return d[alias]
    raise KeyError(
        f"Unknown model alias {alias!r}. Not found in MODEL_IDS / "
        f"TOP_ENGLISH_MTEB_MODEL_IDS / TOP_ENGLISH_RTEB_MODEL_IDS "
        f"(src/retrieval/config.py)."
    )


# ---------------------------------------------------------------------------
# Per-model inference + disk-routing spec.
# ---------------------------------------------------------------------------
Backend = str  # "st" | "vllm" | "pylate" | "jina_v4_hf"
DiskTarget = str  # "home" | "scratch"

# Tokenizer kwargs for decoder-LLM embedders that pool the last / EOS token:
# left padding keeps the final real token at the sequence end (right padding
# would put pad tokens there). Recommended by the Qwen3-Embedding card and
# applies to all its derivatives (F2LLM, Octen, MoD, ICT) + Mistral (LGAI).
_LEFT_PAD = {"padding_side": "left"}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    backend: Backend = "st"
    init_batch: int = 256
    multiprocess_ok: bool = True
    trust_remote_code: bool = True
    dtype: str | None = None  # e.g. "bfloat16"; None -> ST/library default
    doc_prompt: str | None = None  # prepended to each DOCUMENT text at encode time
    # Query-ONLY prompt overrides (read by retrieve_dense._query_encode_kwargs, never by the
    # document/index path). Use these when a model's query instruction is NOT discoverable by
    # the generic auto-detect: query_prompt_name selects a non-standard key in model.prompts
    # (e.g. jasper's 's2p_query'); query_prompt passes a literal instruction string for repos
    # that ship no prompts config at all (e.g. LGAI). At most one should be set.
    query_prompt_name: str | None = None  # -> encode(prompt_name=...)
    query_prompt: str | None = None  # -> encode(prompt=...)
    encode_kwargs: dict = field(default_factory=dict)  # e.g. task=, prompt_name= (SHARED doc+query)
    model_kwargs: dict = field(default_factory=dict)  # extra SentenceTransformer model_kwargs
    config_kwargs: dict = field(default_factory=dict)  # extra SentenceTransformer config_kwargs
    tokenizer_kwargs: dict = field(default_factory=dict)
    disk_target: DiskTarget = "scratch"
    notes: str = ""


def _llm_dense(alias: str, init_batch: int, *, multiprocess_ok: bool = True,
               doc_prompt: str | None = None, query_prompt: str | None = None,
               disk_target: str = "scratch") -> ModelSpec:
    """Helper for the common Qwen3/Qwen2.5/Mistral decoder-LLM dense embedder."""
    return ModelSpec(
        alias=alias,
        backend="st",
        init_batch=init_batch,
        multiprocess_ok=multiprocess_ok,
        trust_remote_code=True,
        dtype="bfloat16",
        doc_prompt=doc_prompt,
        query_prompt=query_prompt,
        tokenizer_kwargs=dict(_LEFT_PAD),
        disk_target=disk_target,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    # ---- vLLM (official, low-risk embed/pooling support) ----------------
    "Qwen3-Embedding-8B": ModelSpec(
        alias="Qwen3-Embedding-8B", backend="vllm", init_batch=8192, dtype="bfloat16",
        disk_target="scratch", notes="vLLM runner=pooling; last-token; plain-text docs.",
    ),
    "Qwen3-Embedding-4B": ModelSpec(
        alias="Qwen3-Embedding-4B", backend="vllm", init_batch=8192, dtype="bfloat16",
        disk_target="scratch", notes="vLLM runner=pooling; last-token; plain-text docs.",
    ),
    # ---- sentence-transformers dense -----------------------------------
    "F2LLM-v2-14B": _llm_dense("F2LLM-v2-14B", 32),
    "F2LLM-v2-8B": _llm_dense("F2LLM-v2-8B", 64),
    "F2LLM-v2-4B": _llm_dense("F2LLM-v2-4B", 128),
    # Octen-8B: encode docs via the model's own 'document' prompt (auto-detected
    # in _dense_core), not a hand-rolled "- " prefix.
    "Octen-Embedding-8B": _llm_dense("Octen-Embedding-8B", 64),
    "Octen-Embedding-4B": _llm_dense("Octen-Embedding-4B", 128),
    "Octen-Embedding-0.6B": _llm_dense("Octen-Embedding-0.6B", 1024),
    "QZhou-Embedding": _llm_dense("QZhou-Embedding", 64),  # 7B Qwen2.5, mean pool
    # LGAI ships NO config_sentence_transformers.json, so the generic query auto-detect finds
    # nothing and queries would go in bare. It's an instruction-tuned Mistral-7B (E5-Mistral
    # lineage), so pin the standard E5/Qwen retrieval instruction on the query side only
    # (documents stay plain). Keeps it comparable to the other in-config instruction models.
    "LGAI-Embedding-Preview": _llm_dense(
        "LGAI-Embedding-Preview", 64,
        query_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    ),  # 7B Mistral
    # MoD is a LoRA adapter on Qwen3-Embedding-4B -> not picklable into ST
    # worker pools -> single GPU (same class as jina v3/v4 parametrize issue).
    "MoD-Embedding": _llm_dense("MoD-Embedding", 128, multiprocess_ok=False),
    "ICT-TIME-and-Querit-embedding-v1": _llm_dense("ICT-TIME-and-Querit-embedding-v1", 128),
    # Jasper / Stella family (Qwen3-0.6B base, mean pool after token compression).
    "Jasper-Token-Compression-600M": ModelSpec(
        alias="Jasper-Token-Compression-600M", backend="st", init_batch=1024,
        dtype="bfloat16", trust_remote_code=True, tokenizer_kwargs=dict(_LEFT_PAD),
        disk_target="scratch",
    ),
    # Vision-language model, TEXT-ONLY mode. is_text_encoder goes in config_kwargs
    # only (the model constructor rejects it as a model_kwarg).
    # jasper's query instruction lives under the prompt key 's2p_query' (sentence->passage), NOT
    # 'query', so the generic query auto-detect misses it and would encode queries as plain text.
    # Pin query_prompt_name='s2p_query' (the retrieval-correct prompt; 's2s_query' is for semantic
    # similarity). Documents stay plain (the card wants no document prompt).
    "jasper_en_vision_language_v1": ModelSpec(
        alias="jasper_en_vision_language_v1", backend="st", init_batch=256, dtype="bfloat16",
        trust_remote_code=True, query_prompt_name="s2p_query",
        config_kwargs={"is_text_encoder": True, "vector_dim": 1024},
        disk_target="scratch", notes="Stella-based; config is_text_encoder disables vision.",
    ),
    # jina v5: prompts are keyed 'query'/'document' (NOT 'passage'); let the
    # _dense_core auto-detect pick 'document'. The model also requires a task to
    # be set before encoding -> default_task at load (same as omni). LoRA-style
    # internals -> single GPU.
    "jina-embeddings-v5-text-small": ModelSpec(
        alias="jina-embeddings-v5-text-small", backend="st", init_batch=1024,
        multiprocess_ok=False, trust_remote_code=True,
        model_kwargs={"default_task": "retrieval"},
        tokenizer_kwargs=dict(_LEFT_PAD), disk_target="scratch",
    ),
    # jina v5 omni: TEXT-ONLY (frozen text backbone, bit-identical to v5-text-small).
    "jina-embeddings-v5-omni-small": ModelSpec(
        alias="jina-embeddings-v5-omni-small", backend="st", init_batch=256, dtype="bfloat16",
        multiprocess_ok=False, trust_remote_code=True,
        model_kwargs={"default_task": "retrieval"},
        tokenizer_kwargs=dict(_LEFT_PAD), disk_target="scratch",
    ),
    # ---- multi-vector / late-interaction (durable home) ----------------
    # colbert is indexed at document_length=512 (full paragraph; ColBERT's BERT-backbone max),
    # not pylate's default 180. At full length its token-vector table is ~253 GB, which does NOT
    # fit the ~155 GB durable home partition, so — like jina-v4-late — it lives on /tmp scratch
    # (3 TB). The durable retrieval candidates are written separately, so losing the scratch table
    # on a wipe only costs a re-index, not the results.
    "colbert-v2": ModelSpec(
        alias="colbert-v2", backend="pylate", init_batch=64, disk_target="scratch",
        notes="pylate ColBERTv2; 128-d/token; is_query=False; document_length=512.",
    ),
    # NOTE: both multi-vector models now live on /tmp scratch at full fidelity (~473 tok/doc).
    # home holds only the shared checkpoint state.json.
    "jina-embeddings-v4-late": ModelSpec(
        alias="jina-embeddings-v4-late", backend="jina_v4_hf", init_batch=16,
        dtype="bfloat16", disk_target="scratch",
        notes="HF AutoModel; return_multivector=True; task=retrieval prompt=passage.",
    ),
}


def spec_for(alias: str) -> ModelSpec:
    return MODEL_SPECS.get(alias, ModelSpec(alias=alias))


def root_for(alias: str, override: str = "auto"):
    """Resolve the LanceDB root directory for a model.

    override: "home" | "scratch" force a root; "auto" consults the spec.
    """
    if override == "home":
        return LANCEDB_PATH
    if override == "scratch":
        return LANCEDB_SCRATCH_PATH
    return LANCEDB_PATH if spec_for(alias).disk_target == "home" else LANCEDB_SCRATCH_PATH


# ---------------------------------------------------------------------------
# Curated alias lists (the deduplicated 18-model frontier-v2 set).
# ---------------------------------------------------------------------------
VLLM_ALIASES = ["Qwen3-Embedding-8B", "Qwen3-Embedding-4B"]

# sentence-transformers dense models (14), ordered big -> small for scheduling.
ST_DENSE_ALIASES = [
    "F2LLM-v2-14B",
    "F2LLM-v2-8B",
    "Octen-Embedding-8B",
    "QZhou-Embedding",
    "LGAI-Embedding-Preview",
    "F2LLM-v2-4B",
    "Octen-Embedding-4B",
    "MoD-Embedding",
    "ICT-TIME-and-Querit-embedding-v1",
    "jasper_en_vision_language_v1",
    "jina-embeddings-v5-omni-small",
    "jina-embeddings-v5-text-small",
    "Jasper-Token-Compression-600M",
    "Octen-Embedding-0.6B",
]

MV_ALIASES = ["colbert-v2", "jina-embeddings-v4-late"]

# The full deduplicated set (16 dense + 2 multi-vector = 18).
ALL_ALIASES = VLLM_ALIASES + ST_DENSE_ALIASES + MV_ALIASES


def all_dense_aliases() -> list[str]:
    return VLLM_ALIASES + ST_DENSE_ALIASES


def expected_union_from_config() -> list[str]:
    """The dedup union of the two TOP_ENGLISH_* dicts (sanity check helper).

    Returns aliases in MTEB-then-RTEB order, de-duplicated, first occurrence
    kept. Used by tests / status to confirm we cover exactly the requested set.
    """
    seen: dict[str, None] = {}
    for d in (TOP_ENGLISH_MTEB_MODEL_IDS, TOP_ENGLISH_RTEB_MODEL_IDS):
        for alias in d:
            seen.setdefault(alias, None)
    return list(seen)
