import os
from pathlib import Path

# Resolve all paths relative to the project root so the scripts can be run
# from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LANCEDB_PATH = PROJECT_ROOT / "lancedb_data"

# Sentence and proposition tables are huge (35-200 GB each). On the cluster
# they overflow the NFS /home quota, so route them to local-disk scratch
# (e.g. /tmp/lancedb_scratch) by default. Override with $LANCEDB_SCRATCH_PATH.
# Paragraph tables remain under LANCEDB_PATH because they're small (3-10 GB)
# and the checkpoint/state file lives alongside them.
LANCEDB_SCRATCH_PATH = Path(os.environ.get("LANCEDB_SCRATCH_PATH", "/tmp/lancedb_scratch"))

LANCEDB_PATH_BY_GRANULARITY: dict[str, Path] = {
    "paragraph": LANCEDB_PATH,
    "sentence": LANCEDB_SCRATCH_PATH,
    "proposition": LANCEDB_SCRATCH_PATH,
}


def lancedb_path_for(granularity: str) -> Path:
    """Return the LanceDB root directory for the given granularity.

    Defaults to LANCEDB_PATH if the granularity isn't in the per-granularity
    map (treats unknown granularities as paragraph-class).
    """
    return LANCEDB_PATH_BY_GRANULARITY.get(granularity, LANCEDB_PATH)

COLLECTION_NAME = "thesis"

# The active v2 benchmark query set (1,100 relational queries).
RELATIONS_PATH = PROJECT_ROOT / "data/queries/queries.json"

# Minimax-era classified-paragraph dirs (the preliminary MiniMax-M2.5 labeling
# pass described in paper §4.1). The data dirs were removed from the working
# tree in the 2026-07 cleanup (recoverable from git history); the constants are
# retained because the shared LanceDB indexer library imports them at module
# level.
CLASSIFIED_PARAGRAPHS_MINIMAX_PATH = PROJECT_ROOT / "data/classified_splade_paragraphs_minimax"
CLASSIFIED_SENTENCES_MINIMAX_PATH = PROJECT_ROOT / "data/classified_splade_sentences_minimax"
CLASSIFIED_PROPOSITIONS_MINIMAX_PATH = PROJECT_ROOT / "data/classified_splade_propositions_minimax"

# Centralized summary / insights output: human-readable rollups land here
# (instead of being scattered next to the raw per-event JSONs / jsonl files).
SUMMARY_PATH = PROJECT_ROOT / "summary"

CLASSIFIED_PARAGRAPHS_PATHS: dict[str, Path] = {
    "minimax": CLASSIFIED_PARAGRAPHS_MINIMAX_PATH,
}

# 2D map: source -> granularity -> per-event JSON directory.
# Used by the indexer to resolve the input data dir for any (source, granularity).
CLASSIFIED_SOURCE_PATHS: dict[str, dict[str, Path]] = {
    "minimax": {
        "paragraph": CLASSIFIED_PARAGRAPHS_MINIMAX_PATH,
        "sentence": CLASSIFIED_SENTENCES_MINIMAX_PATH,
        "proposition": CLASSIFIED_PROPOSITIONS_MINIMAX_PATH,
    },
}
RESULTS_PATH = PROJECT_ROOT / "results"

# --- Reranking study (src/retrieval/rerank/) ---------------------------------
# Cached reranker scores over the frontier-v2 deduplicated candidate pool.
# Layout: <root>/<reranker_alias>/<variant>/<qid>.parquet, columns {id, score}.
#
# The pool (data/candidates/<qid>/all_candidates_<qid>.json) is exactly the
# union of every retriever's top-500, and the GPT-5.2 labels cover exactly that pool.
# A POINTWISE reranker's score s(q,d) does not depend on the candidate set, so scoring
# the pool ONCE per (query, reranker) yields every retriever's reranked top-500 by a
# plain sort of its own ids — exact, not an approximation (2.18M pairs, not 9.9M).
#
# LISTWISE rerankers are set-dependent and cannot share the cache; they write
# per-(reranker, retriever) permutations under
# <root>/<reranker_alias>/listwise/<retriever>/<qid>.json.
FRONTIER_RERANK_SCORES_PATH = PROJECT_ROOT / "data" / "frontier_rerank_scores"

# --- False-positive taxonomy (paper §7 "Anatomy of the False Positives") -----
# Raw per-event classification payloads: the deduplicated (query, paragraph) FP
# population extracted from every system's top-10, its batch inputs/outputs, and
# per-event taxonomy predictions. Layout mirrors data/labels.
FP_TAXONOMY_DATA_PATH = PROJECT_ROOT / "data" / "fp_taxonomy"
# Human-readable rollups: per-system FP composition CSVs, REPORT.md, plots.
SUMMARY_FP_TAXONOMY_PATH = SUMMARY_PATH / "fp_taxonomy"

MODEL_IDS: dict[str, str] = {
    "e5-base": "intfloat/e5-base-v2",
    "gte-small": "thenlper/gte-small",
    "gte-base-en-v1.5": "Alibaba-NLP/gte-base-en-v1.5",
    "bge-small": "BAAI/bge-small-en", 
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "jina-embeddings-v2-small": "jinaai/jina-embeddings-v2-small-en",
    "e5-small": "intfloat/e5-small-v2",
    "jina-embeddings-v3": "jinaai/jina-embeddings-v3",
    "bge-m3": "BAAI/bge-m3",
    "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B",
    "Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
    "llama-embed-nemotron-8b": "nvidia/llama-embed-nemotron-8b",
    "KaLM-Embedding-Gemma3-12B-2511": "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "jina-embeddings-v4": "jinaai/jina-embeddings-v4",
    # Late-interaction (multi-vector / ColBERT-style) variant of jina-v4.
    # Same upstream model; the alias differentiator drives indexer/evaluator
    # dispatch, table naming, and result-dir naming so the dense and
    # late-interaction outputs can sit side-by-side.
    "jina-embeddings-v4-late": "jinaai/jina-embeddings-v4",
    "GritLM-7B": "GritLM/GritLM-7B",
    "colbert-v2": "colbert-ir/colbertv2.0",
}

TOP_ENGLISH_MTEB_MODEL_IDS: dict[str, str] = {
    "Jasper-Token-Compression-600M": "infgrad/Jasper-Token-Compression-600M",      # rank 1  | Apache-2.0
    "QZhou-Embedding": "Kingsoft-LLM/QZhou-Embedding",                          # rank 2  | Apache-2.0
    "Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",                                    # rank 4  | Apache-2.0
    "LGAI-Embedding-Preview": "annamodels/LGAI-Embedding-Preview",                # rank 5  | CC BY-NC 4.0 (non-commercial!)
    "Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-4B",                                    # rank 6  | Apache-2.0
    "F2LLM-v2-8B": "codefuse-ai/F2LLM-v2-8B",                                    # rank 9  | Apache-2.0
    "F2LLM-v2-14B": "codefuse-ai/F2LLM-v2-14B",                                  # rank 10 | Apache-2.0
    "jina-embeddings-v5-text-small": "jinaai/jina-embeddings-v5-text-small",          # rank 11 | CC BY-NC 4.0 (non-commercial!)
    "jina-embeddings-v5-omni-small": "jinaai/jina-embeddings-v5-omni-small",          # rank 11 | CC BY-NC 4.0 (non-commercial!), multimodal
    "jasper_en_vision_language_v1": "NovaSearch/jasper_en_vision_language_v1",    # rank 13 | open, multimodal
}

TOP_ENGLISH_RTEB_MODEL_IDS: dict[str, str] = {
    "Octen-Embedding-8B": "Octen/Octen-Embedding-8B",              # rank 1  | Apache-2.0  (≡ rank 3 INT8)
    "Octen-Embedding-4B": "Octen/Octen-Embedding-4B",              # rank 6  | Apache-2.0  (≡ rank 7 INT8)
    "MoD-Embedding": "bflhc/MoD-Embedding",                                  # rank 9  | ⚠️ repo/license UNVERIFIED — see note
    "Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",                # rank 11 | Apache-2.0
    "F2LLM-v2-14B": "codefuse-ai/F2LLM-v2-14B",              # rank 12 | Apache-2.0
    "F2LLM-v2-8B": "codefuse-ai/F2LLM-v2-8B",                # rank 15 | Apache-2.0
    "Octen-Embedding-0.6B": "Octen/Octen-Embedding-0.6B",          # rank 17 | Apache-2.0
    "F2LLM-v2-4B": "codefuse-ai/F2LLM-v2-4B",                # rank 18 | Apache-2.0
    "ICT-TIME-and-Querit-embedding-v1": "ICT-TIME-and-Querit/ICT-TIME-and-Querit-embedding-v1",              # rank 19 | ⚠️ repo/status UNVERIFIED
    "jina-embeddings-v5-text-small": "jinaai/jina-embeddings-v5-text-small",  # rank 21 | CC BY-NC 4.0 (non-commercial!)
}

# =============================================================================
# Reranker selection for the paper.
#
# NOTE: Unlike retrievers, rerankers have NO standalone public leaderboard
# (MTEB's "Reranking" column scores *embedding* models by cosine similarity,
# not cross-encoder rerankers; RTEB has no reranking track at all). Selection
# is therefore CRITERION-BASED, not rank-based:
#   Axis 1 (performance): best-in-class open model per family + leading API model.
#   Axis 2 (representativeness): span the major reranker architecture families
#           so results characterize *paradigms*, not variants of one approach.
# All models reranked over a FIXED first stage (BM25 top-100, Pyserini) and,
# secondarily, over your best dense retriever top-100. Metric: nDCG@10 on
# BEIR + TREC DL19/DL20. Selection cutoff: 2026-07-08.
# =============================================================================

# Architecture/license facts below were verified against each repo's raw model card and
# config.json on 2026-07-08 (see src/retrieval/rerank/reranker_registry.py for the exact
# scoring contract of each). Three of the original groupings/licenses were wrong and are
# corrected here.
TOP_RERANKER_MODEL_IDS: dict[str, str] = {
    # --- Encoder cross-encoders (full query-doc attention; accuracy-per-pair ceiling) ---
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",                       # Apache-2.0 | XLMRobertaForSequenceClassification, 568M; raw logit. Most-used open cross-encoder baseline
    "ms-marco-MiniLM-L12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",     # Apache-2.0 | BertForSequenceClassification, 33M; raw logit (Identity activation pinned in config.json).
                                                                           #              Efficiency anchor. HARD 512-token limit (BERT position embeddings).

    # --- LLM-based (decoder) rerankers, pointwise (current SOTA on MTEB-R) ---
    # NOTE: mxbai-rerank-large-v2 lives here, NOT with the encoders. Despite the name it is
    # Qwen2ForCausalLM (1.54B, a.k.a. ProRank-1.5B): the score is logit("1") - logit("0") at
    # the last position, so LEFT PADDING IS MANDATORY (right padding reads a pad position and
    # silently corrupts every score). Requires sentence-transformers >= 5.4.0.
    "mxbai-rerank-large-v2": "mixedbread-ai/mxbai-rerank-large-v2",        # Apache-2.0 | RL-trained; strongest of the mxbai family
    "Qwen3-Reranker-8B": "Qwen/Qwen3-Reranker-8B",                         # Apache-2.0 | Qwen3ForCausalLM, 8.2B; P(yes) via log_softmax over yes/no logits. LEFT PAD.
    "Qwen3-Reranker-4B": "Qwen/Qwen3-Reranker-4B",                         # Apache-2.0 | same family mid-size; isolates the model-scale axis. LEFT PAD.
    "bge-reranker-v2-gemma": "BAAI/bge-reranker-v2-gemma",                 # Apache-2.0 (verified; NOT a Gemma license) | GemmaForCausalLM, 2.5B; raw 'Yes' logit, no softmax.
                                                                           #              Instruction is a SUFFIX. Different LLM lineage -> tests family, not just scale.
    # Qwen3-Reranker MUST be scored with transformers, NOT vLLM: vllm#21681 reports scores
    # compressing until relevant/irrelevant stop separating (vLLM 0.907/0.810/0.807/0.076 vs
    # transformers 0.957/0.001/0.000/0.000). No maintainer root-cause as of 2026-07-08.

    # --- LLM-based, LISTWISE (cross-candidate context; distinct paradigm) ---
    # Set-dependent scores: these CANNOT reuse the pooled score cache and must be run once per
    # (reranker, retriever). Evaluated at their native depth-100 protocol.
    "rank_zephyr_7b_v1_full": "castorini/rank_zephyr_7b_v1_full",         # MIT (verified) | canonical RankGPT-style listwise reranker. Emits a permutation, no scores.
                                                                           #                  Requires vLLM: rank_llm@main has no HF-generate fallback.
                                                                           #                  context_size=4096 => sees only ~40% of a 475-token paragraph.
                                                                           #                  MUST force greedy (temperature=0): rank_llm defaults SamplingParams
                                                                           #                  to vLLM's temperature=1.0, but RankZephyr is a greedy generator.
                                                                           #                  MUST force enforce_eager=True: vLLM 0.24 CUDA-graph EngineCore
                                                                           #                  crash mid-run (see backends/listwise_rank_zephyr.py).
    "jina-reranker-v3": "jinaai/jina-reranker-v3",                        # ⚠️ CC BY-NC 4.0 (non-commercial!) | 0.6B "last-but-not-late" interaction; one forward pass,
                                                                           #                                     per-doc cosine. API is .rerank() only — no compute_score().

    # --- Late-interaction (token-level; efficiency/accuracy trade-off) ---
    "colbertv2.0": "colbert-ir/colbertv2.0",                             # MIT | 110M ColBERT MaxSim via pylate. BERT backbone, HARD 512-token cap. Set-independent => pooled cache.
    "GTE-ModernColBERT-v1": "lightonai/GTE-ModernColBERT-v1",            # Apache-2.0 | ModernBERT-backbone ColBERT (128-dim MaxSim via pylate); 8K context => sees the
                                                                         #              full paragraph, unlike colbertv2.0's 512 cap. Set-independent => pooled cache.
}

def get_classified_collection_name(alias: str, granularity: str) -> str:
    return f"{COLLECTION_NAME}-qdrant-classified-{alias}-{granularity}"
