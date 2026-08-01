"""Configuration for the frontier-v2 evaluation: paths, the 18-model registry
(family / size / retriever type), and the external MTEB / RTEB rankings.

Paths derive from the central ``config.py`` (single source of truth for
PROJECT_ROOT / RESULTS_PATH / SUMMARY_PATH) so the evaluation outputs land in
the conventional ``results/`` (machine-readable) and ``summary/`` (human-
readable) trees used throughout this repository.

The external rankings are each model's Borda rank on the public English
leaderboards as captured in the **3 June 2026 snapshot used in the paper**
(archived captures ship as ``leaderboard_snapshot/`` in the data release):
``MTEB(eng, v2)`` and ``RTEB(eng, beta)``. Lower rank = better; ``None`` means the
model is outside the snapshot's captured top ranks. The 16 dense models were
selected *from* the top of these leaderboards; the two late-interaction baselines
(``colbert-v2``, ``jina-embeddings-v4-late``) and a handful of lower-ranked dense
models fall outside the captured top-N and so carry no snapshot rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import _bootstrap  # noqa: F401  (appends src/retrieval to sys.path so `config` imports)
from config import PROJECT_ROOT, RESULTS_PATH, SUMMARY_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Inputs (raw): per-query, per-model ranked candidate dumps + pooled GPT-5.2 labels.
FRONTIER_CANDIDATES_DIR = PROJECT_ROOT / "data" / "candidates"
FRONTIER_CLASSIFIED_DIR = PROJECT_ROOT / "data" / "labels"
# Delta labels for the R>500 queries (same GPT-5.2/short2b setup, labeled beyond the
# depth-500 pool so top-R becomes fully judged; see the paper's extended-pooling appendix).
FRONTIER_DELTA_DIR = FRONTIER_CLASSIFIED_DIR / "extended_pooling_delta"
# The 1100 relational queries (event_id 0-1099); query text is the `sentence` field.
V2_RELATIONS_PATH = PROJECT_ROOT / "data" / "queries" / "queries.json"

# Outputs: machine-readable per-query metrics under results/, human-readable
# rollups / report / plots under summary/.
RESULTS_DIR = RESULTS_PATH / "first_stage"
SUMMARY_DIR = SUMMARY_PATH / "evaluation"
PLOTS_DIR = SUMMARY_DIR / "plots"

# --- Reranking study --------------------------------------------------------
# Outputs of evaluate_rerank.py. Its inputs (the cached reranker scores) live under
# config.FRONTIER_RERANK_SCORES_PATH and are addressed via the rerank package's
# `score_store`, the single source of truth for that layout.
RESULTS_RERANK_DIR = RESULTS_PATH / "rerank"
SUMMARY_RERANK_DIR = SUMMARY_PATH / "rerank"
PLOTS_RERANK_DIR = SUMMARY_RERANK_DIR / "plots"

# Depths at which a reranked list is scored. 500 is the design depth (it matches the
# labeled pool, so P@R stays exact); 100 exists so the pointwise models can be compared
# against the two listwise models at their native published depth. Both fall out of one
# pooled scoring pass, so the second depth is free.
RERANK_DEPTHS = (100, 500)

# Imported, not redeclared: rerank_listwise.py's --depth default comes from the same
# constant, so the two cannot drift apart and mislabel a results column. The evaluator
# additionally cross-checks each permutation file's recorded `depth` against what it
# reports (see evaluate_rerank._listwise_depth), so even a hand-run `--depth 50` shard
# is caught rather than silently labelled "@100".
from reranker_registry import LISTWISE_DEPTH as LISTWISE_RERANK_DEPTH  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------------------------
RETRIEVAL_DEPTH = 500          # each model returned its top-500 per query
MAX_RANK = 3000                # deepest rank materialised in the candidate dumps; caps
                               # the deeper list loaded to compute exact P@R when R > 500
K_VALUES = (1, 5, 10, 20, 50, 100, 200, 500)
SUCCESS_K_VALUES = (1, 5, 10)
HEADLINE_NDCG_K = 10           # nDCG@10 is the standard IR headline cutoff
# R-size buckets for the difficulty analysis (closed integer ranges; None upper = open).
R_BUCKETS = (
    (1, 4),
    (5, 9),
    (10, 49),
    (50, 99),
    (100, 499),
    (500, None),
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelInfo:
    alias: str
    family: str                       # for size-ladder grouping
    params_b: float | None            # parameter count in billions (None if not confidently known)
    retriever_type: str               # "dense" | "late_interaction"
    mteb_rank: int | None             # English MTEB leaderboard rank (lower=better), None if absent
    rteb_rank: int | None             # English RTEB leaderboard rank (lower=better), None if absent
    is_baseline: bool = False         # carried-over baseline not on either leaderboard


# Ordered registry of the 18 evaluated models. params_b is filled only where the
# size is unambiguous (encoded in the alias); left None otherwise to avoid
# asserting unverified parameter counts in a released artifact.
#
# MTEB(eng,v2) and RTEB(eng,beta) Borda ranks below are read from the 3 June 2026
# leaderboard captures in leaderboard_snapshot/ (the snapshot used in the
# paper). Lower = better; None = outside the snapshot's captured top ranks.
MODELS: tuple[ModelInfo, ...] = (
    # F2LLM-v2 size ladder (codefuse-ai)
    ModelInfo("F2LLM-v2-4B", "F2LLM-v2", 4.0, "dense", mteb_rank=14, rteb_rank=18),
    ModelInfo("F2LLM-v2-8B", "F2LLM-v2", 8.0, "dense", mteb_rank=9, rteb_rank=15),
    ModelInfo("F2LLM-v2-14B", "F2LLM-v2", 14.0, "dense", mteb_rank=10, rteb_rank=12),
    # Octen size ladder (Octen models are outside the MTEB top-N snapshot)
    ModelInfo("Octen-Embedding-0.6B", "Octen", 0.6, "dense", mteb_rank=None, rteb_rank=17),
    ModelInfo("Octen-Embedding-4B", "Octen", 4.0, "dense", mteb_rank=None, rteb_rank=6),
    ModelInfo("Octen-Embedding-8B", "Octen", 8.0, "dense", mteb_rank=None, rteb_rank=1),
    # Qwen3-Embedding size ladder (4B outside the RTEB top-N snapshot)
    ModelInfo("Qwen3-Embedding-4B", "Qwen3-Embedding", 4.0, "dense", mteb_rank=6, rteb_rank=None),
    ModelInfo("Qwen3-Embedding-8B", "Qwen3-Embedding", 8.0, "dense", mteb_rank=4, rteb_rank=11),
    # jina v5 small (modality variants, not a size ladder) — tied ranks on both boards
    ModelInfo("jina-embeddings-v5-text-small", "jina-v5", None, "dense", mteb_rank=11, rteb_rank=21),
    ModelInfo("jina-embeddings-v5-omni-small", "jina-v5", None, "dense", mteb_rank=11, rteb_rank=21),
    # Standalone dense models
    ModelInfo("Jasper-Token-Compression-600M", "Jasper", 0.6, "dense", mteb_rank=1, rteb_rank=None),
    ModelInfo("jasper_en_vision_language_v1", "Jasper", None, "dense", mteb_rank=13, rteb_rank=None),
    ModelInfo("QZhou-Embedding", "QZhou", None, "dense", mteb_rank=2, rteb_rank=None),
    ModelInfo("LGAI-Embedding-Preview", "LGAI", None, "dense", mteb_rank=5, rteb_rank=None),
    ModelInfo("MoD-Embedding", "MoD", None, "dense", mteb_rank=None, rteb_rank=9),
    ModelInfo("ICT-TIME-and-Querit-embedding-v1", "ICT-Querit", None, "dense", mteb_rank=None, rteb_rank=19),
    # Late-interaction baselines — outside the captured top-N of both June-3 snapshots.
    ModelInfo("jina-embeddings-v4-late", "jina-v4", None, "late_interaction", mteb_rank=None, rteb_rank=None, is_baseline=True),
    ModelInfo("colbert-v2", "colbert", 0.11, "late_interaction", mteb_rank=None, rteb_rank=None, is_baseline=True),
)

MODEL_ALIASES: tuple[str, ...] = tuple(m.alias for m in MODELS)
MODEL_BY_ALIAS: dict[str, ModelInfo] = {m.alias: m for m in MODELS}

# Convenience views used by the analyses.
MTEB_RANK: dict[str, int] = {m.alias: m.mteb_rank for m in MODELS if m.mteb_rank is not None}
RTEB_RANK: dict[str, int] = {m.alias: m.rteb_rank for m in MODELS if m.rteb_rank is not None}
# Families with a genuine size ladder (>= 2 sizes) for the scaling analysis.
SIZE_LADDERS: dict[str, list[str]] = {
    "F2LLM-v2": ["F2LLM-v2-4B", "F2LLM-v2-8B", "F2LLM-v2-14B"],
    "Octen": ["Octen-Embedding-0.6B", "Octen-Embedding-4B", "Octen-Embedding-8B"],
    "Qwen3-Embedding": ["Qwen3-Embedding-4B", "Qwen3-Embedding-8B"],
}


def qid(event_id: int) -> str:
    """Zero-padded 4-digit query/event directory name (e.g. 36 -> '0036')."""
    return str(event_id).zfill(4)
