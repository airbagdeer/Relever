"""Backend dispatch for the rerank harness.

Imports are lazy and per-backend on purpose: the venv matrix means no single
interpreter can import every backend. ``.venv`` has transformers but no ``pylate``
(which pins ``transformers<=4.56.2``) and no ``rank_llm``/``vllm``. Importing this
package must therefore never pull in a backend the caller did not ask for.

    from backends import build_reranker
    reranker = build_reranker(spec_for("Qwen3-Reranker-8B"), device="cuda")
    scores = reranker.score(query, docs, instruction=...)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Parent-level modules (`reranker_registry`, `_rerank_bootstrap`) are flat, per the
# convention in frontier-v2. Guarantee the rerank dir is importable even when this
# subpackage is reached from a test runner rather than from a script in that dir.
_RERANK_DIR = Path(__file__).resolve().parent.parent
if str(_RERANK_DIR) not in sys.path:
    sys.path.insert(0, str(_RERANK_DIR))

from .base import ListwiseItem, ListwiseReranker, ListwiseResult, Reranker  # noqa: E402
from reranker_registry import RerankerSpec  # noqa: E402

__all__ = [
    "ListwiseItem",
    "ListwiseReranker",
    "ListwiseResult",
    "Reranker",
    "build_listwise_reranker",
    "build_reranker",
]


def build_reranker(spec: RerankerSpec, device: str | None = None, **kwargs) -> Reranker:
    """Construct a POINTWISE backend (one that implements ``score``).

    Listwise models have a different contract — their score depends on the candidate set
    — so asking for one here is an error rather than a silent cast. Use
    ``build_listwise_reranker``.

    Raises ``ImportError`` with the required venv named, rather than a bare
    ``ModuleNotFoundError``, when a backend is asked for in the wrong interpreter.
    """
    if spec.paradigm != "pointwise":
        raise ValueError(
            f"[{spec.alias}] is {spec.paradigm}; it has no set-independent score(). "
            f"Use build_listwise_reranker()."
        )
    backend = spec.backend
    try:
        if backend == "identity":
            from .identity import IdentityReranker  # noqa: PLC0415

            return IdentityReranker(spec, device=device, **kwargs)

        if backend == "st_cross_encoder":
            from .st_cross_encoder import STCrossEncoderReranker  # noqa: PLC0415

            return STCrossEncoderReranker(spec, device=device, **kwargs)

        if backend == "hf_seq_cls":
            from .hf_seq_cls import HFSeqClsReranker  # noqa: PLC0415

            return HFSeqClsReranker(spec, device=device, **kwargs)

        if backend == "hf_causal_yesno":
            from .hf_causal_yesno import HFCausalYesNoReranker  # noqa: PLC0415

            return HFCausalYesNoReranker(spec, device=device, **kwargs)

        if backend == "hf_causal_yes_logit":
            from .hf_causal_yes_logit import HFCausalYesLogitReranker  # noqa: PLC0415

            return HFCausalYesLogitReranker(spec, device=device, **kwargs)

        if backend == "pylate_colbert":
            from .pylate_colbert import PylateColbertReranker  # noqa: PLC0415

            return PylateColbertReranker(spec, device=device, **kwargs)

    except ImportError as exc:
        raise ImportError(
            f"[{spec.alias}] backend {backend!r} could not be imported in this interpreter. "
            f"It expects venv {spec.venv!r}. Original error: {exc}"
        ) from exc

    raise ValueError(f"[{spec.alias}] unknown pointwise backend {backend!r}.")


def build_listwise_reranker(
    spec: RerankerSpec, device: str | None = None, **kwargs
) -> ListwiseReranker:
    """Construct a LISTWISE backend (one that implements ``rerank_many``).

    These score set-dependently, so they cannot share the pooled score cache and must be
    run once per (reranker, retriever). ``rerank_many`` takes many items at once because
    rank_zephyr parallelises across queries, not across its sequential sliding windows.
    """
    if spec.paradigm != "listwise":
        raise ValueError(
            f"[{spec.alias}] is {spec.paradigm}; use build_reranker() and the pooled cache."
        )
    backend = spec.backend
    try:
        if backend == "jina_v3":
            from .listwise_jina_v3 import JinaV3Reranker  # noqa: PLC0415

            return JinaV3Reranker(spec, device=device, **kwargs)

        if backend == "rank_zephyr":
            from .listwise_rank_zephyr import RankZephyrReranker  # noqa: PLC0415

            return RankZephyrReranker(spec, device=device, **kwargs)

    except ImportError as exc:
        raise ImportError(
            f"[{spec.alias}] backend {backend!r} could not be imported in this interpreter. "
            f"It expects venv {spec.venv!r}. Original error: {exc}"
        ) from exc

    raise ValueError(f"[{spec.alias}] unknown listwise backend {backend!r}.")
