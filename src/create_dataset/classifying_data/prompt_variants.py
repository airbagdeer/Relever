"""Shared registry of relation-classification prompt variants.

Single source of truth for the prompt-variant sweep used by BOTH the frontier
(OpenAI) cost runner and the open-source (vLLM) cost runner. Kept deliberately
import-light: it pulls only from ``relation_classification_prompt`` (pure string
constants, no imports) and ``relation_classification_parser`` (json/re/pydantic,
no ``openai``), so it is safe to import inside a vLLM-only venv which has
no ``openai`` package. Do NOT add heavy/optional imports here.

Variants
--------
- short      — flat, trap-hardened baseline prompt.
- short2     — short + loosened over-rejection (generous category/paraphrase matching).
- short2b    — short2 + lexical-match guard (best human-gold agreement for gpt-5.2:medium).
- decomposed — same rules as short, but explicit sub-checks + structured output.
"""
from __future__ import annotations

try:  # package import
    from .relation_classification_prompt import (
        relation_classification_prompt_short,
        relation_classification_prompt_decomposed,
        relation_classification_prompt_short_v2,
        relation_classification_prompt_short_v2b,
    )
    from .relation_classification_parser import (
        FORMAT_INSTRUCTIONS,
        FORMAT_INSTRUCTIONS_DECOMPOSED,
    )
except ImportError:  # run as a plain script (cluster launches by file path)
    from relation_classification_prompt import (  # type: ignore
        relation_classification_prompt_short,
        relation_classification_prompt_decomposed,
        relation_classification_prompt_short_v2,
        relation_classification_prompt_short_v2b,
    )
    from relation_classification_parser import (  # type: ignore
        FORMAT_INSTRUCTIONS,
        FORMAT_INSTRUCTIONS_DECOMPOSED,
    )

# variant name -> (prompt template with a {format_instructions} slot, format instructions)
PROMPT_VARIANTS: dict[str, tuple[str, str]] = {
    "short": (relation_classification_prompt_short, FORMAT_INSTRUCTIONS),
    "short2": (relation_classification_prompt_short_v2, FORMAT_INSTRUCTIONS),
    "short2b": (relation_classification_prompt_short_v2b, FORMAT_INSTRUCTIONS),
    "decomposed": (relation_classification_prompt_decomposed, FORMAT_INSTRUCTIONS_DECOMPOSED),
}


def build_system_prompt(variant: str) -> str:
    """Render the system prompt for ``variant`` (fills in its format instructions)."""
    template, fmt = PROMPT_VARIANTS[variant]
    return template.format(format_instructions=fmt)

