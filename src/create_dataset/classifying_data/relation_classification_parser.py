"""
Parser for relation-classification LLM outputs.

Extracted into its own module so backfill / analysis scripts can import the
parser without pulling in `vllm` (which is only installed on the cluster).

Returns 1 (True), 0 (False), or -1 (unparseable).

Handles two prompt families:
  - "short"      -> a flat ``{"evidences_relation": bool}`` object.
  - "decomposed" -> ``{"checks": {...}, "evidences_relation": bool}`` (the final
    field is the same, so all downstream agreement logic is unchanged).

And several reasoning-trace wrappers (so the answer is read, not the reasoning):
  - ``<think>...</think>``                 (minimax / qwen3-thinking / deepseek-r1;
                                            the chat template may auto-open <think>,
                                            so only the CLOSING tag is present)
  - gpt-oss harmony channels               (the answer follows a `final` channel marker)

NB: vLLM's OFFLINE ``LLM.generate()`` path does NOT populate ``reasoning_content``
(only the online server does), so reasoning stripping must happen here.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser


class ClassificationResult(BaseModel):
    """Flat binary classification result ('short' prompt)."""

    evidences_relation: bool


class DecomposedChecks(BaseModel):
    """The five sub-checks emitted by the 'decomposed' prompt."""

    valid_instance: bool
    constant_present: bool
    relation_asserted: bool
    direction_ok: bool
    asserted_as_fact: bool


class DecomposedResult(BaseModel):
    """Decomposed classification result ('decomposed' prompt)."""

    checks: DecomposedChecks
    evidences_relation: bool


_pydantic_parser = PydanticOutputParser(pydantic_object=ClassificationResult)
_decomposed_parser = PydanticOutputParser(pydantic_object=DecomposedResult)

FORMAT_INSTRUCTIONS: str = _pydantic_parser.get_format_instructions()
FORMAT_INSTRUCTIONS_DECOMPOSED: str = _decomposed_parser.get_format_instructions()


# Markers after which the FINAL answer begins. The last occurrence of any marker
# wins, so reasoning text (which may itself contain JSON / "true" / "false") is
# discarded and only the post-reasoning answer is parsed.
_ANSWER_MARKERS = (
    "</think>",
    "<|channel|>final<|message|>",
    "assistant<|channel|>final",
)


def _strip_think(text: str) -> str:
    """Return only the post-reasoning answer segment.

    Cuts everything up to and including the LAST answer marker (closing think tag
    or harmony final-channel marker). If no marker is present the text is returned
    unchanged — for a non-reasoning model the whole output is the answer; for a
    reasoning model that truncated mid-trace there is no answer and downstream
    parsing fails cleanly to -1.
    """
    cut = -1
    for marker in _ANSWER_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            cut = max(cut, idx + len(marker))
    return text[cut:] if cut != -1 else text


def _coerce_bool_value(val: object) -> int | None:
    """Map a JSON value for ``evidences_relation`` to 1/0, or None if unusable."""
    if isinstance(val, bool):
        return 1 if val else 0
    # Pydantic rejects ints for bool fields, but models commonly emit
    # {"evidences_relation": 0/1}, which is unambiguous and worth recovering.
    if isinstance(val, int) and val in (0, 1):
        return int(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1"):
            return 1
        if s in ("false", "0"):
            return 0
    return None


def parse_classification_output(raw_output: str) -> int:
    """
    Parse the model output to extract the boolean classification.
    Returns 1 for True, 0 for False, -1 for Failure.

    Accepts (post-answer-marker):
      - Decomposed schema:      {"checks": {...}, "evidences_relation": true/false}
      - Flat pydantic schema:   {"evidences_relation": true/false}
      - JSON with int/str value:{"evidences_relation": 0/1 | "true"/"false"}
      - Bare integer:           "0" or "1"
      - Tail-keyword match:     "...true" or "...false"
    """
    answer_text = _strip_think(raw_output).strip()

    # Bare integer fast path: model emitted just "0" or "1" after the answer marker.
    if answer_text in ("0", "1"):
        return int(answer_text)

    # Structured-schema parse: decomposed first (its `checks` field distinguishes
    # it), then the flat schema. Both expose `.evidences_relation`.
    for parser in (_decomposed_parser, _pydantic_parser):
        try:
            result = parser.parse(answer_text)
            return 1 if result.evidences_relation else 0
        except Exception:
            pass

    text_lower = answer_text.lower()

    # Scan every flat JSON object for the final key (the nested `checks` object is
    # skipped by the non-nesting regex; a flat sibling `evidences_relation` is caught).
    for blob in re.findall(r"\{[^{}]*\}", text_lower):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "evidences_relation" in data:
            coerced = _coerce_bool_value(data["evidences_relation"])
            if coerced is not None:
                return coerced

    # Direct key regex — robust to nesting and to trailing commas / minor JSON
    # malformation that json.loads would reject.
    km = re.search(r'"evidences_relation"\s*:\s*"?(true|false|1|0)"?', text_lower)
    if km:
        return 1 if km.group(1) in ("true", "1") else 0

    tail = text_lower[-30:]
    if "true" in tail and "false" not in tail:
        return 1
    if "false" in tail and "true" not in tail:
        return 0

    if "true" in text_lower and "false" not in text_lower:
        return 1
    if "false" in text_lower and "true" not in text_lower:
        return 0

    return -1
