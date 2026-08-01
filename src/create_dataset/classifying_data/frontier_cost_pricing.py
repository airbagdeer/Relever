"""Pricing table + self-metered cost computation for the frontier-model cost estimate.

Token counts come straight from each OpenAI ``response.usage``, so cost is computed
locally and never relies on the (aggregated, smoke-test-polluted) billing dashboard.
Reasoning tokens are billed as output tokens and are already included in
``completion_tokens`` -- they are tracked separately only for insight, not added twice.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens (standard tier, short context)."""

    input: float
    cached_input: float
    output: float


# Verified against OpenAI docs (June 2026). CONFIRM before trusting the dollar
# figures -- the measured token counts are exact regardless of these numbers, so
# correcting a price is a one-line edit + re-run of the report step.
PRICING: dict[str, ModelPricing] = {
    "gpt-5.5": ModelPricing(input=5.00, cached_input=0.50, output=30.00),
    "gpt-5.4": ModelPricing(input=2.50, cached_input=0.25, output=15.00),
    "gpt-5.2": ModelPricing(input=1.75, cached_input=0.175, output=14.00),
    "gpt-5.1": ModelPricing(input=1.25, cached_input=0.125, output=10.00),
    # DeepSeek V4 Pro (permanent discount, May 2026). OpenAI-compatible usage.
    "deepseek-v4-pro": ModelPricing(input=0.435, cached_input=0.003625, output=0.87),
    # Open-source models run locally on H100s (amortized GPU runtime pricing; the
    # per-paragraph $ comes from wall-clock GPU-hours in the jsonl, not these rates).
    "qwen3.5-397b-moe": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "deepseek-v4-flash": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "glm-4.7": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "minimax-m2.5": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "qwen3-coder-480b-moe": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "gpt-oss-120b": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "nemotron-3-super-120b": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "llama-3.3-70b": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "deepseek-r1-distill-llama-70b": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
    "qwen3-235b-thinking": ModelPricing(input=0.0, cached_input=0.0, output=0.0),
}

# Open-source aliases (run on our own GPUs) — no provider batch discount applies.
_OPEN_SOURCE_MODELS = frozenset({
    "qwen3.5-397b-moe",
    "deepseek-v4-flash",
    "glm-4.7",
    "minimax-m2.5",
    "qwen3-coder-480b-moe",
    "gpt-oss-120b",
    "nemotron-3-super-120b",
    "llama-3.3-70b",
    "deepseek-r1-distill-llama-70b",
    "qwen3-235b-thinking",
})

# Cheapest realistic *bulk* multiplier vs the synchronous rate, per provider.
# Only OpenAI offers a true Batch API (50% off input+output, 24h SLA). DeepSeek has
# NO batch API — it has off-peak-window discounts (16:30-00:30 UTC), but V4 off-peak
# rates are unconfirmed — so its bulk price defaults to the standard/sync rate.
_BATCH_DISCOUNT_BY_PROVIDER = {"openai": 0.5, "deepseek": 1.0}


def batch_discount_for(model: str) -> float:
    # Open-source models running on our own local/remote GPUs have no batch discounts
    if model in _OPEN_SOURCE_MODELS:
        return 1.0
    provider = "deepseek" if model.startswith("deepseek") else "openai"
    return _BATCH_DISCOUNT_BY_PROVIDER.get(provider, 0.5)


@dataclass(frozen=True)
class UsageCost:
    """Result of pricing a single response's token usage."""

    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_as_measured: float  # uses observed cached_tokens (warm-cache, realistic)
    cost_uncached: float  # prices ALL input at the standard rate (upper bound)


def price_usage(usage: dict, pricing: ModelPricing) -> UsageCost:
    """Compute the USD cost of one response from its token counts."""
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int(usage.get("reasoning_tokens") or 0)
    non_cached_in = max(prompt - cached, 0)
    cost_measured = (
        non_cached_in / 1e6 * pricing.input
        + cached / 1e6 * pricing.cached_input
        + completion / 1e6 * pricing.output
    )
    cost_uncached = prompt / 1e6 * pricing.input + completion / 1e6 * pricing.output
    return UsageCost(
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cost_as_measured=cost_measured,
        cost_uncached=cost_uncached,
    )


def estimate_population(
    minimax_dir: Path, sample_files: int = 200, seed: int = 13
) -> dict:
    """Estimate full-run size + positive rate from a sample of minimax label files.

    Reports both the *raw* entry count (what minimax actually classified, with the
    splade+bm25 duplicates the pipeline leaves in) and the *deduped* count (unique
    paragraph ids per event -- what a dedup'd reclassification would send).
    """
    files = sorted(minimax_dir.glob("*.json"))
    n_events_total = len(files)
    if n_events_total == 0:
        raise FileNotFoundError(f"No minimax label files under {minimax_dir}")

    rng = random.Random(seed)
    chosen = files if n_events_total <= sample_files else rng.sample(files, sample_files)

    raw_counts: list[int] = []
    uniq_counts: list[int] = []
    pos = neg = 0
    for f in chosen:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        raw_labeled = 0
        seen: dict[str, int] = {}
        for item in data:
            lbl = str(item.get("classification"))
            if lbl not in ("0", "1"):
                continue
            raw_labeled += 1
            rid = item.get("id")
            if rid is not None and rid not in seen:
                seen[rid] = int(lbl)
        raw_counts.append(raw_labeled)
        uniq_counts.append(len(seen))
        pos += sum(1 for v in seen.values() if v == 1)
        neg += sum(1 for v in seen.values() if v == 0)

    n_files = len(uniq_counts) or 1
    mean_raw = sum(raw_counts) / n_files
    mean_uniq = sum(uniq_counts) / n_files
    labeled = pos + neg
    p_pos = pos / labeled if labeled else 0.0
    return {
        "n_events_total": n_events_total,
        "sample_files": len(chosen),
        "mean_raw_per_event": mean_raw,
        "mean_unique_per_event": mean_uniq,
        "n_full_raw_est": int(round(mean_raw * n_events_total)),
        "n_full_dedup_est": int(round(mean_uniq * n_events_total)),
        "p_pos": p_pos,
        "labeled_unique_in_sample": labeled,
    }
