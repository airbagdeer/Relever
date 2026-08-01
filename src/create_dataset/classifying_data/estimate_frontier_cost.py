"""Estimate the cost of reclassifying the full dataset with frontier OpenAI models.

Runs a paired smoke test: the SAME stratified 100-paragraph sample is classified by
every (model x reasoning_effort) config, with exact per-request token usage captured
straight from ``response.usage`` and priced locally (so the billing dashboard is never
needed). Extrapolates to the full deduped dataset and reports inter-labeler agreement.

A separate Claude Opus 4.8 labeling pass (one agent per paragraph, max reasoning) is run
out-of-band and dropped in as ``opus-4.8.jsonl``; the report picks it up automatically.

Usage:
    uv run python src/create_dataset/classifying_data/estimate_frontier_cost.py --build-sample-only
    uv run python src/create_dataset/classifying_data/estimate_frontier_cost.py --limit 5 --models gpt-5.1 --efforts medium
    uv run python src/create_dataset/classifying_data/estimate_frontier_cost.py            # full smoke test + report
    uv run python src/create_dataset/classifying_data/estimate_frontier_cost.py --report-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

try:  # exception classes (defensive across SDK versions)
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        BadRequestError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    APIConnectionError = APIError = APITimeoutError = BadRequestError = RateLimitError = Exception

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.retrieval.config import (  # noqa: E402
    CLASSIFIED_PARAGRAPHS_MINIMAX_PATH,
    PROJECT_ROOT,
    RELATIONS_PATH,
    SUMMARY_PATH,
)

try:
    from .relation_classification_parser import parse_classification_output
    from .prompt_variants import PROMPT_VARIANTS, build_system_prompt
    from . import frontier_cost_pricing as pricing_mod
    from . import frontier_cost_report as report_mod
except ImportError:  # run as a plain script
    from relation_classification_parser import parse_classification_output
    from prompt_variants import PROMPT_VARIANTS, build_system_prompt
    import frontier_cost_pricing as pricing_mod
    import frontier_cost_report as report_mod

# --- Paths & config -------------------------------------------------------- #
RAW_DIR = PROJECT_ROOT / "data/judge_validation/short"
REPORT_DIR = SUMMARY_PATH / "judge_validation" / "short"
SAMPLE_PATH = RAW_DIR / "_sample.json"
POPULATION_PATH = RAW_DIR / "_population.json"

DEFAULT_MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1"]
DEFAULT_EFFORTS = ["medium", "xhigh"]

# minimax positives whose reasoning hedges are noisier (audit finding) -> deprioritize.
HEDGE_MARKERS = ("indirect", "uncertain", "implies", "could be considered",
                 "if it is considered", "somewhat", "not definitiv", "tenuous")

# Prompt variants live in prompt_variants.py (shared with the open-source runner).
# Default; main() reassigns this global from --prompt-variant before any labeling.
SYSTEM_PROMPT = build_system_prompt("short")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_CLIENTS: dict[str, AsyncOpenAI] = {}


# --------------------------------------------------------------------------- #
# Providers (OpenAI + DeepSeek, both via the OpenAI-compatible SDK)
# --------------------------------------------------------------------------- #
def provider_of(model: str) -> str:
    return "deepseek" if model.startswith("deepseek") else "openai"


def ensure_provider_env() -> None:
    """The DeepSeek key may live in the main repo's .env (worktrees have their own)."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        if common:
            cpath = Path(common) if os.path.isabs(common) else (PROJECT_ROOT / common)
            main_env = cpath.resolve().parent / ".env"
            if main_env.exists():
                load_dotenv(main_env, override=False)
    except Exception:
        pass


def get_client(model: str) -> AsyncOpenAI:
    """Return a cached client for the model's provider (raises if its key is missing)."""
    provider = provider_of(model)
    if provider not in _CLIENTS:
        if provider == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY")
            if not key:
                raise SystemExit("DEEPSEEK_API_KEY not set (.env).")
            _CLIENTS[provider] = AsyncOpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        else:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise SystemExit("OPENAI_API_KEY not set (.env).")
            _CLIENTS[provider] = AsyncOpenAI(api_key=key)
    return _CLIENTS[provider]


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _load_relations() -> dict[int, dict]:
    data = json.loads(Path(RELATIONS_PATH).read_text())
    return {int(r["event_id"]): r for r in data}


def build_sample(
    n_per_class: int,
    seed: int,
    per_event_cap: int = 3,
    exclude_ids: set[str] | None = None,
    write: bool = True,
) -> list[dict]:
    """Dedup + stratify n_per_class minimax-positive / n_per_class minimax-negative.

    `exclude_ids` are treated as already-seen so they are never re-drawn (used to add
    FRESH paragraphs when expanding an existing sample). `write=False` returns the
    drawn records without overwriting _sample.json (the caller persists the union).
    """
    relations = _load_relations()
    files = sorted(CLASSIFIED_PARAGRAPHS_MINIMAX_PATH.glob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(files)

    pos_clean, pos_hedged, neg_pool = [], [], []
    seen_ids: set[str] = set(exclude_ids or ())
    seen_content: set[str] = set()

    def rec_of(item: dict, eid: int) -> dict:
        q = relations[eid]
        return {
            "id": item["id"],
            "event_id": eid,
            "minimax_label": int(str(item["classification"])),
            "content": item.get("content", ""),
            "source_query": item.get("source_query", ""),
            "query_info": {
                "sentence": q.get("sentence", ""),
                "abstract_entity_type": q.get("abstract_entity_type", ""),
                "constant_entity": q.get("constant_entity", ""),
                "relation": q.get("relation", ""),
            },
        }

    target = int(n_per_class * 2.0)  # buffer for selection diversity
    for f in files:
        eid = int(f.stem)
        if eid not in relations:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        n_pos = n_neg = 0
        for item in data:
            rid = item.get("id")
            lbl = str(item.get("classification"))
            content = (item.get("content") or "").strip()
            if not rid or rid in seen_ids or lbl not in ("0", "1") or not content:
                continue
            norm = content.lower()[:200]
            if norm in seen_content:
                continue
            if lbl == "1" and n_pos < per_event_cap:
                rec = rec_of(item, eid)
                raw = (item.get("raw_output") or "").lower()
                (pos_hedged if any(h in raw for h in HEDGE_MARKERS) else pos_clean).append(rec)
                seen_ids.add(rid); seen_content.add(norm); n_pos += 1
            elif lbl == "0" and n_neg < per_event_cap:
                neg_pool.append(rec_of(item, eid))
                seen_ids.add(rid); seen_content.add(norm); n_neg += 1
        if len(pos_clean) >= target and len(neg_pool) >= target:
            break

    rng.shuffle(pos_clean); rng.shuffle(pos_hedged); rng.shuffle(neg_pool)
    positives = (pos_clean + pos_hedged)[:n_per_class]  # prefer non-hedged
    negatives = neg_pool[:n_per_class]
    if len(positives) < n_per_class or len(negatives) < n_per_class:
        raise RuntimeError(
            f"Insufficient sample: {len(positives)} pos / {len(negatives)} neg "
            f"(needed {n_per_class} each). Read {len(files)} event files."
        )
    sample = positives + negatives
    rng.shuffle(sample)
    if write:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_PATH.write_text(json.dumps(sample, indent=2))
    n_hedged_used = sum(1 for r in positives if r in pos_hedged)
    dest = SAMPLE_PATH if write else "(not written)"
    print(f"Built sample: {len(positives)} pos ({n_hedged_used} hedged) + {len(negatives)} neg "
          f"across {len({r['event_id'] for r in sample})} events -> {dest}")
    return sample


def load_sample() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text())


def expand_sample(target_total: int, seed: int, per_event_cap: int = 3) -> list[dict]:
    """Grow _sample.json to ``target_total`` items, ADDING fresh paragraphs while
    keeping the existing ones (and thus their already-paid frontier/Opus labels).

    The existing ids are excluded from the new draw; the union is written back with
    the existing records FIRST (stable order), so previously-labeled ids keep their
    positions and the runner's resume-by-id re-labels only the new tail.
    """
    existing = load_sample()
    existing_ids = {r["id"] for r in existing}
    n_add = target_total - len(existing)
    if n_add <= 0:
        print(f"Sample already has {len(existing)} >= {target_total}; nothing to add.")
        return existing
    add_per_class = (n_add + 1) // 2
    drawn = build_sample(add_per_class, seed, per_event_cap, exclude_ids=existing_ids, write=False)
    new = [r for r in drawn if r["id"] not in existing_ids][:n_add]
    combined = existing + new
    SAMPLE_PATH.write_text(json.dumps(combined, indent=2))
    print(f"Expanded sample {len(existing)} -> {len(combined)} "
          f"(added {len(new)} new ids) -> {SAMPLE_PATH}")
    return combined


def build_population() -> dict:
    pop = pricing_mod.estimate_population(CLASSIFIED_PARAGRAPHS_MINIMAX_PATH)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    POPULATION_PATH.write_text(json.dumps(pop, indent=2))
    print(f"Population estimate: n_full_dedup~{pop['n_full_dedup_est']:,} "
          f"(raw~{pop['n_full_raw_est']:,}), p_pos={pop['p_pos']*100:.2f}% -> {POPULATION_PATH}")
    return pop


def load_population() -> dict:
    return json.loads(POPULATION_PATH.read_text())


def _apply_n_full(population: dict, n_full: int | None) -> dict:
    """Override the extrapolation size (e.g. a 5M-invocation scenario)."""
    if not n_full:
        return population
    return {**population, "n_full_dedup_est": n_full, "n_full_raw_est": n_full, "n_full_override": n_full}


def emit_opus_inputs(sample: list[dict], skip_existing: bool = True) -> Path:
    """Write one ready-to-judge prompt file per paragraph for the Opus fan-out.

    With ``skip_existing`` (default), ids already present in ``opus-4.8.jsonl`` are
    skipped, so after a sample expansion only the NEW paragraphs are emitted for the
    fan-out (avoids re-paying for already-labeled ids).
    """
    done: set[str] = set()
    opus_path = RAW_DIR / f"{report_mod.OPUS_LABELER}.jsonl"
    if skip_existing and opus_path.exists():
        done = {
            json.loads(line)["id"]
            for line in opus_path.read_text().splitlines()
            if line.strip()
        }
    d = RAW_DIR / "opus_inputs"
    d.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, rec in enumerate(sample):
        if rec["id"] in done:
            continue
        payload = {"index": i, "id": rec["id"], "judge_prompt": SYSTEM_PROMPT + _user_content(rec)}
        (d / f"{i}.json").write_text(json.dumps(payload, indent=2))
        written += 1
    print(f"Wrote {written} Opus input files (skipped {len(done)} already labeled) -> {d}")
    return d


# --------------------------------------------------------------------------- #
# OpenAI runner (self-metering)
# --------------------------------------------------------------------------- #
def _user_content(rec: dict) -> str:
    q = rec["query_info"]
    return (
        "\n===INPUT===\n"
        f"Sentence: {q.get('sentence', '')}\n"
        f"Abstract Entity Type: {q.get('abstract_entity_type', '')}\n"
        f"Constant Entity: {q.get('constant_entity', '')}\n"
        f"Relation: {q.get('relation', '')}\n"
        f"Paragraph: {rec.get('content', '')}\n\n"
        "===OUTPUT===\n"
    )


async def classify_one(
    model: str, effort: str, rec: dict, max_tokens: int, max_retries: int = 4
) -> dict:
    """Classify one paragraph; return a fully-priced record (ok or failed)."""
    client = get_client(model)
    supports_effort = provider_of(model) == "openai"  # DeepSeek has no reasoning_effort knob
    token_param = "max_completion_tokens" if supports_effort else "max_tokens"
    user_content = _user_content(rec)
    effort_used = effort
    last_err = ""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                token_param: max_tokens,
            }
            if supports_effort:
                kwargs["reasoning_effort"] = effort_used
            resp = await client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            msg = str(e).lower()
            if supports_effort and effort_used != "high" and "xhigh" in msg:  # fall back
                effort_used = "high"
                continue
            last_err = f"bad_request: {e}"
            break
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            last_err = str(e)
            await asyncio.sleep((2 ** attempt) + 1)
            continue
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            await asyncio.sleep(1)
            continue

        choice = resp.choices[0]
        content = choice.message.content or ""
        u = resp.usage
        cached = reasoning = 0
        try:
            if u.prompt_tokens_details:
                cached = u.prompt_tokens_details.cached_tokens or 0
        except Exception:
            pass
        try:
            if u.completion_tokens_details:
                reasoning = u.completion_tokens_details.reasoning_tokens or 0
        except Exception:
            pass
        usage = {
            "prompt_tokens": u.prompt_tokens,
            "cached_tokens": cached,
            "completion_tokens": u.completion_tokens,
            "reasoning_tokens": reasoning,
        }
        cost = pricing_mod.price_usage(usage, pricing_mod.PRICING[model])
        return {
            "id": rec["id"],
            "event_id": rec["event_id"],
            "minimax_label": rec["minimax_label"],
            "prediction": parse_classification_output(content),
            "finish_reason": choice.finish_reason,
            "effort_used": effort_used,
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "cost_as_measured": cost.cost_as_measured,
            "cost_uncached": cost.cost_uncached,
            "raw": content[:300],
        }

    return {
        "id": rec["id"],
        "event_id": rec["event_id"],
        "minimax_label": rec["minimax_label"],
        "prediction": -1,
        "finish_reason": "error",
        "effort_used": effort_used,
        "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
        "cost_as_measured": 0.0, "cost_uncached": 0.0,
        "error": last_err[:300],
    }


async def run_config(
    model: str, effort: str, sample: list[dict],
    concurrency: int, max_tokens: int, limit: int | None,
) -> None:
    out_file = RAW_DIR / report_mod.config_filename(model, effort)
    done_ids = {json.loads(l)["id"] for l in out_file.read_text().splitlines() if l.strip()} \
        if out_file.exists() else set()
    todo = [r for r in sample if r["id"] not in done_ids]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"[{model}:{effort}] all {len(sample)} already done.")
        return

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    fh = out_file.open("a")

    async def worker(rec: dict) -> dict:
        async with sem:
            res = await classify_one(model, effort, rec, max_tokens)
        async with write_lock:
            fh.write(json.dumps(res) + "\n")
            fh.flush()
        return res

    tasks = [worker(r) for r in todo]
    results = []
    for cor in async_tqdm.as_completed(tasks, total=len(tasks), desc=f"{model}:{effort}"):
        results.append(await cor)
    fh.close()

    spent = sum(r["cost_as_measured"] for r in results)
    failed = sum(1 for r in results if r["prediction"] == -1)
    mean_reason = sum(r["reasoning_tokens"] for r in results) / max(len(results), 1)
    print(f"[{model}:{effort}] {len(results)} done, {failed} failed, "
          f"mean_reasoning_tok={mean_reason:.0f}, spent=${spent:.3f}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run_all(configs, sample, concurrency, max_tokens, limit) -> None:
    for model, _ in configs:  # preflight: each provider's key must exist
        get_client(model)
    for model, effort in configs:  # config-by-config keeps each prompt cache warm
        await run_config(model, effort, sample, concurrency, max_tokens, limit)


def discover_configs() -> list[tuple[str, str]]:
    """All (model, effort) configs that have an on-disk results file."""
    out = []
    for f in sorted(RAW_DIR.glob("*__*.jsonl")):
        model, _, effort = f.stem.rpartition("__")
        if model:
            out.append((model, effort))
    return out


def _consensus_filename(label: str) -> str:
    """Map a consensus labeler key to its jsonl filename.

    "gpt-5.1:xhigh" -> "gpt-5.1__xhigh.jsonl"; "opus-4.8" -> "opus-4.8.jsonl".
    """
    if ":" in label:
        model, effort = label.split(":", 1)
        return report_mod.config_filename(model, effort)
    return f"{label}.jsonl"


def _apply_run_tag(tag: str, configs: list[tuple[str, str]]) -> None:
    """Isolate a prompt-variant run in data/judge_validation/<tag>/.

    Reuses the base _sample.json/_population.json (same ids, same extrapolation base)
    and copies the frozen-consensus labels in, so the variant report computes its
    %agree_consensus against the IDENTICAL gold — candidates (the new variant configs)
    are scored apples-to-apples against the same frontier consensus.
    """
    global RAW_DIR, REPORT_DIR, SAMPLE_PATH, POPULATION_PATH
    base_raw, base_report = RAW_DIR, REPORT_DIR
    RAW_DIR = base_raw.parent / tag
    REPORT_DIR = base_report.parent / tag
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH = base_raw / "_sample.json"          # reuse the same paragraphs
    POPULATION_PATH = base_raw / "_population.json"
    run_keys = {report_mod.config_key(m, e) for m, e in configs}
    for label in report_mod.CONSENSUS_LABELERS:
        if label in run_keys:
            print(f"NOTE: consensus labeler {label} is also being relabeled this run; "
                  f"the variant gold will use this run's labels for it.")
            continue
        src = base_raw / _consensus_filename(label)
        dst = RAW_DIR / _consensus_filename(label)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    # Human gold is the primary reference but not a consensus member; copy it in
    # (idempotently) so the variant report also computes the %agree_human column.
    human_src = base_raw / "human__gold.jsonl"
    human_dst = RAW_DIR / "human__gold.jsonl"
    if human_src.exists() and not human_dst.exists():
        shutil.copy2(human_src, human_dst)
    print(f"Run tag '{tag}': RAW_DIR={RAW_DIR}")


def main() -> None:
    global SYSTEM_PROMPT
    load_dotenv()
    ensure_provider_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--efforts", nargs="+", default=DEFAULT_EFFORTS)
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-completion-tokens", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=None, help="classify only first N sample paras/config")
    ap.add_argument("--n-full", type=int, default=None,
                    help="override full-dataset size for the cost extrapolation (e.g. 5000000 invocations)")
    ap.add_argument("--prompt-variant", choices=list(PROMPT_VARIANTS), default="short",
                    help="system-prompt variant (decomposed = explicit sub-checks + bounded reasoning)")
    ap.add_argument("--run-tag", default=None,
                    help="isolate outputs in data/judge_validation/<tag>/ (defaults to the "
                         "variant name for non-'short' variants)")
    ap.add_argument("--expand-to", type=int, default=None,
                    help="grow _sample.json to N items, adding fresh ids and KEEPING existing labels")
    ap.add_argument("--build-sample-only", action="store_true")
    ap.add_argument("--emit-opus-inputs", action="store_true", help="write per-paragraph judge files")
    ap.add_argument("--skip-openai", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    configs = [(m, e) for m in args.models for e in args.efforts]

    # Select the system prompt for this run (read by classify_one / emit_opus_inputs).
    SYSTEM_PROMPT = build_system_prompt(args.prompt_variant)

    # Isolate prompt-variant / tagged runs in their own directory. Plain "short" with
    # no explicit --run-tag stays in the canonical base dir.
    tag = args.run_tag or (args.prompt_variant if args.prompt_variant != "short" else None)
    if tag:
        _apply_run_tag(tag, configs)

    if args.report_only:
        population = _apply_n_full(load_population(), args.n_full)
        report = report_mod.write_report(
            RAW_DIR, REPORT_DIR, discover_configs() or configs, population, load_sample()
        )
        print(f"Report -> {report}")
        return

    # Sample expansion (P2): only against the canonical base sample, never a variant.
    if args.expand_to and not tag:
        expand_sample(args.expand_to, args.seed + 1)

    sample = build_sample(args.n_per_class, args.seed) if not SAMPLE_PATH.exists() else load_sample()
    population = _apply_n_full(
        build_population() if not POPULATION_PATH.exists() else load_population(), args.n_full
    )

    if args.emit_opus_inputs:
        emit_opus_inputs(sample)
        return

    if args.build_sample_only:
        return

    if not args.skip_openai:
        asyncio.run(run_all(configs, sample, args.concurrency, args.max_completion_tokens, args.limit))

    report = report_mod.write_report(
        RAW_DIR, REPORT_DIR, discover_configs() or configs, population, sample
    )
    print(f"Report -> {report}")


if __name__ == "__main__":
    main()
