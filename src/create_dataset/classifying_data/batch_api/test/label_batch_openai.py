"""Label the frontier-cost 200-paragraph sample with gpt-5.2 (medium) via OpenAI's Batch API.

This wraps the *validated best* synchronous labeler — ``gpt-5.2`` @ ``reasoning_effort="medium"``
with the ``short2b`` system prompt (validated in the paper's judge-selection study, which found
96.0% agreement with the 200-pair human gold) — in OpenAI's Batch API (50% cheaper, 24h SLA). The request
``body`` per paragraph is byte-identical to ``estimate_frontier_cost.classify_one``; only the
transport (batch vs synchronous) differs.

Target set: ``data/judge_validation/short/_sample.json`` — the same 200 paragraphs used throughout
the cost study. Human-gold ground truth (``human__gold.jsonl``) now covers all 200; agreement vs
human is reported on the overlap, exactly as the synchronous pipeline does.

Usage:
    # Build the input file only — no API calls, no spend (pre-flight check):
    uv run python -m src.create_dataset.classifying_data.batch_api.test.label_batch_openai submit --dry-run

    # End-to-end: submit, poll to completion, fetch + score + report:
    uv run python -m src.create_dataset.classifying_data.batch_api.test.label_batch_openai run

    # Or drive the steps manually (all resumable via batch_state.json):
    uv run python -m src.create_dataset.classifying_data.batch_api.test.label_batch_openai submit
    uv run python -m src.create_dataset.classifying_data.batch_api.test.label_batch_openai status
    uv run python -m src.create_dataset.classifying_data.batch_api.test.label_batch_openai fetch

    # Independent re-run (test-retest): namespace every output under a per-run dir
    # (data/judge_validation/test_retest/<tag>/); works with all subcommands:
    uv run python -m src.create_dataset.classifying_data.batch_api.test.label_batch_openai submit --run-tag retest_run1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# --- imports from the existing pipeline (repo root on path for the fallback) - #
_PKG_DIR = Path(__file__).resolve().parent      # .../classifying_data/batch_api/test
_PROJECT_ROOT = _PKG_DIR.parents[4]             # repo root
sys.path.insert(0, str(_PROJECT_ROOT))          # so `src.*` resolves as a plain script too

from src.create_dataset.classifying_data.relation_classification_parser import (  # noqa: E402
    parse_classification_output,
)
from src.create_dataset.classifying_data.prompt_variants import build_system_prompt  # noqa: E402
from src.create_dataset.classifying_data import frontier_cost_pricing as pricing_mod  # noqa: E402

_MODULE = "src.create_dataset.classifying_data.batch_api.test.label_batch_openai"

# --- config (mirrors the validated synchronous run) -------------------------- #
MODEL = "gpt-5.2"
EFFORT = "medium"
VARIANT = "short2b"
MAX_COMPLETION_TOKENS = 16000
ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"

# --- paths ------------------------------------------------------------------- #
FCE_DIR = _PROJECT_ROOT / "data/judge_validation/short"
SAMPLE_PATH = FCE_DIR / "_sample.json"
POPULATION_PATH = FCE_DIR / "_population.json"
HUMAN_GOLD_PATH = FCE_DIR / "human__gold.jsonl"
# Prior *synchronous* short2b run for the same (model, effort) — batch-vs-sync sanity check.
SYNC_SHORT2B_PATH = _PROJECT_ROOT / "data/judge_validation/short2b" / "gpt-5.2__medium.jsonl"

RETEST_DIR = _PROJECT_ROOT / "data/judge_validation/test_retest"
OUT_DIR = RETEST_DIR / "june_batch"
INPUT_PATH = OUT_DIR / "batch_input.jsonl"
STATE_PATH = OUT_DIR / "batch_state.json"
OUTPUT_RAW_PATH = OUT_DIR / "batch_output_raw.jsonl"
ERRORS_RAW_PATH = OUT_DIR / "batch_errors_raw.jsonl"
PREDICTIONS_PATH = OUT_DIR / "predictions.jsonl"

REPORT_DIR = _PROJECT_ROOT / "summary/judge_validation/test_retest"
REPORT_PATH = REPORT_DIR / "BATCH_TEST_REPORT.md"

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


def _apply_run_tag(tag: str | None) -> None:
    """Namespace all OUTPUT paths under a per-run subdir (test-retest runs).

    Inputs shared across runs (SAMPLE_PATH, HUMAN_GOLD_PATH, SYNC_SHORT2B_PATH,
    POPULATION_PATH) stay put. With no tag, the legacy (June-run) paths are used.
    """
    global OUT_DIR, INPUT_PATH, STATE_PATH, OUTPUT_RAW_PATH, ERRORS_RAW_PATH
    global PREDICTIONS_PATH, REPORT_DIR, REPORT_PATH
    if not tag:
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise SystemExit(f"Bad --run-tag {tag!r}: use only [A-Za-z0-9_-]+")
    OUT_DIR = RETEST_DIR / tag
    INPUT_PATH = OUT_DIR / "batch_input.jsonl"
    STATE_PATH = OUT_DIR / "batch_state.json"
    OUTPUT_RAW_PATH = OUT_DIR / "batch_output_raw.jsonl"
    ERRORS_RAW_PATH = OUT_DIR / "batch_errors_raw.jsonl"
    PREDICTIONS_PATH = OUT_DIR / "predictions.jsonl"
    REPORT_DIR = _PROJECT_ROOT / "summary/judge_validation/test_retest" / tag
    REPORT_PATH = REPORT_DIR / "BATCH_TEST_REPORT.md"


# --------------------------------------------------------------------------- #
# Request construction (byte-identical to the synchronous labeler)
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _user_content(rec: dict) -> str:
    """Mirror of ``estimate_frontier_cost._user_content`` (kept verbatim so the batch
    request matches the validated synchronous request exactly)."""
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


def load_sample() -> list[dict]:
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"Sample not found: {SAMPLE_PATH}")
    return json.loads(SAMPLE_PATH.read_text())


def build_input_file(sample: list[dict], system_prompt: str) -> int:
    """Write one Batch API request line per sample record. Returns the line count."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with INPUT_PATH.open("w") as fh:
        for rec in sample:
            body = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _user_content(rec)},
                ],
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "reasoning_effort": EFFORT,
            }
            line = {
                "custom_id": rec["id"],  # paragraph UUID — unique, < 64 chars
                "method": "POST",
                "url": ENDPOINT,
                "body": body,
            }
            fh.write(json.dumps(line) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# OpenAI client / batch state
# --------------------------------------------------------------------------- #
def _client() -> OpenAI:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (.env).")
    return OpenAI()


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise SystemExit(f"No batch state at {STATE_PATH}. Run 'submit' first.")
    return json.loads(STATE_PATH.read_text())


def _counts(batch) -> str:
    rc = getattr(batch, "request_counts", None)
    if rc is None:
        return "counts: (pending)"
    return f"counts: total={rc.total} completed={rc.completed} failed={rc.failed}"


# --------------------------------------------------------------------------- #
# Result parsing + scoring
# --------------------------------------------------------------------------- #
def parse_outputs(out_text: str) -> int:
    """Parse the batch output JSONL into predictions.jsonl. Returns the record count."""
    sample = {r["id"]: r for r in load_sample()}
    records: list[dict] = []
    for line in out_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        rec = sample.get(cid, {})
        base = {"id": cid, "event_id": rec.get("event_id"), "minimax_label": rec.get("minimax_label")}

        err = obj.get("error")
        resp = obj.get("response")
        if err or not resp or resp.get("status_code") != 200:
            status = resp.get("status_code") if resp else None
            records.append({
                **base, "prediction": -1, "finish_reason": "error",
                "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                "cost_as_measured": 0.0, "cost_uncached": 0.0,
                "error": (json.dumps(err)[:300] if err else f"http_status={status}"), "raw": "",
            })
            continue

        body = resp.get("body") or {}
        choice = (body.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        u = body.get("usage") or {}
        ptd = u.get("prompt_tokens_details") or {}
        ctd = u.get("completion_tokens_details") or {}
        usage = {
            "prompt_tokens": u.get("prompt_tokens", 0),
            "cached_tokens": ptd.get("cached_tokens", 0) or 0,
            "completion_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": ctd.get("reasoning_tokens", 0) or 0,
        }
        cost = pricing_mod.price_usage(usage, pricing_mod.PRICING[MODEL])
        records.append({
            **base,
            "prediction": parse_classification_output(content),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": usage["cached_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "cost_as_measured": cost.cost_as_measured,
            "cost_uncached": cost.cost_uncached,
            "raw": content[:300],
        })

    with PREDICTIONS_PATH.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return len(records)


def _load_predictions() -> list[dict]:
    return [json.loads(l) for l in PREDICTIONS_PATH.read_text().splitlines() if l.strip()]


def _load_label_map(path: Path, key: str = "prediction") -> dict[str, int]:
    """id -> int label from a jsonl of {id, <key>} rows (only 0/1 kept)."""
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        val = row.get(key)
        if val in (0, 1):
            out[row["id"]] = int(val)
    return out


def _agree(pred: dict[str, int], ref: dict[str, int]) -> tuple[int, int]:
    """(#matching, #comparable) over ids present in both with valid 0/1 on each side."""
    ids = [i for i in ref if i in pred and pred[i] in (0, 1) and ref[i] in (0, 1)]
    return sum(1 for i in ids if pred[i] == ref[i]), len(ids)


def write_report(state: dict, batch) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preds = _load_predictions()
    pred_map = {p["id"]: p["prediction"] for p in preds}
    n = len(preds)
    n_fail = sum(1 for p in preds if p["prediction"] == -1)
    n_pos = sum(1 for p in preds if p["prediction"] == 1)
    n_neg = sum(1 for p in preds if p["prediction"] == 0)

    gold = _load_label_map(HUMAN_GOLD_PATH)
    minimax = {p["id"]: p["minimax_label"] for p in preds if p["minimax_label"] in (0, 1)}
    sync = _load_label_map(SYNC_SHORT2B_PATH)

    h_agree, h_n = _agree(pred_map, gold)
    m_agree, m_n = _agree(pred_map, minimax)
    s_agree, s_n = _agree(pred_map, sync)

    sum_uncached = sum(p["cost_uncached"] for p in preds)
    sum_measured = sum(p["cost_as_measured"] for p in preds)
    disc = pricing_mod.batch_discount_for(MODEL)
    batch_cost = sum_uncached * disc
    n_priced = sum(1 for p in preds if p["prediction"] != -1) or 1
    per_para = batch_cost / n_priced
    tok = {k: sum(p[k] for p in preds)
           for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")}

    extrap = ""
    if POPULATION_PATH.exists():
        n_full = json.loads(POPULATION_PATH.read_text()).get("n_full_dedup_est")
        if n_full:
            extrap = (f"- Extrapolated to the full deduped dataset (~{n_full:,} paragraphs): "
                      f"**${per_para * n_full:,.0f}** at the batch rate.\n")

    def pct(a: int, b: int) -> str:
        return f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a (no overlap)"

    lines = [
        "# Batch API labeling — 200-sample test run",
        "",
        f"_Generated {_now_iso()}_",
        "",
        "## Config",
        f"- Model `{MODEL}` · reasoning effort `{EFFORT}` · prompt variant `{VARIANT}`",
        f"- Endpoint `{ENDPOINT}` · Batch API ({COMPLETION_WINDOW} window) · "
        f"max_completion_tokens {MAX_COMPLETION_TOKENS}",
        f"- Batch id `{state['batch_id']}` · submitted {state.get('submitted_at')} · "
        f"final status `{batch.status}`",
        f"- Request {_counts(batch)}",
        "",
        "## Outcomes",
        f"- Predictions: **{n}** (positive {n_pos}, negative {n_neg}, unparseable/`-1` **{n_fail}**)",
        "",
        "## Agreement",
        f"- **vs human gold** (overlap): **{pct(h_agree, h_n)}** "
        f"— headline; the synchronous `gpt-5.2:medium`+`short2b` run scored 96.0%",
        f"- vs minimax label (all valid): {pct(m_agree, m_n)}",
    ]
    if s_n:
        lines.append(
            f"- vs prior **synchronous** `short2b` run (same model/effort): "
            f"{pct(s_agree, s_n)} identical — batch-vs-sync sanity check"
        )
    lines += [
        "",
        "## Cost (self-metered from `response.usage`)",
        f"- Tokens: prompt {tok['prompt_tokens']:,} (cached {tok['cached_tokens']:,}), "
        f"completion {tok['completion_tokens']:,} (reasoning {tok['reasoning_tokens']:,})",
        f"- Sync-equivalent, uncached upper bound: ${sum_uncached:.4f}",
        f"- **Batch (×{disc:g}): ${batch_cost:.4f}** → **${per_para * 1000:.2f} / 1k paragraphs** "
        f"(${per_para:.5f} each, over {n_priced} priced)",
        f"- Warm-cache as-measured (reference): ${sum_measured:.4f}",
    ]
    if extrap:
        lines += ["", "## Extrapolation", extrap]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_submit(args: argparse.Namespace) -> None:
    if not getattr(args, "run_tag", None) and (STATE_PATH.exists() or INPUT_PATH.exists()):
        raise SystemExit(
            f"Refusing untagged submit: legacy outputs exist under {OUT_DIR} "
            "(the preserved June run). Pass --run-tag <name> to namespace this run."
        )
    sample = load_sample()
    system_prompt = build_system_prompt(VARIANT)
    n = build_input_file(sample, system_prompt)
    size_mb = INPUT_PATH.stat().st_size / 1e6
    print(f"Built {INPUT_PATH}")
    print(f"  {n} requests · {size_mb:.2f} MB · model={MODEL} effort={EFFORT} "
          f"prompt={VARIANT} max_tok={MAX_COMPLETION_TOKENS}")
    print(f"  system prompt: {len(system_prompt)} chars (variant '{VARIANT}')")
    if getattr(args, "dry_run", False):
        print("--dry-run: no upload, no batch created.")
        return

    client = _client()
    with INPUT_PATH.open("rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint=ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={"description": f"relation-classification {VARIANT} {MODEL}:{EFFORT} "
                                 f"200-sample {args.run_tag or 'test'}"},
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": up.id,
        "model": MODEL, "effort": EFFORT, "variant": VARIANT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "endpoint": ENDPOINT, "n_requests": n,
        "run_tag": getattr(args, "run_tag", None),
        "submitted_at": _now_iso(),
        "status": batch.status,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"Submitted batch {batch.id} (status={batch.status}). State -> {STATE_PATH}")


def cmd_status(args: argparse.Namespace) -> None:
    state = load_state()
    batch = _client().batches.retrieve(state["batch_id"])
    print(f"batch {batch.id}: status={batch.status}  {_counts(batch)}")
    if batch.status in {"failed", "expired", "cancelled"} and getattr(batch, "errors", None):
        print("errors:", batch.errors)


def cmd_fetch(args: argparse.Namespace) -> None:
    state = load_state()
    client = _client()
    batch = client.batches.retrieve(state["batch_id"])
    print(f"batch {batch.id}: status={batch.status}  {_counts(batch)}")
    if batch.status != "completed":
        print(f"Not completed (status={batch.status}); nothing to fetch.")
        if getattr(batch, "errors", None):
            print("errors:", batch.errors)
        return
    if not batch.output_file_id:
        print("Completed but no output_file_id present; cannot fetch results.")
        return

    out_text = client.files.content(batch.output_file_id).text
    OUTPUT_RAW_PATH.write_text(out_text)
    print(f"Downloaded results -> {OUTPUT_RAW_PATH}")
    if batch.error_file_id:
        ERRORS_RAW_PATH.write_text(client.files.content(batch.error_file_id).text)
        print(f"Downloaded request errors -> {ERRORS_RAW_PATH}")

    n_pred = parse_outputs(out_text)
    print(f"Parsed {n_pred} predictions -> {PREDICTIONS_PATH}")
    write_report(state, batch)
    print(f"Report -> {REPORT_PATH}")


def cmd_run(args: argparse.Namespace) -> None:
    cmd_submit(args)
    if getattr(args, "dry_run", False):
        return
    state = load_state()
    client = _client()
    deadline = time.monotonic() + args.max_wait
    while True:
        batch = client.batches.retrieve(state["batch_id"])
        print(f"[{_now_iso()}] status={batch.status}  {_counts(batch)}")
        if batch.status in TERMINAL_STATES:
            break
        if time.monotonic() >= deadline:
            print(f"\nMax wait ({args.max_wait}s) exceeded; batch still '{batch.status}'.")
            tag_opt = f" --run-tag {args.run_tag}" if getattr(args, "run_tag", None) else ""
            print(f"Resume later:  uv run python -m {_MODULE} fetch{tag_opt}")
            return
        time.sleep(args.poll_interval)

    if batch.status == "completed":
        cmd_fetch(args)
    else:
        print(f"Batch ended as '{batch.status}'.")
        if getattr(batch, "errors", None):
            print("errors:", batch.errors)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-tag", default=None,
                        help="namespace all outputs under .../batch_api/<tag>/ "
                             "(legacy June-run paths when omitted)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", parents=[common],
                              help="build input file, upload, create the batch")
    p_submit.add_argument("--dry-run", action="store_true",
                          help="build the input file only — no API calls, no spend")
    p_submit.set_defaults(func=cmd_submit)

    sub.add_parser("status", parents=[common],
                   help="print batch status + request counts").set_defaults(func=cmd_status)
    sub.add_parser("fetch", parents=[common],
                   help="download results, parse, score, write report").set_defaults(func=cmd_fetch)

    p_run = sub.add_parser("run", parents=[common],
                           help="submit + poll to completion + fetch (end-to-end)")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--poll-interval", type=int, default=30, help="seconds between status polls")
    p_run.add_argument("--max-wait", type=int, default=3600,
                       help="give up polling after N seconds (batch keeps running; use 'fetch' later)")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    _apply_run_tag(args.run_tag)
    args.func(args)


if __name__ == "__main__":
    main()
