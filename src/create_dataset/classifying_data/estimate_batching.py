"""Measure the cost/accuracy of *same-query* multi-paragraph batching.

The task scores ~870 candidate paragraphs against one relation query, so the natural
batch puts N paragraphs from the SAME query in one call (the system prompt + the query
framing are written once for all N). This script runs gpt-5.2 @ medium at N = 1, 5, 10,
20 on a fixed 5-query x 20-paragraph sample, self-meters cost/paragraph, and checks
whether batching changes the verdict vs the N=1 baseline.

    uv run python src/create_dataset/classifying_data/estimate_batching.py
    uv run python src/create_dataset/classifying_data/estimate_batching.py --report-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

try:
    from . import estimate_frontier_cost as efc
    from . import frontier_cost_pricing as pricing_mod
    from .relation_classification_prompt import relation_classification_prompt_short
except ImportError:  # run as a plain script
    import estimate_frontier_cost as efc
    import frontier_cost_pricing as pricing_mod
    from relation_classification_prompt import relation_classification_prompt_short

MODEL = "gpt-5.2"
EFFORT = "medium"
BATCH_SIZES = [1, 5, 10, 20]
N_EVENTS = 5
PER_EVENT = 20
N_FULL = 2_200_000  # match the headline scenario
N_FULL_LABEL = "2.2M"
BATCH_SAMPLE_PATH = efc.RAW_DIR / "_batch_sample.json"

_BATCH_OUTPUT_INSTR = (
    'You will be given the query once, then a numbered list of paragraphs. Return ONLY a '
    'JSON object mapping each paragraph number (as a string) to true or false — e.g. '
    '{"1": true, "2": false}. Include every number exactly once; no other text.'
)
BATCH_SYSTEM = relation_classification_prompt_short.format(format_instructions=_BATCH_OUTPUT_INSTR)


# --------------------------------------------------------------------------- #
# Sampling: 5 queries x 20 paragraphs (so N=20 batches fit within one query)
# --------------------------------------------------------------------------- #
def build_batch_sample(seed: int = 42) -> list[dict]:
    relations = efc._load_relations()
    files = sorted(efc.CLASSIFIED_PARAGRAPHS_MINIMAX_PATH.glob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(files)
    events: list[list[dict]] = []
    for f in files:
        eid = int(f.stem)
        if eid not in relations:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        q = relations[eid]
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        paras: list[dict] = []
        for item in data:
            rid = item.get("id")
            lbl = str(item.get("classification"))
            content = (item.get("content") or "").strip()
            if not rid or rid in seen_ids or lbl not in ("0", "1") or not content:
                continue
            norm = content.lower()[:200]
            if norm in seen_content:
                continue
            seen_ids.add(rid)
            seen_content.add(norm)
            paras.append({
                "id": rid,
                "event_id": eid,
                "minimax_label": int(lbl),
                "content": content,
                "query_info": {
                    "sentence": q.get("sentence", ""),
                    "abstract_entity_type": q.get("abstract_entity_type", ""),
                    "constant_entity": q.get("constant_entity", ""),
                    "relation": q.get("relation", ""),
                },
            })
            if len(paras) >= PER_EVENT:
                break
        if len(paras) >= PER_EVENT:
            events.append(paras)
        if len(events) >= N_EVENTS:
            break
    if len(events) < N_EVENTS:
        raise RuntimeError(f"Only {len(events)} events with >= {PER_EVENT} paragraphs found.")
    sample = [p for ev in events for p in ev]
    efc.RAW_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_SAMPLE_PATH.write_text(json.dumps(sample, indent=2))
    print(f"Built batch sample: {N_EVENTS} queries x {PER_EVENT} = {len(sample)} paragraphs "
          f"-> {BATCH_SAMPLE_PATH}")
    return sample


def load_batch_sample() -> list[dict]:
    return json.loads(BATCH_SAMPLE_PATH.read_text())


# --------------------------------------------------------------------------- #
# Batched classification
# --------------------------------------------------------------------------- #
def _batch_user_content(batch: list[dict]) -> str:
    q = batch[0]["query_info"]
    paras = "\n\n".join(f"[{i + 1}] {rec['content']}" for i, rec in enumerate(batch))
    return (
        "\n===QUERY===\n"
        f"Sentence: {q['sentence']}\n"
        f"Abstract Entity Type: {q['abstract_entity_type']}\n"
        f"Constant Entity: {q['constant_entity']}\n"
        f"Relation: {q['relation']}\n\n"
        f"Apply the rules to EACH of the {len(batch)} paragraphs below — all concern the SAME "
        "query above. Judge every paragraph independently.\n\n"
        "===PARAGRAPHS===\n"
        f"{paras}\n\n"
        "===OUTPUT===\n"
    )


def parse_batch(raw: str, n: int) -> dict[int, int]:
    labels = {i: -1 for i in range(1, n + 1)}
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return labels
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return labels
    for i in range(1, n + 1):
        v = obj.get(str(i), obj.get(i))
        if isinstance(v, bool):
            labels[i] = 1 if v else 0
        elif isinstance(v, (int, float)) and v in (0, 1):
            labels[i] = int(v)
        elif isinstance(v, str):
            t = v.strip().lower()
            labels[i] = 1 if t in ("true", "1", "yes") else 0 if t in ("false", "0", "no") else -1
    return labels


async def classify_batch(client, batch: list[dict], max_tokens: int, max_retries: int = 4) -> list[dict]:
    n = len(batch)
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": BATCH_SYSTEM},
                    {"role": "user", "content": _batch_user_content(batch)},
                ],
                reasoning_effort=EFFORT,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            await asyncio.sleep((2 ** attempt) + 1)
            continue
        choice = resp.choices[0]
        labels = parse_batch(choice.message.content or "", n)
        u = resp.usage
        cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
        reasoning = (u.completion_tokens_details.reasoning_tokens or 0) if u.completion_tokens_details else 0
        cost = pricing_mod.price_usage(
            {"prompt_tokens": u.prompt_tokens, "cached_tokens": cached,
             "completion_tokens": u.completion_tokens, "reasoning_tokens": reasoning},
            pricing_mod.PRICING[MODEL],
        )
        return [
            {
                "id": rec["id"], "event_id": rec["event_id"], "minimax_label": rec["minimax_label"],
                "N": n, "prediction": labels[i + 1], "finish_reason": choice.finish_reason,
                "per_para_cost": cost.cost_as_measured / n,
                "per_para_reasoning": reasoning / n,
                "batch_size": n,
            }
            for i, rec in enumerate(batch)
        ]
    return [
        {"id": rec["id"], "event_id": rec["event_id"], "minimax_label": rec["minimax_label"],
         "N": n, "prediction": -1, "finish_reason": "error", "per_para_cost": 0.0,
         "per_para_reasoning": 0.0, "batch_size": n, "error": last_err[:200]}
        for rec in batch
    ]


def _out_file(n: int):
    return efc.RAW_DIR / f"batch_{MODEL}-{EFFORT}_N{n}.jsonl"


async def run_n(n: int, sample: list[dict], concurrency: int, max_tokens: int) -> None:
    by_event: dict[int, list[dict]] = {}
    for rec in sample:
        by_event.setdefault(rec["event_id"], []).append(rec)
    batches = [paras[i:i + n] for paras in by_event.values() for i in range(0, len(paras), n)]

    client = efc.get_client(MODEL)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    fh = _out_file(n).open("w")

    async def do(batch):
        async with sem:
            recs = await classify_batch(client, batch, max_tokens)
        async with lock:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
        return recs

    tasks = [do(b) for b in batches]
    all_recs = []
    for cor in async_tqdm.as_completed(tasks, total=len(tasks), desc=f"N={n} ({len(batches)} calls)"):
        all_recs.extend(await cor)
    fh.close()
    cpp = sum(r["per_para_cost"] for r in all_recs) / max(len(all_recs), 1)
    failed = sum(1 for r in all_recs if r["prediction"] == -1)
    print(f"[N={n}] {len(batches)} calls, {len(all_recs)} paragraphs, {failed} failed, "
          f"cost/para=${cpp:.6f}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report() -> str:
    import pandas as pd

    def read(n):
        f = _out_file(n)
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []

    base = {r["id"]: r["prediction"] for r in read(1) if r["prediction"] in (0, 1)}
    rows = []
    for n in BATCH_SIZES:
        recs = read(n)
        if not recs:
            continue
        calls = N_EVENTS * (PER_EVENT // n)
        cpp = sum(r["per_para_cost"] for r in recs) / len(recs)
        rpp = sum(r["per_para_reasoning"] for r in recs) / len(recs)
        failed = sum(1 for r in recs if r["prediction"] == -1)
        shared = [r for r in recs if r["id"] in base and r["prediction"] in (0, 1)]
        agree = (sum(1 for r in shared if r["prediction"] == base[r["id"]]) / len(shared) * 100) \
            if shared else float("nan")
        rows.append({
            "N": n, "calls": calls, "mean_reasoning_per_para": round(rpp, 1),
            "usd_per_1k_para": round(cpp * 1000, 4),
            f"full_{N_FULL_LABEL}_sync_usd": round(cpp * N_FULL, 2),
            f"full_{N_FULL_LABEL}_batch_usd": round(cpp * N_FULL * pricing_mod.batch_discount_for(MODEL), 2),
            "%agree_vs_N1": round(agree, 1) if agree == agree else None,
            "n_failed": failed,
        })
    df = pd.DataFrame(rows)
    efc.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(efc.REPORT_DIR / "batching_metrics.csv", index=False)

    base_cpp = df.loc[df["N"] == 1, "usd_per_1k_para"]
    base_v = float(base_cpp.iloc[0]) if len(base_cpp) else float("nan")
    L = [
        f"# Same-query batching — {MODEL} @ {EFFORT}\n",
        f"Sample: {N_EVENTS} queries x {PER_EVENT} paragraphs = {N_EVENTS * PER_EVENT}. "
        f"One call packs N same-query paragraphs (shared system prompt + query framing) and "
        f"returns N labels. Cost is self-metered per batch and divided by N. Accuracy = "
        f"agreement with the N=1 (one-paragraph-per-call) verdicts.\n",
        df.to_markdown(index=False),
        "",
        f"- N=1 baseline = **${base_v}/1k para**. The `%agree_vs_N1` column is the batching "
        "accuracy cost (100% = batching never changes a label).",
        f"- `full_{N_FULL_LABEL}_*` extrapolate cost/para to {N_FULL:,} invocations (sync and 50% Batch API).",
        "- Caveat: once the shared prefix is cache-warm its cost is already small, so the input "
        "saving from batching shrinks; the call-count drop (N x fewer calls) is the durable win.",
    ]
    out = efc.REPORT_DIR / "BATCHING.md"
    out.write_text("\n".join(L))
    return str(out)


async def _run_all(concurrency: int, max_tokens: int) -> None:
    efc.get_client(MODEL)  # preflight key
    sample = load_batch_sample() if BATCH_SAMPLE_PATH.exists() else build_batch_sample()
    for n in BATCH_SIZES:
        await run_n(n, sample, concurrency, max_tokens)


def main() -> None:
    load_dotenv()
    efc.ensure_provider_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-completion-tokens", type=int, default=32000)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--build-sample-only", action="store_true")
    args = ap.parse_args()

    if args.build_sample_only:
        build_batch_sample()
        return
    if not args.report_only:
        asyncio.run(_run_all(args.concurrency, args.max_completion_tokens))
    print(f"Report -> {build_report()}")


if __name__ == "__main__":
    main()
