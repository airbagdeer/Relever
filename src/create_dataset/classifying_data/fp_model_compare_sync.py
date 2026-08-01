"""Judge-model comparison for the FP presence bits, on the 100-pair human gold.

Runs each (model, prompt_version) configuration over the SAME 100 gold pairs via the
synchronous chat-completions API (the population is small, so no Batch API round-trip)
and reports per-bit and exact-8-cell agreement with the human gold — the same
selection protocol as the benchmark's judge study (candidate configs scored on one
author-labeled gold set, best kappa wins).

Request bodies mirror the batch pipeline exactly (same system prompt builder, same
user content, same max_completion_tokens / reasoning_effort); only the transport is
synchronous. Raw per-pair outputs are cached under
``data/fp_taxonomy/validation/model_compare/<model>__<version>.jsonl`` so re-runs are free;
the rollup lands in ``summary/fp_taxonomy/model_compare.{md,csv}``.

    uv run python src/create_dataset/classifying_data/fp_model_compare_sync.py \\
        --models gpt-5.2 gpt-5.6-luna --versions v3 v4
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import NotFoundError  # noqa: E402

from src.create_dataset.classifying_data.batch_api.label_fp_taxonomy_batch import (  # noqa: E402
    PILOT_DIR,
    _texts_for_event,
    load_predictions_for_event,
)
from src.create_dataset.classifying_data.batch_api import label_frontier_batch as lfb  # noqa: E402
from src.create_dataset.classifying_data.fp_taxonomy_agreement import (  # noqa: E402
    cohen_kappa,
    load_gold,
)
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (  # noqa: E402
    BITS,
    build_fp_bits_prompt,
    parse_fp_bits_output,
)
from src.retrieval.config import SUMMARY_FP_TAXONOMY_PATH  # noqa: E402

COMPARE_DIR = PILOT_DIR / "model_compare"
DEFAULT_GOLD = PILOT_DIR / "fp_human__gold.jsonl"
MAX_COMPLETION_TOKENS = lfb.MAX_COMPLETION_TOKENS

_print_lock = threading.Lock()


def _resolve_model(client, requested: str) -> str:
    """Return a usable model id; if ``requested`` 404s, search the model list for
    the closest name (helps with dated ids like gpt-5.6-luna-2026-XX)."""
    try:
        client.models.retrieve(requested)
        return requested
    except NotFoundError:
        pass
    names = [m.id for m in client.models.list()]
    needles = [requested, requested.replace("gpt-", "")] + requested.split("-")[-1:]
    candidates = [n for n in names if requested in n] or \
                 [n for n in names if all(p in n for p in needles[-1:])]
    if not candidates:
        raise SystemExit(f"Model '{requested}' not found and no close match in "
                         f"{len(names)} available models.")
    best = sorted(candidates, key=len)[0]
    print(f"NOTE: model '{requested}' not found; using closest match '{best}'.")
    return best


def _one_call(client, model: str, effort: str | None, system_prompt: str,
              qinfo: dict, text: str) -> tuple[dict | None, dict, str]:
    """One sync classification call. Returns (bits|None, usage, raw_content)."""
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": lfb._user_content(qinfo, text)}],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    if effort:
        kwargs["reasoning_effort"] = effort
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — retry once without reasoning_effort
        if effort and "reasoning" in str(exc).lower():
            kwargs.pop("reasoning_effort", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    content = (resp.choices[0].message.content or "") if resp.choices else ""
    u = resp.usage
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
    }
    return parse_fp_bits_output(content), usage, content


def run_config(client, model: str, version: str, effort: str | None,
               gold: list[dict], texts: dict[str, str], qmeta: dict[int, dict],
               workers: int, force: bool) -> Path:
    """Classify all gold pairs under one (model, version) config; cache to jsonl."""
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPARE_DIR / f"{model.replace('/', '_')}__{version}.jsonl"
    done: dict[str, dict] = {}
    if out_path.exists() and not force:
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    todo = [g for g in gold if g["id"] not in done]
    if not todo:
        print(f"[{model} × {version}] all {len(gold)} cached -> {out_path.name}")
        return out_path
    system_prompt = build_fp_bits_prompt(version)
    print(f"[{model} × {version}] classifying {len(todo)} pairs "
          f"({len(done)} cached) with {workers} workers...")

    tok: Counter = Counter()
    fails: list[str] = []

    def _work(g: dict) -> tuple[str, dict | None, dict]:
        qinfo = qmeta[g["event_id"]]
        bits, usage, _ = _one_call(client, model, effort, system_prompt,
                                   qinfo, texts[g["id"]])
        if bits is None:  # one retry on parse failure
            bits, usage2, _ = _one_call(client, model, effort, system_prompt,
                                        qinfo, texts[g["id"]])
            for k, v in usage2.items():
                usage[k] += v
        return g["id"], bits, usage

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, g): g for g in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            g = futs[fut]
            try:
                cid, bits, usage = fut.result()
            except Exception as exc:  # noqa: BLE001 — record and continue
                with _print_lock:
                    print(f"  ERROR {g['id'][:8]}: {type(exc).__name__}: {exc}")
                fails.append(g["id"])
                continue
            for k, v in usage.items():
                tok[k] += v
            if bits is None:
                fails.append(cid)
                continue
            done[cid] = {"id": cid, "event_id": g["event_id"], **bits,
                         "model": model, "prompt_version": version}
            if i % 25 == 0:
                with _print_lock:
                    print(f"  [{i}/{len(todo)}]")

    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for r in done.values():
            fh.write(json.dumps(r) + "\n")
    tmp.replace(out_path)
    n_req = max(len(todo) - len(fails), 1)
    print(f"[{model} × {version}] done: {len(done)}/{len(gold)} "
          + (f"({len(fails)} failed) " if fails else "")
          + f"· {tok['prompt_tokens'] / n_req:,.0f} in / "
            f"{tok['completion_tokens'] / n_req:,.0f} out tok/req")
    return out_path


def _agreement_row(name: str, preds: dict[str, dict], gold: list[dict]) -> dict | None:
    joined = [( g, preds[g["id"]] ) for g in gold if g["id"] in preds
              and all(isinstance(preds[g["id"]].get(b), bool) for b in BITS)]
    if not joined:
        return None
    n = len(joined)
    row: dict = {"config": name, "n": n}
    for b in BITS:
        pairs: list[tuple[str, str]] = [(str(g[b]), str(p[b])) for g, p in joined]
        row[f"{b}_pct"] = sum(1 for a, c in pairs if a == c) / n
        row[f"{b}_kappa"] = cohen_kappa(pairs)
    cell = lambda r: "/".join("1" if r[b] else "0" for b in BITS)  # noqa: E731
    cpairs: list[tuple[str, str]] = [(cell(g), cell(p)) for g, p in joined]
    row["cell_pct"] = sum(1 for a, c in cpairs if a == c) / n
    row["cell_kappa"] = cohen_kappa(cpairs)
    row["mean_bit_kappa"] = sum(row[f"{b}_kappa"] for b in BITS) / len(BITS)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["gpt-5.2", "gpt-5.6-luna"])
    ap.add_argument("--versions", nargs="+", default=["v3", "v4"])
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--force", action="store_true", help="ignore cached outputs")
    args = ap.parse_args()

    gold = load_gold(Path(args.gold))
    if not gold:
        raise SystemExit("No usable gold rows.")
    qmeta = lfb.load_query_meta()
    texts: dict[str, str] = {}
    by_event: dict[int, set[str]] = {}
    for g in gold:
        by_event.setdefault(g["event_id"], set()).add(g["id"])
    for eid, ids in by_event.items():
        texts.update(_texts_for_event(eid, ids))

    client = lfb._client()
    rows: list[dict] = []

    # Reference row: the existing v3 batch-pipeline predictions.
    batch_preds: dict[str, dict] = {}
    for eid in by_event:
        for cid, r in load_predictions_for_event(eid).items():
            if r.get("prompt_version") == "v3" and all(isinstance(r.get(b), bool) for b in BITS):
                batch_preds[cid] = r
    ref = _agreement_row("gpt-5.2 × v3 (batch pipeline)", batch_preds, gold)
    if ref:
        rows.append(ref)

    for requested in args.models:
        model = _resolve_model(client, requested)
        for version in args.versions:
            out_path = run_config(client, model, version, args.effort, gold,
                                  texts, qmeta, args.workers, args.force)
            preds = {}
            for line in out_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    preds[r["id"]] = r
            row = _agreement_row(f"{model} × {version} (sync)", preds, gold)
            if row:
                rows.append(row)

    # ---- report --------------------------------------------------------------
    hdr = (f"{'config':<34} {'inst%':>6} {'κ':>6} {'rel%':>6} {'κ':>6} "
           f"{'const%':>6} {'κ':>6} {'cell%':>6} {'κ':>6} {'meanκ':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:<34} "
              f"{r['instance_present_pct']:>6.1%} {r['instance_present_kappa']:>6.3f} "
              f"{r['relation_present_pct']:>6.1%} {r['relation_present_kappa']:>6.3f} "
              f"{r['constant_present_pct']:>6.1%} {r['constant_present_kappa']:>6.3f} "
              f"{r['cell_pct']:>6.1%} {r['cell_kappa']:>6.3f} {r['mean_bit_kappa']:>6.3f}")

    SUMMARY_FP_TAXONOMY_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_FP_TAXONOMY_PATH / "model_compare.csv"
    keys = ["config", "n"] + [f"{b}_{s}" for b in BITS for s in ("pct", "kappa")] + \
           ["cell_pct", "cell_kappa", "mean_bit_kappa"]
    with csv_path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    md_path = SUMMARY_FP_TAXONOMY_PATH / "model_compare.md"
    with md_path.open("w") as fh:
        fh.write("# FP presence-bits judge comparison — vs 100-pair human gold\n\n")
        fh.write("| config | n | instance % / κ | relation % / κ | constant % / κ "
                 "| exact cell % / κ | mean bit κ |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['config']} | {r['n']} "
                     f"| {r['instance_present_pct']:.1%} / {r['instance_present_kappa']:.3f} "
                     f"| {r['relation_present_pct']:.1%} / {r['relation_present_kappa']:.3f} "
                     f"| {r['constant_present_pct']:.1%} / {r['constant_present_kappa']:.3f} "
                     f"| {r['cell_pct']:.1%} / {r['cell_kappa']:.3f} "
                     f"| {r['mean_bit_kappa']:.3f} |\n")
    print(f"\n-> {md_path}\n-> {csv_path}")


if __name__ == "__main__":
    main()
