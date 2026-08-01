"""Classify the false-positive population into the 5-category taxonomy via the Batch API.

Input population: ``data/fp_taxonomy/fp_population.jsonl`` (written by
``src/retrieval/lancedb/frontier-v2/evaluation/extract_fp_population.py``) — the
deduplicated union of every system's top-10 judged-negative paragraphs. Each unique
(query, paragraph) pair is classified ONCE with gpt-5.2 @ medium reasoning effort —
the benchmark judge's exact configuration and input format; only the system prompt
(the taxonomy in ``fp_taxonomy_prompt``) differs.

Unlike the per-query judge pipeline (1050 batches), the FP population is small enough
to run as a handful of **shards**: the pilot is one shard ("pilot"); the full run is
``full_000``, ``full_001``, ... of ``--shard-size`` requests each. The manifest
(``_manifest.json``) tracks shards; predictions always land in the **per-event layout**
``data/fp_taxonomy/<qid>/fp_taxonomy_<qid>.jsonl`` with each record carrying the prompt
version, so a prompt iteration re-classifies (overwrites by id) and a resumed full run
skips anything already classified under the current version.

Lifecycle::

    # 1. stratified pilot selection (no API calls):
    uv run python -m src.create_dataset.classifying_data.batch_api.label_fp_taxonomy_batch \\
        select-pilot --n 200 --n-human 100

    # 2. build batch input(s) (no API calls):
    uv run python -m ...label_fp_taxonomy_batch build --scope pilot
    uv run python -m ...label_fp_taxonomy_batch build --scope full

    # 3. submit / poll / collect:
    uv run python -m ...label_fp_taxonomy_batch submit [--limit N] [--dry-run]
    uv run python -m ...label_fp_taxonomy_batch status
    uv run python -m ...label_fp_taxonomy_batch fetch

    # 4. pilot distribution + usage + projected full-run cost:
    uv run python -m ...label_fp_taxonomy_batch report
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the judge pipeline's helpers so transport + user content stay identical.
from src.create_dataset.classifying_data.batch_api import label_frontier_batch as lfb
from src.create_dataset.classifying_data import frontier_cost_pricing as pricing_mod
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (
    BITS,
    FP_TAXONOMY_PROMPT_VERSION,
    build_fp_taxonomy_system_prompt,
    parse_fp_bits_output,
)
from src.retrieval.config import FP_TAXONOMY_DATA_PATH

# --- config (transport mirrors the validated judge run) ---------------------- #
MODEL = lfb.MODEL                                # gpt-5.2
EFFORT = lfb.EFFORT                              # medium
MAX_COMPLETION_TOKENS = lfb.MAX_COMPLETION_TOKENS
ENDPOINT = lfb.ENDPOINT
COMPLETION_WINDOW = lfb.COMPLETION_WINDOW
TERMINAL_STATES = lfb.TERMINAL_STATES

FP_ROOT = FP_TAXONOMY_DATA_PATH
POPULATION_PATH = FP_ROOT / "fp_population.jsonl"
PILOT_DIR = FP_ROOT / "validation"
PILOT_PAIRS_PATH = PILOT_DIR / "pilot_pairs.json"
BATCH_DIR = FP_ROOT / "batch"
MANIFEST_PATH = FP_ROOT / "_manifest.json"

DEFAULT_SHARD_SIZE = 12_500


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _qid(event_id: int) -> str:
    return str(event_id).zfill(4)


def predictions_path(event_id: int) -> Path:
    q = _qid(event_id)
    return FP_ROOT / q / f"fp_taxonomy_{q}.jsonl"


def shard_input_path(shard: str) -> Path:
    return BATCH_DIR / f"{shard}_input.jsonl"


def shard_output_path(shard: str) -> Path:
    return BATCH_DIR / f"{shard}_output.jsonl"


def shard_errors_path(shard: str) -> Path:
    return BATCH_DIR / f"{shard}_errors.jsonl"


def load_population() -> list[dict]:
    if not POPULATION_PATH.exists():
        raise SystemExit(f"Population not found: {POPULATION_PATH} — run "
                         "extract_fp_population.py first.")
    rows = [json.loads(l) for l in POPULATION_PATH.read_text().splitlines() if l.strip()]
    return rows


def load_predictions_for_event(event_id: int) -> dict[str, dict]:
    """{chunk_id: record} for one event (empty if no file yet)."""
    p = predictions_path(event_id)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _write_predictions_for_event(event_id: int, records: dict[str, dict]) -> None:
    p = predictions_path(event_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for r in records.values():
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, p)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        m = json.loads(MANIFEST_PATH.read_text())
        m.setdefault("build_rounds", 0)
        return m
    return {"config": None, "updated_at": None, "build_rounds": 0, "shards": {}}


def save_manifest(m: dict) -> None:
    m["updated_at"] = lfb._now_iso()
    FP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    os.replace(tmp, MANIFEST_PATH)


def _empty_shard(scope: str) -> dict:
    return {
        "scope": scope, "n_requests": None, "input_bytes": None, "built_at": None,
        "batch_id": None, "input_file_id": None, "submitted_at": None, "status": None,
        "request_counts": None, "output_file_id": None, "error_file_id": None,
        "fetched": False, "fetched_at": None, "n_parsed": None, "n_parse_failed": None,
        "tokens": None, "cost_uncached": None, "cost_batch": None,
    }


def _texts_for_event(event_id: int, needed: set[str]) -> dict[str, str]:
    """{chunk_id: text} for the needed ids of one event (from all_candidates)."""
    q = _qid(event_id)
    path = lfb.CANDIDATES_DIR / q / f"all_candidates_{q}.json"
    data = json.loads(path.read_text())
    return {c["id"]: c.get("text", "") for c in data.get("candidates", [])
            if c.get("id") in needed}


# --------------------------------------------------------------------------- #
# select-pilot — stratified sample over (family, R_q bucket)
# --------------------------------------------------------------------------- #
def _stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional stratified sample over (li, r_bucket), largest-remainder rounding,
    at least one pair from every non-empty stratum."""
    strata: dict[tuple[bool, str], list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["li"], r["r_bucket"])].append(r)
    total = len(rows)
    if n >= total:
        return list(rows)

    quotas: dict[tuple[bool, str], float] = {k: n * len(v) / total for k, v in strata.items()}
    counts = {k: max(1, int(q)) for k, q in quotas.items()}
    # largest-remainder top-up / trim to hit n exactly (never above stratum size)
    def _adjustable(k: tuple[bool, str], up: bool) -> bool:
        return counts[k] < len(strata[k]) if up else counts[k] > 1
    while sum(counts.values()) < n:
        k = max((k for k in counts if _adjustable(k, True)),
                key=lambda k: quotas[k] - counts[k])
        counts[k] += 1
    while sum(counts.values()) > n:
        k = min((k for k in counts if _adjustable(k, False)),
                key=lambda k: quotas[k] - counts[k])
        counts[k] -= 1

    rng = random.Random(seed)
    picked: list[dict] = []
    for k in sorted(strata, key=str):
        pool = sorted(strata[k], key=lambda r: (r["event_id"], r["id"]))
        picked.extend(rng.sample(pool, counts[k]))
    return picked


def cmd_select_pilot(args: argparse.Namespace) -> None:
    rows = load_population()
    pilot = _stratified_sample(rows, args.n, args.seed)
    human_ids = {(r["event_id"], r["id"])
                 for r in _stratified_sample(pilot, args.n_human, args.seed + 1)}
    out = [{**r, "human": (r["event_id"], r["id"]) in human_ids} for r in pilot]
    out.sort(key=lambda r: (r["event_id"], r["id"]))

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_PAIRS_PATH.write_text(json.dumps(out, indent=2))

    by_stratum = Counter((r["li"], r["r_bucket"]) for r in out)
    print(f"Pilot: {len(out)} pairs (seed {args.seed}), human subset: {len(human_ids)}")
    for (li, rb), c in sorted(by_stratum.items(), key=str):
        print(f"  {'late-interaction' if li else 'dense-only':>16} · R_q {rb:>7}: {c}")
    print(f"-> {PILOT_PAIRS_PATH}")


# --------------------------------------------------------------------------- #
# build — write shard input file(s)
# --------------------------------------------------------------------------- #
def _pairs_for_scope(scope: str, pairs_file: str | None = None) -> list[dict]:
    """Pairs for a build scope. ``pairs_file`` (an extract_fp_population.py JSONL, e.g.
    a per-system subset) overrides scope and is read verbatim; otherwise "pilot" reads
    the stratified selection and any other scope reads the full population."""
    if pairs_file:
        path = Path(pairs_file)
        if not path.exists():
            raise SystemExit(f"--pairs-file not found: {path}")
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if scope == "pilot":
        if not PILOT_PAIRS_PATH.exists():
            raise SystemExit("No pilot selection — run 'select-pilot' first.")
        return json.loads(PILOT_PAIRS_PATH.read_text())
    return load_population()


def _needs_classification(pair: dict, done: dict[str, dict]) -> bool:
    rec = done.get(pair["id"])
    return rec is None or rec.get("prompt_version") != FP_TAXONOMY_PROMPT_VERSION


def cmd_build(args: argparse.Namespace) -> None:
    """Write shard input file(s) for every pair not yet classified under the current
    prompt version. Each build round gets a unique shard-name prefix
    (``<scope>_<version>_b<round>_<idx>``), so re-pilots after a prompt iteration and
    parse-failure retries never collide with an already-submitted shard."""
    system_prompt = build_fp_taxonomy_system_prompt()
    query_meta = lfb.load_query_meta()
    pairs = _pairs_for_scope(args.scope, getattr(args, "pairs_file", None))

    # group by event so texts + done-check load once per event
    by_event: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_event[p["event_id"]].append(p)

    m = load_manifest()
    cfg = {"model": MODEL, "effort": EFFORT, "prompt_version": FP_TAXONOMY_PROMPT_VERSION,
           "max_completion_tokens": MAX_COMPLETION_TOKENS,
           "endpoint": ENDPOINT, "completion_window": COMPLETION_WINDOW}
    if m.get("config") and m["config"] != cfg:
        print(f"NOTE: manifest config changes {m['config']} -> {cfg}")
    m["config"] = cfg

    shard_size = args.shard_size
    prefix = f"{args.scope}_{FP_TAXONOMY_PROMPT_VERSION}_b{m['build_rounds'] + 1}"
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, int]] = []   # (shard name, n_requests)
    fh = None
    cur_name = ""
    count = 0

    def _ensure_open() -> None:
        nonlocal fh, cur_name, count
        if fh is None:
            cur_name = f"{prefix}_{len(written):03d}"
            fh = shard_input_path(cur_name).with_suffix(".jsonl.tmp").open("w")
            count = 0

    def _roll() -> None:
        nonlocal fh
        if fh is not None:
            fh.close()
            tmp = shard_input_path(cur_name).with_suffix(".jsonl.tmp")
            os.replace(tmp, shard_input_path(cur_name))
            written.append((cur_name, count))
            fh = None

    n_done = n_todo = 0
    for eid in sorted(by_event):
        ev_pairs = by_event[eid]
        done = load_predictions_for_event(eid)
        todo = [p for p in ev_pairs if _needs_classification(p, done)]
        n_done += len(ev_pairs) - len(todo)
        if not todo:
            continue
        q = query_meta.get(eid)
        if q is None:
            raise SystemExit(f"No query metadata for event {eid}")
        texts = _texts_for_event(eid, {p["id"] for p in todo})
        for p in todo:
            if p["id"] not in texts:
                raise SystemExit(f"Paragraph text missing for {eid}/{p['id']}")
            if count >= shard_size:
                _roll()
            _ensure_open()
            line = lfb._request_line(f"{_qid(eid)}|{p['id']}", system_prompt, q,
                                     texts[p["id"]], MODEL, ENDPOINT,
                                     MAX_COMPLETION_TOKENS, EFFORT)
            assert fh is not None
            fh.write(json.dumps(line) + "\n")
            count += 1
            n_todo += 1
    _roll()

    if written:
        m["build_rounds"] += 1
        for name, n_req in written:
            e = m["shards"].setdefault(name, _empty_shard(args.scope))
            e.update(scope=args.scope, n_requests=n_req,
                     input_bytes=shard_input_path(name).stat().st_size,
                     built_at=lfb._now_iso())
    save_manifest(m)

    print(f"Scope {args.scope}: {n_todo:,} requests across {len(written)} shard(s) "
          f"({n_done:,} already classified under {FP_TAXONOMY_PROMPT_VERSION}); "
          f"system prompt {len(system_prompt):,} chars")
    for name, _ in written:
        e = m["shards"][name]
        print(f"  {name}: {e['n_requests']:,} req · {e['input_bytes'] / 1e6:.1f} MB")


# --------------------------------------------------------------------------- #
# submit / status — same wave mechanics as the judge pipeline
# --------------------------------------------------------------------------- #
def cmd_submit(args: argparse.Namespace) -> None:
    m = load_manifest()
    if not m["shards"]:
        raise SystemExit("No shards — run 'build' first.")
    wave = [s for s, e in sorted(m["shards"].items())
            if not e["batch_id"] and (e.get("n_requests") or 0) > 0
            and (args.scope is None or e["scope"] == args.scope)]
    if args.limit:
        wave = wave[: args.limit]
    print(f"Unsubmitted shards in scope: {len(wave)}")
    if not wave:
        return
    if args.dry_run:
        for s in wave:
            print(f"  would submit {s}: {m['shards'][s]['n_requests']:,} requests")
        return

    client = lfb._client()
    for name in wave:
        path = shard_input_path(name)
        try:
            with path.open("rb") as fh:
                up = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=up.id, endpoint=ENDPOINT,
                completion_window=COMPLETION_WINDOW,
                metadata={"description": f"fp-taxonomy {FP_TAXONOMY_PROMPT_VERSION} "
                                         f"{MODEL}:{EFFORT} shard={name}",
                          "shard": name},
            )
        except Exception as exc:  # noqa: BLE001 — stop the wave cleanly
            print(f"  STOP at {name}: {type(exc).__name__}: {exc}")
            break
        e = m["shards"][name]
        e.update(batch_id=batch.id, input_file_id=up.id,
                 submitted_at=lfb._now_iso(), status=batch.status)
        save_manifest(m)
        print(f"  {name}: batch {batch.id} ({batch.status})")


def cmd_status(args: argparse.Namespace) -> None:
    m = load_manifest()
    client = lfb._client()
    tally: Counter = Counter()
    for name, e in sorted(m["shards"].items()):
        if not e["batch_id"]:
            tally["unsubmitted"] += 1
            continue
        if e["status"] in TERMINAL_STATES and not args.all:
            tally[e["status"]] += 1
            continue
        batch, err = lfb._safe_retrieve(client, e["batch_id"])
        if err:
            tally[err if err == "not_found" else "retrieve_error"] += 1
            print(f"  {name}: {err}")
            continue
        assert batch is not None  # err is None -> retrieve succeeded
        e.update(status=batch.status, request_counts=lfb._counts_dict(batch),
                 output_file_id=batch.output_file_id, error_file_id=batch.error_file_id)
        rc = e["request_counts"] or {}
        tally[batch.status] += 1
        print(f"  {name}: {batch.status} "
              f"({rc.get('completed', '?')}/{rc.get('total', '?')} done)")
    save_manifest(m)
    for st in sorted(tally):
        print(f"  {st}: {tally[st]}")


# --------------------------------------------------------------------------- #
# fetch — download shard outputs -> per-event fp_taxonomy_<qid>.jsonl
# --------------------------------------------------------------------------- #
def _parse_shard_output(out_text: str) -> tuple[dict[int, dict[str, dict]], Counter, dict]:
    """Parse one shard's output JSONL into {event_id: {chunk_id: record}} + stats."""
    per_event: dict[int, dict[str, dict]] = defaultdict(dict)
    tok: Counter = Counter()
    pricing = pricing_mod.PRICING.get(MODEL)
    cost_uncached = 0.0
    n_parsed = n_failed = 0

    for line in out_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid_full = obj.get("custom_id") or ""
        err = obj.get("error")
        resp = obj.get("response")
        if err or not resp or resp.get("status_code") != 200:
            continue  # leave missing so a retry build repicks it
        qid_s, _, chunk_id = cid_full.partition("|")
        body = resp.get("body") or {}
        choice = (body.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        parsed = parse_fp_bits_output(content)

        u = body.get("usage") or {}
        ptd = u.get("prompt_tokens_details") or {}
        ctd = u.get("completion_tokens_details") or {}
        usage = {
            "prompt_tokens": u.get("prompt_tokens", 0),
            "cached_tokens": ptd.get("cached_tokens", 0) or 0,
            "completion_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": ctd.get("reasoning_tokens", 0) or 0,
        }
        for k, v in usage.items():
            tok[k] += v
        if pricing is not None:
            cost_uncached += pricing_mod.price_usage(usage, pricing).cost_uncached

        if parsed is None:
            n_failed += 1
            continue
        n_parsed += 1
        per_event[int(qid_s)][chunk_id] = {
            "id": chunk_id, "event_id": int(qid_s),
            **{b: parsed[b] for b in BITS},
            "prompt_version": FP_TAXONOMY_PROMPT_VERSION,
        }
    stats = {"n_parsed": n_parsed, "n_parse_failed": n_failed,
             "cost_uncached": round(cost_uncached, 4),
             "cost_batch": round(cost_uncached * pricing_mod.batch_discount_for(MODEL), 4)}
    return per_event, tok, stats


def cmd_fetch(args: argparse.Namespace) -> None:
    m = load_manifest()
    client = lfb._client()
    for name, e in sorted(m["shards"].items()):
        if not e["batch_id"] or (e["fetched"] and not args.force):
            continue
        batch, err = lfb._safe_retrieve(client, e["batch_id"])
        if err:
            print(f"  {name}: {err}; skipping.")
            continue
        assert batch is not None  # err is None -> retrieve succeeded
        e["status"] = batch.status
        e["request_counts"] = lfb._counts_dict(batch)
        if batch.status != "completed":
            print(f"  {name}: status={batch.status}; not ready.")
            continue
        if not batch.output_file_id:
            print(f"  {name}: completed with no output (quota failure?) — resubmit later.")
            continue

        out_text = client.files.content(batch.output_file_id).text
        shard_output_path(name).write_text(out_text)
        if batch.error_file_id:
            shard_errors_path(name).write_text(client.files.content(batch.error_file_id).text)

        per_event, tok, stats = _parse_shard_output(out_text)
        for eid, recs in per_event.items():
            merged = load_predictions_for_event(eid)
            merged.update(recs)  # new prompt version wins by id
            _write_predictions_for_event(eid, merged)

        e.update(fetched=True, fetched_at=lfb._now_iso(),
                 output_file_id=batch.output_file_id, error_file_id=batch.error_file_id,
                 n_parsed=stats["n_parsed"], n_parse_failed=stats["n_parse_failed"],
                 tokens=dict(tok), cost_uncached=stats["cost_uncached"],
                 cost_batch=stats["cost_batch"])
        save_manifest(m)
        print(f"  {name}: {stats['n_parsed']:,} parsed "
              f"(+{stats['n_parse_failed']} parse-failures) "
              f"over {len(per_event)} events · ${stats['cost_batch']:.2f} batch")
    print("Done. 'report' summarizes the pilot; 'build --scope full' skips classified pairs.")


# --------------------------------------------------------------------------- #
# report — pilot distribution + usage + projected full-run cost
# --------------------------------------------------------------------------- #
def _pilot_predictions() -> tuple[list[dict], int]:
    pairs = json.loads(PILOT_PAIRS_PATH.read_text())
    by_event: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_event[p["event_id"]].append(p)
    joined, missing = [], 0
    for eid, ev_pairs in by_event.items():
        done = load_predictions_for_event(eid)
        for p in ev_pairs:
            rec = done.get(p["id"])
            if rec is None or rec.get("prompt_version") != FP_TAXONOMY_PROMPT_VERSION:
                missing += 1
                continue
            joined.append({**p, **rec})
    return joined, missing


CELL_GLOSS: dict[tuple[bool, bool, bool], str] = {
    (True, True, True): "all present (relation not between the queried pair)",
    (True, False, True): "instance + constant, no relation",
    (False, True, True): "relation + constant, no instance",
    (True, True, False): "instance + relation, no constant",
    (False, False, True): "constant only",
    (True, False, False): "instance only",
    (False, True, False): "relation only",
    (False, False, False): "none present",
}


def _cell(r: dict) -> tuple[bool, bool, bool]:
    a, b, c = (bool(r[k]) for k in BITS)
    return (a, b, c)


def cmd_report(_args: argparse.Namespace) -> None:
    m = load_manifest()
    joined, missing = _pilot_predictions()
    n = len(joined)
    print(f"=== Pilot ({FP_TAXONOMY_PROMPT_VERSION}) — {n} classified, {missing} missing ===")
    if n:
        cells = Counter(_cell(r) for r in joined)
        print("  instance/relation/constant")
        for cell, cnt in cells.most_common():
            bits_s = "/".join("1" if v else "0" for v in cell)
            print(f"  {bits_s}: {cnt:>4}  ({cnt / n:5.1%})  {CELL_GLOSS[cell]}")
        for b in BITS:
            print(f"  marginal {b}: {sum(1 for r in joined if r[b]) / n:.1%}")
        for li in (True, False):
            sub = [r for r in joined if r["li"] == li]
            if not sub:
                continue
            marg = " · ".join(f"{b.split('_')[0]} {sum(1 for r in sub if r[b]) / len(sub):.0%}"
                              for b in BITS)
            top = Counter(_cell(r) for r in sub).most_common(2)
            tops = ", ".join(f"{'/'.join('1' if v else '0' for v in c)} {cnt / len(sub):.0%}"
                             for c, cnt in top)
            print(f"  [{'late-interaction' if li else 'dense-only'} n={len(sub)}] "
                  f"{marg} · top cells: {tops}")

    # usage + cost projection from fetched shards
    tok: Counter = Counter()
    n_req = 0
    cost_batch = 0.0
    for e in m["shards"].values():
        if e.get("fetched") and e.get("tokens"):
            for k, v in e["tokens"].items():
                tok[k] += v
            n_req += (e.get("n_parsed") or 0) + (e.get("n_parse_failed") or 0)
            cost_batch += e.get("cost_batch") or 0
    if n_req:
        pop = len(load_population())
        cache_rate = tok["cached_tokens"] / tok["prompt_tokens"] if tok["prompt_tokens"] else 0
        per_pair = cost_batch / n_req
        pricing = pricing_mod.PRICING[MODEL]
        disc = pricing_mod.batch_discount_for(MODEL)
        per_pair_meas = disc * (
            (tok["prompt_tokens"] - tok["cached_tokens"]) / 1e6 * pricing.input
            + tok["cached_tokens"] / 1e6 * pricing.cached_input
            + tok["completion_tokens"] / 1e6 * pricing.output) / n_req
        print(f"\n=== Usage over {n_req:,} classified requests ===")
        print(f"  prompt {tok['prompt_tokens'] / n_req:,.0f} tok/req "
              f"(cache hit {cache_rate:.0%}) · completion {tok['completion_tokens'] / n_req:,.0f} "
              f"tok/req (reasoning {tok['reasoning_tokens'] / n_req:,.0f})")
        print(f"  batch cost/pair: ${per_pair:.5f} uncached-priced · "
              f"${per_pair_meas:.5f} at observed cache rate")
        print(f"\n=== Projection to full population ({pop:,} pairs) ===")
        print(f"  uncached upper bound: ${per_pair * pop:,.0f}")
        print(f"  at observed cache rate: ${per_pair_meas * pop:,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("select-pilot", help="stratified pilot sample (no API calls)")
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--n-human", dest="n_human", type=int, default=100)
    sp.add_argument("--seed", type=int, default=13)
    sp.set_defaults(func=cmd_select_pilot)

    pb = sub.add_parser("build", help="write shard input file(s) (no API calls)")
    pb.add_argument("--scope", required=True,
                    help="shard/manifest tag: 'pilot', 'full', or any custom label "
                         "(use a custom tag with --pairs-file for a subset run)")
    pb.add_argument("--pairs-file", dest="pairs_file", default=None,
                    help="classify the pairs in this extract_fp_population.py JSONL "
                         "instead of the pilot/full population (e.g. a per-system subset)")
    pb.add_argument("--shard-size", dest="shard_size", type=int, default=DEFAULT_SHARD_SIZE)
    pb.set_defaults(func=cmd_build)

    ps = sub.add_parser("submit", help="upload + create batches for unsubmitted shards")
    ps.add_argument("--scope", default=None, help="only submit shards with this scope tag")
    ps.add_argument("--limit", type=int)
    ps.add_argument("--dry-run", action="store_true")
    ps.set_defaults(func=cmd_submit)

    pt = sub.add_parser("status", help="refresh shard batch states")
    pt.add_argument("--all", action="store_true")
    pt.set_defaults(func=cmd_status)

    pf = sub.add_parser("fetch", help="download + parse into per-event predictions")
    pf.add_argument("--force", action="store_true")
    pf.set_defaults(func=cmd_fetch)

    pr = sub.add_parser("report", help="pilot distribution + usage + cost projection")
    pr.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
