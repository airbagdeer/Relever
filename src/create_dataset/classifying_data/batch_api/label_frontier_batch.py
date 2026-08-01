"""Classify the full frontier-v2 candidate set with gpt-5.2 (medium) via OpenAI's Batch API.

Generalizes the validated 200-sample test (``batch_api/test/label_batch_openai.py``) to all 1050
v2 queries (~2.06M candidate paragraphs). **One Batch API job per query**: each query's candidates
become ``data/labels/<qid>/batch_input_<qid>.jsonl`` and its results land
back in that same folder. A single JSON manifest (``_manifest.json``) is the checkpoint, so
submission runs in user-controlled waves and every stage is independently resumable.

The per-request ``body`` is byte-identical to the validated synchronous labeler
(``estimate_frontier_cost`` / the batch test): ``gpt-5.2`` @ ``reasoning_effort="medium"`` with the
``short2b`` system prompt — only the transport (batch vs synchronous) differs.

Lifecycle (build first; submit later, in waves):

    # 1. Build all 1050 input files — no API calls, no spend:
    uv run python -m src.create_dataset.classifying_data.batch_api.label_frontier_batch build

    # 2. Submit the next wave of N built-but-unsubmitted queries (repeat as enqueued-token
    #    limits allow — "first x files", then "next y files", ...):
    uv run python -m ...label_frontier_batch submit --limit 50
    uv run python -m ...label_frontier_batch submit --dry-run --limit 50   # preview, no spend

    # 3. Poll submitted batches (updates statuses in the manifest):
    uv run python -m ...label_frontier_batch status

    # 4. Download + parse every completed batch into predictions_<qid>.jsonl:
    uv run python -m ...label_frontier_batch fetch

A query's row in the manifest moves input_built -> submitted (batch_id) -> completed -> fetched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, NotFoundError, OpenAI
from openai.types import Batch

# --- imports from the existing pipeline (repo root on path for the fallback) - #
_PKG_DIR = Path(__file__).resolve().parent  # .../classifying_data/batch_api


def _find_project_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory holding ``pyproject.toml`` (the repo root)."""
    for p in (start, *start.parents):
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError("project root (pyproject.toml) not found above " + str(start))


_PROJECT_ROOT = _find_project_root(_PKG_DIR)
sys.path.insert(0, str(_PROJECT_ROOT))  # so `src.*` resolves as a plain script / in workers too

from src.create_dataset.classifying_data.relation_classification_parser import (  # noqa: E402
    parse_classification_output,
)
from src.create_dataset.classifying_data.prompt_variants import build_system_prompt  # noqa: E402
from src.create_dataset.classifying_data import frontier_cost_pricing as pricing_mod  # noqa: E402

# --- config defaults (mirror the validated synchronous / batch-test run) ----- #
MODEL = "gpt-5.2"
EFFORT = "medium"
VARIANT = "short2b"
MAX_COMPLETION_TOKENS = 16000
ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"

# --- paths ------------------------------------------------------------------- #
CANDIDATES_DIR = _PROJECT_ROOT / "data" / "candidates"
OUT_ROOT = _PROJECT_ROOT / "data" / "labels"
# The query set (event_id 0-1099), read from the released data/queries/queries.json.
RELATIONS_PATH = _PROJECT_ROOT / "data" / "queries" / "queries.json"
MANIFEST_PATH = OUT_ROOT / "_manifest.json"

# Per-query artifacts live in OUT_ROOT/<qid>/.
TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}
_QUERY_FIELDS = ("sentence", "abstract_entity_type", "constant_entity", "relation")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm_qid(s: str) -> str:
    """Normalise a user-supplied id to the 4-digit zero-padded folder name (5 -> '0005')."""
    return str(s).strip().zfill(4)


def input_path(qid: str) -> Path:
    return OUT_ROOT / qid / f"batch_input_{qid}.jsonl"


def output_raw_path(qid: str) -> Path:
    return OUT_ROOT / qid / f"batch_output_{qid}.jsonl"


def errors_raw_path(qid: str) -> Path:
    return OUT_ROOT / qid / f"batch_errors_{qid}.jsonl"


def predictions_path(qid: str) -> Path:
    return OUT_ROOT / qid / f"predictions_{qid}.jsonl"


def _user_content(q: dict, paragraph: str) -> str:
    """Mirror of the validated labeler's user message (kept verbatim so each batch request matches
    the synchronous request exactly); ``paragraph`` is the candidate's ``text``."""
    return (
        "\n===INPUT===\n"
        f"Sentence: {q.get('sentence', '')}\n"
        f"Abstract Entity Type: {q.get('abstract_entity_type', '')}\n"
        f"Constant Entity: {q.get('constant_entity', '')}\n"
        f"Relation: {q.get('relation', '')}\n"
        f"Paragraph: {paragraph}\n\n"
        "===OUTPUT===\n"
    )


def load_query_meta() -> dict[int, dict]:
    """event_id -> {sentence, abstract_entity_type, constant_entity, relation} from the v2 query set."""
    if not RELATIONS_PATH.exists():
        raise SystemExit(f"Query set not found: {RELATIONS_PATH}")
    rows = json.loads(RELATIONS_PATH.read_text())
    return {r["event_id"]: {k: r.get(k, "") for k in _QUERY_FIELDS} for r in rows}


def discover_qids() -> list[str]:
    """Sorted 4-digit folder names under CANDIDATES_DIR that hold an all_candidates_<qid>.json."""
    if not CANDIDATES_DIR.exists():
        raise SystemExit(f"Candidates dir not found: {CANDIDATES_DIR}")
    qids = []
    for d in CANDIDATES_DIR.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 4:
            if (d / f"all_candidates_{d.name}.json").exists():
                qids.append(d.name)
    return sorted(qids)


# --------------------------------------------------------------------------- #
# Manifest (the checkpoint) — single source of truth, atomically rewritten
# --------------------------------------------------------------------------- #
def _empty_entry(event_id: int) -> dict:
    return {
        "event_id": event_id,
        "input_built": False, "n_requests": None, "input_bytes": None, "built_at": None,
        "batch_id": None, "input_file_id": None, "submitted_at": None, "status": None,
        "request_counts": None, "output_file_id": None, "error_file_id": None,
        "fetched": False, "fetched_at": None,
        "n_predictions": None, "n_pos": None, "n_neg": None, "n_failed": None,
        "tokens": None, "cost_uncached": None, "cost_batch": None,
    }


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"config": None, "updated_at": None, "queries": {}}


def save_manifest(m: dict) -> None:
    m["updated_at"] = _now_iso()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    os.replace(tmp, MANIFEST_PATH)


def _require_manifest() -> dict:
    m = load_manifest()
    if not m.get("queries"):
        raise SystemExit("No manifest yet — run 'build' first.")
    return m


# --------------------------------------------------------------------------- #
# OpenAI client
# --------------------------------------------------------------------------- #
def _client() -> OpenAI:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (.env).")
    return OpenAI()


def _counts_dict(batch) -> dict | None:
    rc = getattr(batch, "request_counts", None)
    if rc is None:
        return None
    return {"total": rc.total, "completed": rc.completed, "failed": rc.failed}


def _safe_retrieve(client: OpenAI, batch_id: str) -> tuple[Batch | None, str | None]:
    """Retrieve a batch, tolerating a vanished / unresolvable id so one bad batch can't abort a run.

    Returns ``(batch, None)`` on success; ``(None, "not_found")`` if the id 404s (the batch is
    gone — it produced no output and cost nothing, so it is safe to clear and resubmit); or
    ``(None, "error: ...")`` for a transient API failure (leave state as-is and move on). A 404 is
    the state left behind when a ``submit`` is interrupted, or when OpenAI expires/cancels a batch.
    """
    try:
        return client.batches.retrieve(batch_id), None
    except NotFoundError:
        return None, "not_found"
    except APIError as exc:  # transient (network / 5xx / rate) — don't crash the whole loop
        return None, f"error: {type(exc).__name__}: {exc}"


def _request_line(cid: str, system_prompt: str, qinfo: dict, text: str,
                  model: str, endpoint: str, max_tokens: int, effort: str) -> dict:
    """One Batch API request line for a candidate (shared by build and retry)."""
    return {
        "custom_id": cid,
        "method": "POST",
        "url": endpoint,
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _user_content(qinfo, text)},
            ],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": effort,
        },
    }


def _dedup_candidates(data: dict) -> list[tuple[str, str]]:
    """(id, text) for each unique-by-id candidate, preserving order."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for c in data.get("candidates", []):
        cid = c.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append((cid, c.get("text", "")))
    return out


def _done_ids(qid: str) -> set[str]:
    """Candidate ids that already have a prediction on disk for this query."""
    pf = predictions_path(qid)
    if not pf.exists():
        return set()
    return {json.loads(line)["id"] for line in pf.read_text().splitlines() if line.strip()}


def _candidate_ids(qid: str) -> set[str]:
    """Unique candidate ids in this query's (new) candidate set."""
    data = json.loads((CANDIDATES_DIR / qid / f"all_candidates_{qid}.json").read_text())
    return {c["id"] for c in data.get("candidates", []) if c.get("id")}


# --------------------------------------------------------------------------- #
# build — write one batch_input_<qid>.jsonl per query (no API calls)
# --------------------------------------------------------------------------- #
# Worker context, set once per process by the pool initializer (kept module-global so the worker
# function is a plain top-level picklable callable).
_CTX: dict = {}


def _build_init(system_prompt: str, query_meta: dict, cfg: dict, skip_done: bool = False) -> None:
    _CTX["system_prompt"] = system_prompt
    _CTX["query_meta"] = {int(k): v for k, v in query_meta.items()}
    _CTX["cfg"] = cfg
    _CTX["skip_done"] = skip_done


def _build_qid(qid: str) -> dict:
    """Build the batch input file for one query. Returns a small stats dict (no big payloads).

    When ``skip_done`` is set, candidates that already have a prediction on disk
    (``predictions_<qid>.jsonl`` — e.g. reused labels seeded by ``seed-from-old``) are not
    re-emitted, so only genuinely-new paragraphs become requests. ``n_total`` still reports the full
    unique-candidate count so the manifest's completeness / retry accounting stays correct.
    """
    cfg = _CTX["cfg"]
    sp = _CTX["system_prompt"]
    skip_done = _CTX.get("skip_done", False)
    cand_path = CANDIDATES_DIR / qid / f"all_candidates_{qid}.json"
    try:
        data = json.loads(cand_path.read_text())
    except Exception as e:  # noqa: BLE001 — report, don't crash the whole pool
        return {"qid": qid, "status": "error", "error": f"read candidates: {e}"}

    event_id = data.get("event_id")
    q = _CTX["query_meta"].get(event_id)
    if q is None:
        return {"qid": qid, "status": "error", "error": f"no query metadata for event_id={event_id}"}

    done = _done_ids(qid) if skip_done else set()

    out_dir = OUT_ROOT / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = input_path(qid)
    tmp = out_path.with_suffix(".jsonl.tmp")

    seen: set[str] = set()
    n = dups = skipped = reused = 0
    with tmp.open("w") as fh:
        for c in data.get("candidates", []):
            cid = c.get("id")
            if not cid:
                skipped += 1
                continue
            if cid in seen:  # custom_id must be unique within a batch
                dups += 1
                continue
            seen.add(cid)
            if cid in done:  # already labeled on disk (reused) — don't resend
                reused += 1
                continue
            line = _request_line(cid, sp, q, c.get("text", ""), cfg["model"],
                                  cfg["endpoint"], cfg["max_completion_tokens"], cfg["effort"])
            fh.write(json.dumps(line) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return {
        "qid": qid, "status": "ok", "event_id": event_id,
        "n_requests": n, "n_total": len(seen), "input_bytes": out_path.stat().st_size,
        "dups": dups, "skipped": skipped, "reused": reused,
    }


def _select_qids(arg_qids, all_qids: list[str]) -> list[str]:
    if not arg_qids:
        return all_qids
    known = set(all_qids)
    chosen = []
    for raw in arg_qids:
        qid = _norm_qid(raw)
        if qid not in known:
            raise SystemExit(f"--qids: '{raw}' -> '{qid}' has no candidate folder.")
        chosen.append(qid)
    return chosen


def cmd_build(args: argparse.Namespace) -> None:
    cfg = {
        "model": args.model, "effort": args.effort, "variant": args.variant,
        "max_completion_tokens": args.max_tokens,
        "endpoint": ENDPOINT, "completion_window": COMPLETION_WINDOW,
    }
    system_prompt = build_system_prompt(cfg["variant"])
    query_meta = load_query_meta()
    all_qids = discover_qids()

    m = load_manifest()
    if m.get("config") and m["config"] != cfg and not args.force:
        raise SystemExit(
            f"Manifest config {m['config']} differs from requested {cfg}. "
            "Re-run with --force to rebuild under the new config."
        )
    m["config"] = cfg
    for qid in all_qids:
        m["queries"].setdefault(qid, _empty_entry(int(qid)))

    target = _select_qids(args.qids, all_qids)
    todo = [q for q in target if args.force or not m["queries"][q]["input_built"]]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Config: model={cfg['model']} effort={cfg['effort']} variant={cfg['variant']} "
          f"max_tok={cfg['max_completion_tokens']}")
    print(f"System prompt: {len(system_prompt):,} chars (variant '{cfg['variant']}')")
    print(f"Queries: {len(all_qids)} total · {len(target)} targeted · "
          f"{len(target) - len(todo)} already built · {len(todo)} to build")
    if not todo:
        save_manifest(m)
        print("Nothing to build.")
        return

    workers = max(1, args.workers)
    done = errs = 0
    tot_req = tot_bytes = tot_dups = tot_reused = 0
    print(f"Building with {workers} worker(s)"
          + (" · --skip-done (reusing on-disk predictions)" if args.skip_done else "") + "...")

    def _apply(res: dict) -> None:
        nonlocal done, errs, tot_req, tot_bytes, tot_dups, tot_reused
        qid = res["qid"]
        e = m["queries"][qid]
        if res["status"] == "ok":
            e.update(input_built=True, n_requests=res["n_requests"], input_bytes=res["input_bytes"],
                     n_candidates=res["n_total"], built_at=_now_iso(), event_id=res["event_id"])
            e.pop("build_error", None)
            # Fully-reused query: nothing new to classify, so it's already complete (the seeded
            # predictions cover every candidate) and must not be submitted (no empty batch).
            if res["n_requests"] == 0:
                e["complete"] = True
            tot_req += res["n_requests"]
            tot_bytes += res["input_bytes"]
            tot_dups += res["dups"]
            tot_reused += res.get("reused", 0)
            done += 1
        else:
            e.update(input_built=False, build_error=res["error"])
            errs += 1
            print(f"  ERROR {qid}: {res['error']}")

    if workers == 1:
        _build_init(system_prompt, query_meta, cfg, args.skip_done)
        for i, qid in enumerate(todo, 1):
            _apply(_build_qid(qid))
            if i % 25 == 0 or i == len(todo):
                save_manifest(m)
                print(f"  [{i}/{len(todo)}] built={done} err={errs} "
                      f"req={tot_req:,} disk={tot_bytes / 1e9:.1f}GB")
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_build_init,
            initargs=(system_prompt, query_meta, cfg, args.skip_done),
        ) as ex:
            futs = {ex.submit(_build_qid, qid): qid for qid in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                _apply(fut.result())
                if i % 25 == 0 or i == len(todo):
                    save_manifest(m)
                    print(f"  [{i}/{len(todo)}] built={done} err={errs} "
                          f"req={tot_req:,} disk={tot_bytes / 1e9:.1f}GB")

    save_manifest(m)
    print(f"\nDone. Built {done} queries · {tot_req:,} requests · {tot_bytes / 1e9:.1f} GB"
          + (f" · {tot_reused:,} reused (skipped)" if tot_reused else "")
          + (f" · {tot_dups:,} duplicate custom_ids skipped" if tot_dups else "")
          + (f" · {errs} errors" if errs else ""))
    print(f"Manifest -> {MANIFEST_PATH}")


# --------------------------------------------------------------------------- #
# seed-from-old — pre-populate predictions_<qid>.jsonl by reusing an earlier
# run's labels, matched to the *new* candidate set by candidate id. Run this
# BEFORE 'build --skip-done' so build emits requests only for new paragraphs.
# --------------------------------------------------------------------------- #
def cmd_seed_from_old(args: argparse.Namespace) -> None:
    old_root = Path(args.old_root)
    if not old_root.exists():
        raise SystemExit(f"--old-root not found: {old_root}")
    qids = _select_qids(args.qids, discover_qids())

    reused = drop_stale = drop_fail = 0
    seeded_q = skipped_existing = no_old = 0
    for qid in qids:
        oldp = old_root / qid / f"predictions_{qid}.jsonl"
        if not oldp.exists():  # e.g. queries added since the earlier run — nothing to reuse
            no_old += 1
            continue
        target = predictions_path(qid)
        if target.exists() and not args.force:  # never clobber real / already-seeded results
            skipped_existing += 1
            continue

        cand = _candidate_ids(qid)  # only reuse labels whose paragraph still exists
        kept: list[dict] = []
        for line in oldp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r.get("id")
            if cid not in cand:  # candidate removed by the re-retrieval — drop stale label
                drop_stale += 1
                continue
            if r.get("prediction") == -1 and not args.reuse_failed:  # re-classify parse failures
                drop_fail += 1
                continue
            kept.append({"id": cid, "event_id": r.get("event_id"),
                         "prediction": r.get("prediction")})

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for r in kept:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, target)
        reused += len(kept)
        seeded_q += 1

    print(f"Seeded {seeded_q} query(ies) from {old_root}")
    print(f"  reused labels: {reused:,}"
          + (f" · dropped {drop_stale:,} stale (candidate removed)" if drop_stale else "")
          + (f" · dropped {drop_fail:,} failed (-1, will re-classify)" if drop_fail else ""))
    if no_old:
        print(f"  {no_old} query(ies) had no earlier predictions (all candidates are new)")
    if skipped_existing:
        print(f"  {skipped_existing} query(ies) already had predictions on disk "
              f"(skipped; pass --force to overwrite)")
    print("\nNext: build only the new paragraphs, then submit/fetch as usual:")
    print("  uv run python -m src.create_dataset.classifying_data.batch_api."
          "label_frontier_batch build --skip-done")


# --------------------------------------------------------------------------- #
# submit — upload + create a batch for each query in the wave
# --------------------------------------------------------------------------- #
def cmd_submit(args: argparse.Namespace) -> None:
    m = _require_manifest()
    cfg = m["config"]
    all_qids = sorted(m["queries"])

    # A query with 0 requests (all candidates reused via seed-from-old) has an empty input file and
    # must never be uploaded — the Batch API rejects empty batches.
    def _submittable(q: str) -> bool:
        return not m["queries"][q]["batch_id"] and (m["queries"][q].get("n_requests") or 0) > 0

    if args.qids:
        wave = [q for q in _select_qids(args.qids, all_qids) if _submittable(q)]
    else:
        wave = [q for q in all_qids if m["queries"][q]["input_built"] and _submittable(q)]
    if args.limit:  # applies to both --qids and the default selection, so waves can be paced
        wave = wave[: args.limit]

    eligible_total = sum(1 for q in all_qids
                         if m["queries"][q]["input_built"] and _submittable(q))
    print(f"{eligible_total} queries built-and-unsubmitted; this wave: {len(wave)}")
    if not wave:
        return
    if args.dry_run:
        print("--dry-run: would submit these queries (no upload, no spend):")
        print("  " + " ".join(wave))
        total_req = sum(m["queries"][q]["n_requests"] or 0 for q in wave)
        print(f"  {len(wave)} batches · {total_req:,} total requests")
        return

    client = _client()
    submitted = 0
    for qid in wave:
        path = input_path(qid)
        if not path.exists():
            print(f"  SKIP {qid}: input file missing ({path})")
            continue
        try:
            with path.open("rb") as fh:
                up = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=up.id,
                endpoint=cfg["endpoint"],
                completion_window=cfg["completion_window"],
                metadata={
                    "description": f"relation-classification {cfg['variant']} "
                                   f"{cfg['model']}:{cfg['effort']} qid={qid}",
                    "qid": qid,
                },
            )
        except Exception as exc:  # noqa: BLE001 — stop the wave cleanly, keep prior progress
            print(f"  STOP at {qid}: {type(exc).__name__}: {exc}")
            print(f"  Submitted {submitted} this run; progress is saved. If this is an "
                  f"enqueued-token limit, wait for batches to finish, then re-run 'submit'.")
            break
        e = m["queries"][qid]
        e.update(batch_id=batch.id, input_file_id=up.id,
                 submitted_at=_now_iso(), status=batch.status)
        save_manifest(m)  # persist after EACH submit — a live batch_id is money in flight
        submitted += 1
        print(f"  {qid}: batch {batch.id} ({batch.status})")

    print(f"Submitted {submitted} batch(es). Poll with 'status', collect with 'fetch'.")


# --------------------------------------------------------------------------- #
# status — refresh submitted batches' states
# --------------------------------------------------------------------------- #
def cmd_status(args: argparse.Namespace) -> None:
    m = _require_manifest()
    all_qids = sorted(m["queries"])
    if args.qids:
        qids = _select_qids(args.qids, all_qids)
    else:
        qids = [q for q in all_qids if m["queries"][q]["batch_id"]]

    client = _client()
    tally: Counter = Counter()
    refreshed = 0
    for i, qid in enumerate(qids, 1):
        e = m["queries"][qid]
        if not e["batch_id"]:
            continue
        # Skip re-fetching already-terminal batches unless --all (saves API calls across big waves).
        if e["status"] in TERMINAL_STATES and not args.all:
            tally[e["status"]] += 1
            continue
        batch, err = _safe_retrieve(client, e["batch_id"])
        if err == "not_found":
            e["status"] = "not_found"
            tally["not_found"] += 1
            refreshed += 1
            print(f"  {qid}: batch {e['batch_id']} not found (404) — clear + resend with "
                  f"'reset --qids {qid}' then 'submit --qids {qid}'.")
        elif err:
            tally["retrieve_error"] += 1
            print(f"  {qid}: {err}; left as-is.")
        else:
            assert batch is not None  # err is None -> retrieve succeeded
            e.update(status=batch.status, request_counts=_counts_dict(batch),
                     output_file_id=batch.output_file_id, error_file_id=batch.error_file_id)
            tally[batch.status] += 1
            refreshed += 1
        if i % 25 == 0:
            save_manifest(m)
    save_manifest(m)

    submitted = sum(1 for q in all_qids if m["queries"][q]["batch_id"])
    fetched = sum(1 for q in all_qids if m["queries"][q]["fetched"])
    print(f"Refreshed {refreshed} batch(es). Submitted total: {submitted}; fetched: {fetched}.")
    for st in sorted(tally):
        print(f"  {st}: {tally[st]}")


# --------------------------------------------------------------------------- #
# fetch — download + parse completed batches into predictions_<qid>.jsonl
# --------------------------------------------------------------------------- #
def _parse_to_predictions(qid: str, event_id: int, out_text: str, model: str) -> dict:
    """Parse one batch output JSONL into predictions_<qid>.jsonl. Returns count/cost stats."""
    new_recs: dict[str, dict] = {}
    tok: Counter = Counter()
    cost_uncached = 0.0
    pricing = pricing_mod.PRICING.get(model)

    for line in out_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        err = obj.get("error")
        resp = obj.get("response")
        if err or not resp or resp.get("status_code") != 200:
            # An error entry in the *output* stream (rare — request errors normally land in the
            # error file). Leave it MISSING so a later retry repicks it; do not record a -1.
            continue

        body = resp.get("body") or {}
        choice = (body.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        new_recs[cid] = {"id": cid, "event_id": event_id,
                         "prediction": parse_classification_output(content)}

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

    # Merge with predictions already on disk (additive / retry-safe): a retry batch only carries
    # the previously-missing ids, so we union rather than overwrite.
    merged: dict[str, dict] = {}
    pf = predictions_path(qid)
    if pf.exists():
        for line in pf.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                merged[r["id"]] = r
    merged.update(new_recs)

    pf.parent.mkdir(parents=True, exist_ok=True)
    tmp = pf.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for r in merged.values():
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, pf)

    cost_batch = cost_uncached * pricing_mod.batch_discount_for(model)
    return {
        "n_predictions": len(merged), "n_new": len(new_recs),
        "n_pos": sum(1 for r in merged.values() if r["prediction"] == 1),
        "n_neg": sum(1 for r in merged.values() if r["prediction"] == 0),
        "n_failed": sum(1 for r in merged.values() if r["prediction"] == -1),
        "new_tokens": dict(tok),
        "new_cost_uncached": round(cost_uncached, 4), "new_cost_batch": round(cost_batch, 4),
    }


def cmd_fetch(args: argparse.Namespace) -> None:
    m = _require_manifest()
    model = m["config"]["model"]
    all_qids = sorted(m["queries"])
    if args.qids:
        qids = _select_qids(args.qids, all_qids)
    else:
        qids = [q for q in all_qids if m["queries"][q]["batch_id"] and not m["queries"][q]["fetched"]]

    client = _client()
    fetched = skipped = 0
    grand_cost = 0.0
    for qid in qids:
        e = m["queries"][qid]
        if not e["batch_id"]:
            continue
        if e.get("complete") and not args.force:  # don't re-fetch / double-count a finished query
            continue
        batch, err = _safe_retrieve(client, e["batch_id"])
        if err == "not_found":
            e["status"] = "not_found"
            save_manifest(m)  # persist so 'reset --failed' can later auto-detect it
            print(f"  {qid}: batch not found (404); skipping "
                  f"(clear + resend: 'reset --qids {qid}' then 'submit --qids {qid}').")
            skipped += 1
            continue
        if err:
            print(f"  {qid}: {err}; skipping.")
            skipped += 1
            continue
        assert batch is not None  # err is None -> retrieve succeeded
        e["status"] = batch.status
        e["request_counts"] = _counts_dict(batch)
        if batch.status != "completed":
            print(f"  {qid}: status={batch.status}; not ready.")
            skipped += 1
            continue
        if not batch.output_file_id:
            print(f"  {qid}: completed but no output_file_id; skipping (run 'reset --failed').")
            skipped += 1
            continue

        out_text = client.files.content(batch.output_file_id).text
        output_raw_path(qid).write_text(out_text)
        if batch.error_file_id:
            errors_raw_path(qid).write_text(client.files.content(batch.error_file_id).text)

        stats = _parse_to_predictions(qid, e["event_id"], out_text, model)
        prev_tok = e.get("tokens") or {}
        tokens = {k: prev_tok.get(k, 0) + stats["new_tokens"].get(k, 0)
                  for k in set(prev_tok) | set(stats["new_tokens"])}
        ground = e.get("n_candidates") or e.get("n_requests") or stats["n_predictions"]
        complete = stats["n_predictions"] >= ground
        e.update(
            fetched=True, fetched_at=_now_iso(), complete=complete,
            output_file_id=batch.output_file_id, error_file_id=batch.error_file_id,
            n_predictions=stats["n_predictions"], n_pos=stats["n_pos"],
            n_neg=stats["n_neg"], n_failed=stats["n_failed"], tokens=tokens,
            cost_uncached=round((e.get("cost_uncached") or 0) + stats["new_cost_uncached"], 4),
            cost_batch=round((e.get("cost_batch") or 0) + stats["new_cost_batch"], 4),
        )
        save_manifest(m)
        fetched += 1
        grand_cost += stats["new_cost_batch"]
        flag = "" if complete else f"   ** INCOMPLETE {stats['n_predictions']:,}/{ground:,} **"
        print(f"  {qid}: +{stats['n_new']:,} -> {stats['n_predictions']:,} preds "
              f"(pos {stats['n_pos']}, neg {stats['n_neg']}, fail {stats['n_failed']}) "
              f"· ${stats['new_cost_batch']:.2f}{flag}")

    incomplete = sum(1 for q in qids if m["queries"][q].get("fetched")
                     and not m["queries"][q].get("complete"))
    print(f"Fetched {fetched} batch(es); skipped {skipped}. Batch cost this run: ${grand_cost:.2f}")
    if incomplete:
        print(f"** {incomplete} fetched query(ies) are INCOMPLETE (some requests failed). "
              f"Run 'retry' to rebuild just the missing ids, then submit + fetch. **")


# --------------------------------------------------------------------------- #
# reset — clear a failed batch's submission state so 'submit' can resend it.
# Use after a quota/billing failure: the batch reports status "completed" but has
# no output_file_id because every request was rejected (insufficient_quota) — those
# requests are never processed, so they cost nothing and are safe to retry.
# --------------------------------------------------------------------------- #
_RESET_FIELDS = (
    "batch_id", "input_file_id", "submitted_at", "status", "request_counts",
    "output_file_id", "error_file_id", "fetched_at",
    "n_predictions", "n_pos", "n_neg", "n_failed", "tokens", "cost_uncached", "cost_batch",
)


def cmd_reset(args: argparse.Namespace) -> None:
    m = _require_manifest()
    all_qids = sorted(m["queries"])
    if not args.qids and not args.failed:
        raise SystemExit("Pass --failed (auto-detect) or --qids <ids>.")
    if args.qids:
        cands = _select_qids(args.qids, all_qids)
    else:  # --failed: manifest says terminal (or vanished), no usable output, not yet fetched
        cands = [q for q in all_qids
                 if m["queries"][q]["batch_id"] and not m["queries"][q].get("fetched")
                 and ((m["queries"][q].get("status") in TERMINAL_STATES
                       and not m["queries"][q].get("output_file_id"))
                      or m["queries"][q].get("status") == "not_found")]
    if not cands:
        print("No reset candidates.")
        return

    client = _client()
    reset_ids: list[str] = []
    for q in cands:
        e = m["queries"][q]
        batch, err = _safe_retrieve(client, e["batch_id"])  # verify live before discarding state
        if err == "not_found":  # batch is gone -> produced no output, cost nothing -> safe to clear
            for k in _RESET_FIELDS:
                e[k] = None
            e["fetched"] = False
            reset_ids.append(q)
            save_manifest(m)
            print(f"  {q}: batch not found (404) — cleared for resubmit.")
            continue
        if err:
            print(f"  {q}: {err}; leaving as-is.")
            continue
        assert batch is not None  # err is None -> retrieve succeeded
        e["status"] = batch.status
        e["request_counts"] = _counts_dict(batch)
        if batch.status not in TERMINAL_STATES:
            print(f"  {q}: still {batch.status} — leaving as-is.")
            continue
        if batch.output_file_id:  # it actually produced results; don't throw them away
            e["output_file_id"] = batch.output_file_id
            print(f"  {q}: {batch.status} WITH output — run 'fetch', not 'reset'.")
            continue
        for k in _RESET_FIELDS:
            e[k] = None
        e["fetched"] = False
        reset_ids.append(q)
        save_manifest(m)

    save_manifest(m)
    print(f"\nReset {len(reset_ids)} failed (no-output) batch(es) -> eligible for 'submit' again.")
    if reset_ids:
        print("  " + " ".join(reset_ids))
        print("\nResubmit (paced; re-run until none remain):")
        print("  uv run python -m src.create_dataset.classifying_data.batch_api."
              "label_frontier_batch \\\n    submit --limit 25 --qids " + " ".join(reset_ids))


# --------------------------------------------------------------------------- #
# retry — rebuild a query's input with ONLY its still-missing candidate ids, so
# 'submit' resends just the gaps (the failed-within-batch requests) and 'fetch'
# merges them into the existing predictions. Used when a fetched query is
# incomplete (n_predictions < n_candidates) because some requests failed (quota
# blips, or a paragraph that blew max_completion_tokens). Defaults to a larger
# token budget so the rare truncation failures also resolve on retry.
# --------------------------------------------------------------------------- #
_RETRY_CLEAR_FIELDS = (
    "batch_id", "input_file_id", "submitted_at", "status", "request_counts",
    "output_file_id", "error_file_id", "fetched_at",
)


def cmd_retry(args: argparse.Namespace) -> None:
    m = _require_manifest()
    cfg = m["config"]
    all_qids = sorted(m["queries"])
    if args.qids:
        targets = _select_qids(args.qids, all_qids)
    else:  # auto: fetched queries whose predictions fall short of the candidate count
        targets = [
            q for q in all_qids
            if m["queries"][q].get("fetched")
            and (m["queries"][q].get("n_predictions") or 0)
            < (m["queries"][q].get("n_candidates") or m["queries"][q].get("n_requests") or 0)
        ]
    if not targets:
        print("No incomplete queries to retry.")
        return

    query_meta = load_query_meta()
    system_prompt = build_system_prompt(cfg["variant"])
    max_tokens = args.max_tokens
    rebuilt: list[str] = []
    for q in targets:
        e = m["queries"][q]
        data = json.loads((CANDIDATES_DIR / q / f"all_candidates_{q}.json").read_text())
        qinfo = query_meta.get(data.get("event_id"))
        if qinfo is None:
            print(f"  {q}: no query metadata; skipping.")
            continue
        cand = _dedup_candidates(data)
        e["n_candidates"] = len(cand)
        done = _done_ids(q)
        missing = [(cid, txt) for cid, txt in cand if cid not in done]
        if not missing:
            e["complete"] = True
            print(f"  {q}: already complete ({len(done)}/{len(cand)}).")
            continue
        (OUT_ROOT / q).mkdir(parents=True, exist_ok=True)
        tmp = input_path(q).with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for cid, txt in missing:
                fh.write(json.dumps(_request_line(
                    cid, system_prompt, qinfo, txt,
                    cfg["model"], cfg["endpoint"], max_tokens, cfg["effort"])) + "\n")
        os.replace(tmp, input_path(q))
        for k in _RETRY_CLEAR_FIELDS:
            e[k] = None
        e["fetched"] = False
        e["complete"] = False
        e["n_requests"] = len(missing)  # the retry batch carries only the gaps
        rebuilt.append(q)
        print(f"  {q}: {len(done)}/{len(cand)} done -> retry {len(missing)} missing")
    save_manifest(m)

    print(f"\nRebuilt {len(rebuilt)} retry input(s) at max_completion_tokens={max_tokens} "
          f"(existing predictions kept; 'fetch' will merge the gaps).")
    if rebuilt:
        print("Resubmit (paced; re-run until none remain), then fetch:")
        print("  uv run python -m src.create_dataset.classifying_data.batch_api."
              "label_frontier_batch \\\n    submit --limit 25 --qids " + " ".join(rebuilt))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="write batch_input_<qid>.jsonl for each query (no API calls)")
    pb.add_argument("--qids", nargs="+", help="only these query ids (default: all 1050)")
    pb.add_argument("--limit", type=int, help="cap how many queries to build this run")
    pb.add_argument("--force", action="store_true", help="rebuild even if already built")
    pb.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="parallel build workers (default: min(8, cpus))")
    pb.add_argument("--skip-done", dest="skip_done", action="store_true",
                    help="skip candidates already labeled on disk (predictions_<qid>.jsonl) — use "
                         "after 'seed-from-old' to build requests only for new paragraphs")
    pb.add_argument("--model", default=MODEL)
    pb.add_argument("--effort", default=EFFORT)
    pb.add_argument("--variant", default=VARIANT)
    pb.add_argument("--max-tokens", dest="max_tokens", type=int, default=MAX_COMPLETION_TOKENS)
    pb.set_defaults(func=cmd_build)

    psd = sub.add_parser("seed-from-old",
                         help="reuse an earlier run's labels for unchanged paragraphs (by id)")
    psd.add_argument("--old-root", dest="old_root", default=str(OUT_ROOT / "old"),
                     help="root holding <qid>/predictions_<qid>.jsonl from the earlier run "
                          "(default: data/labels/old)")
    psd.add_argument("--qids", nargs="+", help="only seed these query ids (default: all)")
    psd.add_argument("--force", action="store_true",
                     help="overwrite an existing predictions_<qid>.jsonl (default: skip, so real or "
                          "already-seeded results are never clobbered)")
    psd.add_argument("--reuse-failed", dest="reuse_failed", action="store_true",
                     help="also reuse -1 (parse-failure) old labels (default: re-classify them)")
    psd.set_defaults(func=cmd_seed_from_old)

    ps = sub.add_parser("submit", help="upload + create a batch for the next wave of queries")
    ps.add_argument("--qids", nargs="+", help="submit exactly these query ids")
    ps.add_argument("--limit", type=int, help="submit the next N built-but-unsubmitted queries")
    ps.add_argument("--dry-run", action="store_true", help="show the wave; no upload, no spend")
    ps.set_defaults(func=cmd_submit)

    pt = sub.add_parser("status", help="refresh + tally submitted batch states")
    pt.add_argument("--qids", nargs="+")
    pt.add_argument("--all", action="store_true", help="re-check already-terminal batches too")
    pt.set_defaults(func=cmd_status)

    pf = sub.add_parser("fetch", help="download + parse completed batches into predictions")
    pf.add_argument("--qids", nargs="+")
    pf.add_argument("--force", action="store_true", help="re-fetch even queries already marked complete")
    pf.set_defaults(func=cmd_fetch)

    pr = sub.add_parser("reset", help="clear failed (no-output) batches so they can be resubmitted")
    pr.add_argument("--qids", nargs="+", help="reset exactly these query ids")
    pr.add_argument("--failed", action="store_true",
                    help="auto-detect submitted batches that finished with no output (quota failures)")
    pr.set_defaults(func=cmd_reset)

    prt = sub.add_parser("retry",
                         help="rebuild inputs with only the MISSING ids of incomplete queries")
    prt.add_argument("--qids", nargs="+", help="retry these query ids (default: all incomplete)")
    prt.add_argument("--max-tokens", dest="max_tokens", type=int, default=32000,
                     help="max_completion_tokens for the retry batch (default 32000, above the "
                          "original 16000 so truncated paragraphs can finish)")
    prt.set_defaults(func=cmd_retry)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
