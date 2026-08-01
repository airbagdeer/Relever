"""Classify the *delta* needed to make R-Precision exact on high-R queries (R > 500).

R-Precision (Precision@R, R = #positives for a query) is evaluated per model in
``frontier-v2/evaluation``. ``R`` is the count of ``prediction == 1`` in the pooled labels
``data/labels/<qid>/predictions_<qid>.jsonl``. Those labels only cover
``all_candidates_<qid>.json`` — the **top-500-per-model union** — so when ``R > 500`` a model's
ranks 501..R are unlabeled and ``metrics.r_precision`` silently truncates (understating Precision@R).

This script classifies the missing paragraphs — every id in some model's top-``ceil((1+overhead)*R)``
that is not labeled yet — so each model's top-R becomes fully labeled. A rank-depth ``overhead``
(default 0.2 = go 20% deeper than R) buys headroom: R can grow up to (1+overhead)x before another
round is needed. Because labelling the delta finds new positives, ``R`` grows and a slightly deeper
delta is needed: a **fixpoint (cat-and-mouse)** loop ``build -> submit -> status -> fetch -> build``
until ``report``/``build`` shows 0 delta (the overhead makes this converge in fewer rounds).

Everything is scoped to a separate output root so the main labels + the evaluation stay untouched::

    data/labels/extended_pooling_delta/
        _manifest.json
        <qid>/batch_input_<qid>.jsonl      # gpt-5.2 / medium / short2b, byte-identical bodies
        <qid>/batch_output_<qid>.jsonl
        <qid>/predictions_<qid>.jsonl      # delta labels; accumulate across rounds

The per-request body is produced by the exact same helpers as ``label_frontier_batch`` (imported,
not duplicated), so a delta request is indistinguishable from a pooled one.

Lifecycle::

    # how many paragraphs (read-only, no corpus, no writes, no spend):
    uv run python -m src.create_dataset.classifying_data.batch_api.label_frontier_delta_batch report

    # build the batch inputs under extended_pooling_delta/ (resolves text; NO API calls, NO spend):
    uv run python -m ...label_frontier_delta_batch build

    # then, in waves, exactly like the main labeler:
    uv run python -m ...label_frontier_delta_batch submit --limit 25 [--dry-run]
    uv run python -m ...label_frontier_delta_batch status
    uv run python -m ...label_frontier_delta_batch fetch
    # ...re-run `build` after each fetch until it reports 0 delta.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Importing the main labeler runs its sys.path bootstrap (repo root -> importable `src.*`) and
# gives us every shared, path-agnostic helper so request bodies stay byte-identical.
from src.create_dataset.classifying_data.batch_api import label_frontier_batch as lfb
from src.create_dataset.classifying_data.prompt_variants import build_system_prompt
from src.create_dataset.classifying_data.relation_classification_parser import (
    parse_classification_output,
)
from src.create_dataset.classifying_data import frontier_cost_pricing as pricing_mod

# --- config (mirror the main labeler exactly) ------------------------------- #
MODEL = lfb.MODEL
EFFORT = lfb.EFFORT
VARIANT = lfb.VARIANT
MAX_COMPLETION_TOKENS = lfb.MAX_COMPLETION_TOKENS
ENDPOINT = lfb.ENDPOINT
COMPLETION_WINDOW = lfb.COMPLETION_WINDOW
TERMINAL_STATES = lfb.TERMINAL_STATES

# The trigger + the deepest rank materialised on disk (each per-model file holds ranks 1..3000).
POOL_DEPTH = 500
MAX_RANK = 3000
# Rank-depth overhead: label each model's top ceil((1 + OVERHEAD) * R) instead of top-R, so R can
# grow up to (1 + OVERHEAD)x before another cat-and-mouse round is needed. NB: 20% deeper in rank is
# far more than 20% more paragraphs — the models diverge at depth, so each rank band adds many new
# unique ids (0.2 => ~1.65x the paragraphs of the exact top-R delta on this dataset).
OVERHEAD = 0.2

# --- paths ------------------------------------------------------------------ #
_PROJECT_ROOT = lfb._PROJECT_ROOT
CANDIDATES_DIR = _PROJECT_ROOT / "data" / "candidates"
MAIN_CLASSIFIED_DIR = _PROJECT_ROOT / "data" / "labels"
DELTA_ROOT = MAIN_CLASSIFIED_DIR / "extended_pooling_delta"
SPLADE_V2_DIR = _PROJECT_ROOT / "data" / "splade_wikipedia_v2"
MANIFEST_PATH = DELTA_ROOT / "_manifest.json"


# --------------------------------------------------------------------------- #
# Path helpers (delta root)
# --------------------------------------------------------------------------- #
# These reference the module-global roots at call time (not via bound defaults) so tests can point
# the whole script at a tmp tree by monkeypatching DELTA_ROOT / MAIN_CLASSIFIED_DIR.
def delta_input_path(qid: str) -> Path:
    return DELTA_ROOT / qid / f"batch_input_{qid}.jsonl"


def delta_output_raw_path(qid: str) -> Path:
    return DELTA_ROOT / qid / f"batch_output_{qid}.jsonl"


def delta_errors_raw_path(qid: str) -> Path:
    return DELTA_ROOT / qid / f"batch_errors_{qid}.jsonl"


def delta_predictions_path(qid: str, delta_root: Path | None = None) -> Path:
    root = DELTA_ROOT if delta_root is None else delta_root
    return root / qid / f"predictions_{qid}.jsonl"


def main_predictions_path(qid: str, main_dir: Path | None = None) -> Path:
    root = MAIN_CLASSIFIED_DIR if main_dir is None else main_dir
    return root / qid / f"predictions_{qid}.jsonl"


# --------------------------------------------------------------------------- #
# Pure core (explicit paths so it is unit-testable without the real dataset)
# --------------------------------------------------------------------------- #
def _read_predictions(path: Path) -> dict[str, int]:
    """{chunk_id: prediction} from a predictions_<qid>.jsonl (empty if the file is absent)."""
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["id"]] = int(d["prediction"])
    return out


def union_labels(qid: str, *, main_dir: Path | None = None,
                 delta_root: Path | None = None) -> dict[str, int]:
    """Merge the read-only pooled labels with the accumulated delta labels for a query.

    The delta labels win on the (empty in practice) intersection; together they are the full set of
    everything classified so far, which is what ``R`` and the skip-done set must be computed from.
    """
    labels = _read_predictions(main_predictions_path(qid, main_dir))
    labels.update(_read_predictions(delta_predictions_path(qid, delta_root)))
    return labels


def _alias_candidate_files(qid: str, *, candidates_dir: Path | None = None) -> list[Path]:
    """Per-model ranked files ``<alias>_<qid>.json`` for a query (excludes the pooled union)."""
    cdir = CANDIDATES_DIR if candidates_dir is None else candidates_dir
    qdir = cdir / qid
    if not qdir.is_dir():
        return []
    return sorted(p for p in qdir.glob(f"*_{qid}.json") if p.name != f"all_candidates_{qid}.json")


def needed_ids_for_R(qid: str, R: int, *, candidates_dir: Path | None = None,
                     max_rank: int | None = None) -> set[str]:
    """Union over every model of the ids at rank <= min(R, max_rank) (each model's top-R)."""
    max_rank = MAX_RANK if max_rank is None else max_rank
    rcap = min(R, max_rank)
    needed: set[str] = set()
    for f in _alias_candidate_files(qid, candidates_dir=candidates_dir):
        doc = json.loads(f.read_text())
        for c in doc.get("candidates", []):
            if int(c.get("rank", 1 << 30)) <= rcap:
                cid = c.get("id")
                if cid:
                    needed.add(cid)
    return needed


@dataclass(frozen=True)
class DeltaInfo:
    qid: str
    event_id: int
    R: int
    rank_target: int             # effective per-model cutoff = ceil((1+overhead)*R), capped at max_rank
    r_capped: bool               # True if the target hit the max_rank ceiling
    n_needed: int                # |union of every model's top-rank_target|
    n_labeled: int               # |already classified (main + delta)|
    delta_ids: list[str] = field(default_factory=list)  # needed, not-yet-labeled, sorted

    @property
    def n_delta(self) -> int:
        return len(self.delta_ids)


def compute_delta(qid: str, *, candidates_dir: Path | None = None, main_dir: Path | None = None,
                  delta_root: Path | None = None, pool_depth: int | None = None,
                  max_rank: int | None = None, overhead: float | None = None) -> DeltaInfo:
    """Delta ids for one query: every model's top-ceil((1+overhead)*R) minus what's already labeled.

    With ``overhead == 0`` this is the exact top-R set (Precision@R just computable). With
    ``overhead == 0.2`` we go 20% deeper in rank for headroom against R growing. Returns an empty
    ``delta_ids`` when ``R <= pool_depth`` (the pooled top-500 already covers top-R). ``R`` is read
    from the main+delta union, so re-running after a fetch uses the grown R (the cat-and-mouse step).
    """
    pool_depth = POOL_DEPTH if pool_depth is None else pool_depth
    max_rank = MAX_RANK if max_rank is None else max_rank
    overhead = OVERHEAD if overhead is None else overhead
    labels = union_labels(qid, main_dir=main_dir, delta_root=delta_root)
    R = sum(1 for v in labels.values() if v == 1)
    event_id = int(qid)
    if R <= pool_depth:
        return DeltaInfo(qid, event_id, R, min(R, max_rank), False, 0, len(labels), [])
    rank_target = math.ceil(R * (1.0 + overhead))
    capped = rank_target > max_rank
    rank_target = min(rank_target, max_rank)
    needed = needed_ids_for_R(qid, rank_target, candidates_dir=candidates_dir, max_rank=max_rank)
    delta = sorted(cid for cid in needed if cid not in labels)
    return DeltaInfo(qid, event_id, R, rank_target, capped, len(needed), len(labels), delta)


def resolve_texts(needed_ids: set[str], *, splade_dir: Path | None = None) -> dict[str, str]:
    """Map delta ids -> paragraph text via a single early-exiting scan of the v2 SPLADE/BM25 dump.

    Field access mirrors ``corpus_loader_v2._iter_rows_v2`` (``item.id`` -> ``item.payload.content``).
    Delta ids are out-of-pool so their text is not in ``all_candidates``; every candidate id was
    retrieved from this dump, so all should resolve (any miss is returned as absent, never guessed).
    """
    splade_dir = SPLADE_V2_DIR if splade_dir is None else splade_dir
    remaining = set(needed_ids)
    out: dict[str, str] = {}
    if not remaining:
        return out
    for f in sorted(splade_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        for key in ("splade_results", "bm25_results"):
            for item in doc.get(key) or []:
                if not isinstance(item, dict):
                    continue
                cid = item.get("id")
                if cid not in remaining:
                    continue
                content = (item.get("payload") or {}).get("content")
                if content and content.strip():
                    out[cid] = content.strip()
                    remaining.discard(cid)
            if not remaining:
                return out
    return out


# --------------------------------------------------------------------------- #
# Manifest (delta root) — same shape as the main labeler + R_at_build / n_delta
# --------------------------------------------------------------------------- #
def _empty_entry(event_id: int) -> dict:
    return {
        "event_id": event_id,
        "R_at_build": None, "overhead_at_build": None, "n_delta": None,
        "input_built": False, "n_requests": None, "input_bytes": None, "built_at": None,
        "n_unresolved": None,
        "batch_id": None, "input_file_id": None, "submitted_at": None, "status": None,
        "request_counts": None, "output_file_id": None, "error_file_id": None,
        "fetched": False, "fetched_at": None,
        "n_predictions": None, "n_pos": None, "n_neg": None, "n_failed": None,
        "tokens": None, "cost_uncached": None, "cost_batch": None, "complete": None,
    }


def _config(overhead: float | None = None) -> dict:
    return {
        "model": MODEL, "effort": EFFORT, "variant": VARIANT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "endpoint": ENDPOINT, "completion_window": COMPLETION_WINDOW,
        "overhead": OVERHEAD if overhead is None else overhead,
    }


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"config": None, "updated_at": None, "queries": {}}


def save_manifest(m: dict) -> None:
    m["updated_at"] = lfb._now_iso()
    DELTA_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    os.replace(tmp, MANIFEST_PATH)


def _require_manifest() -> dict:
    m = load_manifest()
    if not m.get("queries"):
        raise SystemExit("No delta manifest yet — run 'build' first.")
    return m


# A rebuild (new round after R grew) starts a fresh batch: clear only the submission/fetch state,
# keeping the cumulative cost / token / label counts (the delta predictions file also accumulates).
_ROUND_CLEAR_FIELDS = (
    "batch_id", "input_file_id", "submitted_at", "status", "request_counts",
    "output_file_id", "error_file_id", "fetched_at",
)


# --------------------------------------------------------------------------- #
# Query discovery
# --------------------------------------------------------------------------- #
def discover_qids() -> list[str]:
    """4-digit query ids that have both a candidate dir and main pooled predictions."""
    if not CANDIDATES_DIR.exists():
        raise SystemExit(f"Candidates dir not found: {CANDIDATES_DIR}")
    qids = []
    for d in CANDIDATES_DIR.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 4:
            if main_predictions_path(d.name).exists():
                qids.append(d.name)
    return sorted(qids)


def _select_qids(arg_qids, all_qids: list[str]) -> list[str]:
    if not arg_qids:
        return all_qids
    known = set(all_qids)
    chosen = []
    for raw in arg_qids:
        qid = lfb._norm_qid(raw)
        if qid not in known:
            raise SystemExit(f"--qids: '{raw}' -> '{qid}' has no candidate folder + pooled labels.")
        chosen.append(qid)
    return chosen


def _high_r_deltas(qids: list[str], *, overhead: float | None = None) -> list[DeltaInfo]:
    """compute_delta for each query, keeping only those with R > POOL_DEPTH (in id order)."""
    out = []
    for qid in qids:
        info = compute_delta(qid, overhead=overhead)
        if info.R > POOL_DEPTH:
            out.append(info)
    return out


# --------------------------------------------------------------------------- #
# report — read-only delta accounting (no corpus, no writes, no spend)
# --------------------------------------------------------------------------- #
def cmd_report(args: argparse.Namespace) -> None:
    overhead = args.overhead
    qids = _select_qids(args.qids, discover_qids())
    infos = _high_r_deltas(qids, overhead=overhead)
    infos.sort(key=lambda i: i.n_delta, reverse=True)

    total_delta = sum(i.n_delta for i in infos)
    pending = [i for i in infos if i.n_delta > 0]
    print(f"Overhead {overhead:.0%}: label each model's top ceil((1+{overhead:g})*R), deduplicated.")
    print(f"Queries scanned: {len(qids)} · with R>{POOL_DEPTH}: {len(infos)} · "
          f"still needing delta: {len(pending)}")
    print(f"{'qid':>5} {'R':>6} {'top<=':>6} {'labeled':>8} {'need':>7} {'delta':>7}")
    for i in infos:
        cap = "  (clamped to MAX_RANK)" if i.r_capped else ""
        print(f"{i.qid:>5} {i.R:>6} {i.rank_target:>6} {i.n_labeled:>8} {i.n_needed:>7} "
              f"{i.n_delta:>7}{cap}")
    print(f"\nTOTAL delta paragraphs to classify now: {total_delta:,}")
    if not total_delta and infos:
        print(f"Converged — every R>{POOL_DEPTH} query's top-ceil((1+{overhead:g})*R) is labeled.")


# --------------------------------------------------------------------------- #
# build — write batch_input_<qid>.jsonl for the current delta (no API calls)
# --------------------------------------------------------------------------- #
def _write_batch_input(qid: str, delta_ids: list[str], id2text: dict[str, str],
                       qinfo: dict, system_prompt: str, cfg: dict) -> tuple[int, int]:
    """Write the delta batch input for one query. Returns (n_requests, n_unresolved)."""
    out_dir = DELTA_ROOT / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = delta_input_path(qid)
    tmp = out_path.with_suffix(".jsonl.tmp")
    n = unresolved = 0
    with tmp.open("w") as fh:
        for cid in delta_ids:
            text = id2text.get(cid)
            if not text:  # never emit an empty-text request; leave it for a later resolve
                unresolved += 1
                continue
            line = lfb._request_line(cid, system_prompt, qinfo, text, cfg["model"],
                                     cfg["endpoint"], cfg["max_completion_tokens"], cfg["effort"])
            fh.write(json.dumps(line) + "\n")
            n += 1
    os.replace(tmp, out_path)
    return n, unresolved


def cmd_build(args: argparse.Namespace) -> None:
    overhead = args.overhead
    cfg = _config(overhead)
    system_prompt = build_system_prompt(cfg["variant"])
    query_meta = lfb.load_query_meta()
    qids = _select_qids(args.qids, discover_qids())

    m = load_manifest()
    m["config"] = cfg

    infos = _high_r_deltas(qids, overhead=overhead)
    for info in infos:
        m["queries"].setdefault(info.qid, _empty_entry(info.event_id))

    # A query is rebuilt when it was never built, R changed since the last build (a fetch grew it),
    # the overhead factor changed, or --force. Same R + same overhead + already built => skip (no-op).
    todo: list[DeltaInfo] = []
    for info in infos:
        e = m["queries"][info.qid]
        if info.n_delta == 0:
            e.update(R_at_build=info.R, overhead_at_build=overhead, n_delta=0,
                     input_built=True, n_requests=0, built_at=lfb._now_iso(), complete=True)
            continue
        if (not args.force and e.get("input_built") and e.get("R_at_build") == info.R
                and e.get("overhead_at_build") == overhead):
            continue
        todo.append(info)
    if args.limit:
        todo = todo[: args.limit]

    print(f"Config: model={cfg['model']} effort={cfg['effort']} variant={cfg['variant']} "
          f"max_tok={cfg['max_completion_tokens']} overhead={overhead:.0%}")
    n_converged = sum(1 for i in infos if i.n_delta == 0)
    print(f"Queries with R>{POOL_DEPTH}: {len(infos)} · converged (0 delta): {n_converged} · "
          f"(re)building this run: {len(todo)}")
    if not todo:
        save_manifest(m)
        print("Nothing to build.")
        return

    # One SPLADE scan resolves text for the union of every targeted query's delta ids.
    all_ids: set[str] = set()
    for info in todo:
        all_ids.update(info.delta_ids)
    print(f"Resolving text for {len(all_ids):,} unique delta ids "
          f"(single scan of {SPLADE_V2_DIR.name}/) ...")
    id2text = resolve_texts(all_ids, splade_dir=SPLADE_V2_DIR)
    missing = len(all_ids) - len(id2text)
    if missing:
        print(f"  WARNING: {missing:,} delta id(s) had no text in the SPLADE dump "
              f"(skipped, not emitted as requests).")

    tot_req = tot_bytes = tot_unresolved = 0
    for info in todo:
        qinfo = query_meta.get(info.event_id)
        e = m["queries"][info.qid]
        if qinfo is None:
            e.update(input_built=False, build_error=f"no query metadata for event_id={info.event_id}")
            print(f"  ERROR {info.qid}: no query metadata for event_id={info.event_id}")
            continue
        n_req, unresolved = _write_batch_input(info.qid, info.delta_ids, id2text, qinfo,
                                               system_prompt, cfg)
        # A new round is a new batch — clear this query's submission/fetch state (cumulative
        # cost/token/label counts are preserved; the delta predictions file also accumulates).
        for k in _ROUND_CLEAR_FIELDS:
            e[k] = None
        e.update(R_at_build=info.R, overhead_at_build=overhead, n_delta=info.n_delta,
                 input_built=True, n_requests=n_req,
                 input_bytes=delta_input_path(info.qid).stat().st_size, built_at=lfb._now_iso(),
                 n_unresolved=unresolved, fetched=False, complete=(n_req == 0))
        e.pop("build_error", None)
        tot_req += n_req
        tot_bytes += e["input_bytes"]
        tot_unresolved += unresolved
        cap = " (clamped to MAX_RANK)" if info.r_capped else ""
        print(f"  {info.qid}: R={info.R} top<={info.rank_target} delta={info.n_delta} "
              f"-> {n_req:,} requests" + (f" · {unresolved} unresolved" if unresolved else "") + cap)
    save_manifest(m)

    print(f"\nDone. Built {len(todo)} query(ies) · {tot_req:,} requests · {tot_bytes / 1e6:.1f} MB"
          + (f" · {tot_unresolved:,} unresolved (skipped)" if tot_unresolved else ""))
    print(f"Manifest -> {MANIFEST_PATH}")
    print("Not submitted. Next: 'submit --dry-run --limit N' to preview, then 'submit'.")


# --------------------------------------------------------------------------- #
# submit / status / fetch — scoped to the delta root (loops mirror the main labeler)
# --------------------------------------------------------------------------- #
def cmd_submit(args: argparse.Namespace) -> None:
    m = _require_manifest()
    cfg = m["config"]
    all_qids = sorted(m["queries"])

    def _submittable(q: str) -> bool:
        e = m["queries"][q]
        return e["input_built"] and not e["batch_id"] and (e.get("n_requests") or 0) > 0

    if args.qids:
        wave = [q for q in _select_qids(args.qids, all_qids) if _submittable(q)]
    else:
        wave = [q for q in all_qids if _submittable(q)]
    if args.limit:
        wave = wave[: args.limit]

    eligible = sum(1 for q in all_qids if _submittable(q))
    print(f"{eligible} delta batch(es) built-and-unsubmitted; this wave: {len(wave)}")
    if not wave:
        return
    if args.dry_run:
        total_req = sum(m["queries"][q]["n_requests"] or 0 for q in wave)
        print("--dry-run: would submit (no upload, no spend):")
        print("  " + " ".join(wave))
        print(f"  {len(wave)} batches · {total_req:,} total requests")
        return

    client = lfb._client()
    submitted = 0
    for qid in wave:
        path = delta_input_path(qid)
        if not path.exists():
            print(f"  SKIP {qid}: input file missing ({path})")
            continue
        try:
            with path.open("rb") as fh:
                up = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=up.id, endpoint=cfg["endpoint"],
                completion_window=cfg["completion_window"],
                metadata={"description": f"relation-classification DELTA {cfg['variant']} "
                                         f"{cfg['model']}:{cfg['effort']} qid={qid}", "qid": qid},
            )
        except Exception as exc:  # noqa: BLE001 — stop the wave cleanly, keep prior progress
            print(f"  STOP at {qid}: {type(exc).__name__}: {exc}")
            print(f"  Submitted {submitted} this run; progress saved. Wait for capacity, re-run submit.")
            break
        e = m["queries"][qid]
        e.update(batch_id=batch.id, input_file_id=up.id,
                 submitted_at=lfb._now_iso(), status=batch.status)
        save_manifest(m)  # persist after EACH submit — a live batch_id is money in flight
        submitted += 1
        print(f"  {qid}: batch {batch.id} ({batch.status})")
    print(f"Submitted {submitted} batch(es). Poll with 'status', collect with 'fetch'.")


def cmd_status(args: argparse.Namespace) -> None:
    m = _require_manifest()
    all_qids = sorted(m["queries"])
    qids = _select_qids(args.qids, all_qids) if args.qids else \
        [q for q in all_qids if m["queries"][q]["batch_id"]]

    client = lfb._client()
    tally: Counter = Counter()
    refreshed = 0
    for i, qid in enumerate(qids, 1):
        e = m["queries"][qid]
        if not e["batch_id"]:
            continue
        if e["status"] in TERMINAL_STATES and not args.all:
            tally[e["status"]] += 1
            continue
        batch, err = lfb._safe_retrieve(client, e["batch_id"])
        if err == "not_found":
            e["status"] = "not_found"
            tally["not_found"] += 1
            refreshed += 1
            print(f"  {qid}: batch {e['batch_id']} not found (404) — rebuild with 'build --force "
                  f"--qids {qid}' then 'submit --qids {qid}'.")
        elif err:
            tally["retrieve_error"] += 1
            print(f"  {qid}: {err}; left as-is.")
        else:
            assert batch is not None
            e.update(status=batch.status, request_counts=lfb._counts_dict(batch),
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


def _parse_to_delta_predictions(qid: str, event_id: int, out_text: str, model: str) -> dict:
    """Parse a batch output JSONL into the delta predictions file (additive union-by-id merge)."""
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
            continue  # leave missing so a later retry re-picks it; never record a -1 here
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

    merged: dict[str, dict] = {}
    pf = delta_predictions_path(qid)
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
    qids = _select_qids(args.qids, all_qids) if args.qids else \
        [q for q in all_qids if m["queries"][q]["batch_id"] and not m["queries"][q]["fetched"]]

    client = lfb._client()
    fetched = skipped = 0
    grand_cost = 0.0
    for qid in qids:
        e = m["queries"][qid]
        if not e["batch_id"]:
            continue
        if e.get("complete") and not args.force:
            continue
        batch, err = lfb._safe_retrieve(client, e["batch_id"])
        if err == "not_found":
            e["status"] = "not_found"
            save_manifest(m)
            print(f"  {qid}: batch not found (404); rebuild + resubmit.")
            skipped += 1
            continue
        if err:
            print(f"  {qid}: {err}; skipping.")
            skipped += 1
            continue
        assert batch is not None
        e["status"] = batch.status
        e["request_counts"] = lfb._counts_dict(batch)
        if batch.status != "completed":
            print(f"  {qid}: status={batch.status}; not ready.")
            skipped += 1
            continue
        if not batch.output_file_id:
            print(f"  {qid}: completed but no output_file_id; skipping.")
            skipped += 1
            continue

        out_text = client.files.content(batch.output_file_id).text
        delta_output_raw_path(qid).write_text(out_text)
        if batch.error_file_id:
            delta_errors_raw_path(qid).write_text(client.files.content(batch.error_file_id).text)

        stats = _parse_to_delta_predictions(qid, e["event_id"], out_text, model)
        prev_tok = e.get("tokens") or {}
        tokens = {k: prev_tok.get(k, 0) + stats["new_tokens"].get(k, 0)
                  for k in set(prev_tok) | set(stats["new_tokens"])}
        # This round's batch carried n_requests ids; complete when they are all accounted for.
        ground = e.get("n_requests") or stats["n_predictions"]
        complete = stats["n_new"] >= ground
        e.update(
            fetched=True, fetched_at=lfb._now_iso(), complete=complete,
            output_file_id=batch.output_file_id, error_file_id=batch.error_file_id,
            n_predictions=stats["n_predictions"], n_pos=stats["n_pos"],
            n_neg=stats["n_neg"], n_failed=stats["n_failed"], tokens=tokens,
            cost_uncached=round((e.get("cost_uncached") or 0) + stats["new_cost_uncached"], 4),
            cost_batch=round((e.get("cost_batch") or 0) + stats["new_cost_batch"], 4),
        )
        save_manifest(m)
        fetched += 1
        grand_cost += stats["new_cost_batch"]
        flag = "" if complete else f"   ** INCOMPLETE {stats['n_new']:,}/{ground:,} **"
        print(f"  {qid}: +{stats['n_new']:,} delta labels "
              f"(pos {stats['n_pos']}, neg {stats['n_neg']}, fail {stats['n_failed']}) "
              f"· ${stats['new_cost_batch']:.2f}{flag}")

    print(f"Fetched {fetched} batch(es); skipped {skipped}. Batch cost this run: ${grand_cost:.2f}")
    print("Re-run 'build' to recompute R (it grows as delta labels land) and enqueue the next "
          "round; repeat until 'report' shows 0 delta.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("report", help="read-only: per-query R + delta counts (no writes, no spend)")
    pr.add_argument("--qids", nargs="+", help="only these query ids (default: all)")
    pr.add_argument("--overhead", type=float, default=OVERHEAD,
                    help=f"rank-depth overhead: label each model's top ceil((1+overhead)*R) "
                         f"(default {OVERHEAD}; 0 = exact top-R)")
    pr.set_defaults(func=cmd_report)

    pb = sub.add_parser("build", help="write delta batch_input_<qid>.jsonl under extended_pooling_delta/")
    pb.add_argument("--qids", nargs="+", help="only these query ids (default: all R>500)")
    pb.add_argument("--limit", type=int, help="cap how many queries to (re)build this run")
    pb.add_argument("--force", action="store_true", help="rebuild even if R is unchanged")
    pb.add_argument("--overhead", type=float, default=OVERHEAD,
                    help=f"rank-depth overhead: label each model's top ceil((1+overhead)*R) "
                         f"(default {OVERHEAD}; 0 = exact top-R). Changing it rebuilds all queries.")
    pb.set_defaults(func=cmd_build)

    ps = sub.add_parser("submit", help="upload + create a batch for each built delta input")
    ps.add_argument("--qids", nargs="+")
    ps.add_argument("--limit", type=int, help="submit the next N built-but-unsubmitted queries")
    ps.add_argument("--dry-run", action="store_true", help="show the wave; no upload, no spend")
    ps.set_defaults(func=cmd_submit)

    pt = sub.add_parser("status", help="refresh + tally submitted batch states")
    pt.add_argument("--qids", nargs="+")
    pt.add_argument("--all", action="store_true", help="re-check already-terminal batches too")
    pt.set_defaults(func=cmd_status)

    pf = sub.add_parser("fetch", help="download + parse completed batches into delta predictions")
    pf.add_argument("--qids", nargs="+")
    pf.add_argument("--force", action="store_true", help="re-fetch even queries marked complete")
    pf.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
