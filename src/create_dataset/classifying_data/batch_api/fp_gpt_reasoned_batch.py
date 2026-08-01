"""gpt-5.2 reasoned second opinions for the FP-pilot reveal, via the Batch API.

Runs the 100 human-subset pilot pairs through gpt-5.2 (medium effort — the pipeline
classifier's model) with the *reasoned* output schema: the three presence bits plus a
brief paragraph-citing reasoning per criterion — the exact instructions the Opus 4.8
reveal agents received, so the two judges are directly comparable in the labeling
tool's reveal panel.

This is a REVEAL AID, not the pipeline classifier: results land in
``data/fp_taxonomy/validation/gpt_bits_reasoned.jsonl`` (sibling of ``opus_bits.jsonl``)
and never touch the per-event ``fp_taxonomy_<qid>.jsonl`` predictions that the
agreement/composition analyses read. Where the reasoned run's bit differs from the
bare pipeline run's, the labeling tool shows both.

Lifecycle (single batch; state in ``pilot/_gpt_reasoned_state.json``)::

    uv run python -m src.create_dataset.classifying_data.batch_api.fp_gpt_reasoned_batch build
    uv run python -m ...fp_gpt_reasoned_batch submit
    uv run python -m ...fp_gpt_reasoned_batch status
    uv run python -m ...fp_gpt_reasoned_batch fetch
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from src.create_dataset.classifying_data.batch_api import label_frontier_batch as lfb
from src.create_dataset.classifying_data.batch_api.label_fp_taxonomy_batch import (
    PILOT_DIR,
    PILOT_PAIRS_PATH,
    _qid,
    _texts_for_event,
)
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (
    BITS,
    REASONING_FIELDS,
    build_fp_bits_reasoned_prompt,
    parse_fp_bits_reasoned_output,
)

MODEL = lfb.MODEL      # gpt-5.2
EFFORT = lfb.EFFORT    # medium

INPUT_PATH = PILOT_DIR / "gpt_reasoned_input.jsonl"
OUTPUT_RAW_PATH = PILOT_DIR / "gpt_reasoned_output.jsonl"
RESULT_PATH = PILOT_DIR / "gpt_bits_reasoned.jsonl"
STATE_PATH = PILOT_DIR / "_gpt_reasoned_state.json"


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"batch_id": None, "input_file_id": None, "status": None, "fetched": False}


def _save_state(s: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, indent=2))
    os.replace(tmp, STATE_PATH)


def _human_pairs() -> list[dict]:
    pairs = [p for p in json.loads(PILOT_PAIRS_PATH.read_text()) if p.get("human")]
    pairs.sort(key=lambda r: (r["event_id"], r["id"]))
    return pairs


def cmd_build(_args: argparse.Namespace) -> None:
    system_prompt = build_fp_bits_reasoned_prompt()
    query_meta = lfb.load_query_meta()
    pairs = _human_pairs()
    by_event: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_event[p["event_id"]].append(p)

    n = 0
    tmp = INPUT_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for eid in sorted(by_event):
            q = query_meta[eid]
            texts = _texts_for_event(eid, {p["id"] for p in by_event[eid]})
            for p in by_event[eid]:
                line = lfb._request_line(
                    f"{_qid(eid)}|{p['id']}", system_prompt, q, texts[p["id"]],
                    MODEL, lfb.ENDPOINT, lfb.MAX_COMPLETION_TOKENS, EFFORT)
                fh.write(json.dumps(line) + "\n")
                n += 1
    os.replace(tmp, INPUT_PATH)
    print(f"Built {n} requests ({INPUT_PATH.stat().st_size / 1e6:.1f} MB, "
          f"system prompt {len(system_prompt):,} chars) -> {INPUT_PATH}")


def cmd_submit(_args: argparse.Namespace) -> None:
    s = _load_state()
    if s["batch_id"]:
        raise SystemExit(f"Already submitted: {s['batch_id']} (status {s['status']}).")
    if not INPUT_PATH.exists():
        raise SystemExit("No input file — run 'build' first.")
    client = lfb._client()
    with INPUT_PATH.open("rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id, endpoint=lfb.ENDPOINT,
        completion_window=lfb.COMPLETION_WINDOW,
        metadata={"description": f"fp-bits reasoned reveal {MODEL}:{EFFORT} (100 pilot pairs)"},
    )
    s.update(batch_id=batch.id, input_file_id=up.id, status=batch.status)
    _save_state(s)
    print(f"Submitted: {batch.id} ({batch.status})")


def cmd_status(_args: argparse.Namespace) -> None:
    s = _load_state()
    if not s["batch_id"]:
        raise SystemExit("Not submitted yet.")
    client = lfb._client()
    batch, err = lfb._safe_retrieve(client, s["batch_id"])
    if err:
        print(f"{s['batch_id']}: {err}")
        return
    assert batch is not None
    s["status"] = batch.status
    _save_state(s)
    rc = lfb._counts_dict(batch) or {}
    print(f"{s['batch_id']}: {batch.status} "
          f"({rc.get('completed', '?')}/{rc.get('total', '?')} done)")


def cmd_fetch(_args: argparse.Namespace) -> None:
    s = _load_state()
    if not s["batch_id"]:
        raise SystemExit("Not submitted yet.")
    client = lfb._client()
    batch, err = lfb._safe_retrieve(client, s["batch_id"])
    if err:
        raise SystemExit(f"{s['batch_id']}: {err}")
    assert batch is not None
    if batch.status != "completed":
        raise SystemExit(f"status={batch.status}; not ready.")
    if not batch.output_file_id:
        raise SystemExit("completed with no output file (quota failure?).")

    out_text = client.files.content(batch.output_file_id).text
    OUTPUT_RAW_PATH.write_text(out_text)

    n_parsed = n_failed = 0
    tmp = RESULT_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for line in out_text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            resp = obj.get("response")
            if obj.get("error") or not resp or resp.get("status_code") != 200:
                n_failed += 1
                continue
            qid_s, _, chunk_id = (obj.get("custom_id") or "").partition("|")
            body = resp.get("body") or {}
            content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            parsed = parse_fp_bits_reasoned_output(content)
            if parsed is None:
                n_failed += 1
                continue
            fh.write(json.dumps({
                "id": chunk_id, "event_id": int(qid_s),
                "judge": f"{MODEL}-{EFFORT}-reasoned",
                **{b: parsed[b] for b in BITS},
                **{f: parsed[f] for f in REASONING_FIELDS},
            }) + "\n")
            n_parsed += 1
    os.replace(tmp, RESULT_PATH)
    s["fetched"] = True
    _save_state(s)
    print(f"Parsed {n_parsed} (+{n_failed} failures) -> {RESULT_PATH}")
    print("Regenerate the labeling page: uv run python "
          "src/create_dataset/classifying_data/make_fp_label_tool.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build", cmd_build), ("submit", cmd_submit),
                     ("status", cmd_status), ("fetch", cmd_fetch)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
