"""Label the Opus-4.8 reference judge over the emitted opus_inputs/ files.

Reproduces the frozen-consensus Opus member: ONE Anthropic API call per paragraph
with extended thinking (max reasoning), parsed to 1/0, appended to opus-4.8.jsonl.
Resumable — skips ids already in opus-4.8.jsonl, so after a sample expansion it
labels only the new paragraphs. Reads the self-contained judge_prompt files written
by `estimate_frontier_cost.py --emit-opus-inputs`.

    uv run python src/create_dataset/classifying_data/label_opus.py
    uv run python src/create_dataset/classifying_data/label_opus.py --limit 3 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    from .relation_classification_parser import parse_classification_output
    from . import frontier_cost_report as report_mod
except ImportError:  # run as a plain script
    from relation_classification_parser import parse_classification_output
    import frontier_cost_report as report_mod

from src.retrieval.config import PROJECT_ROOT  # noqa: E402

MODEL = "claude-opus-4-8"
EFFORT = "high"  # adaptive-thinking effort (max reasoning) for the gold judge
MAX_TOKENS = 10000

RAW_DIR = PROJECT_ROOT / "data/judge_validation/short"
INPUTS_DIR = RAW_DIR / "opus_inputs"
OUT_PATH = RAW_DIR / f"{report_mod.OPUS_LABELER}.jsonl"


def _load_inputs(inputs_dir: Path) -> list[dict]:
    files = sorted(inputs_dir.glob("*.json"), key=lambda p: int(p.stem))
    return [json.loads(f.read_text()) for f in files]


def _done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {
        json.loads(line)["id"]
        for line in out_path.read_text().splitlines()
        if line.strip()
    }


async def judge(client: AsyncAnthropic, sem: asyncio.Semaphore, rec: dict, max_retries: int = 4) -> dict:
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    thinking={"type": "adaptive"},
                    output_config={"effort": EFFORT},
                    messages=[{"role": "user", "content": rec["judge_prompt"]}],
                )
            except Exception as e:  # noqa: BLE001
                if attempt == max_retries - 1:
                    return {"id": rec["id"], "prediction": -1, "brief_reason": f"error: {str(e)[:120]}"}
                await asyncio.sleep((2 ** attempt) + 1)
                continue
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
            think = "".join(getattr(b, "thinking", "") for b in resp.content if getattr(b, "type", "") == "thinking")
            pred = parse_classification_output(text or think)
            brief = (think or text).strip().replace("\n", " ")[:200]
            return {"id": rec["id"], "prediction": pred, "brief_reason": brief}
    return {"id": rec["id"], "prediction": -1, "brief_reason": "error: retries exhausted"}


async def run(concurrency: int, limit: int | None, inputs_dir: Path, out_path: Path) -> None:
    load_dotenv()
    inputs = _load_inputs(inputs_dir)
    done = _done_ids(out_path)
    todo = [r for r in inputs if r["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"All {len(inputs)} opus inputs already labeled (done={len(done)}).")
        return
    print(f"Labeling {len(todo)} paragraphs with {MODEL} (adaptive thinking, "
          f"effort={EFFORT}); {len(done)} already done. inputs={inputs_dir} out={out_path}")

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    fh = out_path.open("a")

    async def worker(rec: dict) -> dict:
        res = await judge(client, sem, rec)
        async with write_lock:
            fh.write(json.dumps(res) + "\n")
            fh.flush()
        return res

    results = []
    for cor in async_tqdm.as_completed([worker(r) for r in todo], total=len(todo), desc="opus-4.8"):
        results.append(await cor)
    fh.close()

    failed = sum(1 for r in results if r["prediction"] == -1)
    pos = sum(1 for r in results if r["prediction"] == 1)
    print(f"Done: {len(results)} labeled, {pos} positive, {failed} failed -> {OUT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="label only first N todo paragraphs")
    ap.add_argument("--inputs-dir", default=str(INPUTS_DIR),
                    help="dir of emitted judge_prompt files (default: base opus_inputs/)")
    ap.add_argument("--out-path", default=str(OUT_PATH),
                    help="jsonl to append labels to (default: base opus-4.8.jsonl)")
    args = ap.parse_args()
    asyncio.run(run(args.concurrency, args.limit, Path(args.inputs_dir), Path(args.out_path)))


if __name__ == "__main__":
    main()
