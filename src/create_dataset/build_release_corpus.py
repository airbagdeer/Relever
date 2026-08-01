"""Build the deduplicated public-release corpus from the SPLADE candidate dumps.

Streams data/splade_wikipedia_v2/<eid>.json (eid 0-1099), unions paragraph ids
across bm25_results and splade_results exactly like
src/process_results/paper_stats.py::corpus_stats, and writes one gzipped JSONL
record per unique paragraph, first-occurrence order (files scanned in numeric
order), so the output is deterministic:

    {"_id": <qdrant point id>, "title": ..., "text": ..., "url": ...}

The line count must equal the paper's corpus size (\\statNCorpus = 1,080,071).

With ``--shard-lines N`` the corpus is written as ``corpus.part-XX.jsonl.gz``
shards of N lines each under ``--out`` (treated as a directory). Each shard is
an independently valid gzip, so ``zcat corpus.part-*.jsonl.gz`` (in order) or
``cat`` reproduces the single-file corpus exactly. The public release ships
8 shards of 135,009 lines to stay under GitHub's 100 MiB per-file limit.

Usage:
    uv run python src/create_dataset/build_release_corpus.py --out <corpus.jsonl.gz>
    uv run python src/create_dataset/build_release_corpus.py --out <dir> --shard-lines 135009
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLADE_V2_DIR = PROJECT_ROOT / "data" / "splade_wikipedia_v2"
VALID_EIDS = range(0, 1100)


class _ShardedWriter:
    """Rotates gzip shards every ``shard_lines`` lines; a plain file when 0."""

    def __init__(self, out_path: Path, shard_lines: int) -> None:
        self.out_path = out_path
        self.shard_lines = shard_lines
        self.n_lines = 0
        self.n_shards = 0
        self._fh = None
        if not shard_lines:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(out_path, "wt", encoding="utf-8")

    def write(self, line: str) -> None:
        if self.shard_lines and self.n_lines % self.shard_lines == 0:
            if self._fh is not None:
                self._fh.close()
            self.out_path.mkdir(parents=True, exist_ok=True)
            shard = self.out_path / f"corpus.part-{self.n_shards:02d}.jsonl.gz"
            self._fh = gzip.open(shard, "wt", encoding="utf-8", compresslevel=9)
            self.n_shards += 1
        self._fh.write(line)
        self.n_lines += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def build(splade_dir: Path, out_path: Path, shard_lines: int = 0) -> dict:
    seen: set[str] = set()
    n_files = 0
    n_written = 0
    files = sorted(splade_dir.glob("*.json"), key=lambda p: int(p.stem))
    out = _ShardedWriter(out_path, shard_lines)
    try:
        for p in files:
            if int(p.stem) not in VALID_EIDS:
                continue
            with p.open() as f:
                doc = json.load(f)
            n_files += 1
            for section in ("bm25_results", "splade_results"):
                for hit in doc.get(section, []):
                    pid = hit["id"]
                    if pid in seen:
                        continue
                    seen.add(pid)
                    payload = hit.get("payload", {})
                    rec = {
                        "_id": pid,
                        "title": payload.get("title", ""),
                        "text": payload.get("content", ""),
                        "url": payload.get("url", ""),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
            if n_files % 100 == 0:
                print(f"  {n_files} files, {n_written} unique paragraphs", flush=True)
    finally:
        out.close()
    return {"n_files": n_files, "n_unique": n_written, "n_shards": out.n_shards}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splade-dir", type=Path, default=SPLADE_V2_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-lines", type=int, default=0,
                    help="lines per corpus.part-XX.jsonl.gz shard (0 = single file; "
                         "--out is a directory when set)")
    args = ap.parse_args()
    stats = build(args.splade_dir, args.out, args.shard_lines)
    print(json.dumps(stats))
    expected = 1_080_071
    if stats["n_unique"] != expected:
        print(f"WARNING: corpus size {stats['n_unique']} != expected {expected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
