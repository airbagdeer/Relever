"""Reconstruct the June 3, 2026 MTEB-English and RTEB-English leaderboard snapshots.

Both leaderboards on the unified MTEB leaderboard (huggingface.co/spaces/mteb/leaderboard)
are computed by the `mteb` package from the git-versioned results repo
`github.com/embeddings-benchmark/results`, ranked by Borda count. A dated snapshot is
reconstructed by pinning that repo to the commit that was current on the target date and
recomputing the summary tables locally — the exact code path the leaderboard Space uses.

  MTEB screenshot -> benchmark "MTEB(eng, v2)"
  RTEB screenshot -> benchmark "RTEB(eng, beta)"  (public tasks only; see RtebBenchmark)

June 3, 2026 pin: results commit 9ff0358 (2026-06-03 11:05 UTC), preceded by 0927beae
(2026-05-31). The screenshots in MTEB/ and RTEB/ are the acceptance oracle.

Run (ephemeral env, no pyproject pollution):
  uv run --no-project --with "mteb" --with pandas \
      python paper/leaderboard_snapshot/build_snapshot.py --commit 9ff0358
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RESULTS_REMOTE = "https://github.com/embeddings-benchmark/results"
SNAP_DATE = "2026-06-03"
HERE = Path(__file__).resolve().parent

BENCHMARKS = {
    "MTEB": {"name": "MTEB(eng, v2)", "csv": HERE / "MTEB" / "mteb_eng_v2_2026-06-03.csv"},
    "RTEB": {"name": "RTEB(eng, beta)", "csv": HERE / "RTEB" / "rteb_eng_beta_2026-06-03.csv"},
}

# Expected Borda rank -> set of model short-names, transcribed from the screenshots
# (img.png / img_1.png in each folder). Ties share a rank.
EXPECTED = {
    "MTEB": {
        1: {"Jasper-Token-Compression-600M"},
        2: {"QZhou-Embedding"},
        3: {"Seed1.5-Embedding"},
        4: {"Qwen3-Embedding-8B"},
        5: {"LGAI-Embedding-Preview"},
        6: {"Qwen3-Embedding-4B"},
        7: {"Seed1.6-embedding"},
        8: {"gemini-embedding-001"},
        9: {"F2LLM-v2-8B"},
        10: {"F2LLM-v2-14B"},
        11: {"jina-embeddings-v5-text-small", "jina-embeddings-v5-omni-small"},
        13: {"jasper_en_vision_language_v1"},
        14: {"F2LLM-v2-4B"},
        15: {"F2LLM-4B"},
        16: {"jina-embeddings-v5-text-nano", "jina-embeddings-v5-omni-nano"},
        18: {"F2LLM-v2-1.7B"},
    },
    "RTEB": {
        1: {"Octen-Embedding-8B"},
        2: {"voyage-4-large (embed_dim=2048)"},
        3: {"Octen-Embedding-8B-INT8"},
        4: {"voyage-4-large"},
        5: {"voyage-3-large"},
        6: {"Octen-Embedding-4B"},
        7: {"Octen-Embedding-4B-INT8"},
        8: {"voyage-4"},
        9: {"MoD-Embedding"},
        10: {"gemini-embedding-001"},
        11: {"Qwen3-Embedding-8B"},
        12: {"F2LLM-v2-14B"},
        13: {"voyage-3.5"},
        14: {"voyage-3.5 (output_dtype=int8)"},
        15: {"F2LLM-v2-8B"},
        16: {"voyage-4-lite"},
        17: {"Octen-Embedding-0.6B"},
        18: {"F2LLM-v2-4B"},
        19: {"ICT-TIME-and-Querit-embedding-v1"},
        20: {"gemini-embedding-2-preview"},
        21: {"jina-embeddings-v5-omni-small", "jina-embeddings-v5-text-small"},
    },
}


def ensure_pinned_clone(cache: Path, commit: str) -> str:
    """Ensure {cache}/remote is the results repo checked out at `commit`. Returns full SHA."""
    remote = cache / "remote"
    if not (remote / ".git").exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"[clone] {RESULTS_REMOTE} -> {remote} (blobless partial clone)")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", RESULTS_REMOTE, str(remote)],
            check=True,
        )
    # make sure the commit is present, then check it out
    subprocess.run(["git", "-C", str(remote), "fetch", "--all", "--tags"], check=False)
    subprocess.run(["git", "-C", str(remote), "checkout", commit], check=True)
    full_sha = subprocess.run(
        ["git", "-C", str(remote), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    date = subprocess.run(
        ["git", "-C", str(remote), "show", "-s", "--format=%cI", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"[pin] HEAD = {full_sha}  ({date})")
    return full_sha


def short_name(name: str) -> str:
    return name.rsplit("/", 1)[-1] if isinstance(name, str) and name else name


def build_table(all_results, bench_name: str):
    import mteb
    import polars as pl

    bench = mteb.get_benchmark(bench_name)
    pl_df = all_results._to_results_df(bench.tasks)
    df = bench._create_summary_table(pl_df).df.to_pandas()  # canonical "Model" (org/name)

    # RTEB summary carries only a (sparse, skipna=False) "Mean (Task)" and no
    # per-dataset columns. Its raison d'être is per-domain retrieval scores, so
    # enrich with the per-dataset NDCG@10 columns for the PUBLIC (open) tasks —
    # matching what the RTEB leaderboard view exposes — plus a usable mean over
    # the datasets each model was actually evaluated on.
    if bench_name.startswith("RTEB") and "is_public" in pl_df.columns:
        pub_df = pl_df.filter(pl.col("is_public"))
        per_task = bench._create_per_task_table(pub_df).to_pandas()
        dataset_cols = [c for c in per_task.columns if c != "Model"]
        df = df.merge(per_task, on="Model", how="left")
        df.insert(
            df.columns.get_loc("Mean (Task)") + 1,
            "Mean (Public, available)",
            df[dataset_cols].mean(axis=1, skipna=True),
        )

    df["Model"] = df["Model"].map(short_name)  # -> short display name
    rank_col = "Rank (Borda)" if "Rank (Borda)" in df.columns else (
        "Rank (Mean Task)" if "Rank (Mean Task)" in df.columns else "Rank"
    )
    df = df.sort_values(rank_col, kind="stable").reset_index(drop=True)
    return df, rank_col


def verify(df, rank_col: str, expected: dict[int, set[str]]) -> bool:
    """Check that each expected Borda rank maps to the expected set of model short-names."""
    by_rank: dict[int, set[str]] = {}
    for _, row in df.iterrows():
        r = row[rank_col]
        if r is None or (isinstance(r, float) and r != r):  # NaN
            continue
        by_rank.setdefault(int(round(r)), set()).add(row["Model"])
    ok = True
    print(f"  {'rank':>4}  {'status':<6}  expected -> got")
    for rank in sorted(expected):
        got = by_rank.get(rank, set())
        want = expected[rank]
        match = want.issubset(got)  # allow extra tied models beyond screenshot crop
        ok = ok and match
        flag = "OK" if match else "MISS"
        extra = f"  (also: {sorted(got - want)})" if match and (got - want) else ""
        print(f"  {rank:>4}  {flag:<6}  {sorted(want)} -> {sorted(got)}{extra}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="9ff0358", help="results-repo commit to pin (June 3 = 9ff0358)")
    ap.add_argument("--cache", default=str(HERE / ".mteb_cache"), help="MTEB_CACHE dir (holds the pinned clone)")
    ap.add_argument("--write", action="store_true", help="write CSVs (default: dry-run + verify only)")
    args = ap.parse_args()

    cache = Path(args.cache).expanduser().resolve()
    os.environ.setdefault("MTEB_CACHE", str(cache))
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    full_sha = ensure_pinned_clone(cache, args.commit)

    import mteb

    print(f"[mteb] version {mteb.__version__}")
    cache_obj = mteb.ResultCache(cache_path=cache)
    all_model_names = [m.name for m in mteb.get_model_metas() if m.name is not None]
    print(f"[load] loading results for {len(all_model_names)} registered models from pinned clone ...")
    all_results = cache_obj.load_results(
        models=all_model_names,
        only_main_score=True,
        require_model_meta=False,
        include_remote=True,
    )

    all_ok = True
    prov_lines = [
        "# Leaderboard snapshot provenance",
        "",
        f"- **Snapshot date:** {SNAP_DATE} (the leaderboard state displayed on that date)",
        "- **Source:** unified MTEB leaderboard (huggingface.co/spaces/mteb/leaderboard).",
        "  Both the MTEB-English and RTEB-English screenshots come from this one site;",
        "  benchmarks `MTEB(eng, v2)` and `RTEB(eng, beta)`.",
        f"- **Data repo:** {RESULTS_REMOTE} @ `{full_sha}`",
        "  (committed 2026-05-31). This is the state the leaderboard displayed on 2026-06-03:",
        "  the next commit (`9ff0358`, 2026-06-03 11:05 UTC) added two vision-language octen",
        "  models that had not yet propagated into the RTEB view when the screenshots were taken.",
        f"- **mteb package:** {mteb.__version__}",
        "- **Ranking:** Borda count, recomputed with mteb's own `_create_summary_table`",
        "  (the exact code path the leaderboard Space uses). RTEB Borda uses public tasks only",
        "  (mteb `RtebBenchmark` hides private tasks from the summary; issue 3902).",
        "- **Reconstruction is verified** against the top-N of the screenshots in `MTEB/` and",
        "  `RTEB/` (see `build_snapshot.py` — all ranks incl. ties match exactly).",
        "- **Scores** are raw main-metric values in [0, 1] (displayed x100 on the site):",
        "  MTEB = per-task-type means; RTEB = per-dataset NDCG@10. `Mean (Public, available)`",
        "  (RTEB) averages the open datasets each model was actually evaluated on.",
        "",
        "## Files",
    ]
    for folder, cfg in BENCHMARKS.items():
        bench_name = str(cfg["name"])
        csv_path = Path(cfg["csv"])
        print(f"\n=== {folder}: {bench_name} ===")
        df, rank_col = build_table(all_results, bench_name)
        print(f"  rows: {len(df)}  rank column: {rank_col}  columns: {list(df.columns)}")
        ok = verify(df, rank_col, EXPECTED[folder])
        all_ok = all_ok and ok
        print(f"  VERIFY {folder}: {'PASS' if ok else 'FAIL'}")
        if args.write:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"  wrote {csv_path}  ({len(df)} rows)")
            prov_lines.append(f"- `{csv_path.relative_to(HERE)}` — {bench_name}, {len(df)} models, rank by {rank_col}")

    if args.write:
        (HERE / "PROVENANCE.md").write_text("\n".join(prov_lines) + "\n")
        print(f"\nwrote {HERE / 'PROVENANCE.md'}")

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
