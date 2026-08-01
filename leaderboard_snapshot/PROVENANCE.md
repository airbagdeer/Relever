# Leaderboard snapshot provenance

- **Snapshot date:** 2026-06-03 (the leaderboard state displayed on that date)
- **Source:** unified MTEB leaderboard (huggingface.co/spaces/mteb/leaderboard).
  Both the MTEB-English and RTEB-English screenshots come from this one site;
  benchmarks `MTEB(eng, v2)` and `RTEB(eng, beta)`.
- **Data repo:** https://github.com/embeddings-benchmark/results @ `0927beae00f9bfb0c4f928fada2ec39811216327`
  (committed 2026-05-31). This is the state the leaderboard displayed on 2026-06-03:
  the next commit (`9ff0358`, 2026-06-03 11:05 UTC) added two vision-language octen
  models that had not yet propagated into the RTEB view when the screenshots were taken.
- **mteb package:** 2.18.3
- **Ranking:** Borda count, recomputed with mteb's own `_create_summary_table`
  (the exact code path the leaderboard Space uses). RTEB Borda uses public tasks only
  (mteb `RtebBenchmark` hides private tasks from the summary; issue 3902).
- **Reconstruction is verified** against the top-N of the screenshots in `MTEB/` and
  `RTEB/` (see `build_snapshot.py` — all ranks incl. ties match exactly).
- **Scores** are raw main-metric values in [0, 1] (displayed x100 on the site):
  MTEB = per-task-type means; RTEB = per-dataset NDCG@10. `Mean (Public, available)`
  (RTEB) averages the open datasets each model was actually evaluated on.

## Files
- `MTEB/mteb_eng_v2_2026-06-03.csv` — MTEB(eng, v2), 308 models, rank by Rank (Borda)
- `RTEB/rteb_eng_beta_2026-06-03.csv` — RTEB(eng, beta), 249 models, rank by Rank (Borda)
