# frontier-v2 / evaluation

IR evaluation for the frontier-v2 benchmark: turn the 18 models' ranked
candidates (`data/candidates/`, materialised to rank 3000) + GPT-5.2
labels (`data/labels/`, depth-500 pool + the
`extended_pooling_delta/` delta labels for R>500 queries) into metrics, statistics
and a report.

## Run

```bash
# Unit tests (metric functions + delta-label loading / exact-P@R logic)
uv run pytest src/retrieval/lancedb/frontier-v2/evaluation/ -v

# Full pipeline: metrics -> significance -> 5 analyses -> plots -> reports (~45s, CPU only)
uv run python src/retrieval/lancedb/frontier-v2/evaluation/run_all.py
```

Individual modules are runnable standalone (each has a `__main__` that reads the
persisted CSVs), e.g. `uv run python .../evaluation/analysis_benchmark_ranking.py`.

## Layout

| File | Role |
|---|---|
| `metrics.py` | Pure IR metric functions: nDCG@k (binary), MRR, P@k, Recall@k, R-precision, MAP, Success@k. No project imports. |
| `test_metrics.py` / `test_data_loading_delta.py` / `conftest.py` | Unit tests (hand-computed values + `sklearn.ndcg_score` cross-check; delta-merge + exact-P@R logic on synthetic data). |
| `eval_config.py` | Paths + 18-model registry (family/size/retriever-type) + MTEB(eng v2) / RTEB(eng beta) Borda-rank dicts (from the 3 June 2026 leaderboard snapshot in `leaderboard_snapshot/ (data release)`). |
| `data_loading.py` | Join ranked candidates ↔ pooled labels → per-query relevance vectors + R. |
| `evaluate_frontier.py` | Core pass → per-query CSVs + leaderboard. |
| `significance.py` | Pairwise Wilcoxon + Holm-Bonferroni + rank-biserial. |
| `analysis_*.py` | Size scaling, fusion/oracle, score separation, difficulty, benchmark-vs-task. |
| `plots.py` / `reporting.py` / `make_reports.py` | Figures, markdown helpers, report generation. |
| `run_all.py` | Single entry point orchestrating everything. |

## Outputs

- `results/first_stage/<alias>/metrics.csv` — per-query raw metrics (machine-readable).
- `results/first_stage/all_models_per_query.csv` — combined long format.
- `summary/evaluation/` — `REPORT.md` (full), `CORE_SUMMARY.md` (short), `LEADERBOARD.md`,
  `STATISTICAL_TESTS.md`, the 5 per-analysis `.md` files, supporting CSVs, and `plots/`.

## Method notes

- **Relevance = GPT-5.2 label == 1**; `R` = positives in the merged label set: the depth-500
  union pool of all 18 models plus, for R>500 queries, the `extended_pooling_delta/` delta labels
  covering each model's top-R band (TREC-style pooling). Recall/R-precision are relative to the
  pool, not all of Wikipedia.
- **71 R=0 queries excluded** from every aggregate so all models share the same 1029-query set.
- **R-precision is reported only where it is exact** (every model's top-R fully labeled): the
  1007-query set = 968 with R ≤ 500 + 39 delta-converged. The 22 high-R queries whose delta
  labelling never converged get `r_precision = NaN` (`p_at_r_exact = False`) and drop out of
  every P@R mean and significance test — see `the paper's extended-pooling appendix`.
  The standard suite stays at the top-500 design depth; only `r_precision` reads deeper ranks.
- Failed labels (`-1`, zero in the current data) and any unlabelled candidate are treated as
  non-relevant.
