"""End-to-end driver for the frontier-v2 evaluation.

Runs: core metric pass → pairwise significance → the five analyses → plots →
data-driven markdown reports + CSVs. One command reproduces every machine-
generated artifact under ``results/first_stage/`` and ``summary/evaluation/``.

    uv run python src/retrieval/lancedb/frontier-v2/evaluation/run_all.py

The flagship narrative files (REPORT.md, CORE_SUMMARY.md) are authored on top of
these outputs and are not overwritten here.
"""

from __future__ import annotations

import analysis_benchmark_ranking as a_bench
import analysis_difficulty as a_diff
import analysis_fusion_oracle as a_fusion
import analysis_score_separation as a_sep
import analysis_size_scaling as a_size
import eval_config as cfg
import evaluate_frontier as ev
import make_reports as mr
import plots
import significance as sig


def main() -> None:
    cfg.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Core metric evaluation (all queries × 18 models) ...")
    res = ev.evaluate_all()
    ev.write_outputs(res)
    leaderboard = res.leaderboard
    per_query = res.per_query
    meta = {
        "n_total": res.n_queries_total,
        "n_eval": res.n_queries_eval,
        "n_excluded": len(res.excluded_event_ids),
        "n_pr_exact": res.n_queries_pr_exact,
        "n_pr_excluded": len(res.pr_excluded_event_ids),
        "n_failed": res.total_failed_labels,
        "n_missing": res.total_missing_labels,
    }
    print(f"      evaluated={meta['n_eval']} excluded(R=0)={meta['n_excluded']} "
          f"exact-P@R={meta['n_pr_exact']} (r_precision excludes {meta['n_pr_excluded']})")

    print("[2/6] Pairwise significance ...")
    sig_frames = sig.run(per_query)

    print("[3/6] Analyses: benchmark / size / difficulty (fast) ...")
    bench_res = a_bench.compute(leaderboard)
    size_res = a_size.compute(leaderboard)
    diff_res = a_diff.compute(per_query)
    # persist their CSVs
    bench_res["tables"]["ndcg@10"].to_csv(cfg.SUMMARY_DIR / "benchmark_ranking_ndcg10.csv", index=False)
    size_res["table"].to_csv(cfg.SUMMARY_DIR / "size_scaling.csv", index=False)
    diff_res["summary"].to_csv(cfg.SUMMARY_DIR / "difficulty_buckets.csv", index=False)
    diff_res["matrix_ndcg10"].to_csv(cfg.SUMMARY_DIR / "difficulty_model_bucket_ndcg10.csv")

    print("[4/6] Analyses: fusion/oracle + score-separation (raw passes) ...")
    fusion_res = a_fusion.compute(leaderboard, pr_exclude=set(res.pr_excluded_event_ids))
    fusion_headline = a_fusion.headline_frame(fusion_res)
    fusion_oracle = a_fusion.oracle_frame(fusion_res)
    fusion_headline.to_csv(cfg.SUMMARY_DIR / "fusion_vs_best.csv", index=False)
    fusion_oracle.to_csv(cfg.SUMMARY_DIR / "union_oracle_recall.csv", index=False)
    sep_res = a_sep.compute(leaderboard)
    sep_res["table"].to_csv(cfg.SUMMARY_DIR / "score_separation.csv", index=False)

    print("[5/6] Plots ...")
    plots.plot_leaderboard_bar(leaderboard, "ndcg@10", "leaderboard_ndcg10.png", f"nDCG@10 ({meta['n_eval']} queries, R>0)")
    plots.plot_leaderboard_bar(
        leaderboard, "r_precision", "leaderboard_r_precision.png",
        f"R-precision ({meta['n_pr_exact']} exact-P@R queries; "
        f"{meta['n_pr_excluded']} high-R non-converged excluded)")
    plots.plot_metric_heatmap(leaderboard, mr.HEADLINE_COLS, "metric_heatmap.png")
    plots.plot_size_scaling(size_res["table"], "size_scaling.png")
    plots.plot_recall_curves(fusion_oracle, fusion_res["fusion_row"], "recall_curves.png")
    plots.plot_score_separation(sep_res["table"], sep_res["spearman_auc_vs_ndcg10"]["rho"], "score_separation.png")
    plots.plot_difficulty(diff_res["summary"], diff_res["matrix_ndcg10"], diff_res["best_model"], "difficulty_buckets.png")
    plots.plot_benchmark_scatter(bench_res["tables"]["ndcg@10"], bench_res["correlations"]["ndcg@10"], "ndcg@10", "benchmark_vs_task_ndcg10.png")

    print("[6/6] Markdown reports ...")
    mr.write_leaderboard_md(leaderboard, meta)
    mr.write_statistical_tests_md(sig_frames)
    mr.write_size_scaling_md(size_res, meta["n_eval"])
    mr.write_difficulty_md(diff_res, meta["n_eval"])
    mr.write_score_separation_md(sep_res)
    mr.write_fusion_oracle_md(fusion_headline, fusion_oracle, n_rp=fusion_res["n_rp"])
    mr.write_benchmark_ranking_md(bench_res)

    print(f"\nDone. Outputs in:\n  {cfg.RESULTS_DIR}\n  {cfg.SUMMARY_DIR}")


if __name__ == "__main__":
    main()
