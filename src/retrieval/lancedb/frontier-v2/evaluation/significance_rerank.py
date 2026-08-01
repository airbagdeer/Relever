"""Statistical significance for the reranking study (paper §7 / Appendix).

Same protocol as the first-stage tests (``significance.py``): paired two-sided
**Wilcoxon signed-rank** over per-query scores, **Holm-Bonferroni** correction
within each test family, matched-pairs **rank-biserial** effect size. Complete
cases per pair: 1,029 queries for depth-capped metrics, 1,007 for r_precision
(the same exact-P@R query set as the published first-stage leaderboard —
reranking cannot change exactness, see evaluate_rerank.py).

Two test families, corrected independently per metric:

  * ``vs_first_stage``  — every (reranker system x retriever) cell against that
    retriever's first-stage list: is the improvement significant?
    22 systems x 18 retrievers = 396 tests per metric.
  * ``instruct``        — the task_instruct variant against its own default for
    the instruction-aware models (Qwen3-Reranker-8B, bge-reranker-v2-gemma) at
    both depths: does naming the relation help? 4 pairs x 18 retrievers = 72
    tests per metric.

Outputs (the source of the paper's rerank significance columns):
    summary/rerank/significance_rerank_vs_first_stage__<metric>.csv
    summary/rerank/significance_rerank_instruct__<metric>.csv

Run:
    uv run python src/retrieval/lancedb/frontier-v2/evaluation/significance_rerank.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import eval_config as cfg
from significance import fmt_sig, holm_bonferroni, rank_biserial

PER_QUERY_CSV = cfg.RESULTS_RERANK_DIR / "all_systems_per_query.csv"
PER_SYSTEM_DIR = cfg.RESULTS_RERANK_DIR / "per_system"
BASELINE_SYSTEM = "first_stage"
METRICS = ("r_precision", f"ndcg@{cfg.HEADLINE_NDCG_K}")

# The instruction ablation: task_instruct vs the model's own card default.
INSTRUCT_MODELS = ("Qwen3-Reranker-8B", "bge-reranker-v2-gemma")


def load_per_query() -> pd.DataFrame:
    usecols = ["system", "retriever", "event_id", "R", *METRICS]
    if PER_QUERY_CSV.exists():
        df = pd.read_csv(PER_QUERY_CSV, usecols=usecols)
    else:
        # Release layout: one standalone CSV per system under per_system/ (the
        # combined CSV exceeds GitHub's per-file limit). Same rows either way.
        parts = sorted(PER_SYSTEM_DIR.glob("*.csv"))
        if not parts:
            raise FileNotFoundError(
                f"neither {PER_QUERY_CSV} nor {PER_SYSTEM_DIR}/*.csv exists"
            )
        df = pd.concat(
            (pd.read_csv(p, usecols=usecols) for p in parts), ignore_index=True
        )
    return df[df["R"] > 0]


def _paired(mat: pd.DataFrame, sys_a: str, sys_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Complete-case paired arrays for two systems on one retriever's queries."""
    sub = mat[[sys_a, sys_b]].dropna()
    return sub[sys_a].to_numpy(dtype=float), sub[sys_b].to_numpy(dtype=float)


def _test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if np.allclose(a, b):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:  # all differences zero
        return 0.0, 1.0
    return float(stat), float(p)


def family_tests(per_query: pd.DataFrame, metric: str,
                 pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Run one Holm family. ``pairs`` = (retriever, system_a, system_b) triples;
    the test is system_a vs system_b on that retriever's per-query scores."""
    recs = []
    p_raw: list[float] = []
    for retriever, sub in per_query.groupby("retriever"):
        mat = sub.pivot(index="event_id", columns="system", values=metric)
        for r, sys_a, sys_b in pairs:
            if r != retriever:
                continue
            a, b = _paired(mat, sys_a, sys_b)
            stat, p = _test(a, b)
            recs.append({
                "retriever": retriever, "system": sys_a, "baseline": sys_b,
                "n": int(len(a)),
                "mean_system": float(a.mean()), "mean_baseline": float(b.mean()),
                "delta": float(a.mean() - b.mean()),
                "stat": stat, "p_raw": p, "rank_biserial": rank_biserial(a, b),
            })
            p_raw.append(p)
    df = pd.DataFrame(recs)
    df["p_holm"] = holm_bonferroni(p_raw)
    df["sig"] = [fmt_sig(p) for p in df["p_holm"]]
    return df


def wins_rollup(fam: pd.DataFrame) -> pd.DataFrame:
    """Per system: of the 18 retrievers, how many are significantly improved / hurt."""
    fam = fam.assign(
        sig_up=(fam["p_holm"] < 0.05) & (fam["delta"] > 0),
        sig_down=(fam["p_holm"] < 0.05) & (fam["delta"] < 0),
    )
    out = fam.groupby("system").agg(
        n_retrievers=("retriever", "size"),
        n_sig_improved=("sig_up", "sum"),
        n_sig_hurt=("sig_down", "sum"),
        mean_delta=("delta", "mean"),
    )
    return out.sort_values("mean_delta", ascending=False)


def main() -> None:
    per_query = load_per_query()
    systems = sorted(s for s in per_query["system"].unique() if s != BASELINE_SYSTEM)
    retrievers = list(cfg.MODEL_ALIASES)
    print(f"{len(systems)} reranker systems x {len(retrievers)} retrievers")

    vs_pairs = [(r, s, BASELINE_SYSTEM) for s in systems for r in retrievers]
    instr_pairs = [
        (r, f"{m}__task_instruct@{d}", f"{m}__default@{d}")
        for m in INSTRUCT_MODELS for d in cfg.RERANK_DEPTHS for r in retrievers
    ]

    cfg.SUMMARY_RERANK_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        safe = metric.replace("@", "").replace("_", "")
        fam_vs = family_tests(per_query, metric, vs_pairs)
        fam_vs.to_csv(cfg.SUMMARY_RERANK_DIR
                      / f"significance_rerank_vs_first_stage__{safe}.csv", index=False)
        fam_in = family_tests(per_query, metric, instr_pairs)
        fam_in.to_csv(cfg.SUMMARY_RERANK_DIR
                      / f"significance_rerank_instruct__{safe}.csv", index=False)

        print(f"\n=== {metric}: significantly improved retrievers per system "
              f"(Holm over {len(fam_vs)} tests) ===")
        print(wins_rollup(fam_vs).to_string(float_format=lambda v: f"{v:+.4f}"))

        print(f"\n=== {metric}: task_instruct vs default "
              f"(Holm over {len(fam_in)} tests) ===")
        print(wins_rollup(fam_in).to_string(float_format=lambda v: f"{v:+.4f}"))

        not_sig = fam_vs[fam_vs["p_holm"] >= 0.05]
        if len(not_sig):
            print(f"\n  non-significant vs first stage ({metric}):")
            print(not_sig[["system", "retriever", "delta", "p_holm"]]
                  .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
