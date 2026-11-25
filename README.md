# RELEVER — code

Benchmark-construction, labeling, and evaluation code for **RELEVER**, a 1,100-query
benchmark for **relation evidence retrieval**: given a query that decomposes into an
abstract entity type *T*, a constant (anchor) entity *e*, and an exact relation *r*,
retrieve Wikipedia paragraphs that *assert* the relation — not paragraphs with a
similar-but-different relation or similar-but-wrong entities.

This repository contains the **entire release**: the benchmark data (queries, corpus,
relevance labels), the per-query result tables behind the paper's numbers, and the
construction + evaluation code.

The benchmark data (queries, corpus, relevance labels) is also available on Hugging Face
at [deer2222/RELEVER](https://huggingface.co/datasets/deer2222/RELEVER), loadable with
`datasets.load_dataset("deer2222/RELEVER", "<queries|corpus|qrels>")`.

## Setup

Requires Python **3.12.7** (exact pin) and [uv](https://github.com/astral-sh/uv).

```bash
git clone <this repository> && cd relever
uv sync
```

That's all — code and data live together:

```
src/                 construction, labeling, and evaluation code
data/queries/        the 1,100 queries + generation prompt
data/corpus/         the corpus, as 8 standalone gzip shards (see below)
data/labels/         relevance labels (qrels) + extended-pooling delta
data/fp_taxonomy/    false-positive taxonomy data
data/judge_validation/  the 200-pair human gold behind judge validation
results/             per-query metrics (first_stage/, rerank/per_system/)
summary/             leaderboard + significance + analysis tables
leaderboard_snapshot/  archived MTEB/RTEB captures (model selection)
```

All paths resolve relative to the repository root via `src/retrieval/config.py`; scripts
run from any working directory.

**The corpus** ships as 8 shard files (`data/corpus/corpus.part-0*.jsonl.gz`, each under
GitHub's 100 MiB per-file limit). Every shard is an independently valid gzip and they
concatenate to the exact single-file corpus:

```bash
zcat data/corpus/corpus.part-*.jsonl.gz | wc -l    # 1,080,071 — or stream it directly
cat  data/corpus/corpus.part-*.jsonl.gz > corpus.jsonl.gz   # optional single file
```

## Evaluating your own retrieval system

The labels (`data/labels/`, one qrels file per query, `{id, event_id, prediction}` with
`prediction ∈ {1, 0, -1}`) score any ranked list over the released corpus:

1. Retrieve over the corpus (1,080,071 paragraphs, `{_id, title, text, url}`;
   shards under `data/corpus/`) for each of the 1,100 queries in
   `data/queries/queries.json` (query text = the `sentence` field).
2. Write one run file per query at `data/candidates/<qid>/<alias>_<qid>.json`
   (`qid` = `event_id` zero-padded to 4 digits; `alias` = your system name). Format:

   ```json
   {"event_id": 0, "query": "…", "model": "your-alias", "k": 3000, "n_candidates": 3000,
    "candidates": [{"id": "<corpus _id>", "rank": 1, "score": 12.3}, …]}
   ```

   Only `candidates[].id` and `candidates[].rank` are consumed (`score` optional).
   Materialize to rank 3,000 (or at least to the query's R) so exact Precision@R is
   computable for the deep queries; the standard metric suite reads the top 500.
3. Register the alias in `MODELS` in
   `src/retrieval/lancedb/frontier-v2/evaluation/eval_config.py` and run
   `uv run python src/retrieval/lancedb/frontier-v2/evaluation/run_all.py`.
   The evaluator scores every alias registered in `MODELS`, and the original 18
   systems' run files are not distributed — so first trim `MODELS` to the aliases
   whose run files exist under `data/candidates/` (e.g., just your own entry).

Unjudged paragraphs count as non-relevant (standard pooled-evaluation convention); the
labeled pool is the union of the 18 evaluated systems' top-500 plus a extended-pooling
delta for the 61 queries with R > 500 (`data/labels/extended_pooling_delta/`).

## What reproduces from the released data

| Reported numbers | Command |
|---|---|
| Pairwise significance (18×18, Holm-corrected), difficulty buckets, benchmark-vs-task ranking, size scaling | `evaluation/significance.py`, `analysis_difficulty.py`, `analysis_benchmark_ranking.py`, `analysis_size_scaling.py` (each standalone, reading `results/first_stage/all_models_per_query.csv` + `summary/evaluation/leaderboard.csv`) |
| Reranking significance tables | `evaluation/significance_rerank.py` (reads `results/rerank/per_system/*.csv`) |
| FP-taxonomy tables | `classifying_data/fp_subset_report.py`, `fp_mcnemar.py`, `src/process_results/fp_paper_stats.py` |
| Dataset / label statistics | `src/process_results/paper_stats.py --skip-corpus` |

Aggregate leaderboard values are means of the released per-query tables. The full
18-system metric recomputation (`run_all.py` stage 1) additionally needs ranked lists
under `data/candidates/` — the original 18 systems' run files are not distributed
(≈5 GB), but any subset can be regenerated with the indexing/retrieval code below, and
new systems are evaluated exactly the same way.

## Pipeline provenance (how the benchmark was built)

1. **Query generation** — `src/create_dataset/generate_more_relations.py` with the
   byte-exact prompt `data/queries/query_generation_prompt.txt` (30-domain steering,
   embedding dedup). `prepare_relations_v2.py` documents the historical zero-positive
   seed filter (its inputs are internal working data and are not distributed).
2. **Candidate corpus** — BM25 + SPLADE over the `EmergentMethods/en_qdrant_wikipedia`
   snapshot via Qdrant (see "Rebuilding the candidate pool" below), then
   `src/create_dataset/build_release_corpus.py` deduplicates the per-query pools into
   the corpus shards under `data/corpus/` (`--shard-lines 135009`).
3. **Indexing + retrieval (18 systems)** — `src/retrieval/lancedb/frontier-v2/`
   (`index_*.py`, `retrieve_*.py`; shared LanceDB machinery in
   `src/retrieval/lancedb/minimax/`), pooled by `aggregate_candidates.py`.
4. **Relevance labeling** — `src/create_dataset/classifying_data/batch_api/`
   `label_frontier_batch.py` (GPT-5.2, `reasoning_effort="medium"`, the byte-exact
   `short2b` prompt in `relation_classification_prompt.py`, OpenAI Batch API, one pair
   per request) and `label_frontier_delta_batch.py` (R > 500 extended pooling).
5. **Judge selection & validation** — `classifying_data/estimate_frontier_cost.py`
   (11 judge configs × 200-pair human gold), `label_opus.py`, `make_label_tool.py` +
   `merge_human_labels.py` (human annotation tooling), plus the agreement /
   prompt-variant / test-retest analysis scripts. The study's 200-pair human
   gold ships in `data/judge_validation/short/` — `human__gold.jsonl`
   (`{id, prediction}` in annotation order; first 100 = tuning split, last
   100 = held-out) plus `_sample.json` (each pair's query and paragraph
   text); the judge model outputs and test-retest runs are not distributed,
   and the study's results are reported in the paper's judge appendix.
6. **Reranking study** — `src/retrieval/rerank/` (10 rerankers; pointwise pool-scoring
   via `score_pool.py`, listwise via `rerank_listwise.py`), evaluated by
   `evaluation/evaluate_rerank.py` (requires the ~1 GB reranker score cache, which is
   regenerable but not distributed; the per-query rerank metrics it produced ship in
   `results/rerank/`).
7. **False-positive taxonomy** — `evaluation/extract_fp_population.py` →
   `batch_api/label_fp_taxonomy_batch.py` (byte-exact prompt in
   `fp_taxonomy_prompt.py`) → rollups in step "FP-taxonomy tables" above.

### Rebuilding the candidate pool (optional, stage 2)

Requires Docker and ~350 GB of disk for the restored Qdrant collection:

```bash
# 1. Download the EmergentMethods/en_qdrant_wikipedia snapshot (Hugging Face) and
#    restore it into a local storage dir:
QDRANT_SNAPSHOT_PATH=/path/to/wikipedia-en-qdrant-2025-09-03.snapshot \
QDRANT_STORAGE_DIR=/path/to/qdrant_storage \
uv run python src/qdrant_dataset/manual_restore.py

# 2. Serve it (image pinned; tolerates the snapshot's legacy shard_key_mapping.json):
docker run -d --name qdrant-wikipedia -p 6333:6333 -p 6334:6334 \
    -v /path/to/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.16.3

# 3. Retrieve SPLADE+BM25 candidates for all queries (resumable):
QDRANT_URL=http://localhost:6333 uv run python -m src.qdrant_dataset.run_batch_retrieval \
    --input-file data/queries/queries.json --output-dir data/splade_wikipedia_v2 \
    --per-method-limit 1000
```

### GPU environments (stages 3 & 6)

Most models run in the main `uv sync` venv. Exceptions (dependency conflicts, each
documented where it is used):

| venv | used by | why |
|---|---|---|
| `.venv-colbert` | `index_colbert.py`, ColBERT reranking (`pylate`) | pylate pins `transformers<=4.56.2` (see the note in `pyproject.toml`) |
| `.venv-jina-v4` | `index_jina_v4_late.py` | jina-v4 custom modeling code |
| a vLLM venv | `index_vllm.py` / `retrieve_vllm.py` (Qwen3-Embedding) | vLLM's own torch pin |
| `.venv-rankllm` (py3.11) | `rank_zephyr` listwise reranker | `rank-llm` must be installed from source (see `src/retrieval/rerank/README.md`) |

Hardware used for the paper: a single node with four NVIDIA A100 GPUs; large
(7–14B) embedding models need reduced batch sizes (2048–4096 vs. the 8192 default).

## License

Code is released under the MIT License (see `LICENSE`). The data under `data/`,
`results/` and `summary/` is released under CC BY-SA 4.0 — the corpus paragraphs
derive from English Wikipedia (CC BY-SA) and carry per-paragraph attribution
(`title`, `url`); see `data/README.md` (the dataset card) for details.
