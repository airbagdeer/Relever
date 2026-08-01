---
license: cc-by-sa-4.0
language:
  - en
pretty_name: "RELEVER — relation evidence retrieval benchmark"
size_categories:
  - 1M<n<10M
task_categories:
  - text-retrieval
---

# RELEVER — data release

The data half of **RELEVER**, a 1,100-query benchmark for **relation evidence
retrieval** over English Wikipedia. Each query decomposes into an abstract entity
type *T*, a constant (anchor) entity *e*, and an exact relation *r*; a paragraph is
relevant only if it *asserts* the queried relation between the queried entities.
This directory is the data component of the RELEVER repository (code and data ship
together); see the repository root `README.md` for setup, evaluation instructions,
and how each paper table regenerates.

## Contents

| Path | What it is |
|---|---|
| `data/queries/queries.json` | **The benchmark**: 1,100 queries (JSON array). Query text = the `sentence` field; each object also carries `event_id` (0–1099), `abstract_entity_type`, `constant_entity`, `relation`, and generation metadata. |
| `data/queries/query_generation_prompt.txt` | Byte-exact query-generation prompt (paper appendix). |
| `data/corpus/corpus.part-{00..07}.jsonl.gz` | **The corpus**: 1,080,071 unique paragraphs, one JSON per line: `{_id, title, text, url}` (`title` + `url` = CC BY-SA attribution to the source Wikipedia article). Deduplicated union of all per-query candidate pools, shipped as 8 standalone gzip shards (read in order; `cat`/`zcat` of `corpus.part-*.jsonl.gz` reproduces the single file exactly). |
| `data/labels/<qid>/predictions_<qid>.jsonl` | **The relevance labels (qrels)**: one file per query, `{id, event_id, prediction}` with `prediction ∈ {1, 0, -1}` (relevant / not relevant / judge parse failure — zero in this data). 2,180,641 judged pairs; unjudged pairs count as non-relevant (pooled evaluation). |
| `data/labels/extended_pooling_delta/` | Extended-pooling delta for the 61 queries with R > 500 (58,878 additional pairs, same judge setup), making exact Precision@R computable down to rank R. |
| `data/fp_taxonomy/` | False-positive taxonomy: per-query presence bits for 1,029 queries (`instance_present` / `relation_present` / `constant_present`), the 48,616-pair FP population, the 17,310-pair two-system subset, and `validation/` (100-pair human gold + model comparisons). |
| `data/judge_validation/short/` | **Judge-validation human gold**: `human__gold.jsonl` — the 200-pair author-labeled gold behind the judge-selection study (`{id, prediction}`, `prediction ∈ {1, 0}`, in annotation order: pairs 1–100 = the tuning split, 40 positive / 60 negative; pairs 101–200 = the held-out split, 36 / 64). `_sample.json` maps each pair `id` to its query (`event_id`, `source_query`, `query_info`) and paragraph text (`content`). Judge model outputs and test-retest runs are not distributed. |
| `results/first_stage/all_models_per_query.csv` | Per-query metrics for all 18 evaluated systems (18 × 1,100 rows: nDCG@k, precision/recall@k, exact P@R, MRR, MAP, …). |
| `results/rerank/per_system/*.csv` | Per-query metrics for every (reranker × variant × depth) configuration — 23 standalone CSVs (one per system), jointly 23 × 18 retrievers × 1,100 queries. |
| `summary/evaluation/` | Machine-generated rollups: leaderboard, the full pairwise significance tables (153 = C(18,2) pairs, Holm-corrected, for nDCG@10 and P@R), difficulty buckets, union-oracle recall, fusion-vs-best, score separation, benchmark-ranking correlation. |
| `summary/rerank/` | Rerank leaderboard + 4 significance tables. |
| `leaderboard_snapshot/` | Archived MTEB / RTEB leaderboard captures (3 June 2026) used for model selection, with `build_snapshot.py` + provenance notes. |

Not distributed (regenerable or internal): the 18 systems' raw ranked lists (≈5 GB;
any system can be re-run with the released code, and new systems are evaluated
against the released labels), the per-query first-stage BM25+SPLADE dumps, the
reranker score cache, the judge-selection study's model outputs and test-retest
runs (the 200-pair human gold and its sample ship in `data/judge_validation/`;
the study's results are reported in the paper's judge appendix), raw judge
request/response bodies, and internal lineage files (query seed and id-map).

## Provenance and licensing

- **Corpus**: paragraphs from English Wikipedia via the public
  `EmergentMethods/en_qdrant_wikipedia` snapshot (2025-09-03). Wikipedia text is
  CC BY-SA 4.0; the corpus is released under **CC BY-SA 4.0** with per-paragraph
  attribution (`title`, `url`). The rest of the release (queries, labels, result
  tables) is released under the same CC BY-SA 4.0 terms for simplicity.
- **Queries**: LLM-generated (GPT-5.4) with 30-domain steering and embedding-based
  deduplication; the byte-exact prompt ships in `data/queries/`.
- **Labels**: produced by an LLM judge — GPT-5.2, `reasoning_effort="medium"`, the
  `short2b` prompt (byte-exact in the code release), one (query, paragraph) pair per
  request via the OpenAI Batch API. Judge validity is documented in the paper's
  judge-selection appendix: 96.0% agreement / Cohen's κ 0.914 with a 200-pair
  human gold (97.0% on the tuning half / 95.0% held-out), Fleiss κ 0.710 across
  four frontier models, and 94.3% mean pairwise test-retest agreement across
  three independent re-runs.
- **Intended use**: research on retrieval evaluation. The labels are model
  judgments, not human ground truth, and are pool-relative (union of 18 systems'
  top-500 + extended-pooling delta); treat unjudged pairs as non-relevant.
