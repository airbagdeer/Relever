# Reranking the frontier-v2 candidates

Answers the standing objection to the headline result: *"you only tested
first-stage retrieval — a cross-encoder reranker would fix this."*

Produces an **18 retrievers × 9 rerankers** grid of ΔPrecision@R against the 1100-query v2
benchmark, reusing the existing GPT-5.2 labels. Nothing needs re-labelling.

## The idea that makes it cheap

`data/candidates/<qid>/all_candidates_<qid>.json` is *exactly* the union of every
retriever's top-500, carries the paragraph text, and the labels cover *exactly* that pool.
Verified on disk (`tests/test_equivalence.py`): every retriever's top-500 is inside it, and
fully labeled — 0 violations.

A **pointwise** reranker's score `s(q, d)` does not depend on the candidate set. So:

> Score the pool **once** per (query, reranker). Every retriever's reranked top-500 is
> then a sort of its own ids by the cached score.

Exact, not an approximation — **2.18M** (query, doc) pairs instead of 9.9M, and rerank@10 /
@50 / @100 / @500 all fall out of one pass. `test_pool_scoring_equals_direct_reranking`
asserts this rather than trusting it.

**Listwise** rerankers (`rank_zephyr`, `jina-reranker-v3`) score set-dependently, so the
cache cannot serve them. They run once per (reranker, retriever) at their published
depth-100 protocol, and the pointwise models get a free depth-100 column for comparison.

## Layout

```
reranker_registry.py   the 9 specs: backend, paradigm, max_length, dtype, padding_side, venv
score_store.py         on-disk layout (parquet + permutation JSON). Imported by the evaluator too.
score_pool.py          POINTWISE driver: pool -> <scores>/<rr>/<variant>/<qid>.parquet
rerank_listwise.py     LISTWISE driver: per (rr, retriever) top-100 -> permutation JSON
backends/              one module per scoring contract
tests/                 unit + integration + `gpu`-marked card-fidelity tests
```

Evaluation lives with its siblings and reuses `metrics.py` / `significance.py` unchanged:
`src/retrieval/lancedb/frontier-v2/evaluation/evaluate_rerank.py`.

## Running

```bash
# pointwise: score the pool once per reranker (see --help for GPU sharding flags)
uv run python src/retrieval/rerank/score_pool.py --reranker <name>

# listwise (published depth-100 protocol)
uv run python src/retrieval/rerank/rerank_listwise.py --reranker <name>

# evaluate whatever has a complete cache
uv run python src/retrieval/lancedb/frontier-v2/evaluation/evaluate_rerank.py
```

Pointwise models run one at a time, each query-sharded across all GPUs (`event_id % NGPU`).
Shards write disjoint per-query files, so they never coordinate and a killed job resumes by
skipping what exists.

## Things that fail silently if you get them wrong

* **`padding_side`.** `mxbai`, `Qwen3-Reranker-*` and `bge-gemma` read `logits[:, -1, :]`.
  Right padding reads a pad token and returns plausible noise for every pair, with no error.
  Guarded by `assert_padding_side` and by `test_batch_invariance`, which checks that
  `score(q, [d1..dn]) == [score(q, [di])]` — the definition of "pointwise", and the property
  the score cache depends on.
* **Score scales are not comparable.** `P(yes) ∈ [0,1]` (Qwen3), a raw `Yes` logit
  (bge-gemma), a log-odds difference (mxbai), a raw logit (bge-m3, ms-marco), a cosine
  (colbert, jina-v3). Rank-only metrics like P@R are invariant to this. **No score fusion,
  thresholding, or cross-model averaging is valid on these numbers.**
* **Qwen3-Reranker under vLLM.** [vllm#21681](https://github.com/vllm-project/vllm/issues/21681):
  scores compress until relevant and irrelevant stop separating. We score with
  `transformers`.
* **Truncation.** `ms-marco-MiniLM` and `colbertv2.0` are hard-capped at 512 tokens and cut
  ~15% of paragraphs; `rank_zephyr`'s 4096-token context leaves it seeing ~40% of each
  paragraph. Declared per-model as `expected_truncation`, logged every run, and warned on
  when exceeded. No silent caps.

## Isolated venvs

Two backends cannot share `.venv`; both name the venv they need in their `ImportError`.

```bash
uv venv .venv-colbert --python 3.12.7      # pylate pins transformers<=4.56.2
uv pip install --python .venv-colbert/bin/python 'pylate>=1.4.0' pyarrow tqdm numpy torch
```

`rank_llm` **must be installed from source.** The PyPI wheel (0.25.7) eagerly imports its
LiT5 module, which needs `transformers.modeling_utils.find_pruneable_heads_and_indices` —
removed in transformers 5.x, which `vllm>=0.24` requires. `main` guards that import and
makes the `listwise` subpackage lazy, so `ZephyrReranker` loads. (Its card says to install
from source for exactly this reason.) It is ~9 GB, so keep it off a quota-tight `/home`:

```bash
uv venv /tmp/venv-rankllm --python 3.11
UV_CACHE_DIR=/tmp/uv_cache uv pip install --python /tmp/venv-rankllm/bin/python \
    'rank-llm @ git+https://github.com/castorini/rank_llm.git' pyarrow
ln -sfn /tmp/venv-rankllm .venv-rankllm
```

## Running on a quota-tight cluster

The reranker weights total ~40 GB (Qwen3-8B alone is 16 GB; bge-gemma ships fp32 at 10 GB).
Point the HF cache at scratch or the run will fill `/home` around hour 30:

```bash
export HF_HOME=/tmp/hf_cache
```

## Tests

```bash
uv run pytest src/retrieval/rerank/tests -m "not gpu"   # 83 tests, ~2 s
uv run pytest src/retrieval/rerank/tests -m gpu         # downloads real weights
```

The `gpu` set checks each backend against the example output printed on its own model card
(e.g. `bge-reranker-v2-m3` scoring `('query', 'passage')` → `-5.65234375`, cross-checked
against the card's `normalize=True` value via `sigmoid`) and asserts batch invariance,
which is what catches a wrong `padding_side`.

Verified on real weights so far: `ms-marco-MiniLM-L12-v2` (card values + invariance),
`bge-reranker-v2-m3` (card value + invariance), `mxbai-rerank-large-v2` (invariance — the
left-padding contract on a 1.5B decoder). The rest run on the cluster.
