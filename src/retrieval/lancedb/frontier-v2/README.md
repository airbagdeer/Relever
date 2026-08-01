# frontier-v2 — index the `splade_wikipedia_v2` paragraph corpus

Embeds and indexes the **~1.04M unique paragraphs** in `data/splade_wikipedia_v2/`
(union + dedup of every file's `splade_results` + `bm25_results`, keyed on the
paragraph UUID `id == payload.point_id`) into LanceDB, paragraph granularity
only. Classifications are **not** required for indexing — they're back-joined on
the paragraph UUID at eval time.

## Model set (18)

Deduplicated union of `TOP_ENGLISH_MTEB_MODEL_IDS` + `TOP_ENGLISH_RTEB_MODEL_IDS`
(16 dense) + `colbert-v2` + `jina-embeddings-v4-late` (2 multi-vector). See
`model_registry.MODEL_SPECS`.

- **vLLM** (dedicated vLLM venv): `Qwen3-Embedding-8B`, `Qwen3-Embedding-4B`.
- **sentence-transformers** (`.venv`): the other 14 dense models.
- **pylate** (`.venv-colbert`): `colbert-v2`.
- **transformers** (`.venv-jina-v4`): `jina-embeddings-v4-late`.

## Disk routing

Per model via `--lancedb-root auto` → `MODEL_SPECS[alias].disk_target`:

- **scratch** (`LANCEDB_SCRATCH_PATH`=`/tmp/lancedb_scratch`, ephemeral, large):
  all 18 models. The two multi-vector tables — `colbert-v2` (indexed at
  `document_length=512`, full paragraph) and `jina-embeddings-v4-late` — are each
  ~253 GB at the v2 corpus's ~473 tok/doc, too large for the ~155 GB durable home
  partition, so they live here alongside the 16 dense tables. Corpus cache also
  lives here.
- **home** (`LANCEDB_PATH`, durable, small): holds only the shared checkpoint
  state (`_minimax_checkpoint/state.json`) — no model tables.

The checkpoint is shared across all backends and roots (table names are unique per
alias). The durable retrieval candidates (`data/candidates/`) are written
separately, so a scratch wipe only costs a re-index, not the results.

## Design

All new code lives here; nothing under `minimax/` or `config.py` is modified.
The stable minimax library functions are **reused by import** via `_bootstrap`
(checkpoint store, `ParagraphCorpus` + cache helpers, the dense
encode/bucketing/OOM machinery, and the two multi-vector index functions). The
three same-named `index_*` modules are loaded by explicit path
(`_bootstrap.load_minimax`) to avoid name collisions.

## Files

| file | role |
|------|------|
| `_bootstrap.py` | sys.path wiring + `load_minimax()` |
| `paths.py` | `SPLADE_WIKIPEDIA_V2_PATH`, `V2_CORPUS_CACHE_DIR` |
| `corpus_loader_v2.py` | v2 nested-JSON → deduped `ParagraphCorpus` |
| `model_registry.py` | alias union resolution + `ModelSpec`/`MODEL_SPECS` + routing |
| `_dense_core.py` | spec-driven dense encode loop (reuses minimax helpers) |
| `index_dense.py` | CLI: sentence-transformers dense models |
| `index_vllm.py` | CLI: vLLM Qwen3-Embedding (runs in the vLLM venv) |
| `index_colbert.py` | CLI: ColBERTv2 wrapper (runs in `.venv-colbert`) |
| `index_jina_v4_late.py` | CLI: jina-v4 late-interaction wrapper (`.venv-jina-v4`) |
| `status_v2.py` | status / row-count / disk-location report |

## Run

Each indexer/retriever is a standalone CLI (see each module's ``--help``); pick
GPUs via ``CUDA_VISIBLE_DEVICES`` and run one backend per process, e.g.:

```bash
uv run python src/retrieval/lancedb/frontier-v2/index_dense.py --lancedb-root auto
uv run python src/retrieval/lancedb/frontier-v2/retrieve_dense.py

# Status:
uv run python src/retrieval/lancedb/frontier-v2/status_v2.py
```

(The paper's runs orchestrated these same CLIs one model per GPU on a
multi-GPU node.)

Resumable: re-running skips finished models (rows == U) via the checkpoint
store; per-model failures are isolated so one bad model never aborts a slot.
