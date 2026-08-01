# batch_api — OpenAI Batch API labeling

Labels the relation-classification dataset with the **validated best** labeler —
`gpt-5.2` at `reasoning_effort="medium"` with the **`short2b`** system prompt (the config that
reached the best human-gold agreement in the judge-selection study; see the paper appendix) — through
OpenAI's [Batch API](https://developers.openai.com/api/docs/guides/batch) (50% cheaper than the
synchronous API, 24h SLA).

The per-paragraph request `body` is **byte-identical** to the synchronous labeler in
[`estimate_frontier_cost.py`](../estimate_frontier_cost.py) (`classify_one`): same model, same
`messages` (system = `short2b`, user = the `===INPUT===` block), same `reasoning_effort` and
`max_completion_tokens`. Only the transport differs, so this is a faithful batch reproduction of the
validated run — not a new prompt or model.

## Test target

`data/judge_validation/short/_sample.json` — the **same 200 paragraphs** used throughout the cost
study. Human-gold ground truth (`data/judge_validation/short/human__gold.jsonl`) covers **128** of
them; agreement-vs-human is reported on that overlap (exactly as the synchronous pipeline does).

## Usage

```bash
# Pre-flight: build the input .jsonl only — NO API calls, NO spend.
uv run python -m src.create_dataset.classifying_data.batch_api.label_batch_openai submit --dry-run

# End-to-end: build + upload + create batch + poll to completion + fetch + score + report.
uv run python -m src.create_dataset.classifying_data.batch_api.label_batch_openai run

# Or step-by-step (every step is resumable via batch_state.json):
uv run python -m src.create_dataset.classifying_data.batch_api.label_batch_openai submit
uv run python -m src.create_dataset.classifying_data.batch_api.label_batch_openai status
uv run python -m src.create_dataset.classifying_data.batch_api.label_batch_openai fetch
```

`run` polls every `--poll-interval` seconds (default 30) up to `--max-wait` seconds (default 3600).
If the batch hasn't finished by then it keeps running on OpenAI's side — just re-run `fetch` later;
the batch id is persisted in `batch_state.json`.

## Configuration

Editable constants at the top of `label_batch_openai.py`:
`MODEL="gpt-5.2"`, `EFFORT="medium"`, `VARIANT="short2b"`, `MAX_COMPLETION_TOKENS=16000`,
`ENDPOINT="/v1/chat/completions"`, `COMPLETION_WINDOW="24h"`. Requires `OPENAI_API_KEY` in `.env`.

## Outputs

Raw artifacts under `data/judge_validation/test_retest/<run>/` (the distributed
copies keep only `predictions.jsonl`; the other files below exist for fresh runs):

| File | Contents |
|------|----------|
| `batch_input.jsonl` | the uploaded batch requests (one line per paragraph) |
| `batch_state.json` | `batch_id`, `input_file_id`, config, `n_requests`, `submitted_at` |
| `batch_output_raw.jsonl` | raw results downloaded from OpenAI |
| `batch_errors_raw.jsonl` | per-request error file (only if any request errored) |
| `predictions.jsonl` | parsed `{id, event_id, minimax_label, prediction (1/0/-1), tokens…, cost…}` |

Human-readable report (per the repo's `summary/` convention):
`summary/judge_validation/test_retest/BATCH_TEST_REPORT.md` — config, agreement (vs human gold,
vs minimax, vs the prior synchronous `short2b` run as a batch-vs-sync sanity check), and self-metered
cost (batch rate + extrapolation to the full deduped dataset).

## Scaling to the full dataset later

This is a **test run** on the 200-sample. To label a larger set (e.g. `data/splade_wikipedia_v2/`),
the only change needed is the record source feeding `build_input_file` — each record needs an `id`
and a `query_info` (`sentence`, `abstract_entity_type`, `constant_entity`, `relation`) plus
`content`. A single batch allows up to 50,000 requests / 200 MB; shard larger sets across batches.
