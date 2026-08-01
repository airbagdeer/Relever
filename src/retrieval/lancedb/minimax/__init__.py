"""Indexing pipeline for `data/classified_splade_paragraphs_minimax/`.

Provides:
- `corpus_loader`: dedupe + aggregate paragraphs across files (one entry per
  unique paragraph_id, with the list of (event_id, source_query, classification)
  occurrences preserved in metadata).
- `checkpoint`: per-model resume-safe state and bounded drop-table logic.
- `index_dense`: multi-GPU SentenceTransformer indexer using
  `model.encode_multi_process`.
- `index_colbert`: pylate-based ColBERTv2 indexer producing a multi-vector
  LanceDB table.

Tables produced:
    thesis-qdrant-classified-{alias}-paragraph

Driver script: `run_index_minimax.sh` (hybrid/sequential GPU scheduling).
"""
