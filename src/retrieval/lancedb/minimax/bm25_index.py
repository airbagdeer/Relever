"""Build, cache, and query a BM25 index over the minimax corpus per granularity.

Used by ``evaluate_minimax.py --ensemble-with-bm25`` and ``--bm25-only`` to
score BM25 against (or alongside) each dense / ColBERT model. The BM25 index
is built over the *same* corpus (same row IDs, same texts) that the dense
indexer wrote to LanceDB, so a row's BM25 score lines up 1:1 with its dense
score by ``row_id``.

Backend: ``bm25s`` (SciPy-sparse-matrix BM25 — typically 10-100x faster than
``rank_bm25`` for both build and per-query retrieval, and multi-threaded out
of the box). We use BM25 Okapi to match the math of the prior baseline.

Cache layout (under ``LANCEDB_PATH / _minimax_bm25_cache``):

  - ``bm25_<gran>_<sig>/``            — bm25s on-disk index directory (its
                                       native format: sparse matrix +
                                       vocab/IDF + doc-length stats).
  - ``bm25_queries_<gran>_<sig>.pkl`` — top-``N`` BM25 hits per query so
                                       per-model evals can ensemble without
                                       reloading the full BM25 index.

The signature ``<sig>`` is derived from (source_dir, granularity, corpus
size) so an updated corpus naturally rebuilds.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from corpus_loader import ParagraphCorpus, load_paragraph_corpus

logger = logging.getLogger(__name__)

CACHE_DIRNAME = "_minimax_bm25_cache"
# bm25s parameters: Okapi BM25 with the common k1=1.5, b=0.75 defaults so
# results are comparable with `src/retrieval/bm25/evaluate.py` and the
# broader BM25 literature.
BM25S_METHOD = "lucene"  # see bm25s docs; "lucene" is the default and gives
                          # standard Okapi-style scoring across implementations.
BM25S_K1 = 1.5
BM25S_B = 0.75


def default_bm25_cache_dir(lancedb_path: Path) -> Path:
    """Return the standard BM25 cache directory under a LanceDB root."""
    return lancedb_path / CACHE_DIRNAME


def _tokenize_basic(text: str) -> list[str]:
    """Tokenizer matching `src/retrieval/bm25/evaluate.py`: lowercase + whitespace split.

    We deliberately keep this dumb and string-based so the BM25 baseline
    stays comparable across the project. Stop-word removal / stemming /
    Lucene-style filters would be a separate experiment.
    """
    return text.lower().split()


def _signature(source_dir: Path, granularity_level: str, n_rows: int) -> str:
    """Stable signature for caching. Changes whenever the corpus changes."""
    h = hashlib.sha256()
    h.update(str(source_dir.resolve()).encode())
    h.update(b";gran=")
    h.update(granularity_level.encode())
    h.update(b";n=")
    h.update(str(n_rows).encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class BM25Bundle:
    """Container the ensemble path uses to drive BM25 lookups."""

    bm25: object  # bm25s.BM25 — typed loosely so callers don't need bm25s installed at import time
    row_ids: tuple[str, ...]            # row_id per BM25 corpus index (== corpus.paragraph_ids)
    parent_paragraph_ids: tuple[str, ...]  # parent paragraph_id per corpus index
    cache_path: Path                    # bm25s on-disk directory


@dataclass(frozen=True)
class BM25QueryHits:
    """Precomputed BM25 top-N hits for the 100 queries on one granularity."""

    granularity: str
    n_per_query: int
    indices: np.ndarray                 # (Q, N) int64, BM25 corpus row indices
    scores: np.ndarray                  # (Q, N) float32
    cache_path: Path


def _row_to_parent(row_id: str) -> str:
    """Map a corpus row_id back to its parent paragraph UUID.

    Paragraph granularity: ``row_id == parent`` (no '_').
    Sentence / proposition: ``row_id = '<parent_uuid>_<idx>'``; paragraph UUIDs
    themselves don't contain underscores, so splitting on the first '_' is
    safe.
    """
    return row_id.split("_", 1)[0] if "_" in row_id else row_id


def _detect_n_threads() -> int:
    """Conservative thread count for BM25 build.

    bm25s respects ``$OMP_NUM_THREADS`` / ``$BM25S_NUM_THREADS`` and parallelizes
    across docs internally. On the cluster we have 128 cores; pin to 32 by
    default to leave room for other users (override with ``$BM25S_NUM_THREADS``).
    """
    if "BM25S_NUM_THREADS" in os.environ:
        try:
            return max(1, int(os.environ["BM25S_NUM_THREADS"]))
        except ValueError:
            pass
    avail = os.cpu_count() or 4
    return min(32, max(1, avail - 2))


def build_or_load_bm25(
    *,
    source_dir: Path,
    granularity_level: str,
    cache_dir: Path,
    corpus: Optional[ParagraphCorpus] = None,
) -> BM25Bundle:
    """Return a :class:`BM25Bundle` for the given granularity.

    If the bm25s on-disk index already exists at ``cache_dir`` (matching
    signature), it is loaded instead of rebuilt.
    """
    if corpus is None:
        corpus = load_paragraph_corpus(
            source_dir=source_dir,
            granularity_level=granularity_level,
        )

    sig = _signature(source_dir, granularity_level, len(corpus))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"bm25_{granularity_level}_{sig}"

    row_ids = tuple(corpus.paragraph_ids)
    parent_ids = tuple(_row_to_parent(rid) for rid in row_ids)

    # Local import: bm25s is only required when an eval actually exercises
    # this module, not when evaluate_minimax.py is imported for --help.
    import bm25s

    # bm25s.save() writes params.index.json + vocab.index.json + several .npy
    # CSC-matrix blobs. params.index.json is the smallest fixed-name sentinel
    # and is always present when a save completed successfully.
    if cache_path.is_dir() and (cache_path / "params.index.json").exists():
        logger.info("Loading BM25 (bm25s) from cache: %s", cache_path)
        bm25 = bm25s.BM25.load(str(cache_path), load_corpus=False)
        return BM25Bundle(bm25=bm25, row_ids=row_ids, parent_paragraph_ids=parent_ids, cache_path=cache_path)

    n_threads = _detect_n_threads()
    logger.info(
        "Building BM25 (bm25s, method=%s, k1=%.2f, b=%.2f) over %d %s rows from %s using %d threads ...",
        BM25S_METHOD, BM25S_K1, BM25S_B, len(corpus), granularity_level, source_dir, n_threads,
    )

    t0 = time.time()
    tokenized = bm25s.tokenize(
        corpus.texts,
        lower=True,
        stopwords=None,
        stemmer=None,
        return_ids=False,
        show_progress=False,
    )
    logger.info("Tokenization done in %.1fs", time.time() - t0)

    t1 = time.time()
    bm25 = bm25s.BM25(method=BM25S_METHOD, k1=BM25S_K1, b=BM25S_B)
    bm25.index(tokenized, show_progress=False)
    logger.info("BM25 index built in %.1fs", time.time() - t1)

    # Atomic-ish save: bm25s writes a directory of binary blobs; we save to a
    # tmp dir then rename. Avoids leaving a half-written cache if interrupted.
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    bm25.save(str(tmp_path))
    if cache_path.exists():
        shutil.rmtree(cache_path)
    tmp_path.rename(cache_path)
    total_mb = sum(p.stat().st_size for p in cache_path.rglob("*") if p.is_file()) / 1024 / 1024
    logger.info("bm25s cache written (%.1f MB) -> %s", total_mb, cache_path)

    return BM25Bundle(bm25=bm25, row_ids=row_ids, parent_paragraph_ids=parent_ids, cache_path=cache_path)


def build_or_load_query_hits(
    *,
    bundle: BM25Bundle,
    granularity_level: str,
    queries: list[str],
    n_per_query: int,
    cache_dir: Path,
) -> BM25QueryHits:
    """Return precomputed top-N BM25 hits per query.

    Cached to disk keyed by (granularity, corpus, n, query_hash). Cheap to
    load (~tens of MB even at N=10000, 100 queries).
    """
    import bm25s

    sig = hashlib.sha256()
    sig.update(str(bundle.cache_path).encode())
    sig.update(b";n=")
    sig.update(str(n_per_query).encode())
    sig.update(b";qs=")
    for q in queries:
        sig.update(q.encode())
        sig.update(b"|")
    short = sig.hexdigest()[:16]
    cache_path = cache_dir / f"bm25_queries_{granularity_level}_{short}.pkl"

    if cache_path.exists():
        logger.info("Loading precomputed BM25 query hits from %s", cache_path)
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        return BM25QueryHits(
            granularity=obj["granularity"],
            n_per_query=obj["n_per_query"],
            indices=obj["indices"],
            scores=obj["scores"],
            cache_path=cache_path,
        )

    n_threads = _detect_n_threads()
    logger.info(
        "Computing BM25 top-%d hits for %d queries on granularity=%s (n_threads=%d) ...",
        n_per_query, len(queries), granularity_level, n_threads,
    )
    t0 = time.time()
    tokenized_q = bm25s.tokenize(
        queries,
        lower=True,
        stopwords=None,
        stemmer=None,
        return_ids=False,
        show_progress=False,
    )
    # bm25s.retrieve handles all queries in one call and is multi-threaded.
    results = bundle.bm25.retrieve(
        tokenized_q,
        k=n_per_query,
        n_threads=n_threads,
        show_progress=True,
    )
    # results is a Results object with .documents (Q, N) int64 indices and .scores (Q, N) float32.
    indices = np.asarray(results.documents, dtype=np.int64)
    scores = np.asarray(results.scores, dtype=np.float32)
    logger.info("BM25 retrieve done in %.1fs", time.time() - t0)

    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "granularity": granularity_level,
                "n_per_query": n_per_query,
                "indices": indices,
                "scores": scores,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.replace(cache_path)
    logger.info(
        "BM25 query-hits cache written (%.1f MB) -> %s",
        cache_path.stat().st_size / 1024 / 1024,
        cache_path,
    )

    return BM25QueryHits(
        granularity=granularity_level,
        n_per_query=n_per_query,
        indices=indices,
        scores=scores,
        cache_path=cache_path,
    )


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-array min-max scale to [0, 1]. Degenerate arrays (all-equal,
    empty) return zeros so they contribute nothing to a 50/50 mix."""
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr.astype(np.float32) - lo) / (hi - lo)).astype(np.float32, copy=False)
