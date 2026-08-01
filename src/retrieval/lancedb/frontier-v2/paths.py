"""frontier-v2 path constants.

Kept inside frontier-v2 (rather than added to ``src/retrieval/config.py``) so
the whole v2 indexing effort is self-contained in this package. Path *roots*
(``PROJECT_ROOT``, ``LANCEDB_PATH``, ``LANCEDB_SCRATCH_PATH``) are still read
from the central ``config.py`` so there is a single source of truth for where
the project and the LanceDB partitions live.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path so `config` imports)
from config import LANCEDB_PATH, LANCEDB_SCRATCH_PATH, PROJECT_ROOT

# The v2 SPLADE/BM25 candidate dump: 1050 per-event JSON files, each a dict with
# `splade_results` + `bm25_results` lists. ~1.04M unique paragraphs after dedup.
SPLADE_WIKIPEDIA_V2_PATH = PROJECT_ROOT / "data" / "splade_wikipedia_v2"

# The v2 deduplicated-corpus pickle cache. Routed to the large ephemeral /tmp
# scratch partition (it is ~1-2 GB and fully regenerable from the JSONs) so it
# does not eat into the small, near-full durable home partition.
V2_CORPUS_CACHE_DIR = LANCEDB_SCRATCH_PATH / "_v2_corpus_cache"

# The query set (event_id 0-1099); the query text is the per-object ``sentence``
# field. Mirrors the central ``config.RELATIONS_PATH``.
V2_RELATIONS_PATH = PROJECT_ROOT / "data" / "queries" / "queries.json"

# Output root for the per-query / per-model retrieved candidate dumps. Each query
# gets a ``<qid>/`` subdir (qid = event_id zero-padded to 4 digits) holding one
# ``<alias>_<qid>.json`` per model plus a deduplicated ``all_candidates_<qid>.json``.
FRONTIER_CANDIDATES_DIR = PROJECT_ROOT / "data" / "candidates"

# Re-export the two LanceDB roots for convenience.
HOME_ROOT = LANCEDB_PATH               # durable NFS home (small)
SCRATCH_ROOT = LANCEDB_SCRATCH_PATH    # ephemeral local /tmp (large)
