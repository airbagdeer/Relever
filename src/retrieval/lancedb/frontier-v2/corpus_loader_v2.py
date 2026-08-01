"""Deduplicated paragraph corpus loader for the ``splade_wikipedia_v2`` dump.

The v2 JSON shape differs from the ``classified_splade_*`` files that
``minimax/corpus_loader.py`` parses. Each v2 file is a *dict*::

    {
      "query": "...",
      "event_id": 0,
      "splade_results": [ {"id": "<uuid>", "score": ..., "payload": {"content": "...", ...}, ...}, ... ],
      "bm25_results":   [ {"id": "<uuid>", ...}, ... ]
    }

We build the same :class:`ParagraphCorpus` (reused from the minimax loader) by
unioning ``splade_results`` + ``bm25_results``, keying each paragraph on
``item["id"]`` (== ``payload["point_id"]``, the stable paragraph UUID) and
taking the text from ``payload["content"]``. Deduplication is global across all
1050 files (the same paragraph is retrieved for many events).

No classifications are available yet, so metadata is minimal: we keep
``paragraph_id == row_id`` (so labels can be back-joined on the UUID later) and
record the set of ``event_ids`` that retrieved each paragraph for provenance.

Caching mirrors the minimax loader (pickle keyed on file mtimes/sizes), with a
distinct signature tag so v1 and v2 caches never collide.
"""

from __future__ import annotations

import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional

import _bootstrap  # noqa: F401  (sets sys.path so minimax `corpus_loader` imports)
from corpus_loader import (  # minimax/corpus_loader.py
    ParagraphCorpus,
    _cache_paths,
    _list_json_files,
    _meta_signature,
    _read_one,
)

logger = logging.getLogger(__name__)

# Granularity tag folded into the cache signature so the v2 paragraph cache is
# never confused with a v1 ("paragraph") cache for the same file set.
_V2_SIGNATURE_TAG = "splade_v2_paragraph"

_RESULT_KEYS = ("splade_results", "bm25_results")


def _iter_rows_v2(payload: Any) -> Iterable[tuple[str, str]]:
    """Yield ``(row_id, text)`` for every candidate in one v2 event file.

    Iterates both ``splade_results`` and ``bm25_results``. ``row_id`` is the
    item-level ``id`` (the paragraph UUID, equal to ``payload.point_id``);
    ``text`` is ``payload.content``. Items without an id or with empty content
    are skipped. ``payload`` may be a non-dict (``_read_one`` returns ``[]`` for
    files that fail to decode) — those yield nothing.
    """
    if not isinstance(payload, dict):
        return
    for key in _RESULT_KEYS:
        results = payload.get(key)
        if not results:
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            row_id = item.get("id")
            if not row_id:
                continue
            item_payload = item.get("payload") or {}
            content = item_payload.get("content")
            if not content:
                continue
            content = content.strip()
            if not content:
                continue
            yield row_id, content


def load_v2_paragraph_corpus(
    source_dir: Path,
    limit_files: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    max_workers: int = 8,
) -> ParagraphCorpus:
    """Build a globally-deduplicated paragraph corpus from the v2 dump.

    Returns one entry per unique paragraph UUID. Text is taken from the first
    occurrence; differing content for the same id across files is logged and
    the first kept (paragraph text is stable per UUID, so this should be rare).
    """
    files = _list_json_files(source_dir, limit_files=limit_files)
    if not files:
        logger.warning("No JSON files found under %s", source_dir)
        return ParagraphCorpus([], [], [])

    cache_pkl: Optional[Path] = None
    cache_meta: Optional[Path] = None
    signature: Optional[str] = None
    if use_cache and cache_dir is not None and limit_files is None:
        signature = _meta_signature(files, _V2_SIGNATURE_TAG)
        cache_pkl, cache_meta = _cache_paths(cache_dir, signature)
        if cache_pkl.exists():
            try:
                with open(cache_pkl, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, ParagraphCorpus) and len(cached) > 0:
                    logger.info("Loaded v2 corpus from cache: %s (U=%d)", cache_pkl, len(cached))
                    return cached
            except (OSError, pickle.PickleError) as exc:
                logger.warning("v2 cache read failed (%s); re-loading from JSON.", exc)

    # row_id -> {"text": str, "event_ids": list[str]}
    aggregated: dict[str, dict] = {}
    text_mismatches = 0

    logger.info("Scanning %d v2 JSON files from %s ...", len(files), source_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for event_id, payload in pool.map(_read_one, files):
            for row_id, text in _iter_rows_v2(payload):
                rec = aggregated.get(row_id)
                if rec is None:
                    aggregated[row_id] = {"text": text, "event_ids": [str(event_id)]}
                else:
                    if rec["text"] != text:
                        text_mismatches += 1
                        if text_mismatches <= 5:
                            logger.warning(
                                "row_id %s has differing content across files; keeping first. "
                                "first=%r later=%r",
                                row_id, rec["text"][:80], text[:80],
                            )
                    if str(event_id) not in rec["event_ids"]:
                        rec["event_ids"].append(str(event_id))

    if text_mismatches:
        logger.warning("Total v2 row_id content-mismatches: %d (first 5 logged).", text_mismatches)

    paragraph_ids: list[str] = []
    texts: list[str] = []
    metadata_jsons: list[str] = []
    for row_id, rec in aggregated.items():
        meta = {
            "paragraph_id": row_id,
            "row_id": row_id,
            "granularity_level": "paragraph",
            "source": "splade_v2",
            "event_ids": rec["event_ids"],
        }
        paragraph_ids.append(row_id)
        texts.append(rec["text"])
        metadata_jsons.append(json.dumps(meta, ensure_ascii=False))

    corpus = ParagraphCorpus(
        paragraph_ids=paragraph_ids, texts=texts, metadata_jsons=metadata_jsons
    )
    logger.info("v2 corpus built: U=%d unique paragraphs from %d files.", len(corpus), len(files))

    if cache_dir is not None and cache_pkl is not None and cache_meta is not None and signature is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Atomic write (mirrors minimax _write_locked): write to a temp file
            # then rename, so a concurrent reader never sees a partial pickle.
            tmp_pkl = cache_pkl.with_suffix(".tmp")
            with open(tmp_pkl, "wb") as f:
                pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_pkl.replace(cache_pkl)
            with open(cache_meta, "w", encoding="utf-8") as f:
                json.dump(
                    {"signature": signature, "n_files": len(files), "U": len(corpus),
                     "source_dir": str(source_dir), "format": "splade_v2"},
                    f, indent=2,
                )
            logger.info("Cached v2 corpus -> %s", cache_pkl)
        except (OSError, pickle.PickleError) as exc:
            logger.warning("Failed to write v2 corpus cache: %s", exc)

    return corpus
