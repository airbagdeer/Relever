"""Dedupe + aggregate text records from `classified_splade_*` directories.

Supports three granularities:
  - paragraph    -> top-level `id` + `content` (one row per paragraph)
  - sentence     -> top-level `id` + `sentences: list[str]` (one row per sentence,
                    synthetic id ``f"{paragraph_id}_{i}"``)
  - proposition  -> top-level `id` + `propositions: list[str]` (analogous)

The same parent paragraph (UUID `id`) often appears across multiple per-event
files with different `event_id` / `source_query` / `classification`. The loader
collects all occurrences into a single record per *corpus row* (paragraph,
sentence, or proposition) so the embedding pass runs once per unique text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

CACHE_DIRNAME = "_minimax_corpus_cache"

_GRANULARITY_LIST_FIELD = {
    "sentence": "sentences",
    "proposition": "propositions",
}


@dataclass
class ParagraphCorpus:
    """Generic, granularity-agnostic corpus container.

    Field names retained for back-compat: ``paragraph_ids`` here means
    "row ids" (paragraph id, sentence id like ``pid_0``, or proposition id
    like ``pid_3`` depending on granularity).
    """

    paragraph_ids: list[str]
    texts: list[str]
    metadata_jsons: list[str]

    def __len__(self) -> int:
        return len(self.paragraph_ids)


def _read_one(path: Path) -> tuple[str, list[dict]]:
    event_id = path.stem
    try:
        with open(path, "r", encoding="utf-8") as f:
            return event_id, json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from %s: %s", path, exc)
        return event_id, []
    except OSError as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return event_id, []


def _list_json_files(source_dir: Path, limit_files: Optional[int] = None) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix == ".json"]
    files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if limit_files is not None:
        files = files[:limit_files]
    return files


def _meta_signature(files: list[Path], granularity_level: str) -> str:
    h = hashlib.sha256()
    h.update(b"granularity=")
    h.update(granularity_level.encode())
    h.update(b";")
    for p in files:
        try:
            stat = p.stat()
        except OSError:
            continue
        h.update(p.name.encode())
        h.update(b"|")
        h.update(str(stat.st_size).encode())
        h.update(b"|")
        h.update(str(int(stat.st_mtime)).encode())
        h.update(b";")
    return h.hexdigest()


def _cache_paths(cache_dir: Path, signature: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"corpus_{signature}.pkl", cache_dir / "corpus_meta.json"


def _iter_rows(item: dict, granularity_level: str) -> Iterable[tuple[str, str, str]]:
    """Yield ``(row_id, text, parent_paragraph_id)`` for one input item.

    For paragraph granularity, ``row_id == parent_paragraph_id``.
    For sentence/proposition, ``row_id == f"{parent_paragraph_id}_{i}"``.
    """
    parent_pid = item.get("id")
    if not parent_pid:
        return
    if granularity_level == "paragraph":
        content = item.get("content") or ""
        if content:
            yield parent_pid, content, parent_pid
        return

    field = _GRANULARITY_LIST_FIELD.get(granularity_level)
    if field is None:
        raise ValueError(f"Unsupported granularity_level: {granularity_level!r}")

    children = item.get(field) or []
    for i, text in enumerate(children):
        if not text:
            continue
        text = text.strip()
        if not text:
            continue
        yield f"{parent_pid}_{i}", text, parent_pid


def load_paragraph_corpus(
    source_dir: Path,
    granularity_level: str = "paragraph",
    limit_files: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    max_workers: int = 8,
) -> ParagraphCorpus:
    """Build a deduplicated corpus from a directory of per-event JSONs.

    Each input JSON file is named ``<event_id>.json`` and contains a list of
    objects whose schema depends on ``granularity_level`` (see module docstring).

    Output: one entry per unique ``row_id``, with all occurrences aggregated
    into metadata. Text is taken from the first occurrence; if a subsequent
    occurrence has different content for the same id, a warning is logged.
    """
    if granularity_level not in ("paragraph", "sentence", "proposition"):
        raise ValueError(f"Unsupported granularity_level: {granularity_level!r}")

    files = _list_json_files(source_dir, limit_files=limit_files)
    if not files:
        return ParagraphCorpus([], [], [])

    cache_pkl: Optional[Path] = None
    cache_meta: Optional[Path] = None
    signature: Optional[str] = None
    if use_cache and cache_dir is not None and limit_files is None:
        signature = _meta_signature(files, granularity_level)
        cache_pkl, cache_meta = _cache_paths(cache_dir, signature)
        if cache_pkl.exists():
            try:
                with open(cache_pkl, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, ParagraphCorpus) and len(cached) > 0:
                    logger.info("Loaded corpus from cache: %s (U=%d)", cache_pkl, len(cached))
                    return cached
            except (OSError, pickle.PickleError) as exc:
                logger.warning("Cache read failed (%s); re-loading from JSON.", exc)

    # row_id -> {"text": str, "parent_paragraph_id": str, "occurrences": list[dict]}
    aggregated: dict[str, dict] = {}
    text_mismatches = 0

    logger.info(
        "Scanning %d JSON files from %s (granularity=%s) ...",
        len(files),
        source_dir,
        granularity_level,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for event_id, items in pool.map(_read_one, files):
            for item in items:
                occurrence = {
                    "event_id": str(event_id),
                    "source_query": item.get("source_query", "") or "",
                    "classification": int(item.get("classification", -1)),
                }
                for row_id, text, parent_pid in _iter_rows(item, granularity_level):
                    rec = aggregated.get(row_id)
                    if rec is None:
                        aggregated[row_id] = {
                            "text": text,
                            "parent_paragraph_id": parent_pid,
                            "occurrences": [occurrence],
                        }
                    else:
                        if rec["text"] != text:
                            text_mismatches += 1
                            if text_mismatches <= 5:
                                logger.warning(
                                    "row_id %s has differing content across files; "
                                    "keeping first occurrence (event=%s). first=%r, later=%r",
                                    row_id,
                                    rec["occurrences"][0]["event_id"],
                                    rec["text"][:100],
                                    text[:100],
                                )
                        rec["occurrences"].append(occurrence)

    if text_mismatches > 0:
        logger.warning(
            "Total row_id content-mismatches: %d (only first 5 logged in detail).",
            text_mismatches,
        )

    paragraph_ids: list[str] = []
    texts: list[str] = []
    metadata_jsons: list[str] = []
    for row_id, rec in aggregated.items():
        occurrences = rec["occurrences"]
        event_ids: list[str] = []
        positive_event_ids: list[str] = []
        seen_events: set[str] = set()
        for occ in occurrences:
            eid = occ["event_id"]
            if eid not in seen_events:
                seen_events.add(eid)
                event_ids.append(eid)
            if occ["classification"] == 1 and eid not in positive_event_ids:
                positive_event_ids.append(eid)
        meta = {
            "paragraph_id": rec["parent_paragraph_id"],
            "row_id": row_id,
            "granularity_level": granularity_level,
            "occurrences": occurrences,
            "event_ids": event_ids,
            "positive_event_ids": positive_event_ids,
            "any_positive": bool(positive_event_ids),
        }
        paragraph_ids.append(row_id)
        texts.append(rec["text"])
        metadata_jsons.append(json.dumps(meta, ensure_ascii=False))

    corpus = ParagraphCorpus(
        paragraph_ids=paragraph_ids,
        texts=texts,
        metadata_jsons=metadata_jsons,
    )
    logger.info(
        "Corpus built: %d unique %ss from %d files (avg %.2f occurrences/row).",
        len(corpus),
        granularity_level,
        len(files),
        sum(len(json.loads(m)["occurrences"]) for m in metadata_jsons) / max(len(corpus), 1),
    )

    if cache_pkl is not None and cache_meta is not None and signature is not None:
        try:
            with open(cache_pkl, "wb") as f:
                pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cache_meta, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "signature": signature,
                        "n_files": len(files),
                        "U": len(corpus),
                        "source_dir": str(source_dir),
                        "granularity_level": granularity_level,
                    },
                    f,
                    indent=2,
                )
            logger.info("Cached corpus -> %s", cache_pkl)
        except (OSError, pickle.PickleError) as exc:
            logger.warning("Failed to write corpus cache: %s", exc)

    return corpus


def default_cache_dir(lancedb_path: Path) -> Path:
    return lancedb_path / CACHE_DIRNAME
