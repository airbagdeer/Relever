import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from fastembed.sparse.bm25 import Bm25
from qdrant_client import QdrantClient, models

# Resolve default paths relative to the project root so the script runs from
# any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COLLECTION_NAME = 'wikipedia_en'
SPLADE_MODEL = 'prithivida/Splade_PP_en_v1'
BM25_MODEL = 'Qdrant/bm25'
BM25_LANGUAGE = 'english'

# --- v2 retrieval defaults (override on the CLI) ---
# v2 keeps the FULL per-query union of the two methods (no truncation): each
# query yields up to per_method_limit hits from EACH of SPLADE and BM25, then
# the two lists are deduplicated against each other so every paragraph is
# stored once per query (~1.5-1.9k unique/query). This is what pushes the
# global unique paragraph universe past 1M after cross-query dedup at index
# time. Contrast with v1, which retrieved 2000/method and truncated to 1000
# total unique per query.
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data/queries/queries.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/splade_wikipedia_v2"
DEFAULT_PER_METHOD_LIMIT = 1000   # retrieve this many from EACH of SPLADE and BM25
DEFAULT_INSTANTIATED_LIMIT = 100  # per instantiated sentence; generic query takes the remainder

load_dotenv()


def is_valid_checkpoint(path: Path) -> bool:
    """A completed-event checkpoint file must parse as JSON and have the
    expected top-level keys. Anything else (empty, truncated, unreadable)
    means the event should be re-processed.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(data, dict)
        and "splade_results" in data
        and "bm25_results" in data
        and "event_id" in data
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to a sibling `.tmp` file and os.replace() it into place.

    Guarantees that `path` either (a) does not exist, or (b) contains the
    fully-serialized payload — never a half-written file that would fool
    the checkpoint check on resume.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _init_failure_logger(output_dir: Path) -> logging.Logger:
    """File-only logger for per-event retrieval failures.

    Appends one timestamped line per failure to ``output_dir/failures.log``.
    Created lazily by ensuring `output_dir` exists before the first write;
    handler attach is idempotent across reruns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_batch_retrieval.failures")
    if not logger.handlers:
        handler = logging.FileHandler(output_dir / "failures.log", mode="a", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    return logger


def build_qdrant_client() -> QdrantClient:
    url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    api_key = os.getenv('QDRANT_API_KEY')

    if 'localhost' in url or '127.0.0.1' in url:
        return QdrantClient(url=url, api_key=api_key, timeout=1200)
    return QdrantClient(url=url, https=True, timeout=1200, api_key=api_key)

def format_point(point) -> dict:
    """Extract relevant fields from ScoredPoint."""
    return {
        "id": point.id,
        "score": point.score,
        "payload": point.payload,
    }

def clean_sentence(sentence: str) -> str:
    """Removes underscores from relation names (e.g. supports_industry -> supports industry)."""
    return sentence.replace("_", " ")

def _batch_search(
    client: QdrantClient,
    sparse_vectors: List[models.SparseVector],
    queries_with_limits: List[Tuple[str, int]],
    using: str,
) -> List[List[dict]]:
    """Issue one Qdrant batch request for a list of (query_text, limit) pairs.

    Returns per-sub-query result lists in the same order as the input.
    Sub-queries with limit <= 0 are skipped (mapped to empty lists) and do
    not contribute a Qdrant request.
    """
    requests: List[models.QueryRequest] = []
    active_indices: List[int] = []
    for i, ((_, limit), vec) in enumerate(zip(queries_with_limits, sparse_vectors)):
        if limit <= 0:
            continue
        requests.append(
            models.QueryRequest(
                query=vec,
                using=using,
                limit=limit,
                with_payload=True,
            )
        )
        active_indices.append(i)

    batches: List[List[dict]] = [[] for _ in queries_with_limits]
    if not requests:
        return batches

    responses = client.query_batch_points(
        collection_name=COLLECTION_NAME,
        requests=requests,
    )

    for idx, resp in zip(active_indices, responses):
        q_text = queries_with_limits[idx][0]
        formatted: List[dict] = []
        for p in resp.points:
            fp = format_point(p)
            fp['source_query'] = q_text
            formatted.append(fp)
        batches[idx] = formatted
    return batches


def _sparse_to_vector(sparse_obj) -> models.SparseVector:
    """Normalize fastembed SparseEmbedding (ndarray-backed) into SparseVector."""
    obj = sparse_obj.as_object()
    return models.SparseVector(
        indices=list(obj['indices']),
        values=list(obj['values']),
    )


def batch_search_splade(
    client: QdrantClient,
    splade_model: SparseTextEmbedding,
    queries_with_limits: List[Tuple[str, int]],
) -> List[List[dict]]:
    """Batch-embed texts and issue one Qdrant batch call against 'text-sparse'."""
    texts = [q for q, _ in queries_with_limits]
    vectors = [_sparse_to_vector(o) for o in splade_model.embed(texts)]
    return _batch_search(client, vectors, queries_with_limits, using='text-sparse')


def batch_search_bm25(
    client: QdrantClient,
    bm25_model: Bm25,
    queries_with_limits: List[Tuple[str, int]],
) -> List[List[dict]]:
    """Batch-embed texts and issue one Qdrant batch call against 'bm25'."""
    texts = [q for q, _ in queries_with_limits]
    vectors = [_sparse_to_vector(o) for o in bm25_model.query_embed(texts)]
    return _batch_search(client, vectors, queries_with_limits, using='bm25')


def deduplicate_results(
    splade_results: List[dict],
    bm25_results: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """Per-relation union dedup — keep every unique paragraph from both methods.

    BM25 keeps all of its hits (deduplicated by ``id``). SPLADE keeps its hits
    that BM25 did not already return, so each paragraph is stored exactly once
    per query. **Nothing is truncated** — the full ``per_method_limit`` union is
    preserved (this is the v2 change vs the old truncate-to-1000 behavior).

    Shared paragraphs are attributed to BM25 here; the choice is cosmetic
    because every downstream consumer (classification, indexing, the unique
    count) re-unions by ``id`` regardless. Within each method, the original
    score/sub-query ordering is preserved and the first occurrence wins.
    """
    seen: set = set()

    final_bm25: List[dict] = []
    for r in bm25_results:
        rid = r.get("id")
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        final_bm25.append(r)

    final_splade: List[dict] = []
    for r in splade_results:
        rid = r.get("id")
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        final_splade.append(r)

    return final_splade, final_bm25


def main(
    input_file: Path,
    output_dir: Path,
    per_method_limit: int,
    instantiated_limit: int,
) -> None:
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_logger = _init_failure_logger(output_dir)

    print("Initializing Qdrant Client...")
    client = build_qdrant_client()

    print(f"Loading SPLADE model ({SPLADE_MODEL})...")
    splade = SparseTextEmbedding(model_name=SPLADE_MODEL)

    print(f"Loading BM25 model ({BM25_MODEL})...")
    bm25 = Bm25(BM25_MODEL, language=BM25_LANGUAGE)

    print(f"Reading queries from {input_file}...")
    with open(input_file, 'r') as f:
        queries = json.load(f)

    # Checkpoint summary: count events already completed (valid output files).
    already_done = sum(
        1 for q in queries
        if q.get("event_id") is not None
        and is_valid_checkpoint(output_dir / f"{q['event_id']}.json")
    )
    print(f"Found {len(queries)} queries. Resuming: {already_done} already complete, "
          f"{len(queries) - already_done} remaining.")
    print(f"Strategy: retrieve {per_method_limit} from EACH method, keep the full "
          f"per-relation union (no truncation).")

    for item in tqdm(queries, desc="Processing Queries"):
        raw_generic_query = item.get("sentence")
        event_id = item.get("event_id")
        instantiated = item.get("instantiated_sentences", [])

        if raw_generic_query is None or event_id is None:
            continue

        generic_query = clean_sentence(raw_generic_query)

        output_file = output_dir / f"{event_id}.json"

        # Checkpoint: skip events whose output file already exists AND parses
        # cleanly. Corrupt/truncated files (e.g. from a kill during a pre-
        # atomic-write version) are re-processed.
        if is_valid_checkpoint(output_file):
            continue

        # Prepare list of (query, limit) pairs. The per-method budget is split
        # across the instantiated sentences (instantiated_limit each) with the
        # generic query taking whatever remains up to per_method_limit.
        search_requests: List[Tuple[str, int]] = []

        # 1. Add instantiated sentences
        for sent in instantiated:
             clean_sent = clean_sentence(sent)
             search_requests.append((clean_sent, instantiated_limit))

        # 2. Add generic query (remainder to reach per_method_limit)
        used_quota = len(search_requests) * instantiated_limit
        remaining_quota = per_method_limit - used_quota

        if remaining_quota > 0:
             search_requests.append((generic_query, remaining_quota))

        # Execute Searches in two batched calls (one per method). On ANY
        # failure, skip the event entirely — do not write a partial file, so
        # output_file.exists() remains a truthful resumability gate.
        try:
            splade_batches = batch_search_splade(client, splade, search_requests)
            bm25_batches = batch_search_bm25(client, bm25, search_requests)
        except Exception as e:
            msg = f"FAILED event {event_id}: {type(e).__name__}: {e}"
            print(msg + " — no output written, will retry on next run")
            failure_logger.warning(msg)
            continue

        all_splade_results = [r for batch in splade_batches for r in batch]
        all_bm25_results = [r for batch in bm25_batches for r in batch]

        # Per-relation union dedup: keep all unique paragraphs from both methods.
        final_splade, final_bm25 = deduplicate_results(
            all_splade_results,
            all_bm25_results,
        )

        # Save Output
        output_data = {
            "query": generic_query,
            "event_id": event_id,
            "splade_results": final_splade,
            "bm25_results": final_bm25
        }

        atomic_write_json(output_file, output_data)

    print(f"Retrieval complete. Results saved to {output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Retrieve SPLADE + BM25 candidate paragraphs per relation query "
                    "and store the full per-relation union (v2)."
    )
    ap.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE,
                    help=f"Relations JSON to process (default: {DEFAULT_INPUT_FILE})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Per-event output dir (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--per-method-limit", type=int, default=DEFAULT_PER_METHOD_LIMIT,
                    help=f"Retrieve this many from EACH of SPLADE and BM25 "
                         f"(default: {DEFAULT_PER_METHOD_LIMIT})")
    ap.add_argument("--instantiated-limit", type=int, default=DEFAULT_INSTANTIATED_LIMIT,
                    help=f"Per instantiated sentence; the generic query takes the "
                         f"remainder up to --per-method-limit (default: {DEFAULT_INSTANTIATED_LIMIT})")
    args = ap.parse_args()
    main(args.input_file, args.output_dir, args.per_method_limit, args.instantiated_limit)
