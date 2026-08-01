"""Build the v2 relation seed: drop zero-positive queries and renumber.

Historical provenance for the preliminary filtering step (paper Sec. 4.1): the
v1 generation produced 1000 queries (an internal, superseded set that is not
distributed). After minimax classification, 23 of them have **0 positive
paragraphs**, so they contribute nothing to Precision@R. This script drops
those, renumbers the survivors to a contiguous ``0..N-1`` event_id range, and
writes the result as the v2 seed that ``generate_more_relations.py`` then
grows into the released ``data/queries/queries.json``.

The zero-positive set is computed **dynamically** from the minimax
classification output (never hardcoded), so re-running after a classification
change stays correct. The expected count (23) is asserted, so a stale or
partial classification dir fails loudly instead of silently dropping the wrong
queries.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data/queries/v1-queries.json"  # internal v1 set; not distributed
DEFAULT_CLASSIFIED_DIR = PROJECT_ROOT / "data/classified_splade_paragraphs_minimax"
# The seed is a *distinct* file from the final queries.json so that
# generate_more_relations.py can read the seed (immutable) and write the grown
# set to queries.json without aliasing — re-running the generator after
# completion then stays idempotent. (Seed + id-map are internal lineage files
# and are not part of the public data release.)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/queries/queries_seed.json"
DEFAULT_ID_MAP = PROJECT_ROOT / "data/queries/seed_id_map.json"

EXPECTED_ZERO_POSITIVE = 23


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _display(path: Path) -> str:
    """Project-relative path for printing, falling back to the absolute path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def positive_count(classified_file: Path) -> int:
    """Number of paragraphs labelled positive (``classification == 1``).

    Mirrors ``load_ground_truth_counts`` in
    ``src/retrieval/lancedb/evaluate_lancedb.py``.
    """
    data = load_json(classified_file)
    return sum(1 for item in data if item.get("classification") == 1)


def find_zero_positive_event_ids(relations: List[dict], classified_dir: Path) -> List[int]:
    """Return the sorted event_ids whose minimax positive count is 0.

    Raises if any relation has no classification file: we cannot tell whether
    it is zero-positive or merely unclassified, and guessing either way would
    corrupt the seed.
    """
    zero: List[int] = []
    missing: List[int] = []
    for item in relations:
        event_id = item["event_id"]
        classified_file = classified_dir / f"{event_id}.json"
        if not classified_file.exists():
            missing.append(event_id)
            continue
        if positive_count(classified_file) == 0:
            zero.append(event_id)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} relations have no classification file in "
            f"{classified_dir} (e.g. {missing[:10]}). Cannot determine the "
            f"zero-positive set; aborting."
        )
    return sorted(zero)


def renumber(relations: List[dict]) -> tuple[List[dict], Dict[int, int]]:
    """Return relations with contiguous event_ids 0..N-1 (originals untouched).

    Order is fixed by the original event_id so the mapping is deterministic.
    Each item is shallow-copied before its event_id is rewritten, so the input
    list is not mutated.
    """
    ordered = sorted(relations, key=lambda it: it["event_id"])
    id_map: Dict[int, int] = {}
    renumbered: List[dict] = []
    for new_id, item in enumerate(ordered):
        id_map[item["event_id"]] = new_id
        new_item = dict(item)
        new_item["event_id"] = new_id
        renumbered.append(new_item)
    return renumbered, id_map


def main(
    source: Path,
    classified_dir: Path,
    output: Path,
    id_map_file: Path,
    expected_zero: int,
) -> None:
    relations = load_json(source)
    print(f"Loaded {len(relations)} relations from {source}")

    zero_ids = find_zero_positive_event_ids(relations, classified_dir)
    id_to_sentence = {it["event_id"]: it.get("sentence", "") for it in relations}
    print(f"\nFound {len(zero_ids)} zero-positive relations (from {classified_dir.name}):")
    for eid in zero_ids:
        print(f"  event_id={eid:>4}: {id_to_sentence.get(eid, '')[:80]}")

    if len(zero_ids) != expected_zero:
        raise SystemExit(
            f"\nExpected {expected_zero} zero-positive relations but found "
            f"{len(zero_ids)}. Refusing to write the v2 seed — verify that "
            f"{classified_dir} is the complete, final classification "
            f"(or pass --expected-zero {len(zero_ids)} if this change is intended)."
        )

    zero_set = set(zero_ids)
    kept = [it for it in relations if it["event_id"] not in zero_set]
    renumbered, id_map = renumber(kept)

    save_json(renumbered, output)
    save_json({str(old): new for old, new in id_map.items()}, id_map_file)

    last = len(renumbered) - 1
    print(
        f"\nDropped {len(zero_ids)}, kept {len(renumbered)} "
        f"(event_id 0..{last}, contiguous)."
    )
    print(f"Wrote v2 seed       -> {output}")
    print(f"Wrote old->new map  -> {id_map_file}")
    print(
        f"\nNext: grow the seed to 1050 with\n"
        f"  uv run python src/create_dataset/generate_more_relations.py \\\n"
        f"      --source {_display(output)} \\\n"
        f"      --output data/queries/queries-new.json \\\n"
        f"      --full-output data/queries/queries.json \\\n"
        f"      --target 1050"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Drop zero-positive relations and renumber into the v2 seed."
    )
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help=f"v1 relations file (default: {DEFAULT_SOURCE})")
    ap.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR,
                    help=f"minimax classified paragraphs dir (default: {DEFAULT_CLASSIFIED_DIR})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"v2 seed output file (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--id-map", type=Path, default=DEFAULT_ID_MAP,
                    help=f"old->new event_id map output (default: {DEFAULT_ID_MAP})")
    ap.add_argument("--expected-zero", type=int, default=EXPECTED_ZERO_POSITIVE,
                    help="expected number of zero-positive relations (safety check)")
    args = ap.parse_args()
    main(args.source, args.classified_dir, args.output, args.id_map, args.expected_zero)
