import json
import random
import argparse
from collections import Counter
from pathlib import Path
from typing import List
import numpy as np
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data/queries/queries_seed.json"
OUTPUT_FILE = PROJECT_ROOT / "data/queries/queries.json"
PROMPT_FILE = PROJECT_ROOT / "data/queries/query_generation_prompt.txt"

BATCH_SIZE = 15  # relations requested per API call
BATCH_OVERAGE = 3  # extra requested per batch to absorb duplicates

DEDUP_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.80
STEERING_SAMPLE_SIZE = 15
STEERING_OVERREP_SLOTS = 10  # how many of the 15 slots come from over-represented domains
STEERING_OVERREP_TOPK = 3    # sample from the top-K over-represented domains

DOMAIN_POOL: List[str] = [
    "professional sports and athletic competition",
    "film, television, and streaming entertainment",
    "music and performing arts",
    "visual arts, literature, and museums",
    "political parties, elections, and governance",
    "diplomacy, treaties, and international organizations",
    "military and defense",
    "law, courts, and criminal justice",
    "religion and faith communities",
    "higher education and academia",
    "primary and secondary education",
    "healthcare delivery and hospitals",
    "pharmaceuticals and biotechnology",
    "agriculture, farming, and food production",
    "journalism and news media",
    "book publishing and literature",
    "video games and interactive entertainment",
    "social movements and activism",
    "philanthropy and nonprofit organizations",
    "tourism and hospitality",
    "architecture, construction, and real estate",
    "fashion, luxury goods, and design",
    "transportation and logistics infrastructure",
    "space exploration and astronomy",
    "environmental conservation and climate policy",
    "energy production and utilities",
    "finance, banking, and capital markets",
    "manufacturing and heavy industry",
    "information technology and software",
    "telecommunications and networking",
]


class Relation(BaseModel):
    sentence: str = Field(description="One natural-language sentence stating the relation between T and E1.")
    relation: str = Field(description="A short, machine-friendly name: snake_case, lowercase.")
    constant_entity: str = Field(description="A real or realistic concept, industry, market, organization, etc.")
    abstract_entity_type: str = Field(description="A broad, well-defined class.")
    abstract_entity_instances: List[str] = Field(description="3-8 realistic instances.")
    instantiated_sentences: List[str] = Field(description="One sentence per entity instance.")


class RelationBatch(BaseModel):
    relations: List[Relation]


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def classify_domain(text: str, embedder: SentenceTransformer, domain_embeddings: np.ndarray) -> str:
    """Return the DOMAIN_POOL entry whose embedding is closest to `text`."""
    q = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    sims = (q @ domain_embeddings.T)[0]
    return DOMAIN_POOL[int(np.argmax(sims))]


def item_signal_text(item: dict) -> str:
    """Short text used as the classifier input for a dataset item."""
    return f"{item.get('abstract_entity_type', '')} {item.get('constant_entity', '')} {item.get('sentence', '')}"


def pick_domain(domain_counts: Counter) -> str:
    """Pick an under-represented domain (lowest count; ties broken randomly)."""
    min_count = min(domain_counts.get(d, 0) for d in DOMAIN_POOL)
    candidates = [d for d in DOMAIN_POOL if domain_counts.get(d, 0) == min_count]
    return random.choice(candidates)


def build_steering_block(
    all_sentences: List[str],
    item_domains: List[str],
    domain_counts: Counter,
    sample_size: int,
) -> str:
    """
    Build the 'AVOID' block. Half the budget is pulled from the top-K
    over-represented domains (so the model is explicitly pushed away from
    them), the rest is a random sample over the full set.
    """
    n = len(all_sentences)
    if n == 0:
        return (
            "============================================================\n"
            "AVOID GENERATING SENTENCES SIMILAR TO ALREADY-COLLECTED RELATIONS\n"
            "============================================================\n"
            "(no prior relations yet)"
        )

    # Top-K most over-represented domains
    top_domains = {d for d, _ in domain_counts.most_common(STEERING_OVERREP_TOPK)}
    over_rep_pool = [s for s, dom in zip(all_sentences, item_domains) if dom in top_domains]

    overrep_take = min(STEERING_OVERREP_SLOTS, len(over_rep_pool), sample_size)
    overrep_sample = random.sample(over_rep_pool, overrep_take) if overrep_take > 0 else []

    remaining = sample_size - len(overrep_sample)
    remaining_pool = [s for s in all_sentences if s not in set(overrep_sample)]
    random_take = min(remaining, len(remaining_pool))
    random_sample = random.sample(remaining_pool, random_take) if random_take > 0 else []

    sample = overrep_sample + random_sample
    random.shuffle(sample)
    items = "\n".join(f"- {s}" for s in sample)

    top_str = ", ".join(sorted(top_domains))
    return (
        "============================================================\n"
        "AVOID GENERATING SENTENCES SIMILAR TO THESE EXAMPLES\n"
        "(these are already-collected relations — do not repeat\n"
        "them or anything semantically equivalent;\n"
        f"the dataset is currently over-indexed on: {top_str} —\n"
        "steer clear of those topics unless the assigned DOMAIN matches)\n"
        "============================================================\n"
        f"{items}"
    )


def build_domain_block(domain: str, domain_type_counts: Counter) -> str:
    """
    `domain_type_counts` is the Counter of abstract_entity_types already used
    for this specific domain across prior batches. We surface the top 5 as a
    soft nudge so the model picks new types rather than cycling the same one.
    """
    top_types = domain_type_counts.most_common(5)
    if top_types:
        used_str = ", ".join(f"{t}×{c}" for t, c in top_types)
        prior_note = (
            f"\nEntity types already used for this domain (do not over-use;\n"
            f"prefer introducing NEW entity types): {used_str}.\n"
        )
    else:
        prior_note = ""

    return (
        "\n============================================================\n"
        f"DOMAIN FOR THIS BATCH: {domain}\n"
        "Every single relation in this batch MUST be about this domain.\n"
        "The abstract_entity_type, constant_entity, and relation must all\n"
        "belong naturally to this domain. Do NOT default to business,\n"
        "industry, or technology topics unless the assigned domain is\n"
        "explicitly one of those.\n"
        "\n"
        "Aim for VARIETY within this batch: think of several different\n"
        "abstract_entity_types that fit this domain (e.g. for politics:\n"
        "political parties, lobbying firms, think tanks, ministries,\n"
        "diplomats, NGOs, courts, unions, activist groups, PACs). Do not\n"
        "let any single entity type dominate — a small amount of repetition\n"
        "is fine, but avoid having most items share the same entity type."
        f"{prior_note}"
        "============================================================\n"
    )


def make_prompt_template(
    base_prompt: str,
    parser: PydanticOutputParser,
    all_sentences: List[str],
    item_domains: List[str],
    domain_counts: Counter,
    domain: str,
    domain_type_counts: Counter,
) -> PromptTemplate:
    steering_block = build_steering_block(
        all_sentences, item_domains, domain_counts, STEERING_SAMPLE_SIZE
    )
    domain_block = build_domain_block(domain, domain_type_counts)
    prompt_text = base_prompt.replace(
        "OUTPUT FORMAT",
        f"{steering_block}\n\nOUTPUT FORMAT",
    )
    return PromptTemplate(
        template=prompt_text,
        input_variables=["amount"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "domain_hint": domain_block,
        },
    )


def generate_batch(llm, prompt_template: PromptTemplate, parser: PydanticOutputParser, amount: int) -> List[Relation]:
    chain = prompt_template | llm | parser
    result = chain.invoke({"amount": amount})
    return result.relations


def main(
    target_total: int,
    source_file: Path = SOURCE_FILE,
    output_file: Path = OUTPUT_FILE,
    prompt_file: Path = PROMPT_FILE,
    full_output_file: Path | None = None,
) -> None:
    print(f"--- Generating relations (target total: {target_total}) ---")

    existing_data = load_json(source_file)
    print(f"Loaded {len(existing_data)} existing relations from {source_file}")

    # Merge with output file if it already has progress
    if output_file.exists():
        extended_data = load_json(output_file)
        print(f"Resuming: {len(extended_data)} relations already in {output_file}")
    else:
        extended_data = []

    all_data = existing_data + extended_data
    existing_sentences_list: list = [item["sentence"] for item in all_data]
    existing_sentences: set = set(existing_sentences_list)
    next_event_id = max(item["event_id"] for item in all_data) + 1 if all_data else 0

    still_needed = target_total - len(all_data)
    if still_needed <= 0:
        print(f"Already have {len(all_data)} relations — target of {target_total} reached.")
        if full_output_file is not None:
            save_json(all_data, full_output_file)
            print(f"Wrote combined {len(all_data)} relations to {full_output_file}")
        return

    print(f"Need {still_needed} more relations. Starting from event_id={next_event_id}.")

    with open(prompt_file, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    llm = ChatOpenAI(model="gpt-5.4", temperature=0.9)
    parser = PydanticOutputParser(pydantic_object=RelationBatch)

    print(f"Loading dedup embedder: {DEDUP_MODEL_NAME}")
    embedder = SentenceTransformer(DEDUP_MODEL_NAME)
    existing_embeddings = embedder.encode(
        existing_sentences_list,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    ) if existing_sentences_list else np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
    print(f"Embedded {existing_embeddings.shape[0]} existing sentences "
          f"(dim={existing_embeddings.shape[1]}).")

    # Encode DOMAIN_POOL once; classify all existing items to seed coverage counts.
    print(f"Encoding {len(DOMAIN_POOL)} domains for classification.")
    domain_embeddings = embedder.encode(
        DOMAIN_POOL, normalize_embeddings=True, convert_to_numpy=True
    )
    item_domains: list = [
        classify_domain(item_signal_text(item), embedder, domain_embeddings)
        for item in all_data
    ]
    domain_counts = Counter(item_domains)

    # domain -> Counter of abstract_entity_type usage (cross-batch memory)
    domain_type_counts: dict = {d: Counter() for d in DOMAIN_POOL}
    for item, dom in zip(all_data, item_domains):
        t = item.get("abstract_entity_type")
        if t:
            domain_type_counts[dom][t] += 1

    print("Initial domain coverage (top 10):")
    for d, c in domain_counts.most_common(10):
        print(f"  {c:4d}  {d}")
    missing = [d for d in DOMAIN_POOL if domain_counts.get(d, 0) == 0]
    print(f"Domains with 0 items: {len(missing)}/{len(DOMAIN_POOL)}")

    new_items: list = []
    batch_num = 0

    while len(new_items) < still_needed:
        remaining = still_needed - len(new_items)
        batch_request = min(BATCH_SIZE, remaining) + BATCH_OVERAGE
        batch_num += 1

        batch_domain = pick_domain(domain_counts)
        print(f"\nBatch {batch_num}: requesting {batch_request} candidates "
              f"(need {remaining} more). Domain: {batch_domain} "
              f"(current count: {domain_counts.get(batch_domain, 0)})")

        # Rebuild prompt each batch so the rotating sample is refreshed
        prompt_template = make_prompt_template(
            base_prompt, parser, existing_sentences_list,
            item_domains, domain_counts, batch_domain,
            domain_type_counts[batch_domain],
        )
        prompt_text_preview = prompt_template.format(amount=batch_request)
        print(f"  Prompt length: {len(prompt_text_preview)} chars "
              f"(steering sample: {min(STEERING_SAMPLE_SIZE, len(existing_sentences_list))})")

        try:
            candidates = generate_batch(llm, prompt_template, parser, batch_request)
        except Exception as e:
            print(f"  Error in batch {batch_num}: {e}. Retrying next iteration...")
            continue

        # Stage 1: exact-match filter (cheap)
        staged: list = []
        for rel in candidates:
            if rel.sentence in existing_sentences:
                print(f"  Reject (exact dup): {rel.sentence[:80]}...")
                continue
            staged.append(rel)

        if not staged:
            print(f"  Accepted 0/{len(candidates)} candidates. Total new so far: {len(new_items)}/{still_needed}")
            save_json(extended_data + new_items, output_file)
            continue

        # Stage 2: embedding-based semantic dedup (batched)
        cand_sentences = [rel.sentence for rel in staged]
        cand_emb = embedder.encode(
            cand_sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        accepted = 0
        for i, rel in enumerate(staged):
            if len(new_items) >= still_needed:
                break
            if existing_embeddings.shape[0] > 0:
                sims = cand_emb[i] @ existing_embeddings.T
                max_idx = int(np.argmax(sims))
                max_sim = float(sims[max_idx])
            else:
                max_sim = 0.0
                max_idx = -1

            if max_sim >= SIMILARITY_THRESHOLD:
                closest = existing_sentences_list[max_idx] if max_idx >= 0 else "<none>"
                print(f"  Reject (sim={max_sim:.3f}): {rel.sentence[:70]}...")
                print(f"      closest: {closest[:70]}...")
                continue

            item = rel.model_dump()
            item["event_id"] = next_event_id
            new_items.append(item)
            existing_sentences.add(rel.sentence)
            existing_sentences_list.append(rel.sentence)
            existing_embeddings = np.vstack([existing_embeddings, cand_emb[i:i+1]])

            item_domain = classify_domain(
                item_signal_text(item), embedder, domain_embeddings
            )
            item_domains.append(item_domain)
            domain_counts[item_domain] += 1
            t = item.get("abstract_entity_type")
            if t:
                domain_type_counts[item_domain][t] += 1

            next_event_id += 1
            accepted += 1
            drift = " (DRIFTED from assigned domain)" if item_domain != batch_domain else ""
            print(f"  Accept (max_sim={max_sim:.3f}, dom={item_domain}{drift}): "
                  f"{rel.sentence[:70]}...")

        top5 = ", ".join(f"{d.split(',')[0].split(' and ')[0]}={c}"
                         for d, c in domain_counts.most_common(5))
        min_d, min_c = domain_counts.most_common()[-1] if domain_counts else ("-", 0)
        print(f"  Accepted {accepted}/{len(candidates)} candidates. Total new so far: {len(new_items)}/{still_needed}")
        print(f"  Coverage top5: {top5}  |  min: {min_d.split(',')[0]}={min_c}")

        # Save progress after every batch
        save_json(extended_data + new_items, output_file)

    final_extended = extended_data + new_items
    save_json(final_extended, output_file)
    print(f"\nDone. {len(final_extended)} new relations saved to {output_file}")
    print(f"Grand total (existing + new): {len(existing_data) + len(final_extended)}")

    # Optionally write the combined (existing + new) set as a single file.
    if full_output_file is not None:
        combined = existing_data + final_extended
        save_json(combined, full_output_file)
        print(f"Wrote combined {len(combined)} relations to {full_output_file}")


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser(description="Generate additional relation queries.")
    parser_arg.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Total number of relations to reach across existing + new (default: 1000)",
    )
    parser_arg.add_argument(
        "--source",
        type=Path,
        default=SOURCE_FILE,
        help=f"Existing relations loaded as the base set (default: {SOURCE_FILE})",
    )
    parser_arg.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Resume buffer holding NEW items only, saved after each batch "
             f"(default: {OUTPUT_FILE})",
    )
    parser_arg.add_argument(
        "--full-output",
        type=Path,
        default=None,
        help="If set, also write the combined (source + new) set to this file "
             "at the end. Use a path distinct from --source to keep reruns idempotent.",
    )
    parser_arg.add_argument(
        "--prompt",
        type=Path,
        default=PROMPT_FILE,
        help=f"Base prompt template (default: {PROMPT_FILE})",
    )
    args = parser_arg.parse_args()
    main(args.target, args.source, args.output, args.prompt, args.full_output)
