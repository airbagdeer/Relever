import argparse
import json
import logging
import os
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

import lancedb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

# Add parent directory to sys.path to allow importing from config.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    LANCEDB_PATH,
    MODEL_IDS,
    get_classified_collection_name,
    RELATIONS_PATH,
    RESULTS_PATH,
    CLASSIFIED_PARAGRAPHS_PATHS,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _resolve_ground_truth_path(source: str):
    """Return the per-event-id directory for the requested ground-truth source."""
    if source not in CLASSIFIED_PARAGRAPHS_PATHS:
        raise ValueError(
            f"Unknown ground-truth source {source!r}. "
            f"Expected one of {sorted(CLASSIFIED_PARAGRAPHS_PATHS.keys())}."
        )
    return CLASSIFIED_PARAGRAPHS_PATHS[source]


def load_ground_truth_counts(granularity: str = "paragraph", source: str = "minimax") -> Dict[str, int]:
    """
    Loads ground truth positive counts for each event_id.
    Always uses PARAGRAPHS to determine 'R' (relevant items) logic,
    as the goal is retrieving relevant content.
    """
    counts = {}
    base_path = _resolve_ground_truth_path(source)

    if not base_path.exists():
        logger.warning(f"Warning: Ground truth path {base_path} does not exist.")
        return {}

    files = sorted(
        [f for f in base_path.iterdir() if f.suffix == ".json"],
        key=lambda x: int(x.stem) if x.stem.isdigit() else 0
    )

    for file_path in tqdm(files, desc="Scanning Ground Truth"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            event_id = file_path.stem
            positive_count = sum(1 for item in data if item.get("classification") == 1)
            counts[str(event_id)] = positive_count

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    return counts

def load_ground_truth_labels(source: str = "minimax") -> Dict[str, Dict[str, int]]:
    """
    Loads ground truth classifications mapping event_id -> paragraph_id -> classification.
    """
    labels = {}
    base_path = _resolve_ground_truth_path(source)

    if not base_path.exists():
        logger.warning(f"Warning: Ground truth path {base_path} does not exist.")
        return {}

    files = sorted(
        [f for f in base_path.iterdir() if f.suffix == ".json"],
        key=lambda x: int(x.stem) if x.stem.isdigit() else 0
    )

    for file_path in tqdm(files, desc="Loading Ground Truth Labels"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            event_id = file_path.stem
            labels[event_id] = {
                str(item.get("id")): item.get("classification", -1)
                for item in data if item.get("id")
            }
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    return labels

def compute_avg_precision_recall_at_k(
    cumulative_positives_list: List[List[int]],
    total_relevant_list: List[int],
    k: int,
) -> Tuple[float, float]:
    if len(cumulative_positives_list) != len(total_relevant_list):
        raise ValueError("Mismatch: one total relevant count required per cumulative list.")

    precisions = []
    recalls = []

    for cumulative, total_rel in zip(cumulative_positives_list, total_relevant_list):
        if len(cumulative) < k:
             relevant_at_k = cumulative[-1] if cumulative else 0
        else:
             relevant_at_k = cumulative[k - 1]
             
        precisions.append(relevant_at_k / k)
        recalls.append(relevant_at_k / total_rel if total_rel > 0 else 0)

    return float(np.mean(precisions)), float(np.mean(recalls))

def calc_average_negative_documents(df: pd.DataFrame, k: int) -> float:
    valid_k = k
    while valid_k > 0 and f"top_k_negative_{valid_k}" not in df.columns:
        valid_k -= 1
        
    if valid_k == 0:
        return 0.0

    cumulative_negatives_list = [
        df.iloc[row][f"top_k_negative_{valid_k}"] for row in range(len(df))
    ]
    return float(np.mean(cumulative_negatives_list))

def summarise_results(df: pd.DataFrame, k_values: Iterable[int]) -> dict:
    summary = {}
    
    max_k_col = 0
    for col in df.columns:
        if col.startswith("top_k_positive_"):
            try:
                k_val = int(col.split("_")[-1])
                if k_val > max_k_col:
                    max_k_col = k_val
            except ValueError:
                continue
    
    cumulative_positives_list = [
        [df.iloc[row][f"top_k_positive_{col}"] for col in range(1, max_k_col + 1)]
        for row in range(len(df))
    ]
    
    totals = df["R"].tolist()

    for k in k_values:
        if k > max_k_col:
            continue
        precision_k, recall_k = compute_avg_precision_recall_at_k(cumulative_positives_list, totals, k=k)
        summary[k] = {
            "precision": precision_k,
            "recall": recall_k,
            "avg_negatives": calc_average_negative_documents(df, k),
        }
        
    if "precision_at_R" in df.columns:
        summary["at_R"] = {
            "precision": df["precision_at_R"].mean(),
            "recall": df["recall_at_R"].mean(),
            "avg_negatives": 0.0
        }
        
    return summary

def evaluate_model(
    alias: str,
    model_id: str,
    granularities: List[str],
    k: int = 1000,
    ground_truth_source: str = "minimax",
    table_suffix: str = "",
):
    # 1. Load Model
    logger.info(f"Loading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_kwargs = {}
    try:
        model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=device,
            model_kwargs=model_kwargs
        )
        logger.info(f"Successfully loaded {model_id}.")

    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        return {}

    # Fix for GritLM-7B DynamicCache error
    if "GritLM" in model_id:
        try:
             # GritLM is usually wrapped in a Transformer module at index 0
             if hasattr(model[0], 'auto_model'):
                  model[0].auto_model.config.use_cache = False
                  logger.info("Disabled use_cache for GritLM to prevent DynamicCache errors.")
        except Exception as e:
             logger.warning(f"Could not disable use_cache for GritLM: {e}")

    # if device == "cuda":
    #     model.half()

    # 2. Connect DB
    if not LANCEDB_PATH.exists():
        logger.error(f"LanceDB path {LANCEDB_PATH} not found.")
        return {}
    db = lancedb.connect(LANCEDB_PATH)

    # 3. Load Relations (Queries)
    with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
        relations = json.load(f)
    
    logger.info(f"Encoding {len(relations)} queries...")
    queries = [r["sentence"] for r in relations]
    
    encode_kwargs = {
        "batch_size": 128, 
        "show_progress_bar": True, 
        "convert_to_numpy": True,
        "normalize_embeddings": True
    }
    # Jina v3 and v4 disagree on the task API. v3 wants "retrieval.query";
    # v4 wants task="retrieval" plus prompt_name="query" (query/passage are
    # selected via prompts, not via a suffix on task).
    if "jina-embeddings-v3" in model_id:
         encode_kwargs["task"] = "retrieval.query"
    elif "jina-embeddings-v4" in model_id:
         encode_kwargs["task"] = "retrieval"
         encode_kwargs["prompt_name"] = "query"

    query_embeddings = model.encode(queries, **encode_kwargs)

    summaries = {}
    ground_truth_labels = load_ground_truth_labels(source=ground_truth_source)

    # 4. Evaluate per Granularity
    for granularity in granularities:
        logger.info(f"Evaluating granularity: {granularity}")

        table_name = get_classified_collection_name(alias, granularity) + table_suffix
        try:
             table = db.open_table(table_name)
        except Exception:
             logger.warning(f"Table {table_name} not found in database. Skipping evaluation for this granularity.")
             continue
        ground_truth_counts = load_ground_truth_counts(granularity, source=ground_truth_source)
        
        # Prepare Results DataFrame
        top_k_cols = [f"top_k_positive_{i}" for i in range(1, k + 1)] + \
                     [f"top_k_negative_{i}" for i in range(1, k + 1)]
        columns = ["event_id", "event", "R", "precision_at_R", "recall_at_R"] + top_k_cols
        results_data = []
        problematic_ids = []

        # 5. Search Loop
        logger.info("Searching...")
        
        for i, relation in enumerate(tqdm(relations, desc="Querying")):
            event_id = str(relation["event_id"])
            query_vec = query_embeddings[i]
            R = ground_truth_counts.get(event_id)
            if R is None:
                logger.warning(
                    f"No ground-truth count for event {event_id} (granularity={granularity}); skipping."
                )
                problematic_ids.append(event_id)
                continue
            
            # LanceDB Search
            # Retrieve 10x k items to ensure we find enough unique paragraphs
            buffer_multiplier = 10
            search_limit = k * buffer_multiplier
            
            try:
                # We need to fetch enough candidates to filter down to k unique paragraphs
                # especially for sentence/proposition granularities.
                results = table.search(query_vec).limit(search_limit).to_list()
            except Exception as e:
                logger.error(f"Search failed for {event_id}: {e}")
                problematic_ids.append(event_id)
                continue

            if not results:
                logger.warning(f"No results found for event {event_id}")
                problematic_ids.append(event_id)
                continue

            # Process Results
            count_positives = 0
            count_negatives = 0
            positive_at_r = 0
            
            # Metadata extraction
            # LanceDB results are dicts with 'metadata' field which is a JSON string in our schema
            # We need to parse it.
            
            parsed_metas = []
            for r in results:
                try:
                    meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
                    parsed_metas.append(meta)
                except:
                    parsed_metas.append({})

            # Deduplication Logic (Paragraphs)
            # If granularity != paragraph, we might map multiple sentences to same paragraph
            # We need to count unique paragraphs found.
            seen_pids = set()
            unique_paragraphs = []
            
            for m in parsed_metas:
                # Resolve paragraph ID
                # In sentence metadata, we usually have 'id' (sentence_id), but need paragraph mapping?
                # The indexer stored 'paragraph_id' for paragraphs.
                # For sentences, we just stored 'id' (sentence_id). 
                # Wait, 'id' for sentence is usually "pid_index". So split('_')[0] is likely pid.
                # Let's verify standard ID format: "{base_id}_{i}" where base_id is from source.
                # In sentence loading: base_id = item["id"] (which is paragraph id).
                # So yes, split('_')[0] is paragraph ID.
                
                pid = m.get("paragraph_id")
                if not pid:
                    raw_id = m.get("id", "")
                    if "_" in str(raw_id):
                        pid = str(raw_id).split("_")[0]
                    else:
                        pid = raw_id # fallback
                        
                if pid and pid not in seen_pids:
                    seen_pids.add(pid)
                    unique_paragraphs.append(m)
                    
                    # Stop once we have found k unique paragraphs
                    if len(unique_paragraphs) >= k:
                        break

            # Evaluation metrics on UNIQUE PARAGRAPHS
            # This matches logic in evaluate_models.py
            
            row = {
                "event_id": event_id,
                "event": relation["sentence"],
                "R": R
            }
            
            # Fill Top K columns
            for j in range(k):
                is_positive = False
                if j < len(unique_paragraphs):
                    meta = unique_paragraphs[j]
                    
                    # Lookup actual classification from ground truth
                    # We use paragraph_id as the canonical identifier
                    pid = meta.get("paragraph_id")
                    if not pid:
                        raw_id = meta.get("id", "")
                        pid = str(raw_id).split("_")[0] if "_" in str(raw_id) else raw_id
                        
                    if ground_truth_labels.get(event_id, {}).get(str(pid)) == 1:
                        is_positive = True
                
                if is_positive:
                    count_positives += 1
                    if (j + 1) <= R:
                        positive_at_r += 1
                else:
                    if j < len(unique_paragraphs):
                        # Only count as negative if we actually retrieved something relevant to THIS event
                        # Wait, normal retrieval means anything retrieved is a candidate.
                        # If it belongs to another event, it's a negative for THIS query.
                        # So yes, count_negatives increases.
                        count_negatives += 1
                        
                row[f"top_k_positive_{j+1}"] = count_positives
                row[f"top_k_negative_{j+1}"] = count_negatives

            row["precision_at_R"] = positive_at_r / R if R > 0 else 0.0
            row["recall_at_R"] = positive_at_r / R if R > 0 else 0.0
            
            results_data.append(row)

        # Save Results
        out_dir = RESULTS_PATH / "models" / f"{alias}-lancedb"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not results_data:
            logger.warning(f"No results data generated for {granularity}. Creating empty CSV.")
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(results_data)
        
        df.to_csv(out_dir / f"{granularity}.csv", index=False)
        
        # Summarise
        summaries[granularity] = {
            "metrics": summarise_results(df, k_values=(10, 20, 30, 50, 100, 500, 1000)),
            "problematic_relations": problematic_ids
        }
    
    # Clean up model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model alias or ID")
    parser.add_argument("--granularities", nargs="+", default=["paragraph", "sentence", "proposition"])
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument(
        "--ground-truth",
        choices=sorted(CLASSIFIED_PARAGRAPHS_PATHS.keys()),
        default="minimax",
        help="Per-event classified-paragraphs directory used to compute R and labels.",
    )
    parser.add_argument(
        "--retriever",
        choices=["dense", "colbert"],
        default="dense",
        help="Retrieval backend. ColBERT is reserved for a follow-up; this script handles dense.",
    )
    parser.add_argument(
        "--table-suffix",
        type=str,
        default="",
        help="Append to the LanceDB table name (e.g. '_smoke').",
    )
    args = parser.parse_args()

    if args.retriever == "colbert":
        logger.error(
            "ColBERT MaxSim retrieval is not implemented in evaluate_lancedb.py yet. "
            "Index produced by index_colbert.py is multi-vector and needs a dedicated evaluator. "
            "Re-run with --retriever dense once the dense indexer has finished."
        )
        sys.exit(2)

    # Resolve Alias
    model_id = args.model
    alias = args.model
    if args.model in MODEL_IDS:
        model_id = MODEL_IDS[args.model]
    else:
        for a, mid in MODEL_IDS.items():
            if mid == args.model:
                alias = a
                break

    summaries = evaluate_model(
        alias=alias,
        model_id=model_id,
        granularities=args.granularities,
        k=args.k,
        ground_truth_source=args.ground_truth,
        table_suffix=args.table_suffix,
    )
    
    print("\n=== Evaluation Summary ===")
    for gran, data in summaries.items():
        print(f"\nGranularity: {gran}")
        if "at_R" in data["metrics"]:
            m = data["metrics"]["at_R"]
            print(f"  @R: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")
        
        for k in [10, 50, 100]:
            if k in data["metrics"]:
                m = data["metrics"][k]
                print(f"  k={k}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")
