"""Where does a target labeler disagree with the human gold?

Diagnostic for the gpt-5.2:medium reclassification prompt. Joins, per chunk:
- the relation query (abstract entity / constant entity / relation) + paragraph text,
- the target labeler's label (default gpt-5.2:medium, the cheap config we want to fix),
- the human gold label + its brief_reason,
- Opus-4.8's label + brief_reason (the strongest single labeler — the adjudicator),
- the frozen frontier consensus label,
- the decomposed sub-checks for the target (which of the 5 criteria flipped), when present,
- a bibliography/reference-list heuristic flag.

Emits:
- ``reports/<target>_disagreements.md`` — human-readable, grouped by error category;
- ``summary/judge_validation/short/<target>_disagreements.json`` — full machine dump
  (every contested chunk, all labelers, reasons, flags);
- ``summary/judge_validation/short/contested_ids.json`` — the FROZEN contested id set
  (non-unanimous across the 11 model labelers), so prompt-variant before/after comparisons
  use an identical denominator.

Reuses frontier_cost_report (load_predictions / majority_vote / pct_agreement /
CONSENSUS_LABELERS) so labels are loaded identically to the report.

Usage:
    uv run python src/create_dataset/classifying_data/disagreement_analysis.py
    uv run python src/create_dataset/classifying_data/disagreement_analysis.py \
        --target-model gpt-5.2 --target-effort medium
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.config import PROJECT_ROOT, SUMMARY_PATH  # noqa: E402

try:
    from . import frontier_cost_report as report_mod
except ImportError:  # run as a plain script
    import frontier_cost_report as report_mod  # type: ignore


RAW_DIR = PROJECT_ROOT / "data/judge_validation/short"
DECOMP_DIR = PROJECT_ROOT / "data/judge_validation/decomposed"
REPORT_DIR = SUMMARY_PATH / "judge_validation" / "short"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAMPLE_PATH = RAW_DIR / "_sample.json"
HUMAN_PATH = RAW_DIR / "human__gold.jsonl"
OPUS_PATH = RAW_DIR / f"{report_mod.OPUS_LABELER}.jsonl"

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_sample_by_id() -> dict[str, dict]:
    """{id: sample_rec} — source of paragraph content + query_info."""
    return {r["id"]: r for r in json.loads(SAMPLE_PATH.read_text())}


def base_model_configs() -> list[tuple[str, str]]:
    """The (model, effort) configs on disk in the base dir, EXCLUDING human gold / batches."""
    out: list[tuple[str, str]] = []
    for f in sorted(RAW_DIR.glob("*__*.jsonl")):
        if f.name.startswith("batch_") or f.name.startswith("human__"):
            continue
        model, _, effort = f.stem.rpartition("__")
        if model:
            out.append((model, effort))
    return out


def load_model_labelers(sample: list[dict]) -> dict[str, dict[str, int]]:
    """11 model labelers keyed by id: minimax + the gpt/deepseek configs + opus (-1 dropped)."""
    return report_mod.load_predictions(RAW_DIR, base_model_configs(), sample)


def _load_jsonl_field(path: Path, field: str) -> dict[str, object]:
    """{id: rec[field]} from a jsonl, excluding missing/blank values."""
    out: dict[str, object] = {}
    for rec in report_mod._read_jsonl(path):
        rid = rec.get("id")
        if rid and field in rec:
            out[rid] = rec[field]
    return out


def load_human_gold() -> tuple[dict[str, int], dict[str, str]]:
    """({id: 0/1}, {id: brief_reason}) from human__gold.jsonl (0/1 only)."""
    labels: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for rec in report_mod._read_jsonl(HUMAN_PATH):
        rid = rec.get("id")
        if rid and rec.get("prediction") in (0, 1):
            labels[rid] = int(rec["prediction"])
            reasons[rid] = (rec.get("brief_reason") or "").strip()
    return labels, reasons


def load_decomposed_checks(model: str, effort: str) -> dict[str, dict]:
    """{id: {valid_instance, constant_present, relation_asserted, direction_ok, asserted_as_fact}}."""
    path = DECOMP_DIR / report_mod.config_filename(model, effort)
    out: dict[str, dict] = {}
    for rec in report_mod._read_jsonl(path):
        rid = rec.get("id")
        if not rid:
            continue
        try:
            checks = json.loads(rec.get("raw", "") or "{}").get("checks")
        except (json.JSONDecodeError, AttributeError):
            checks = None
        if isinstance(checks, dict):
            out[rid] = checks
    return out


# --------------------------------------------------------------------------- #
# Contested set + bibliography heuristic
# --------------------------------------------------------------------------- #
def contested_ids(model_labelers: dict[str, dict[str, int]]) -> set[str]:
    """Ids where the model labelers are NOT unanimous (both 0 and 1 appear)."""
    all_ids: set[str] = set()
    for preds in model_labelers.values():
        all_ids |= set(preds)
    contested: set[str] = set()
    for rid in all_ids:
        votes = {preds[rid] for preds in model_labelers.values() if rid in preds}
        if len(votes) > 1:
            contested.add(rid)
    return contested


_REF_HEADING = re.compile(
    r"(?im)^\s*(references|bibliography|further reading|external links|"
    r"notes|citations|works cited|sources|see also)\s*:?\s*$"
)
_CITE_MARKUP = re.compile(r"(?i)(\{\{\s*cite|</?ref[\s>]|\bdoi:\s|\barxiv:)")
_IDENT = re.compile(r"(?i)\b(isbn|issn)\b")
_BRACKET_CITE = re.compile(r"\[\d{1,3}\]")
_MASTHEAD = re.compile(
    r"(?i)(published by .{0,60}\b(press|publishing|university press)\b|"
    r"\bretrieved\s+\d{4}\b|\b(vol\.|no\.)\s*\d+)"
)


def is_reference_like(content: str) -> tuple[bool, list[str]]:
    """Heuristic: does this chunk read like a citation list / bibliography / masthead?"""
    signals: list[str] = []
    if _REF_HEADING.search(content):
        signals.append("ref-heading")
    if _CITE_MARKUP.search(content):
        signals.append("cite-markup")
    if _IDENT.search(content):
        signals.append("isbn/issn")
    if len(_BRACKET_CITE.findall(content)) >= 3:
        signals.append("dense-[n]-markers")
    if _MASTHEAD.search(content):
        signals.append("masthead/catalogue")
    return (bool(signals), signals)


# --------------------------------------------------------------------------- #
# Build entries
# --------------------------------------------------------------------------- #
def _sides_with(target: int, human: int, opus: int | None) -> str:
    if opus is None:
        return "opus-missing"
    if opus == target and opus != human:
        return "gpt"  # opus agrees with the target against the human -> candidate human error
    if opus == human and opus != target:
        return "human"  # opus agrees with the human -> clear target error
    return "split/none"


def build_disagreements(
    target_key: str,
    sample: dict[str, dict],
    model_labelers: dict[str, dict[str, int]],
    human: dict[str, int],
    human_reasons: dict[str, str],
    opus_reasons: dict[str, str],
    consensus: dict[str, int],
    decomp: dict[str, dict],
) -> list[dict]:
    target = model_labelers[target_key]
    opus = model_labelers.get(report_mod.OPUS_LABELER, {})
    shared = sorted(set(target) & set(human))
    entries: list[dict] = []
    for rid in shared:
        if target[rid] == human[rid]:
            continue
        rec = sample[rid]
        q = rec.get("query_info", {})
        content = rec.get("content", "")
        ref_flag, ref_signals = is_reference_like(content)
        opus_lbl = opus.get(rid)
        entries.append({
            "id": rid,
            "event_id": rec.get("event_id"),
            "category": "false_positive" if target[rid] == 1 else "false_negative",
            "query": {
                "sentence": q.get("sentence", ""),
                "abstract_entity_type": q.get("abstract_entity_type", ""),
                "constant_entity": q.get("constant_entity", ""),
                "relation": q.get("relation", ""),
            },
            "target_label": target[rid],
            "human_label": human[rid],
            "human_reason": human_reasons.get(rid, ""),
            "opus_label": opus_lbl,
            "opus_reason": opus_reasons.get(rid, ""),
            "consensus_label": consensus.get(rid),
            "minimax_label": model_labelers.get(report_mod.MINIMAX_LABELER, {}).get(rid),
            "decomposed_checks": decomp.get(rid),
            "is_reference_like": ref_flag,
            "ref_signals": ref_signals,
            "opus_sides_with": _sides_with(target[rid], human[rid], opus_lbl),
            "content": content,
        })
    return entries


def contested_overview(
    model_labelers: dict[str, dict[str, int]],
    human: dict[str, int],
    contested: set[str],
) -> list[dict]:
    """Per-labeler agreement with the human gold, full vs on the contested subset."""
    rows: list[dict] = []
    for name in sorted(model_labelers):
        preds = model_labelers[name]
        p_full, n_full = report_mod.pct_agreement(preds, human)
        sub = {i: v for i, v in preds.items() if i in contested}
        p_con, n_con = report_mod.pct_agreement(sub, human)
        rows.append({
            "labeler": name,
            "%agree_human_full": round(p_full * 100, 1) if n_full else None,
            "n_full": n_full,
            "%agree_human_contested": round(p_con * 100, 1) if n_con else None,
            "n_contested": n_con,
        })
    rows.sort(key=lambda r: (r["%agree_human_contested"] is None, -(r["%agree_human_contested"] or 0)))
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def summary_counts(entries: list[dict]) -> dict:
    fp = [e for e in entries if e["category"] == "false_positive"]
    fn = [e for e in entries if e["category"] == "false_negative"]
    return {
        "n_disagreements": len(entries),
        "false_positive": len(fp),
        "false_negative": len(fn),
        "reference_like": sum(1 for e in entries if e["is_reference_like"]),
        "fp_reference_like": sum(1 for e in fp if e["is_reference_like"]),
        "clear_target_errors": sum(1 for e in entries if e["opus_sides_with"] == "human"),
        "candidate_human_errors": sum(1 for e in entries if e["opus_sides_with"] == "gpt"),
        "opus_missing": sum(1 for e in entries if e["opus_sides_with"] == "opus-missing"),
        "consensus_missing": sum(1 for e in entries if e["consensus_label"] is None),
    }


def _checks_str(checks: dict | None) -> str:
    if not checks:
        return "n/a"
    order = ["valid_instance", "constant_present", "relation_asserted", "direction_ok", "asserted_as_fact"]
    return ", ".join(f"{k}={'T' if checks.get(k) else 'F'}" for k in order if k in checks)


def _entry_md(e: dict) -> str:
    q = e["query"]
    lbl = lambda v: {1: "POSITIVE", 0: "negative", None: "—"}[v]
    return "\n".join([
        f"### `{e['id']}` (event {e['event_id']}) — {e['category']}",
        "",
        f"- **Query:** *{q['abstract_entity_type']}* — **{q['relation']}** — *{q['constant_entity']}*",
        f"- **Labels:** target=**{lbl(e['target_label'])}** · human=**{lbl(e['human_label'])}** · "
        f"opus={lbl(e['opus_label'])} · consensus={lbl(e['consensus_label'])} · minimax={lbl(e['minimax_label'])}",
        f"- **Adjudication:** opus sides with **{e['opus_sides_with']}**"
        + ("  ⚠️ _candidate wrong human label_" if e["opus_sides_with"] == "gpt" else ""),
        f"- **Reference-like:** {e['is_reference_like']} {e['ref_signals'] or ''}",
        f"- **Decomposed sub-checks (target):** {_checks_str(e['decomposed_checks'])}",
        f"- **Human note:** {e['human_reason'] or '_(none)_'}",
        f"- **Opus reasoning:** {e['opus_reason'] or '_(none)_'}",
        "",
        "> " + e["content"].replace("\n", "\n> "),
        "",
    ])


def render_markdown(target_key: str, entries: list[dict], counts: dict,
                    overview: list[dict], n_contested: int) -> str:
    import pandas as pd

    fp = [e for e in entries if e["category"] == "false_positive"]
    fn = [e for e in entries if e["category"] == "false_negative"]
    cand_human = [e for e in entries if e["opus_sides_with"] == "gpt"]

    L: list[str] = []
    L.append(f"# {target_key} vs human gold — disagreement analysis\n")
    L.append(
        f"Target labeler **{target_key}** disagrees with the human gold on "
        f"**{counts['n_disagreements']}** of the shared chunks "
        f"({counts['false_positive']} false-positive, {counts['false_negative']} false-negative). "
        f"Contested set (non-unanimous across the 11 model labelers): **{n_contested}** chunks.\n"
    )
    L.append("## Summary\n")
    L.append("| metric | count |\n|---|---:|")
    for k, v in counts.items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append(
        "- **clear_target_errors** = Opus agrees with the human against the target → genuine "
        "prompt failures to fix.\n"
        "- **candidate_human_errors** = Opus agrees with the target against the human → the "
        "human label may be wrong; review, do NOT optimize toward these.\n"
    )

    L.append("\n## A. Agreement with the human gold (per labeler)\n")
    L.append("Full overlap vs restricted to the contested subset.\n")
    L.append(pd.DataFrame(overview).to_markdown(index=False))

    L.append(f"\n## B. False positives — target said POSITIVE, human said negative ({len(fp)})\n")
    for e in fp:
        L.append(_entry_md(e))
    L.append(f"\n## C. False negatives — target said negative, human said POSITIVE ({len(fn)})\n")
    for e in fn:
        L.append(_entry_md(e))

    L.append(f"\n## D. Candidate wrong human labels (Opus + target agree against human) ({len(cand_human)})\n")
    if cand_human:
        L.append("Review these manually — fixing the gold here raises agreement without prompt changes.\n")
        L.append("| id | query | human | target/opus | ref-like |\n|---|---|---|---|---|")
        for e in cand_human:
            qd = e["query"]
            L.append(
                f"| `{e['id']}` | {qd['abstract_entity_type']} — {qd['relation']} — {qd['constant_entity']} "
                f"| {e['human_label']} | {e['target_label']} | {e['is_reference_like']} |"
            )
    else:
        L.append("_None._\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-model", default="gpt-5.2")
    ap.add_argument("--target-effort", default="medium")
    args = ap.parse_args()
    target_key = report_mod.config_key(args.target_model, args.target_effort)

    sample_list = json.loads(SAMPLE_PATH.read_text())
    sample = {r["id"]: r for r in sample_list}
    model_labelers = load_model_labelers(sample_list)
    if target_key not in model_labelers:
        raise SystemExit(f"target {target_key} not found among {sorted(model_labelers)}")
    human, human_reasons = load_human_gold()
    opus_reasons = {k: str(v) for k, v in _load_jsonl_field(OPUS_PATH, "brief_reason").items()}
    decomp = load_decomposed_checks(args.target_model, args.target_effort)

    consensus_names = [n for n in report_mod.CONSENSUS_LABELERS if n in model_labelers]
    consensus = report_mod.majority_vote(model_labelers, consensus_names)
    contested = contested_ids(model_labelers)

    entries = build_disagreements(
        target_key, sample, model_labelers, human, human_reasons, opus_reasons, consensus, decomp
    )
    counts = summary_counts(entries)
    overview = contested_overview(model_labelers, human, contested)

    # frozen contested id set (for apples-to-apples variant comparisons)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "contested_ids.json").write_text(json.dumps(sorted(contested), indent=2))

    safe = target_key.replace(":", "_").replace(".", "")
    dump = {
        "target": target_key,
        "counts": counts,
        "n_contested": len(contested),
        "n_decomposed_available": len(decomp),
        "contested_overview": overview,
        "disagreements": entries,
    }
    json_path = REPORT_DIR / f"{safe}_disagreements.json"
    json_path.write_text(json.dumps(dump, indent=2))
    md_path = REPORTS_DIR / f"{safe}_disagreements.md"
    md_path.write_text(render_markdown(target_key, entries, counts, overview, len(contested)))

    print(f"target={target_key}  disagreements={counts['n_disagreements']} "
          f"(FP={counts['false_positive']}, FN={counts['false_negative']})  "
          f"contested={len(contested)}  decomposed_available={len(decomp)}")
    print(f"  clear target errors (opus=human): {counts['clear_target_errors']}  "
          f"candidate human errors (opus=target): {counts['candidate_human_errors']}")
    print(f"  reference-like disagreements: {counts['reference_like']} "
          f"(of which FP: {counts['fp_reference_like']})")
    print(f"-> {md_path}")
    print(f"-> {json_path}")
    print(f"-> {REPORT_DIR / 'contested_ids.json'}")


if __name__ == "__main__":
    main()
