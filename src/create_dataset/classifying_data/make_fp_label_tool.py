"""Generate a self-contained HTML tool for hand-labeling the FP pilot (v3 bits schema).

Reads the pilot selection (``data/fp_taxonomy/validation/pilot_pairs.json``, ``human: true``
subset by default), resolves paragraph texts + query metadata, and embeds everything
into one offline ``label_fp_taxonomy.html``. For each paragraph the annotator answers
three INDEPENDENT yes/no presence questions —

    1. instance_present  — a concrete instance of the abstract entity type appears
    2. relation_present  — the relation is asserted, as fact, between ANY two entities
    3. constant_present  — the constant entity is referred to

(2^3 = 8 cells; even yes/yes/yes can be a judged-negative when the relation does not
bind the queried pair) — and exports a ready-to-ingest ``fp_human__gold.jsonl``.
Autosaves to localStorage (resumable). Deliberately **blind**: no classifier output is
embedded, so the human labels are an independent reference for
``fp_taxonomy_agreement.py``.

    uv run python src/create_dataset/classifying_data/make_fp_label_tool.py

Then open data/fp_taxonomy/validation/label_fp_taxonomy.html, label, Export, and move the
downloaded fp_human__gold.jsonl into data/fp_taxonomy/validation/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.create_dataset.classifying_data.batch_api.label_fp_taxonomy_batch import (  # noqa: E402
    PILOT_DIR,
    PILOT_PAIRS_PATH,
    _texts_for_event,
    load_predictions_for_event,
)
from src.create_dataset.classifying_data.batch_api.label_frontier_batch import (  # noqa: E402
    load_query_meta,
)
from src.create_dataset.classifying_data.fp_taxonomy_prompt import (  # noqa: E402
    BITS,
    FP_TAXONOMY_PROMPT_VERSION,
    build_fp_taxonomy_system_prompt,
)

OUT_HTML = PILOT_DIR / "label_fp_taxonomy.html"
# Reveal judges with per-criterion reasoning: Opus 4.8 agents and a gpt-5.2
# reasoned Batch API run (fp_gpt_reasoned_batch.py), both blind to the pipeline.
OPUS_BITS_PATH = PILOT_DIR / "opus_bits.jsonl"
GPT_REASONED_PATH = PILOT_DIR / "gpt_bits_reasoned.jsonl"


def _load_judge_bits(path: Path) -> dict[str, dict]:
    """{chunk_id: judge record} from a reveal-judge jsonl (empty if absent)."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if all(isinstance(r.get(b), bool) for b in BITS):
                out[r["id"]] = r
    return out


def build_items(pairs: list[dict], include_reveal: bool = True) -> list[dict]:
    """One display record per pair: query fields + paragraph text.

    With ``include_reveal`` (default), each item carries a ``_reveal`` object with
    the Opus agent's bits + per-criterion reasoning and the gpt-5.2 pipeline's bits
    (current prompt version). The tool shows it only AFTER the item is fully
    labeled, mirroring the binary gold round's pause-and-review flow.
    """
    query_meta = load_query_meta()
    opus = _load_judge_bits(OPUS_BITS_PATH) if include_reveal else {}
    gpt_reasoned = _load_judge_bits(GPT_REASONED_PATH) if include_reveal else {}
    reason_keys = (*BITS, "instance_reasoning", "relation_reasoning", "constant_reasoning")
    by_event: dict[int, list[dict]] = {}
    for p in pairs:
        by_event.setdefault(p["event_id"], []).append(p)

    items: list[dict] = []
    for eid in sorted(by_event):
        q = query_meta[eid]
        texts = _texts_for_event(eid, {p["id"] for p in by_event[eid]})
        gpt_preds = load_predictions_for_event(eid) if include_reveal else {}
        for p in by_event[eid]:
            reveal = None
            if include_reveal:
                o = opus.get(p["id"])
                gr = gpt_reasoned.get(p["id"])
                g = gpt_preds.get(p["id"])
                if g is not None and (g.get("prompt_version") != FP_TAXONOMY_PROMPT_VERSION
                                      or not all(isinstance(g.get(b), bool) for b in BITS)):
                    g = None
                reveal = {
                    "opus": ({k: o.get(k, "") for k in reason_keys} if o else None),
                    "gpt": ({k: gr.get(k, "") for k in reason_keys} if gr else None),
                    "gpt_pipeline": ({b: g[b] for b in BITS} if g else None),
                }
                if all(v is None for v in reveal.values()):
                    reveal = None
            contested = bool(
                reveal and reveal["opus"] and reveal["gpt"]
                and any(reveal["opus"][b] != reveal["gpt"][b] for b in BITS)
            )
            items.append({
                "id": p["id"],
                "event_id": eid,
                "content": texts.get(p["id"], "(text missing)"),
                "sentence": q.get("sentence", ""),
                "abstract_entity_type": q.get("abstract_entity_type", ""),
                "constant_entity": q.get("constant_entity", ""),
                "relation": q.get("relation", ""),
                "_reveal": reveal,
                "_contested": contested,
            })
    return items


def _embed(blob: dict) -> str:
    return (
        json.dumps(blob, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="embed all pilot pairs (default: only the human-subset pairs)")
    ap.add_argument("--contested-first", action="store_true",
                    help="order items with Opus-vs-gpt contested pairs first "
                         "(stable (event_id, id) order within each group)")
    ap.add_argument("--no-reveal", action="store_true",
                    help="strip ALL model labels/reasoning (for public hosting)")
    ap.add_argument("--out", default=None, help=f"output HTML path (default: {OUT_HTML})")
    ap.add_argument("--key", default="fp_bits_v3", help="localStorage progress key")
    args = ap.parse_args()

    if not PILOT_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Pilot selection not found: {PILOT_PAIRS_PATH} — run "
                                "label_fp_taxonomy_batch select-pilot first.")
    pairs = json.loads(PILOT_PAIRS_PATH.read_text())
    if not args.all:
        pairs = [p for p in pairs if p.get("human")]

    include_reveal = not args.no_reveal
    items = build_items(pairs, include_reveal=include_reveal)
    if args.contested_first:
        items.sort(key=lambda it: not it["_contested"])  # stable: keeps page order within groups
        n_con = sum(1 for it in items if it["_contested"])
        print(f"Contested-first ordering: {n_con} contested pairs lead the list.")
    n_opus = sum(1 for it in items if it.get("_reveal") and it["_reveal"]["opus"])
    n_gr = sum(1 for it in items if it.get("_reveal") and it["_reveal"]["gpt"])
    n_gp = sum(1 for it in items if it.get("_reveal") and it["_reveal"]["gpt_pipeline"])
    blob = _embed({"rubric": build_fp_taxonomy_system_prompt(), "items": items,
                   "store_key": args.key,
                   "reveal_available": include_reveal and (n_opus + n_gr + n_gp) > 0})
    out = Path(args.out) if args.out else OUT_HTML
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_HTML_TEMPLATE.replace("__BLOB__", blob), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  {len(items)} pairs embedded; 3 yes/no questions each.")
    if include_reveal:
        print(f"  Post-commit reveal: Opus judge+reasoning on {n_opus}, gpt-5.2 "
              f"reasoned on {n_gr}, gpt-5.2 pipeline bits on {n_gp}.")
    else:
        print("  NO model labels embedded (safe to host publicly).")
    print("  Open in a browser, label, Export -> fp_human__gold.jsonl; move it into")
    print(f"  {PILOT_DIR} and run fp_taxonomy_agreement.py.")


# --------------------------------------------------------------------------- #
# Self-contained HTML (no external deps). __BLOB__ is replaced with the data.
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FP presence labeling</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1f242e; --fg:#e6e9ef; --muted:#9aa3b2;
    --line:#2a3140; --accent:#4c8bf5; --yes:#1f9d57; --no:#d2453f;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f7f9; --panel:#fff; --panel2:#eef1f5; --fg:#1a1d23; --muted:#5b6472;
            --line:#dde2ea; --accent:#2563eb; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--line);
           padding:10px 16px; }
  .bar { display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  #progress { font-weight:700; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .pbar { flex:1; min-width:120px; height:8px; background:var(--panel2); border-radius:6px; overflow:hidden; }
  #fill { height:100%; width:0%; background:var(--accent); transition:width .15s; }
  button { font:inherit; cursor:pointer; border:1px solid var(--line); background:var(--panel2);
           color:var(--fg); padding:6px 10px; border-radius:8px; }
  button:hover { border-color:var(--accent); }
  main { max-width:940px; margin:0 auto; padding:16px; }
  details#rubricBox { background:var(--panel); border:1px solid var(--line); border-radius:10px;
                      padding:8px 12px; margin-bottom:14px; }
  details#rubricBox summary { cursor:pointer; color:var(--accent); font-weight:600; }
  #rubric { white-space:pre-wrap; font-size:13px; color:var(--muted); margin:10px 0 4px; }
  .qpanel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .qrow { display:flex; gap:10px; padding:3px 0; }
  .qrow .k { color:var(--muted); min-width:160px; font-size:13px; text-transform:uppercase; letter-spacing:.03em; }
  .relation { font-weight:700; color:var(--accent); }
  .question { margin:14px 2px 8px; font-size:15px; color:var(--muted); }
  .content { background:var(--panel); border:1px solid var(--line); border-radius:10px;
             padding:16px 18px; max-height:38vh; overflow:auto; white-space:pre-wrap; font-size:15.5px; }
  .qs { display:flex; flex-direction:column; gap:8px; margin:16px 0 6px; }
  .qitem { display:flex; align-items:center; gap:12px; background:var(--panel);
           border:2px solid var(--line); border-radius:10px; padding:10px 14px; }
  .qitem.current { border-color:var(--accent); }
  .qitem .qname { font-weight:700; min-width:210px; }
  .qitem .qdesc { color:var(--muted); font-size:12.8px; flex:1; }
  .qitem .yn { display:flex; gap:8px; }
  .qitem .yn button { min-width:74px; padding:9px 12px; font-weight:700; }
  .qitem .yes { border-color:var(--yes); } .qitem .yes.sel { background:var(--yes); color:#fff; }
  .qitem .no  { border-color:var(--no);  } .qitem .no.sel  { background:var(--no);  color:#fff; }
  .notes input { width:100%; padding:9px 11px; border-radius:8px; border:1px solid var(--line);
                 background:var(--panel); color:var(--fg); font:inherit; margin-top:8px; }
  .toggle { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:13px; }
  .reveal { margin-top:12px; min-height:8px; }
  .revpanel { background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .revhead { margin-bottom:8px; font-size:14.5px; }
  .revq { margin:9px 0; padding:8px 10px; background:var(--panel); border:1px solid var(--line);
          border-radius:8px; }
  .revq .rq { font-weight:700; font-size:13.5px; }
  .revq .rr { color:var(--muted); font-size:13px; margin-top:4px; line-height:1.45; }
  .agree { color:var(--yes); font-weight:700; } .disagree { color:var(--no); font-weight:700; }
  .muted { color:var(--muted); }
  .revgpt { margin-top:8px; font-size:13.5px; }
  .cont { margin-top:12px; width:100%; padding:13px; font-weight:700; border-color:var(--accent);
          color:var(--accent); }
  footer { max-width:940px; margin:0 auto; padding:10px 16px 40px; color:var(--muted); font-size:12.5px; }
  kbd { background:var(--panel2); border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:12px; }
  .done-banner { background:var(--yes); color:#fff; padding:8px 12px; border-radius:8px; margin-bottom:12px;
                 display:none; }
</style>
</head>
<body>
<header>
  <div class="bar">
    <div id="progress">0 / 0</div>
    <div class="pbar"><div id="fill"></div></div>
    <button id="prevBtn" title="k / ←">◀ Prev</button>
    <button id="nextBtn" title="j / →">Next ▶</button>
    <button id="jumpBtn" title="next unlabeled">↻ Unlabeled</button>
    <button id="exportBtn">⬇ Export jsonl</button>
    <button id="importBtn">⬆ Import</button>
    <input id="importFile" type="file" accept=".jsonl,.json,application/json" hidden>
    <label class="toggle"><input type="checkbox" id="revealChk"> pause &amp; show AI judges after each</label>
  </div>
</header>
<main>
  <div class="done-banner" id="doneBanner"></div>
  <details id="rubricBox"><summary>Rubric — the exact matching rules (click to expand)</summary><pre id="rubric"></pre></details>
  <section id="card">
    <div class="qpanel">
      <div class="qrow"><span class="k">Relation (r)</span><span id="relation" class="relation"></span></div>
      <div class="qrow"><span class="k">Abstract entity type (T)</span><span id="aet"></span></div>
      <div class="qrow"><span class="k">Constant entity (e)</span><span id="ce"></span></div>
      <div class="qrow"><span class="k">Query sentence</span><span id="sentence"></span></div>
    </div>
    <div class="question">Answer the three <b>independent</b> presence questions from the paragraph alone.
      <kbd>y</kbd>/<kbd>n</kbd> fills the highlighted question and moves on.</div>
    <div id="content" class="content"></div>
    <div class="qs">
      <div class="qitem" id="qi0">
        <span class="qname">1 · Instance of T present?</span>
        <span class="qdesc">a concrete, <b>named</b> instance of the abstract type — not the bare category, a statistic, or an associated supplier / parent / product / person</span>
        <span class="yn"><button class="yes" id="y0">Yes</button><button class="no" id="n0">No</button></span>
      </div>
      <div class="qitem" id="qi1">
        <span class="qname">2 · Relation present?</span>
        <span class="qdesc">r (or a substance-preserving paraphrase) asserted <b>as fact between ANY two entities</b> — planned / negated / terminated r, planted verbs, and citation-list co-occurrence don't count</span>
        <span class="yn"><button class="yes" id="y1">Yes</button><button class="no" id="n1">No</button></span>
      </div>
      <div class="qitem" id="qi2">
        <span class="qname">3 · Constant entity present?</span>
        <span class="qdesc">e referred to, even as background — a named <b>member</b> of a category-valued e counts; a mere sub-part / neighbor of a specific named e doesn't</span>
        <span class="yn"><button class="yes" id="y2">Yes</button><button class="no" id="n2">No</button></span>
      </div>
    </div>
    <div class="notes"><input id="notes" placeholder="optional note — saved with the label"></div>
    <div class="reveal" id="reveal"></div>
  </section>
</main>
<footer>
  <span id="meta"></span> ·
  Keys: <kbd>y</kbd>/<kbd>n</kbd> answer highlighted question · direct set: <kbd>q</kbd>/<kbd>a</kbd> Q1,
  <kbd>w</kbd>/<kbd>s</kbd> Q2, <kbd>e</kbd>/<kbd>d</kbd> Q3 (top row = yes, home row = no) ·
  <kbd>j</kbd>/<kbd>k</kbd> or <kbd>←</kbd><kbd>→</kbd> navigate · <kbd>r</kbd> toggle the AI-judge reveal.
  An item counts as labeled when all three are answered; the judges appear only after your third answer.
  Progress autosaves in this browser.
</footer>

<script id="blob" type="application/json">__BLOB__</script>
<script>
(function () {
  var DATA = JSON.parse(document.getElementById("blob").textContent);
  var ITEMS = DATA.items;
  var KEY = DATA.store_key || "fp_bits_v3", KEY_IDX = KEY + "_idx";
  var QKEYS = ["instance_present", "relation_present", "constant_present"];
  var labels = {}; try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  var idx = parseInt(localStorage.getItem(KEY_IDX) || "0", 10) || 0;
  if (idx < 0 || idx >= ITEMS.length) idx = 0;

  var $ = function (id) { return document.getElementById(id); };
  $("rubric").textContent = DATA.rubric;
  var QLABEL = ["1 · Instance of T", "2 · Relation", "3 · Constant entity"];
  var KEY_REVEAL = KEY + "_reveal";
  var revealChk = $("revealChk");
  var _rv = localStorage.getItem(KEY_REVEAL);
  revealChk.checked = (_rv === null ? true : _rv === "1");   // default ON: pause & show AI
  if (DATA.reveal_available === false) {
    revealChk.checked = false;
    var rl = revealChk.closest("label"); if (rl) rl.style.display = "none";
  }
  function escapeHtml(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function ynWord(v) { return v === true ? "Yes" : v === false ? "No" : "—"; }
  function tag(v, you) {
    if (v === true || v === false)
      return "<span class='" + (v === you ? "agree" : "disagree") + "'>" + ynWord(v) + "</span>";
    return "<span class='muted'>—</span>";
  }

  function save() {
    localStorage.setItem(KEY, JSON.stringify(labels));
    localStorage.setItem(KEY_IDX, String(idx));
  }
  function isComplete(rec) {
    return !!rec && QKEYS.every(function (k) { return rec[k] === true || rec[k] === false; });
  }
  function firstOpen(rec) {
    for (var i = 0; i < QKEYS.length; i++)
      if (!rec || (rec[QKEYS[i]] !== true && rec[QKEYS[i]] !== false)) return i;
    return -1;
  }
  function counts() {
    var done = 0;
    ITEMS.forEach(function (it) { if (isComplete(labels[it.id])) done++; });
    return { done: done, total: ITEMS.length };
  }

  function render() {
    var it = ITEMS[idx];
    var rec = labels[it.id] || {};
    $("relation").textContent = it.relation || "(none)";
    $("aet").textContent = it.abstract_entity_type;
    $("ce").textContent = it.constant_entity;
    $("sentence").textContent = it.sentence;
    $("content").textContent = it.content;
    $("content").scrollTop = 0;
    $("notes").value = rec.notes || "";
    var open = firstOpen(rec);
    for (var i = 0; i < 3; i++) {
      $("y" + i).classList.toggle("sel", rec[QKEYS[i]] === true);
      $("n" + i).classList.toggle("sel", rec[QKEYS[i]] === false);
      $("qi" + i).classList.toggle("current", i === open);
    }
    var c = counts();
    $("progress").textContent = c.done + " / " + c.total;
    $("fill").style.width = (100 * c.done / c.total) + "%";
    $("meta").textContent = "#" + (idx + 1) + " · id " + it.id.slice(0, 8) + " · event " + it.event_id
      + (it._contested ? " · ⚡ judges disagree here" : "");
    var banner = $("doneBanner");
    if (c.done === c.total) {
      banner.style.display = "block";
      banner.textContent = "All " + c.total + " labeled — click Export jsonl.";
    } else banner.style.display = "none";

    // After all three answers (and if the pause toggle is on), show the AI judges:
    // Opus per-question verdicts + reasoning, and the gpt-5.2 pipeline bits.
    var rv = $("reveal"); rv.innerHTML = "";
    var showAI = (DATA.reveal_available !== false) && revealChk.checked
                 && isComplete(rec) && it._reveal;
    if (showAI) {
      var o = it._reveal.opus, g = it._reveal.gpt, gp = it._reveal.gpt_pipeline;
      var RKEYS = ["instance_reasoning", "relation_reasoning", "constant_reasoning"];
      var gbits = g || gp;   // reasoned run preferred; bare pipeline bits as fallback
      var h = "<div class='revpanel'>";
      h += "<div class='revhead'>Your answers vs the AI judges "
         + "(<span class='agree'>green</span> agrees with you, <span class='disagree'>red</span> differs):</div>";
      for (var i = 0; i < 3; i++) {
        var head = QLABEL[i] + " — you: <b>" + ynWord(rec[QKEYS[i]]) + "</b>";
        if (o) head += " · Opus: " + tag(o[QKEYS[i]], rec[QKEYS[i]]);
        if (gbits) head += " · gpt-5.2: " + tag(gbits[QKEYS[i]], rec[QKEYS[i]]);
        var body = "";
        if (o && o[RKEYS[i]]) body += "<div class='rr'><b>Opus:</b> " + escapeHtml(o[RKEYS[i]]) + "</div>";
        if (g && g[RKEYS[i]]) body += "<div class='rr'><b>gpt-5.2:</b> " + escapeHtml(g[RKEYS[i]]) + "</div>";
        h += "<div class='revq'><div class='rq'>" + head + "</div>" + body + "</div>";
      }
      if (!o) h += "<div class='revgpt muted'>No Opus judge for this item yet.</div>";
      if (!g) h += "<div class='revgpt muted'>No gpt-5.2 reasoning for this item yet"
                 + (gp ? " (bits above are the bare pipeline run)." : ".") + "</div>";
      if (g && gp) {
        var diffs = [];
        for (var j = 0; j < 3; j++) if (g[QKEYS[j]] !== gp[QKEYS[j]])
          diffs.push(QLABEL[j].split(" · ")[1] + " (pipeline said " + ynWord(gp[QKEYS[j]]) + ")");
        if (diffs.length)
          h += "<div class='revgpt muted'>Note: gpt-5.2's bare pipeline run differed — " + diffs.join("; ") + ".</div>";
      }
      h += "<button id='continueBtn' class='cont'>Continue → next  (or press →)</button></div>";
      rv.innerHTML = h;
      var cb = document.getElementById("continueBtn");
      if (cb) cb.onclick = advanceUnlabeled;
    }
    save();
  }

  function go(i) { idx = Math.max(0, Math.min(ITEMS.length - 1, i)); render(); }
  function advanceUnlabeled() {
    for (var step = 1; step <= ITEMS.length; step++) {
      var j = (idx + step) % ITEMS.length;
      if (!isComplete(labels[ITEMS[j].id])) { idx = j; render(); return; }
    }
    go(idx + 1);
  }

  function setBit(qi, val) {
    var it = ITEMS[idx];
    var rec = labels[it.id] || {};
    var wasComplete = isComplete(rec);
    rec[QKEYS[qi]] = val;
    rec.notes = $("notes").value || "";
    rec.ts = Date.now();
    labels[it.id] = rec;
    save();
    var pause = (DATA.reveal_available !== false) && revealChk.checked && it._reveal;
    if (!isComplete(rec)) { render(); }
    else if (pause || wasComplete) { render(); }  // stay & show judges; advance on Continue / →
    else { advanceUnlabeled(); }
  }
  function answerCurrent(val) {
    var open = firstOpen(labels[ITEMS[idx].id]);
    if (open === -1) { advanceUnlabeled(); return; }
    setBit(open, val);
  }

  function exportJsonl() {
    var lines = [], incomplete = 0;
    ITEMS.forEach(function (it) {
      var rec = labels[it.id];
      if (isComplete(rec)) {
        lines.push(JSON.stringify({ id: it.id, event_id: it.event_id,
          instance_present: rec.instance_present, relation_present: rec.relation_present,
          constant_present: rec.constant_present, notes: rec.notes || "" }));
      } else incomplete++;
    });
    if (incomplete &&
        !confirm("Exporting " + lines.length + " labeled; " + incomplete +
                 " incomplete will be OMITTED. Continue?")) return;
    var blob = new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "fp_human__gold.jsonl";
    document.body.appendChild(a); a.click(); a.remove();
  }

  function importFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var added = 0;
      reader.result.split(/\r?\n/).forEach(function (line) {
        line = line.trim(); if (!line) return;
        try {
          var r = JSON.parse(line);
          var ok = r && r.id && QKEYS.every(function (k) { return r[k] === true || r[k] === false; });
          if (ok) {
            labels[r.id] = { instance_present: r.instance_present,
                             relation_present: r.relation_present,
                             constant_present: r.constant_present,
                             notes: r.notes || "", ts: Date.now() };
            added++;
          }
        } catch (e) {}
      });
      save(); render();
      alert("Imported " + added + " labels.");
    };
    reader.readAsText(file);
  }

  for (var qi = 0; qi < 3; qi++) (function (qi) {
    $("y" + qi).onclick = function () { setBit(qi, true); };
    $("n" + qi).onclick = function () { setBit(qi, false); };
  })(qi);
  $("prevBtn").onclick = function () { go(idx - 1); };
  $("nextBtn").onclick = function () { go(idx + 1); };
  $("jumpBtn").onclick = advanceUnlabeled;
  $("exportBtn").onclick = exportJsonl;
  $("importBtn").onclick = function () { $("importFile").click(); };
  $("importFile").onchange = function (e) { if (e.target.files[0]) importFile(e.target.files[0]); };
  $("notes").onchange = function () {
    var it = ITEMS[idx]; var rec = labels[it.id] || {};
    rec.notes = $("notes").value; labels[it.id] = rec; save();
  };
  revealChk.onchange = function () {
    localStorage.setItem(KEY_REVEAL, revealChk.checked ? "1" : "0"); render();
  };

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
      if (e.key === "Enter") e.target.blur();
      return;
    }
    var k = e.key.toLowerCase();
    if (k === "y") { answerCurrent(true); e.preventDefault(); }
    else if (k === "n") { answerCurrent(false); e.preventDefault(); }
    else if (k === "q") { setBit(0, true); e.preventDefault(); }
    else if (k === "a") { setBit(0, false); e.preventDefault(); }
    else if (k === "w") { setBit(1, true); e.preventDefault(); }
    else if (k === "s") { setBit(1, false); e.preventDefault(); }
    else if (k === "e") { setBit(2, true); e.preventDefault(); }
    else if (k === "d") { setBit(2, false); e.preventDefault(); }
    else if (k === "j" || k === "arrowright") { go(idx + 1); e.preventDefault(); }
    else if (k === "k" || k === "arrowleft") { go(idx - 1); e.preventDefault(); }
    else if (k === "r") { revealChk.checked = !revealChk.checked; revealChk.onchange(); e.preventDefault(); }
  });

  render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
