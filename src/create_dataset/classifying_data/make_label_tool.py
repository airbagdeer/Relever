"""Generate a self-contained HTML tool for manually labeling the 200-paragraph sample.

Reads ``data/judge_validation/short/_sample.json`` + the short-prompt rubric, embeds
the paragraphs (and, for an optional post-commit reveal, the minimax / opus / frozen-
consensus labels) into one offline ``label_paragraphs.html``. Open it in any browser,
label each paragraph with one keypress (Positive / Negative / Skip), and export a
ready-to-ingest ``human__gold.jsonl``. Autosaves to localStorage (resumable).

    uv run python src/create_dataset/classifying_data/make_label_tool.py

Then open data/judge_validation/short/label_paragraphs.html, label, Export, and move
the downloaded human__gold.jsonl into data/judge_validation/short/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .relation_classification_prompt import relation_classification_prompt_short
    from . import frontier_cost_report as report_mod
except ImportError:  # run as a plain script
    from relation_classification_prompt import relation_classification_prompt_short
    import frontier_cost_report as report_mod

RAW_DIR = PROJECT_ROOT / "data/judge_validation/short"
SAMPLE_PATH = RAW_DIR / "_sample.json"
OUT_HTML = RAW_DIR / "label_paragraphs.html"


def _discover_configs(raw_dir: Path) -> list[tuple[str, str]]:
    out = []
    for f in sorted(raw_dir.glob("*__*.jsonl")):
        if f.name.startswith("batch_"):
            continue
        model, _, effort = f.stem.rpartition("__")
        if model:
            out.append((model, effort))
    return out


def build_items(sample: list[dict], include_reveal: bool = True) -> list[dict]:
    """One display record per paragraph + a `_reveal` of the model labels.

    With ``include_reveal=False`` (for a publicly-hosted / shareable build) NO model
    labels are loaded or embedded — only the public paragraphs + queries — so the
    research gold (minimax/opus/consensus) is never exposed.
    """
    labelers: dict[str, dict] = {}
    consensus: dict = {}
    opus_reason: dict[str, str] = {}
    model_keys: list[str] = []
    if include_reveal:
        configs = _discover_configs(RAW_DIR)
        try:
            labelers = report_mod.load_predictions(RAW_DIR, configs, sample)
        except Exception as e:  # reveal is optional — never block tool generation
            print(f"(warn) could not load model labels for reveal: {e}")
            labelers = {}
        consensus = report_mod.majority_vote(
            labelers, [n for n in report_mod.CONSENSUS_LABELERS if n in labelers]
        )
        # Opus reasoning (brief_reason) — the one substantive rationale we stored.
        opath = RAW_DIR / f"{report_mod.OPUS_LABELER}.jsonl"
        if opath.exists():
            for line in opath.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                br = (r.get("brief_reason") or "").strip()
                # Drop non-substantive "reasons" that are just the JSON answer
                # (Opus's adaptive thinking sometimes returns no separate reasoning).
                if br.startswith("{"):
                    br = ""
                opus_reason[r["id"]] = br
        # GPT / DeepSeek configs (everything except opus / minimax / human), stable order.
        model_keys = [
            k for k in labelers
            if k not in (report_mod.OPUS_LABELER, report_mod.MINIMAX_LABELER, report_mod.HUMAN_LABELER)
        ]

    minimax = labelers.get(report_mod.MINIMAX_LABELER, {})
    opus = labelers.get(report_mod.OPUS_LABELER, {})

    items = []
    for rec in sample:
        rid = rec["id"]
        q = rec.get("query_info", {})
        reveal = None
        if include_reveal:
            reveal = {
                "consensus": consensus.get(rid),
                "minimax": minimax.get(rid),
                "opus": {"label": opus.get(rid), "reason": opus_reason.get(rid, "")},
                "models": {k: labelers[k].get(rid) for k in model_keys},
            }
        items.append({
            "id": rid,
            "event_id": rec.get("event_id"),
            "content": rec.get("content", ""),
            "source_query": rec.get("source_query", ""),
            "sentence": q.get("sentence", ""),
            "abstract_entity_type": q.get("abstract_entity_type", ""),
            "constant_entity": q.get("constant_entity", ""),
            "relation": q.get("relation", ""),
            "_reveal": reveal,
        })
    return items


def _embed(blob: dict) -> str:
    """JSON for an inline <script>, with HTML-significant chars neutralised."""
    return (
        json.dumps(blob, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_html(items: list[dict], rubric: str, reveal_available: bool = True,
                store_key: str = "relabel_v1") -> str:
    blob = _embed({"rubric": rubric, "items": items, "reveal_available": reveal_available,
                   "store_key": store_key})
    return _HTML_TEMPLATE.replace("__BLOB__", blob)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), default=None,
                    help="embed only sample[START:END] (e.g. --range 100 200 for the second half)")
    ap.add_argument("--no-reveal", action="store_true",
                    help="strip ALL model labels (only the public paragraphs) — for public hosting "
                         "where the gold labels must not be exposed; disables the AI-label reveal")
    ap.add_argument("--ids-file", default=None,
                    help="build only the sample records whose id is listed in this file "
                         "(one id per line; kept in sample order). Overrides --range.")
    ap.add_argument("--key", default=None, help="localStorage progress key (default derived from range)")
    ap.add_argument("--out", default=None, help="output HTML path (default: label_paragraphs.html)")
    args = ap.parse_args()

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample not found: {SAMPLE_PATH}")
    sample = json.loads(SAMPLE_PATH.read_text())
    scope = ""
    if args.ids_file:
        wanted = {ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()}
        sample = [r for r in sample if r["id"] in wanted]  # keep sample order
        scope = f" (ids-file: {len(sample)}/{len(wanted)} matched)"
    elif args.range:
        sample = sample[args.range[0]:args.range[1]]
        scope = f" (sample[{args.range[0]}:{args.range[1]}])"
    include_reveal = not args.no_reveal
    items = build_items(sample, include_reveal=include_reveal)
    # Per-scope storage key so each build keeps its own progress and opens at its first item.
    store_key = args.key or (f"relabel_{args.range[0]}_{args.range[1]}" if args.range else "relabel_v1")
    out = Path(args.out) if args.out else (RAW_DIR / "label_paragraphs.html")
    out.write_text(render_html(items, relation_classification_prompt_short,
                               reveal_available=include_reveal, store_key=store_key),
                   encoding="utf-8")
    n_reveal = sum(1 for it in items if it.get("_reveal") and it["_reveal"].get("consensus") is not None)
    rng = scope
    print(f"Wrote {out}")
    if include_reveal:
        print(f"  {len(items)} paragraphs{rng} embedded WITH model labels + opus reasoning "
              f"({n_reveal} have a consensus label).")
    else:
        print(f"  {len(items)} paragraphs{rng} embedded, NO model labels (safe to host publicly).")
    print("  Open in a browser, label, Export -> human__gold.jsonl; move it into")
    print(f"  {RAW_DIR} and regenerate the report.")


# --------------------------------------------------------------------------- #
# Self-contained HTML (no external deps). __BLOB__ is replaced with the data.
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relation labeling</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1f242e; --fg:#e6e9ef; --muted:#9aa3b2;
    --line:#2a3140; --pos:#1f9d57; --neg:#d2453f; --skip:#b8860b; --accent:#4c8bf5;
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
  .actions { display:flex; gap:6px; flex-wrap:wrap; }
  button { font:inherit; cursor:pointer; border:1px solid var(--line); background:var(--panel2);
           color:var(--fg); padding:6px 10px; border-radius:8px; }
  button:hover { border-color:var(--accent); }
  .toggle { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:13px; }
  main { max-width:900px; margin:0 auto; padding:16px; }
  details#rubricBox { background:var(--panel); border:1px solid var(--line); border-radius:10px;
                      padding:8px 12px; margin-bottom:14px; }
  details#rubricBox summary { cursor:pointer; color:var(--accent); font-weight:600; }
  #rubric { white-space:pre-wrap; font-size:13px; color:var(--muted); margin:10px 0 4px; }
  .qpanel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .qrow { display:flex; gap:10px; padding:3px 0; }
  .qrow .k { color:var(--muted); min-width:150px; font-size:13px; text-transform:uppercase; letter-spacing:.03em; }
  .qrow.muted { color:var(--muted); }
  .relation { font-weight:700; color:var(--accent); }
  .question { margin:14px 2px 8px; font-size:16px; }
  .content { background:var(--panel); border:1px solid var(--line); border-radius:10px;
             padding:16px 18px; max-height:46vh; overflow:auto; white-space:pre-wrap; font-size:15.5px; }
  .decide { display:flex; gap:10px; margin:16px 0 6px; }
  .decide button { flex:1; padding:14px; font-size:16px; font-weight:700; }
  .pos { border-color:var(--pos); } .pos.sel { background:var(--pos); color:#fff; }
  .neg { border-color:var(--neg); } .neg.sel { background:var(--neg); color:#fff; }
  .skip { border-color:var(--skip); } .skip.sel { background:var(--skip); color:#fff; }
  .notes input { width:100%; padding:9px 11px; border-radius:8px; border:1px solid var(--line);
                 background:var(--panel); color:var(--fg); font:inherit; }
  .reveal { margin-top:12px; font-size:14px; color:var(--muted); min-height:20px; }
  .reveal b { color:var(--fg); }
  .agree { color:var(--pos); font-weight:700; } .disagree { color:var(--neg); font-weight:700; }
  .revpanel { background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .revhead { margin-bottom:8px; font-size:14.5px; }
  .revrow { margin:5px 0; font-size:14px; }
  .revreason { margin-top:10px; padding-top:10px; border-top:1px solid var(--line); color:var(--fg);
               font-size:14px; line-height:1.5; }
  .cont { margin-top:12px; width:100%; padding:13px; font-weight:700; border-color:var(--accent); color:var(--accent); }
  footer { max-width:900px; margin:0 auto; padding:10px 16px 40px; color:var(--muted); font-size:12.5px; }
  kbd { background:var(--panel2); border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:12px; }
  .done-banner { background:var(--pos); color:#fff; padding:8px 12px; border-radius:8px; margin-bottom:12px;
                 display:none; }
  #intro { color:var(--muted); font-size:13.5px; margin:0 0 12px; }
  #intro b { color:var(--fg); }
  .storewarn { display:none; background:var(--skip); color:#fff; padding:5px 10px; border-radius:8px;
               font-size:12.5px; }
  #copyWrap { display:none; margin-top:14px; }
  #copyText { width:100%; height:180px; margin-top:8px; padding:10px 12px; border-radius:8px;
              border:1px solid var(--line); background:var(--bg); color:var(--fg); resize:vertical;
              font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
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
    <span class="actions">
      <button id="exportBtn">⬇ Export jsonl</button>
      <button id="copyBtn">⧉ Copy results</button>
      <button id="importBtn">⬆ Import</button>
      <input id="importFile" type="file" accept=".jsonl,.json,application/json" hidden>
    </span>
    <label class="toggle"><input type="checkbox" id="revealChk"> pause &amp; show AI labels after each</label>
    <span id="storeWarn" class="storewarn">Autosave unavailable here — finish in one sitting, then Copy results.</span>
  </div>
</header>
<main>
  <div class="done-banner" id="doneBanner"></div>
  <p id="intro">Read the criteria, then label each paragraph — <b>Positive</b>, <b>Negative</b>, or <b>Skip</b>. When you finish, click <b>Copy results</b> (or <b>Export jsonl</b>) to save your labels and send them back. Progress autosaves in this browser.</p>
  <details id="rubricBox"><summary>Rubric — the exact criteria (click to expand)</summary><pre id="rubric"></pre></details>
  <section id="card">
    <div class="qpanel">
      <div class="qrow"><span class="k">Relation</span><span id="relation" class="relation"></span></div>
      <div class="qrow"><span class="k">Abstract entity type</span><span id="aet"></span></div>
      <div class="qrow"><span class="k">Constant entity</span><span id="ce"></span></div>
      <div class="qrow"><span class="k">Query sentence</span><span id="sentence"></span></div>
      <div class="qrow muted"><span class="k">Example match</span><span id="srcq"></span></div>
    </div>
    <div class="question">Does the paragraph below provide evidence that a concrete instance of
      <b id="aet2"></b> stands in the exact relation <b id="rel2"></b> to <b id="ce2"></b>?
      <span class="muted">(judge from the paragraph alone)</span></div>
    <div id="content" class="content"></div>
    <div class="decide">
      <button id="posBtn" class="pos">Positive · <kbd>1</kbd>/<kbd>p</kbd></button>
      <button id="negBtn" class="neg">Negative · <kbd>0</kbd>/<kbd>n</kbd></button>
      <button id="skipBtn" class="skip">Skip · <kbd>s</kbd></button>
    </div>
    <div class="notes"><input id="notes" placeholder="optional note — saved as brief_reason"></div>
    <div class="reveal" id="reveal"></div>
  </section>
  <div id="copyWrap">
    <div class="qpanel">
      <div class="revhead"><b id="copyHead">Results (JSONL)</b> — select all, copy, and send this back.</div>
      <textarea id="copyText" readonly></textarea>
    </div>
  </div>
</main>
<footer>
  <span id="meta"></span> ·
  Keys: <kbd>1</kbd>/<kbd>p</kbd>/<kbd>t</kbd> positive · <kbd>0</kbd>/<kbd>n</kbd>/<kbd>f</kbd> negative ·
  <kbd>s</kbd>/<kbd>space</kbd> skip · <kbd>j</kbd>/<kbd>k</kbd> or <kbd>←</kbd><kbd>→</kbd> navigate ·
  <kbd>r</kbd> reveal. Progress autosaves in this browser.
</footer>

<script id="blob" type="application/json">__BLOB__</script>
<script>
(function () {
  var DATA = JSON.parse(document.getElementById("blob").textContent);
  var ITEMS = DATA.items, RUBRIC = DATA.rubric;
  var BASE = DATA.store_key || "relabel_v1";   // per-range key: each split keeps its own progress
  var KEY = BASE, KEY_IDX = BASE + "_idx", KEY_REVEAL = BASE + "_reveal";
  var STORAGE_OK = true;
  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { STORAGE_OK = false; return null; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { STORAGE_OK = false; } }
  var labels = {}; try { labels = JSON.parse(lsGet(KEY) || "{}"); } catch (e) {}
  var idx = parseInt(lsGet(KEY_IDX) || "0", 10) || 0;
  if (idx < 0 || idx >= ITEMS.length) idx = 0;

  var $ = function (id) { return document.getElementById(id); };
  $("rubric").textContent = RUBRIC;
  var revealChk = $("revealChk");
  var _rv = lsGet(KEY_REVEAL);
  revealChk.checked = (_rv === null ? true : _rv === "1");   // default ON: pause & show AI
  if (DATA.reveal_available === false) {        // no model labels embedded (public build)
    revealChk.checked = false;
    var rl = revealChk.closest("label"); if (rl) rl.style.display = "none";
  }
  function escapeHtml(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  function save() {
    lsSet(KEY, JSON.stringify(labels));
    lsSet(KEY_IDX, String(idx));
  }
  function counts() {
    var labeled = 0, skipped = 0;
    ITEMS.forEach(function (it) {
      var l = labels[it.id] && labels[it.id].label;
      if (l === 0 || l === 1) labeled++; else if (l === "skip") skipped++;
    });
    return { labeled: labeled, skipped: skipped, total: ITEMS.length };
  }
  function lab(v) { return v === 1 ? "Positive" : v === 0 ? "Negative" : v === "skip" ? "Skip" : "—"; }

  function render() {
    var it = ITEMS[idx];
    var rec = labels[it.id] || {};
    $("relation").textContent = it.relation || "(none)";
    $("aet").textContent = it.abstract_entity_type;
    $("ce").textContent = it.constant_entity;
    $("sentence").textContent = it.sentence;
    $("srcq").textContent = it.source_query || "(none)";
    $("aet2").textContent = it.abstract_entity_type;
    $("rel2").textContent = it.relation;
    $("ce2").textContent = it.constant_entity;
    $("content").textContent = it.content;
    $("content").scrollTop = 0;
    $("notes").value = rec.notes || "";
    $("posBtn").classList.toggle("sel", rec.label === 1);
    $("negBtn").classList.toggle("sel", rec.label === 0);
    $("skipBtn").classList.toggle("sel", rec.label === "skip");

    var c = counts();
    $("progress").textContent = c.labeled + " / " + c.total +
      (c.skipped ? "  (" + c.skipped + " skipped)" : "");
    $("fill").style.width = (100 * c.labeled / c.total) + "%";
    $("meta").textContent = "#" + (idx + 1) + " · id " + it.id.slice(0, 8) + " · event " + it.event_id;

    var banner = $("doneBanner");
    if (c.labeled === c.total) {
      banner.style.display = "block";
      banner.textContent = "All " + c.total + " labeled — click Export jsonl.";
    } else if (c.labeled + c.skipped === c.total && c.skipped) {
      banner.style.display = "block";
      banner.textContent = c.skipped + " skipped still need a decision before export.";
    } else { banner.style.display = "none"; }

    // After a committed 0/1 decision (and if the pause toggle is on), show the AI
    // verdicts + Opus reasoning; the user reviews, then continues.
    var rv = $("reveal"); rv.innerHTML = "";
    var showAI = (DATA.reveal_available !== false) && revealChk.checked
                 && (rec.label === 0 || rec.label === 1) && it._reveal;
    if (showAI) {
      var r = it._reveal, you = rec.label;
      var tag = function (v) {
        if (v === null || v === undefined) return "<span class='muted'>—</span>";
        return "<span class='" + (v === you ? "agree" : "disagree") + "'>" + lab(v) + "</span>";
      };
      var h = "<div class='revpanel'>";
      h += "<div class='revhead'>Your label: <b>" + lab(you) + "</b> — AI verdicts "
         + "(<span class='agree'>green</span> agrees, <span class='disagree'>red</span> differs):</div>";
      h += "<div class='revrow'><b>Consensus (gold):</b> " + tag(r.consensus)
         + " &nbsp;·&nbsp; <b>opus:</b> " + tag(r.opus ? r.opus.label : null)
         + " &nbsp;·&nbsp; minimax: " + tag(r.minimax) + "</div>";
      var ms = r.models ? Object.keys(r.models) : [];
      if (ms.length) {
        h += "<div class='revrow'>" + ms.map(function (k) {
          return k.replace(/:/g, " ") + ": " + tag(r.models[k]);
        }).join(" &nbsp;·&nbsp; ") + "</div>";
      }
      if (r.opus && r.opus.reason) {
        h += "<div class='revreason'><b>Opus reasoning:</b> " + escapeHtml(r.opus.reason) + "</div>";
      }
      h += "<button id='continueBtn' class='cont'>Continue → next  (or press →)</button></div>";
      rv.innerHTML = h;
      var cb = document.getElementById("continueBtn");
      if (cb) cb.onclick = advanceUnlabeled;
    }
    save();
  }

  function setLabel(v) {
    var it = ITEMS[idx];
    labels[it.id] = { label: v, notes: $("notes").value || "", ts: Date.now() };
    save();
    var pause = (v === 0 || v === 1) && revealChk.checked
                && DATA.reveal_available !== false && it._reveal;
    if (v === "skip") { go(idx + 1); }
    else if (pause) { render(); }   // stay & show AI panel; advance on Continue / →
    else { advanceUnlabeled(); }
  }
  function go(i) { idx = Math.max(0, Math.min(ITEMS.length - 1, i)); render(); }
  function advanceUnlabeled() {
    for (var step = 1; step <= ITEMS.length; step++) {
      var j = (idx + step) % ITEMS.length;
      var l = labels[ITEMS[j].id] && labels[ITEMS[j].id].label;
      if (l !== 0 && l !== 1) { idx = j; render(); return; }
    }
    go(idx + 1); // all labeled — just step forward
  }

  function exportJsonl() {
    var lines = [], skipped = 0, unl = 0;
    ITEMS.forEach(function (it) {
      var rec = labels[it.id];
      var l = rec && rec.label;
      if (l === 0 || l === 1) {
        lines.push(JSON.stringify({ id: it.id, prediction: l, brief_reason: (rec.notes || "") }));
      } else if (l === "skip") { skipped++; } else { unl++; }
    });
    if (skipped || unl) {
      if (!confirm("Exporting " + lines.length + " labeled. " + skipped + " skipped + " +
                   unl + " unlabeled will be OMITTED. Continue?")) return;
    }
    var blob = new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "human__gold.jsonl";
    document.body.appendChild(a); a.click(); a.remove();
  }

  // Copy-paste return path — works even where the iframe/browser blocks file downloads.
  function copyResults() {
    var lines = [], skipped = 0, unl = 0;
    ITEMS.forEach(function (it) {
      var rec = labels[it.id];
      var l = rec && rec.label;
      if (l === 0 || l === 1) {
        lines.push(JSON.stringify({ id: it.id, prediction: l, brief_reason: (rec.notes || "") }));
      } else if (l === "skip") { skipped++; } else { unl++; }
    });
    var ta = $("copyText"), wrap = $("copyWrap");
    ta.value = lines.join("\n") + "\n";
    $("copyHead").textContent = "Results — " + lines.length + " labeled" +
      (skipped || unl ? "  (" + skipped + " skipped, " + unl + " unlabeled omitted)" : "");
    wrap.style.display = "block";
    wrap.scrollIntoView({ behavior: "smooth", block: "center" });
    ta.focus(); ta.select();
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(ta.value).then(function () {}, function () {});
      } else { document.execCommand("copy"); }
    } catch (e) {}
  }

  function importFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var text = reader.result, added = 0;
      text.split(/\r?\n/).forEach(function (line) {
        line = line.trim(); if (!line) return;
        try {
          var r = JSON.parse(line);
          if (r && r.id && (r.prediction === 0 || r.prediction === 1)) {
            labels[r.id] = { label: r.prediction, notes: r.brief_reason || "", ts: Date.now() };
            added++;
          }
        } catch (e) {}
      });
      save(); render();
      alert("Imported " + added + " labels.");
    };
    reader.readAsText(file);
  }

  // --- wiring ---
  $("posBtn").onclick = function () { setLabel(1); };
  $("negBtn").onclick = function () { setLabel(0); };
  $("skipBtn").onclick = function () { setLabel("skip"); };
  $("prevBtn").onclick = function () { go(idx - 1); };
  $("nextBtn").onclick = function () { go(idx + 1); };
  $("jumpBtn").onclick = advanceUnlabeled;
  $("exportBtn").onclick = exportJsonl;
  $("copyBtn").onclick = copyResults;
  $("importBtn").onclick = function () { $("importFile").click(); };
  $("importFile").onchange = function (e) { if (e.target.files[0]) importFile(e.target.files[0]); };
  $("notes").onchange = function () {
    var it = ITEMS[idx]; var rec = labels[it.id] || { label: null };
    rec.notes = $("notes").value; labels[it.id] = rec; save();
  };
  revealChk.onchange = function () {
    lsSet(KEY_REVEAL, revealChk.checked ? "1" : "0"); render();
  };

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
      if (e.key === "Enter") e.target.blur();
      return;
    }
    var k = e.key.toLowerCase();
    if (k === "1" || k === "p" || k === "t") { setLabel(1); e.preventDefault(); }
    else if (k === "0" || k === "n" || k === "f") { setLabel(0); e.preventDefault(); }
    else if (k === "s" || k === " ") { setLabel("skip"); e.preventDefault(); }
    else if (k === "j" || k === "arrowright") { go(idx + 1); e.preventDefault(); }
    else if (k === "k" || k === "arrowleft") { go(idx - 1); e.preventDefault(); }
    else if (k === "r") { revealChk.checked = !revealChk.checked; revealChk.onchange(); e.preventDefault(); }
  });

  if (!STORAGE_OK) { var sw = $("storeWarn"); if (sw) sw.style.display = "inline-block"; }
  render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
