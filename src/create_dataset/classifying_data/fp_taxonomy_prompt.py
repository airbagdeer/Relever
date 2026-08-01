"""False-positive taxonomy prompt + parser (paper §7 "Anatomy of the False Positives").

Classifies each judged-negative (query, paragraph) pair that some retriever placed in
its top-10 into exactly ONE of five categories, mirroring the introduction's three
failure modes plus a label-noise audit bucket and a residual:

    looks_positive   the paragraph appears to satisfy all five §3 positivity criteria
                     (label-noise audit — reported separately, labels stay untouched)
    wrong_relation   right entities, wrong/no asserted relation   (intro mode 1)
                       sub_flag: different_relation_asserted | no_relation_asserted
    wrong_entities   right relation, wrong entities               (intro mode 2)
    constant_only    merely about the constant entity             (intro mode 3)
    off_target       none of the above

Mutual exclusivity is forced by a fixed decision order (walk the steps, stop at the
first that applies). The prompt deliberately does NOT reveal that a prior judge
labeled the pair negative, so ``looks_positive`` is an independent second opinion,
not an anchored one.

The pair is presented in the judge's exact input format (Sentence / Abstract Entity
Type / Constant Entity / Relation / Paragraph — see ``label_frontier_batch._user_content``)
and classified by the same configuration as the benchmark's judge (gpt-5.2, medium
reasoning effort, Batch API). This module is import-light (json/re only) so it can be
loaded anywhere, mirroring ``relation_classification_prompt``.

Version history (prompt iteration, mirroring the judge-prompt iteration):
    v1 — initial prompt.
    v2 — manual inspection of the v1 pilot (200 pairs) showed ``off_target`` (52%)
         absorbing two classes it shouldn't: (a) category-valued constant entities
         ("stadium construction projects", "the commercial vehicle industry") were
         read strictly, though the benchmark's judge (short2b) counts a named
         *member* of such a category as the constant entity — v2 inherits the
         judge's matching semantics (category-CE by member, constituents,
         relation-by-substance, lexical/citation guards) so the taxonomy measures
         against the same task definition as the labels; (b) pairs where the
         queried relation is asserted toward a *different* target with e absent
         ("brigade based in Poland" for e = Eastern Europe) belong to
         ``wrong_entities`` by its own definition — v2 sharpens the step-3 vs
         step-5 boundary and adds micro-examples of both shapes.
    v3 — output redesigned (2026-07-12): instead of one mutually-exclusive
         category, the classifier answers three INDEPENDENT presence bits —
         instance_present / relation_present / constant_present — yielding 8
         error cells for judged-negative pairs (all-three-present is still a
         negative when the relation does not bind the queried pair). The v2
         matching semantics carry over verbatim; only the decision layer and
         output schema change. v1/v2 categorical predictions remain on disk
         under their versions.
    v4 — calibrated to the 100-pair human gold (2026-07-15). The v3 disagreements
         were one-sided (relation 12:3, constant 8:1, instance 5:2 under-firing):
         the model kept demanding the full relational frame, exact constant-entity
         wording, and explicit type labels — i.e. sliding back toward judging
         evidence instead of PRESENCE. v4 makes the presence construct explicit
         (head action of a compound relation suffices; close synonyms and
         contextual instantiations of e count; functional-role instances count
         without the type label) — every loosening traceable to a gold
         disagreement (e559 created_series_for, e288 clinical research/trials,
         e292 machine vision, e408/e665 public schools, e645 TETCo-as-operator).
         INTERNAL ONLY: v4's worked examples quoted gold items nearly verbatim
         (item-level leakage into the validation set) — superseded by v5.
    v5 — final round (2026-07-16), from the reasoned analysis of v4's 23 residual
         disagreements: (1) relation_present needs the action's characteristic
         argument shape minus the query binding (operates_in needs SOME locative;
         created_series_for needs only the creating act), and the guards
         (negation/modality, definitional boilerplate, verbless headings and
         statistics, cited-source-only names) explicitly rank above the
         paraphrase loosenings, which v4's loosening was overriding;
         (2) single-member-standing-alone does not make a geographic whole
         present (UK HQ != Europe); (3) all worked examples replaced with
         constructed analogues — no gold item appears in the prompt. Rules
         frozen after this revision (residual includes ~4% run-flip noise).
"""
from __future__ import annotations

import json
import re

CATEGORIES: tuple[str, ...] = (
    "looks_positive",
    "wrong_relation",
    "wrong_entities",
    "constant_only",
    "off_target",
)

SUB_FLAGS: tuple[str, ...] = (
    "different_relation_asserted",
    "no_relation_asserted",
)

FP_TAXONOMY_PROMPT_VERSION = "v5"

# The three independent presence bits (v3+ schema), in the fixed display order:
# instance of the abstract type, relation, constant entity.
BITS: tuple[str, ...] = ("instance_present", "relation_present", "constant_present")

fp_taxonomy_prompt = """
You are auditing the top-ranked results of retrieval systems on a relation-evidence benchmark. A retrieval system returned the PARAGRAPH below among its top results for the QUERY. Classify the paragraph's relationship to the query into exactly ONE category.

**The query** asks for evidence that an instance of an **Abstract Entity Type** stands in an **exact Relation** to a **Constant Entity**:
- **Abstract Entity Type (T)** — a class (e.g. "Technology companies"). Satisfied only by a *concrete, named instance* (e.g. "Dell") — never by the bare category name, an aggregate statistic, or an entity merely *associated with* the class (a supplier, customer, regulator, parent, subsidiary, product, or executive of such firms is not an instance).
- **Constant Entity (e)** — one specific entity (e.g. "NVIDIA"). Must be e itself, not a part, member, or neighbor of it.
- **Relation (r)** — the exact relationship (e.g. "partnered_with"). Adjacent relations (competing, supplying, ending a partnership) are different relations.
- **Sentence** — the templated query combining the three.

A paragraph is **true evidence** only if, judged from the paragraph ALONE: (1) it names a concrete instance x of T; (2) it refers to e itself; (3) it asserts, or unambiguously implies, r between x and e; (4) for a directional relation, in the queried direction (x -> e); (5) as fact — not planned, hypothetical, negated, or terminated.

**Matching semantics — these are the benchmark's own labeling rules; apply them wherever the steps below read "names an instance", "refers to e", or "asserts r":**
- **Category-valued constant entity**: when e is itself a class, market, sector, industry, field, or type of project/place (e.g. "the commercial vehicle industry", "stadium construction projects", "the wearable devices market"), e counts as referred to whenever the paragraph names a concrete member, product, or example of it as the relation's target — exactly as T is satisfied by an instance ("heavy trucks" ARE the commercial vehicle industry; "the London Velodrome" IS a stadium construction project).
- **Specific named constant entity**: a specific, extensive named entity (a region, organization, program) is NOT satisfied by a mere sub-part or neighbor ("the UAE" is not "the Middle East") — but clear constituents jointly standing in for it do count ("U.S. and Canadian regulators" for "North America").
- **Relation by substance**: r counts when its substance is asserted in other words ("allocates substantial financial resources to" satisfies invests_in). Adjacent-but-different relations never count.
- **Lexical / list guard**: a category word inside a brand or product name is not e ("Horizon Organic" does not refer to the organic food sector); entities co-occurring only inside a citation/reference list, infobox, fleet list, or navigation box assert nothing (a content list or data table that states facts in its rows does).

**Decision procedure — walk these steps IN ORDER and stop at the FIRST that applies:**

1. **looks_positive** — the paragraph satisfies ALL five conditions above: you would call it true evidence for the query.

2. **wrong_relation** — the paragraph names a concrete instance x of T AND refers to e, but what it asserts between x and e is not r-as-fact. Set sub_flag:
   - "different_relation_asserted" — a relation between x and e IS asserted, but it is a different one (x competes with e, x is supplied by e, x ended its partnership with e), or r itself appears only as planned / hypothetical / negated / terminated (the asserted state is not r-as-fact);
   - "no_relation_asserted" — x and e merely co-occur; the paragraph asserts nothing between them (whatever it asserts about x or e separately).
   Exception: if what is asserted between x and e is exactly r but only in the reverse direction (e -> x), go to step 3.

3. **wrong_entities** — r (or a clear paraphrase of it) IS asserted somewhere in the paragraph — whether or not e appears in it — but not between an instance of T and e in the queried direction: it targets a different constant entity (including a mere part or neighbor of e), or its subject is not an instance of T, or it holds only in the reverse direction (e -> x).

4. **constant_only** — the paragraph refers to or is about e (including a member of a category-valued e), but names no concrete instance of T (and, per step 3, does not assert r).

5. **off_target** — none of the above: the paragraph never refers to e and asserts r nowhere (e.g. it names an instance of T with no e and no r, or it is about something else entirely).

Judge from the paragraph ALONE — never use outside knowledge about what is true in the real world.

**Worked micro-examples** for the query "Technology companies partnered with NVIDIA." (T = Technology companies, r = partnered_with, e = NVIDIA):
- "Dell and NVIDIA announced a strategic partnership in 2023." -> {"category": "looks_positive", "sub_flag": null}
- "Intel competes with NVIDIA in the GPU market." -> {"category": "wrong_relation", "sub_flag": "different_relation_asserted"}
- "The keynote featured speakers from Foxconn, NVIDIA, and Samsung." -> {"category": "wrong_relation", "sub_flag": "no_relation_asserted"}
- "Microsoft partnered with OpenAI." -> {"category": "wrong_entities", "sub_flag": null}
- "Dell partnered with AMD on data-center GPUs." -> {"category": "wrong_entities", "sub_flag": null}  (r asserted, wrong target; e absent)
- "The Louvre partnered with NVIDIA on a digitization project." -> {"category": "wrong_entities", "sub_flag": null}
- "NVIDIA designs GPUs and AI accelerators." -> {"category": "constant_only", "sub_flag": null}
- "AMD reported record quarterly revenue." -> {"category": "off_target", "sub_flag": null}

And for the category-valued-e query "Automotive suppliers serving the commercial vehicle industry." (T = Automotive suppliers, r = serves, e = the commercial vehicle industry):
- "Knorr-Bremse supplies braking systems for heavy trucks." -> {"category": "looks_positive", "sub_flag": null}  (heavy trucks ARE the commercial vehicle industry)
- "Western Star builds heavy-duty trucks." -> {"category": "constant_only", "sub_flag": null}  (a member of e is present, but no automotive supplier is named)

**Output** — return ONLY this JSON object, with no explanation or extra text. "sub_flag" MUST be null unless "category" is "wrong_relation":
{"category": "<looks_positive | wrong_relation | wrong_entities | constant_only | off_target>", "sub_flag": "<different_relation_asserted | no_relation_asserted | null>"}
"""


# ---------------------------------------------------------------------------
# v3 — three independent presence bits. The definitions and matching-semantics
# blocks mirror the v2 categorical prompt verbatim; only the decision layer and
# the output schema differ.
# ---------------------------------------------------------------------------
fp_bits_prompt = """
You are auditing the top-ranked results of retrieval systems on a relation-evidence benchmark. A retrieval system returned the PARAGRAPH below among its top results for the QUERY. Answer three INDEPENDENT yes/no presence questions about the paragraph.

**The query** asks for evidence that an instance of an **Abstract Entity Type** stands in an **exact Relation** to a **Constant Entity**:
- **Abstract Entity Type (T)** — a class (e.g. "Technology companies"). Satisfied only by a *concrete, named instance* (e.g. "Dell") — never by the bare category name, an aggregate statistic, or an entity merely *associated with* the class (a supplier, customer, regulator, parent, subsidiary, product, or executive of such firms is not an instance).
- **Constant Entity (e)** — one specific entity (e.g. "NVIDIA"). Must be e itself, not a part, member, or neighbor of it.
- **Relation (r)** — the exact relationship (e.g. "partnered_with"). Adjacent relations (competing, supplying, ending a partnership) are different relations.
- **Sentence** — the templated query combining the three.

**Matching semantics — these are the benchmark's own labeling rules; apply them wherever the questions below read "names an instance", "refers to e", or "asserts r":**
- **Category-valued constant entity**: when e is itself a class, market, sector, industry, field, or type of project/place (e.g. "the commercial vehicle industry", "stadium construction projects", "the wearable devices market"), e counts as referred to whenever the paragraph names a concrete member, product, or example of it ("heavy trucks" ARE the commercial vehicle industry; "the London Velodrome" IS a stadium construction project).
- **Specific named constant entity**: a specific, extensive named entity (a region, organization, program) is NOT satisfied by a mere sub-part or neighbor ("the UAE" is not "the Middle East") — but clear constituents jointly standing in for it do count ("U.S. and Canadian regulators" for "North America").
- **Relation by substance**: r counts when its substance is asserted in other words ("allocates substantial financial resources to" satisfies invests_in). Adjacent-but-different relations never count.
- **Lexical / list guard**: a category word inside a brand or product name is not e ("Horizon Organic" does not refer to the organic food sector); entities co-occurring only inside a citation/reference list, infobox, fleet list, or navigation box assert nothing (a content list or data table that states facts in its rows does).

**The three questions — answer each INDEPENDENTLY of the others, judged from the paragraph ALONE (never use outside knowledge):**

1. **instance_present** — does the paragraph name a concrete instance of the Abstract Entity Type? (true even if that instance has nothing to do with the Constant Entity or the Relation.)

2. **relation_present** — does the paragraph assert the Relation (or a paraphrase preserving its substance), as fact, between ANY two entities — whatever they are? (false when r appears only as planned / hypothetical / negated / terminated, only as a planted surface verb with no actual assertion, or only via citation-list / navigation-box co-occurrence.)

3. **constant_present** — does the paragraph refer to the Constant Entity? (true even when e appears only as background, with no relation to anything.)

Do NOT judge whether the paragraph is overall evidence for the query — even three trues do not mean the relation holds between the right pair. Answer only the three presence questions.

**Worked micro-examples** for the query "Technology companies partnered with NVIDIA." (T = Technology companies, r = partnered_with, e = NVIDIA):
- "Intel competes with NVIDIA in the GPU market." -> {"instance_present": true, "relation_present": false, "constant_present": true}  (competing is a different relation)
- "Microsoft partnered with OpenAI." -> {"instance_present": true, "relation_present": true, "constant_present": false}
- "NVIDIA designs GPUs and AI accelerators." -> {"instance_present": false, "relation_present": false, "constant_present": true}
- "The keynote featured speakers from Foxconn, NVIDIA, and Samsung." -> {"instance_present": true, "relation_present": false, "constant_present": true}
- "The Louvre partnered with NVIDIA on a digitization project." -> {"instance_present": false, "relation_present": true, "constant_present": true}  (the Louvre is not a technology company)
- "Dell and NVIDIA announced a strategic partnership in 2023." -> {"instance_present": true, "relation_present": true, "constant_present": true}
- "AMD reported record quarterly revenue." -> {"instance_present": true, "relation_present": false, "constant_present": false}

And for the category-valued-e query "Automotive suppliers serving the commercial vehicle industry." (T = Automotive suppliers, r = serves, e = the commercial vehicle industry):
- "Knorr-Bremse supplies braking systems for heavy trucks." -> {"instance_present": true, "relation_present": true, "constant_present": true}  (heavy trucks ARE the commercial vehicle industry)

**Output** — return ONLY this JSON object, with no explanation or extra text:
{"instance_present": true|false, "relation_present": true|false, "constant_present": true|false}
"""


# ---------------------------------------------------------------------------
# v4 — presence-construct calibration against the 100-pair human gold. One
# principle drives every change vs v3: the bits measure COMPONENT PRESENCE,
# independent of argument binding and exact wording. See the module docstring's
# version history for the per-disagreement provenance.
# ---------------------------------------------------------------------------
fp_bits_prompt_v4 = """
You are auditing the top-ranked results of retrieval systems on a relation-evidence benchmark. A retrieval system returned the PARAGRAPH below among its top results for the QUERY. Answer three INDEPENDENT yes/no presence questions about the paragraph.

**The query** asks for evidence that an instance of an **Abstract Entity Type** stands in an **exact Relation** to a **Constant Entity**:
- **Abstract Entity Type (T)** — a class (e.g. "Technology companies").
- **Constant Entity (e)** — one specific entity or category (e.g. "NVIDIA", "the commercial vehicle industry").
- **Relation (r)** — a relationship (e.g. "partnered_with").
- **Sentence** — the templated query combining the three.

**PRESENCE, not evidence.** Each question asks only whether a component APPEARS in the paragraph. A paragraph can be completely unrelated to what the query claims and still answer yes on all three, because presence never requires the components to be connected to each other. Never make one answer depend on another, and never judge whether the paragraph supports the query.

**The three questions — judge each from the paragraph ALONE (never use outside knowledge):**

1. **instance_present** — does the paragraph contain a concrete, named instance of T? Counts: an entity carrying the type's label; an entity shown performing the type's defining function even without the label ("the series, created by Matthew Weiner" makes Weiner an instance of *Showrunners*; the company running a pipeline is an instance of *Pipeline Operators*); a closely equivalent subtype (a motor-rifle brigade for *Armored Brigades*). Does not count: the bare category name; aggregate statistics; an entity merely associated with the type (its regulator, parent, or product) with nothing in-type shown; a name appearing purely as a bibliographic source.

2. **relation_present** — is the relation's core action or state expressed anywhere in the paragraph, between ANY arguments — even incomplete ones? For a compound relation the head action alone counts: "X created the series" satisfies *created_series_for*; "the port tranships containers" satisfies *handles_cargo_from*; "operates a chain of supermarkets across the country" satisfies *operates_in*. Paraphrases preserving the substance count; past/completed acts count; a relational verb phrase inside a cited title counts ("CBS ... Ink Two-Year Broadcast Deal" expresses *broadcasts_coverage_of*). Does not count: a DIFFERENT relation (competing is not partnering); a bare list of names, titles, or projects with no relational wording at all; a lexical cognate with no act expressed ("contributions" in licensing boilerplate does not express *contributes_to*); a relation that is only negated or hypothetical.

3. **constant_present** — does the paragraph refer to e, even as pure background? Exact wording is NOT required. Counts: close synonyms ("machine vision" for *computer vision*); near-equivalent terms ("clinical research" for *clinical trials*, "ETL tool" for *ETL pipelines*); a named member or example of a category-valued e ("heavy trucks" for *the commercial vehicle industry*); contextual instantiations ("low-income children" for *low-income communities*, plain "schools" in a public-education context for *public schools*); clear constituents of a named whole ("U.S. and Canadian regulators" for *North America*); e's name inside a proper noun when the content is genuinely about e (the "Public Schools Accountability Act" refers to public schools; "China National Aviation Fuel" selling jet fuel refers to *the aviation sector*). Does not count: a qualified e whose qualifier the context does not fit ("haute couture" discussed apart from Paris is not *Paris haute couture*); a mere sub-part or neighbor of a specific named entity ("the UAE" is not *the Middle East*).

**Worked micro-examples** for the query "Technology companies partnered with NVIDIA." (T = Technology companies, r = partnered_with, e = NVIDIA):
- "Intel competes with NVIDIA in the GPU market." -> {"instance_present": true, "relation_present": false, "constant_present": true}  (competing is a different relation)
- "Microsoft partnered with OpenAI." -> {"instance_present": true, "relation_present": true, "constant_present": false}
- "The Louvre partnered with NVIDIA on a digitization project." -> {"instance_present": false, "relation_present": true, "constant_present": true}
- "AMD reported record quarterly revenue." -> {"instance_present": true, "relation_present": false, "constant_present": false}

For "Showrunners created series for AMC." (T = Showrunners, r = created_series_for, e = AMC):
- "Mad Men, created by Matthew Weiner, was broadcast on the cable network AMC." -> {"instance_present": true, "relation_present": true, "constant_present": true}  (Weiner performs the showrunner function; "created" is the head action — no "for AMC" binding needed; AMC appears)

For "Hospitals participating in clinical trials." (T = Hospitals, r = participates_in, e = clinical trials):
- "The NIH Clinical Center is a hospital solely dedicated to clinical research with volunteer patients." -> {"instance_present": true, "relation_present": true, "constant_present": true}  ("clinical research" refers to clinical trials; being dedicated to it expresses participation)

**Output** — return ONLY this JSON object, with no explanation or extra text:
{"instance_present": true|false, "relation_present": true|false, "constant_present": true|false}
"""


# ---------------------------------------------------------------------------
# v5 — final calibration round. Two changes vs v4, both from the reasoned
# disagreement analysis of the v4 sync run (23 residual pairs):
#   (1) relation_present made precise: the core action needs its CHARACTERISTIC
#       argument shape (an "operates_in" needs some locative; "created_series_for"
#       needs only the creating-a-series act), and the guards (negation/modality,
#       definitional boilerplate, verbless headings/statistics, cited-source
#       names) explicitly RANK ABOVE the paraphrase loosenings — v4's loosening
#       was overriding its own guards.
#   (2) every worked example is a CONSTRUCTED ANALOGUE: v4 quoted several gold
#       items nearly verbatim, which is indefensible for a prompt validated on
#       that same gold (item-level leakage). v5 keeps the principles, swaps the
#       instances. Rules were frozen after this revision; residual disagreement
#       includes ~4% run-to-run flip noise that we deliberately do not chase.
# ---------------------------------------------------------------------------
fp_bits_prompt_v5 = """
You are auditing the top-ranked results of retrieval systems on a relation-evidence benchmark. A retrieval system returned the PARAGRAPH below among its top results for the QUERY. Answer three INDEPENDENT yes/no presence questions about the paragraph.

**The query** asks for evidence that an instance of an **Abstract Entity Type** stands in an **exact Relation** to a **Constant Entity**:
- **Abstract Entity Type (T)** — a class (e.g. "Technology companies").
- **Constant Entity (e)** — one specific entity or category (e.g. "NVIDIA", "the commercial vehicle industry").
- **Relation (r)** — a relationship (e.g. "partnered_with").
- **Sentence** — the templated query combining the three.

**PRESENCE, not evidence.** Each question asks only whether a component APPEARS in the paragraph. A paragraph can be completely unrelated to what the query claims and still answer yes on all three, because presence never requires the components to be connected to each other. Never make one answer depend on another, and never judge whether the paragraph supports the query.

**The three questions — judge each from the paragraph ALONE (never use outside knowledge):**

1. **instance_present** — does the paragraph contain a concrete, named instance of T? COUNTS: an entity carrying the type's label; an entity shown performing the type's defining function even without the label (a person the text says wrote and ran a TV series is an instance of *Showrunners*); the named owner or operator of the type's characteristic asset (the company that owns a major gas pipeline is an instance of *Pipeline Operators*); a closely related subtype (a motel chain for *Hotel Chains*). DOES NOT COUNT: the bare category word; aggregate statistics ("over 400 firms"); a placeholder or example name in formatting instructions ("entries such as XYZ Corp"); an entity named ONLY as a cited source, author, or publisher — being cited shows nothing in-type about it, however in-type its name sounds; an entity merely associated with the type (its regulator, parent, or product) with nothing in-type shown.

2. **relation_present** — is the relation's core action or state ACTUALLY EXPRESSED somewhere in the paragraph — performed, held, or completed by someone — in its characteristic shape but NOT necessarily with the query's arguments? Drop only the query-side binding: "the film, written by Nora Ephron" satisfies *wrote_films_for* (no "for" argument needed); "operates stores across three provinces" satisfies *operates_in* (some locative is needed — operating ships is not operating IN anywhere); "the port tranships containers" satisfies *handles_cargo_from*. Substance-preserving paraphrase counts; past or completed acts count; a stated role or designed function of a specific entity counts (an "Arctic patrol vessel" expresses patrolling the Arctic); a verb phrase describing an actual act counts even inside a cited headline ("Toyota Opens Kentucky Plant" expresses opening a plant). THESE GUARDS OVERRIDE EVERYTHING ABOVE — answer no when the only match is: a different or adjacent action, or the same verb aimed at a different characteristic object (coordinating budgets is not coordinating governance); ability, permission, intention, or negation ("may perform", "considers entering", "has no operations in the region"); definitional or legal boilerplate about the concept with nobody doing it ("all contributions become the licensor's property" defines terms — nobody contributes); a verbless heading, list title, ranking name, or statistic ("List of banks in the United Kingdom", "ranked ninth in attendance" express no act).

3. **constant_present** — does the paragraph refer to e, even as pure background? Exact wording is NOT required. COUNTS: close synonyms ("cardiac surgery" for *heart surgery*); near-equivalent terms ("wind farms" for *the wind energy sector*); a named member, product, or example of a category-valued e ("heavy trucks" for *the commercial vehicle industry*); a contextual instantiation whose qualifier the context supports (plain "hospitals" in a national-health-service passage for *public hospitals*); e's name inside a proper noun when the content is genuinely about e (the "Clean Air Act" refers to air-quality regulation; an airline flying jets refers to *the aviation sector*). DOES NOT COUNT: a single sub-part or member standing alone for a named whole (a UK headquarters does not make *Europe* present; "the UAE" is not *the Middle East*) — only constituents jointly standing in for the whole count ("U.S. and Canadian regulators" for *North America*); a qualified e whose qualifier the context does not fit ("jazz festivals" discussed apart from Montreal is not *Montreal jazz festivals*).

**Worked micro-examples** for the query "Technology companies partnered with NVIDIA." (T = Technology companies, r = partnered_with, e = NVIDIA):
- "Intel competes with NVIDIA in the GPU market." -> {"instance_present": true, "relation_present": false, "constant_present": true}  (competing is a different relation)
- "Microsoft partnered with OpenAI." -> {"instance_present": true, "relation_present": true, "constant_present": false}
- "The Louvre partnered with NVIDIA on a digitization project." -> {"instance_present": false, "relation_present": true, "constant_present": true}
- "AMD reported record quarterly revenue." -> {"instance_present": true, "relation_present": false, "constant_present": false}

For "Directors made films for Netflix." (T = Directors, r = made_films_for, e = Netflix):
- "Roma, directed by Alfonso Cuaron, was distributed by Netflix." -> {"instance_present": true, "relation_present": true, "constant_present": true}  (Cuaron performs the director function; directing a film is the core act — no "for Netflix" binding needed)

For "Automotive suppliers serving the commercial vehicle industry." (T = Automotive suppliers, r = serves, e = the commercial vehicle industry):
- "Knorr-Bremse supplies braking systems for heavy trucks." -> {"instance_present": true, "relation_present": true, "constant_present": true}  (heavy trucks ARE the commercial vehicle industry)

**Output** — return ONLY this JSON object, with no explanation or extra text:
{"instance_present": true|false, "relation_present": true|false, "constant_present": true|false}
"""


# Ablation variant (NOT canonical): v5 with the retrieval-audit framing removed
# (ablation question, 2026-07-16: does telling the model these are retrieved
# top-ranked results help?). Everything else is byte-identical to v5.
_V5_FRAMING = ("You are auditing the top-ranked results of retrieval systems on a "
               "relation-evidence benchmark. A retrieval system returned the PARAGRAPH "
               "below among its top results for the QUERY. Answer three INDEPENDENT "
               "yes/no presence questions about the paragraph.")
_V5_NOCTX_OPENING = ("Answer three INDEPENDENT yes/no presence questions about a "
                     "PARAGRAPH with respect to a QUERY.")
fp_bits_prompt_v5_noctx = fp_bits_prompt_v5.replace(_V5_FRAMING, _V5_NOCTX_OPENING)
assert fp_bits_prompt_v5_noctx != fp_bits_prompt_v5, "framing sentence not found"

# Versioned registry of the bits prompts (the categorical v1/v2 live separately
# above; the pipeline's canonical version is FP_TAXONOMY_PROMPT_VERSION).
FP_BITS_PROMPTS: dict[str, str] = {
    "v3": fp_bits_prompt,
    "v4": fp_bits_prompt_v4,
    "v5": fp_bits_prompt_v5,
    "v5noctx": fp_bits_prompt_v5_noctx,  # ablation only
}


def build_fp_bits_prompt(version: str) -> str:
    """The bits system prompt for an explicit version ('v3' | 'v4')."""
    return FP_BITS_PROMPTS[version]


def build_fp_taxonomy_system_prompt() -> str:
    """The system prompt for the FP classifier (canonical version)."""
    return FP_BITS_PROMPTS[FP_TAXONOMY_PROMPT_VERSION]


# ---------------------------------------------------------------------------
# Reasoned variant (reveal aid, NOT the pipeline classifier): same v3 body and
# questions, but the output schema adds a brief paragraph-citing reasoning per
# criterion. This is byte-identical to the instructions given to the Opus 4.8
# reveal agents, so the two judges' reasonings are directly comparable.
# ---------------------------------------------------------------------------
_REASONING_DUTY = """**Your reasoning duty**: for EACH of the three questions, give a brief reasoning (1-2 sentences) that cites the paragraph — name the concrete instance you found (or say none qualifies and why the near-misses fail), quote the phrase that asserts the relation (or say no assertion exists), and say where/how the constant entity appears (or doesn't). If a call is borderline under the matching semantics, say so explicitly and pick the closer answer. Never use outside knowledge; never judge overall evidence — only the three presence questions.

**Output** — your ENTIRE final message must be exactly this one JSON object, no preamble, no markdown fences, no trailing text:
{"instance_present": true|false, "instance_reasoning": "...", "relation_present": true|false, "relation_reasoning": "...", "constant_present": true|false, "constant_reasoning": "..."}
"""

REASONING_FIELDS: tuple[str, ...] = (
    "instance_reasoning", "relation_reasoning", "constant_reasoning",
)


def build_fp_bits_reasoned_prompt() -> str:
    """v3 body + per-criterion reasoning output (the reveal judges' prompt)."""
    return fp_bits_prompt.split("**Output**")[0] + _REASONING_DUTY


def parse_fp_bits_reasoned_output(raw_output: str) -> dict | None:
    """Parse bits + per-criterion reasonings. Bits are required (else None);
    missing reasonings degrade to empty strings rather than failing the pair."""
    text = _strip_think(raw_output).strip()
    # The reasoning strings may legally contain braces-free prose; scan candidate
    # JSON objects greedily from the first '{' to the last '}' first, then fall
    # back to the flat-object scan used by the bare parser.
    candidates = []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates.extend(re.findall(r"\{[^{}]*\}", text))
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        coerced = {b: _coerce_bool(data.get(b)) for b in BITS}
        if all(v is not None for v in coerced.values()):
            out: dict = dict(coerced)
            for f in REASONING_FIELDS:
                v = data.get(f)
                out[f] = v.strip() if isinstance(v, str) else ""
            return out
    bare = parse_fp_bits_output(raw_output)
    if bare is not None:
        return {**bare, **{f: "" for f in REASONING_FIELDS}}
    return None


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
_CATEGORY_SET = frozenset(CATEGORIES)
_SUB_FLAG_SET = frozenset(SUB_FLAGS)

# Same answer markers as the judge parser: reasoning models may wrap the answer.
_ANSWER_MARKERS = ("</think>", "<|channel|>final<|message|>", "assistant<|channel|>final")


def _strip_think(text: str) -> str:
    cut = -1
    for marker in _ANSWER_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            cut = max(cut, idx + len(marker))
    return text[cut:] if cut != -1 else text


def _normalize(category: object, sub_flag: object) -> dict | None:
    """Validate + normalize a (category, sub_flag) pair; None if the category is unusable."""
    if not isinstance(category, str):
        return None
    cat = category.strip().lower().replace("-", "_").replace(" ", "_")
    if cat not in _CATEGORY_SET:
        return None
    flag = None
    if isinstance(sub_flag, str):
        f = sub_flag.strip().lower().replace("-", "_").replace(" ", "_")
        if f in _SUB_FLAG_SET:
            flag = f
    # sub_flag is defined only for wrong_relation; anything else is normalized away.
    if cat != "wrong_relation":
        flag = None
    return {"category": cat, "sub_flag": flag}


def parse_fp_taxonomy_output(raw_output: str) -> dict | None:
    """Parse the classifier's output into {"category": str, "sub_flag": str | None}.

    Returns None when no valid category can be recovered (the pair stays missing on
    disk, so a retry batch repicks it — same convention as the judge pipeline).
    A wrong_relation answer with a missing/invalid sub_flag is still accepted
    (sub_flag None): the category-level taxonomy must not lose a pair over the flag.
    """
    text = _strip_think(raw_output).strip()

    # JSON fast path: scan flat objects for a valid category (last valid one wins,
    # so a reasoning trace that quotes candidate JSON doesn't shadow the answer).
    best: dict | None = None
    for blob in re.findall(r"\{[^{}]*\}", text):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "category" in data:
            norm = _normalize(data.get("category"), data.get("sub_flag"))
            if norm is not None:
                best = norm
    if best is not None:
        return best

    # Regex fallback: tolerate minor JSON malformation.
    low = text.lower()
    cm = re.search(r'"category"\s*:\s*"([a-z_ -]+)"', low)
    if cm:
        fm = re.search(r'"sub_flag"\s*:\s*"([a-z_ -]+)"', low)
        norm = _normalize(cm.group(1), fm.group(1) if fm else None)
        if norm is not None:
            return norm

    # Bare-keyword last resort: exactly one category name present in the tail.
    tail = low[-200:]
    hits = [c for c in CATEGORIES if c in tail]
    if len(hits) == 1:
        flag = next((f for f in SUB_FLAGS if f in tail), None)
        return _normalize(hits[0], flag)
    return None


# --------------------------------------------------------------------------- #
# v3 output parsing — three presence bits
# --------------------------------------------------------------------------- #
def _coerce_bool(val: object) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, int) and val in (0, 1):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return None


def parse_fp_bits_output(raw_output: str) -> dict | None:
    """Parse the v3 classifier's output into {bit: bool for bit in BITS}.

    Returns None when any of the three bits cannot be recovered (the pair stays
    missing on disk so a retry batch repicks it). The last complete JSON object
    wins, so a reasoning trace that quotes candidate JSON doesn't shadow the answer.
    """
    text = _strip_think(raw_output).strip()

    best: dict | None = None
    for blob in re.findall(r"\{[^{}]*\}", text):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        coerced = {b: _coerce_bool(data.get(b)) for b in BITS}
        if all(v is not None for v in coerced.values()):
            best = coerced
    if best is not None:
        return best

    # Regex fallback: tolerate minor JSON malformation, one key at a time.
    low = text.lower()
    out: dict = {}
    for b in BITS:
        m = re.search(rf'"{b}"\s*:\s*"?(true|false|1|0|yes|no)"?', low)
        if not m:
            return None
        out[b] = _coerce_bool(m.group(1))
    return out if all(v is not None for v in out.values()) else None
