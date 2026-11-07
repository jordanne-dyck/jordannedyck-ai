"""
Citation checks and the per-claim grounding judge, ported from the web repo so
the eval scores every model the way the live site labels gpt-4o's answers.

MUST STAY IN SYNC with jordannedyck-ai-web:
  - parse_citations(), citation_labels()  <-  parseCitations(), citationLabels()
    in lib/citations.ts (both in engine/citations.py since 2026-11-07)
  - judge_grounding()                     <-  judgeGrounding() in lib/classify.ts
    (same model, temperature, system text, prompt layout and verdict rule)
Ported 2026-10-15 for the model comparison (run_models.py). Edit both or neither.

Deliberate differences from the site:
  - The judge keeps every claim's verdict, not only the failed ones, so an eval
    result can be read claim by claim.
  - No PII redaction: eval questions are fixed, not visitor text.
  - Lead-in lines are joined to the claim after them before judging
    (fold_lead_ins, 2026-10-30). Mirrored in foldLeadIns() in lib/classify.ts
    (web PR #52). Headings with no colon were added on 2026-11-01; until the
    web side mirrors that, the two differ on headings. Brief:
    nimbalyst-local/web-brief-judge-fold-headings-2026-11-01.md.
"""

import json
import re

from engine.citations import (  # noqa: F401  re-exported: run_models.py and selftest.py use them
    CITE_RE, FIGURE_RE, ID_RE, _claim_figures, _found_in, _push_segment, citation_labels,
    parse_citations,
)
from engine.prompt import SYSTEM_PROMPT

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_COST_PER_1M_IN, JUDGE_COST_PER_1M_OUT = 0.15, 0.60

# Answers longer than this many segments are judged on their first ones only.
MAX_JUDGED_SEGMENTS = 40

# The CORE POSITIONING section, which the judge counts as a source. Read from
# SYSTEM_PROMPT the way classify.ts reads it from buildSystemPrompt('').
_template = SYSTEM_PROMPT.replace("{context}", "")
POSITIONING = re.sub(
    r"\n-{3,}\s*$", "",
    _template[_template.index("# CORE POSITIONING"):_template.index("# HOW TO RESPOND")],
).strip()


# ── lib/citations.ts ──────────────────────────────────────────────────────────
# parse_citations() and citation_labels() live in engine/citations.py since 2026-11-07.

def leaked_citations(visible_answer, chunks):
    """Chunk ids still in the text a visitor sees after stripping.

    Not a site label. A model that cites in a format CITE_RE does not match,
    such as "(ops-009)", would show raw ids to visitors.
    """
    ids = {c["chunk_id"] for c in chunks if c.get("chunk_id")}
    return sorted(i for i in ids if i in visible_answer)


# ── lib/classify.ts judgeGrounding() ──────────────────────────────────────────

JUDGE_SYSTEM = (
    "You check whether each numbered claim in an answer is supported by the source chunks it cites. "
    "Each claim lists the chunk ids it cites; the context gives each chunk under its id. "
    "The context also has a [positioning] block of facts about her that the site states on her behalf. "
    'A claim the positioning supports is "yes", whatever it cites. '
    "Check every specific: numbers, dates, employers, product and project names, job titles, outcomes, "
    "and any statement about how she works or what she prefers. "
    '"yes": the cited chunks say it. '
    '"other": the cited chunks do not say it, but another chunk in the context does. '
    '"no": no chunk in the context says it. '
    "A claim that goes further than its source — a stronger word, a bigger number, or a reading the source does not state — is not supported. "
    'A claim citing "none" is "yes" if it only says the context does not cover something, or makes no factual claim; '
    "otherwise judge it against the whole context. "
    "Return one entry per claim. Judge only against the context. Do not use outside knowledge."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "supported": {"type": "string", "enum": ["yes", "other", "no"]},
                    "reason": {
                        "type": "string",
                        "description": 'Empty for "yes". Otherwise one sentence naming what the source does not say.',
                    },
                },
                "required": ["n", "supported", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


# An uncited segment ending in a colon introduces what follows: "Her hands-on
# builds include:". Alone it states nothing the chunks could carry, and the
# judge marked such lines "no": 49 times in the 215 answers of 10-22. Some do
# carry a detail ("The evidence settled it within a quarter:"), so they are
# joined to the next claim and judged with it, not dropped. Re-judged on
# 2026-10-30 (judge_check.py), folding took the single-turn run from 46
# flagged answers to 17, and agreement with the blind D1 scores held.
LEAD_IN_END_RE = re.compile(r":[*_\s]*$")
LIST_MARK_RE = re.compile(r"^(?:[-*]|\d+\.)\s+")
# A heading with no colon introduces what follows the same way: a markdown
# heading, or a line that is all bold ("**How it was built**"). The judge
# marked 9 such lines "no" in the 10-22 re-judges (2026-11-01). Folding keeps
# a detail in one checked ("**Loblaw Companies Limited (2011-2016)**").
HEADING_RE = re.compile(r"#{1,6}\s+\S.*|\*\*[^*]+\*\*|__[^_]+__")


def is_lead_in(s):
    """An uncited segment that introduces the next one, not a claim on its own."""
    return not s["ids"] and bool(LEAD_IN_END_RE.search(s["text"]) or HEADING_RE.fullmatch(s["text"].strip()))


def fold_lead_ins(segments):
    """Joins each run of lead-in lines and headings to the segment after it,
    which keeps its ids. A lead-in with nothing after it is judged as it is."""
    out, lead = [], []
    for s in segments:
        if is_lead_in(s):
            lead.append(s["text"])
            continue
        if lead:
            s = {"text": " ".join(lead + [LIST_MARK_RE.sub("", s["text"])]), "ids": s["ids"]}
            lead = []
        out.append(s)
    return out + [{"text": t, "ids": []} for t in lead]


def judge_prompt(query, answer, segments, chunks, fold=True):
    """The segments the judge sees, and the user prompt it gets."""
    judged = segments or [{"text": answer, "ids": []}]
    judged = (fold_lead_ins(judged) if fold else judged)[:MAX_JUDGED_SEGMENTS]
    prompt = (
        f"QUESTION:\n{query}\n\nCONTEXT:\n[positioning]\n{POSITIONING}\n\n"
        + "\n\n".join(f"[{c.get('chunk_id') or f'result {i + 1}'}]\n{c['content']}"
                      for i, c in enumerate(chunks))
        + "\n\nCLAIMS:\n"
        + "\n".join(f"{i + 1}. (cites: {', '.join(s['ids']) or 'none'}) {s['text']}"
                    for i, s in enumerate(judged))
    )
    return judged, prompt


def judge_grounding(client, query, answer, segments, chunks, *, fold=True):
    """Judges each cited segment against the chunks it cites, with gpt-4o-mini.
    fold=False is the pre-2026-10-30 way, kept for judge_check.py's comparison."""
    judged, prompt = judge_prompt(query, answer, segments, chunks, fold)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "claims", "strict": True, "schema": JUDGE_SCHEMA}},
        timeout=30,
    )
    claims = json.loads(resp.choices[0].message.content)["claims"]

    # The model's numbering is checked against the segments; anything it made up is dropped.
    verdicts = []
    for c in claims:
        if 1 <= c["n"] <= len(judged):
            s = judged[c["n"] - 1]
            verdicts.append({"n": c["n"], "ids": s["ids"], "text": s["text"][:200],
                             "supported": c["supported"], "reason": c["reason"][:200]})

    failed = [v for v in verdicts if v["supported"] != "yes"]
    unsupported = sum(1 for v in failed if v["supported"] == "no")
    verdict = ("grounded" if unsupported == 0
               else "not_grounded" if unsupported * 2 > len(judged) else "partial")
    u = resp.usage
    cost = (u.prompt_tokens / 1e6) * JUDGE_COST_PER_1M_IN + (u.completion_tokens / 1e6) * JUDGE_COST_PER_1M_OUT
    return {
        "verdict": verdict,
        "checked": len(judged),
        "folded": len(segments) - len(fold_lead_ins(segments)) if fold else 0,
        "no": unsupported,
        "other": sum(1 for v in failed if v["supported"] == "other"),
        "claims": verdicts,
        "cost_usd": round(cost, 6),
    }
