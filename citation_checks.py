"""
Citation checks and the per-claim grounding judge, ported from the web repo so
the eval scores every model the way the live site labels gpt-4o's answers.

MUST STAY IN SYNC with jordannedyck-ai-web:
  - parse_citations(), citation_labels()  <-  parseCitations(), citationLabels()
    in lib/citations.ts
  - judge_grounding()                     <-  judgeGrounding() in lib/classify.ts
    (same model, temperature, system text, prompt layout and verdict rule)
Ported 2026-10-15 for the model comparison (run_models.py). Edit both or neither.

Two deliberate differences from the site:
  - The judge keeps every claim's verdict, not only the failed ones, so an eval
    result can be read claim by claim.
  - No PII redaction: eval questions are fixed, not visitor text.
"""

import json
import re

from run_eval import _CITE_ID, CITE_RE, SYSTEM_PROMPT, extract_figures

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_COST_PER_1M_IN, JUDGE_COST_PER_1M_OUT = 0.15, 0.60

ID_RE = re.compile(_CITE_ID)
FIGURE_RE = re.compile(r"\$?\d[\d,]*\.?\d*\s*(?:%|x|M|B|K|million|billion|\+)?")

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

def _push_segment(segments, text, ids):
    t = re.sub(r"^[\s.,;:!?)]+", "", text).strip()
    # Headings, bullet markers and stray punctuation are not claims.
    if not re.search(r"[a-z]{3}", t, re.IGNORECASE) and not ids:
        return
    segments.append({"text": t, "ids": ids})


def parse_citations(raw):
    """Splits the cited answer into segments, each with the ids cited at its end."""
    segments = []
    for line in raw.split("\n"):
        last = 0
        for m in CITE_RE.finditer(line):
            _push_segment(segments, line[last:m.start()], ID_RE.findall(m.group(0)))
            last = m.end()
        _push_segment(segments, line[last:], [])
    cited = list(dict.fromkeys(i for s in segments for i in s["ids"]))
    return segments, cited


def _claim_figures(text):
    """claimFigures(): extract_figures()'s rule, kept in order of appearance."""
    found = extract_figures(text)
    tokens = [m.group(0).strip() for m in FIGURE_RE.finditer(text)]
    return list(dict.fromkeys(t for t in tokens if t in found))


def _found_in(figure, haystack):
    """Digits only, as run_eval.py's figure check does: "$150M+" matches "150"."""
    return re.sub(r"\D", "", figure) in re.sub(r"\D", "", haystack)


def citation_labels(segments, chunks):
    """invented, unsupported, uncited and top_uncited, as the site stores them."""
    by_id = {c["chunk_id"]: c["content"] for c in chunks if c.get("chunk_id")}
    all_context = "\n".join(c["content"] for c in chunks)
    cited = {i for s in segments for i in s["ids"]}
    invented = [i for i in dict.fromkeys(i for s in segments for i in s["ids"]) if i not in by_id]

    unsupported = []
    for s in segments:
        if not s["ids"]:
            continue
        source = "\n".join(by_id.get(i, "") for i in s["ids"])
        for token in _claim_figures(s["text"]):
            if not _found_in(token, source):
                unsupported.append({"ids": s["ids"], "token": token,
                                    "in_context": _found_in(token, all_context)})

    top = max(chunks, key=lambda c: c["similarity"]) if chunks else None
    return {
        "segments": len(segments),
        "uncited": sum(1 for s in segments if not s["ids"]),
        "invented": invented,
        "unsupported": unsupported,
        "top_uncited": bool(top and top.get("chunk_id") and top["chunk_id"] not in cited),
    }


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


def judge_grounding(client, query, answer, segments, chunks):
    """Judges each cited segment against the chunks it cites, with gpt-4o-mini."""
    judged = (segments or [{"text": answer, "ids": []}])[:MAX_JUDGED_SEGMENTS]
    prompt = (
        f"QUESTION:\n{query}\n\nCONTEXT:\n[positioning]\n{POSITIONING}\n\n"
        + "\n\n".join(f"[{c.get('chunk_id') or f'result {i + 1}'}]\n{c['content']}"
                      for i, c in enumerate(chunks))
        + "\n\nCLAIMS:\n"
        + "\n".join(f"{i + 1}. (cites: {', '.join(s['ids']) or 'none'}) {s['text']}"
                    for i, s in enumerate(judged))
    )
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
        "no": unsupported,
        "other": sum(1 for v in failed if v["supported"] == "other"),
        "claims": verdicts,
        "cost_usd": round(cost, 6),
    }
