"""Replies the site gives without calling the answer model: the career-overview
lead (and its replay) and the refusal to repeat the instructions, plus the
questions that trigger them. Moved here from run_eval.py on 2026-11-07, text
unchanged. run_eval.py's timeline_drift() and refusal_drift() compare them with
the web repo's copies until the site calls this engine."""

import re

# ── career timeline ──────────────────────────────────────────────────────────
# MUST STAY IN SYNC with FIXED_LEAD, TIMELINE_REPLAY, isTimelineReplay(),
# asksForTimeline(), asksForOverview(), asksForCareerHistory(), MARKER_RE and
# CAREER_CHUNKS in jordannedyck-ai-web/lib/timeline.ts (frontend branch feat/career-timeline,
# d0d7ce2). timeline_drift() compares.
# Since 2026-11-04 (option D) route.ts does not call the answer model for an
# overview question: it streams FIXED_LEAD, Jordanne's own text, with the
# timeline, and flags the chat_events row timeline_lead 'fixed' ('replay' for
# TIMELINE_REPLAY). TIMELINE_NOTE is gone. The overview cases (fixed_lead in
# timeline_cases, TL01-TL04) test the route, so run_e2e.py checks them (T2.13);
# --timeline here runs the rest (TL05, TL06) through the model.
# Since 2026-11-05 a career-history question (asksForCareerHistory) also gets
# the card, flagged timeline 'asked', whatever the answer cites. It does not get
# FIXED_LEAD. run_e2e.py checks that the card arrives with explicit: true (T2.17, TL05).
TIMELINE_MARKER = "{{timeline}}"
FIXED_LEAD = (
    "Jordanne's career runs from the human side of helping customers to the digital side. She started in "
    "bilingual frontline support at Shoppers Drug Mart's call centre, and has since built the systems that "
    "decide where people help customers and where technology does: at DECIEM, Regimen Builder, the AI Personal "
    "Shopper and automated service that cut resolution time 86%, while annual DTC revenue grew to $150M+ with "
    f"4M+ DTC customers globally.\n\n{TIMELINE_MARKER}\n\n"
    "Today she runs Warai Neko, her independent practice, bringing that work to small businesses while "
    "continuing to refine her AI building skills with this AI-powered resume and portfolio system among "
    "other hobby projects."
)
TIMELINE_REPLAY = f"Here's her timeline again.\n\n{TIMELINE_MARKER}"
_ASKS_TIMELINE = r"\btime ?line\b"
_ASKS_OVERVIEW = (r"\bcareer\s+(?:history|path|journey|story|arc|overview|so far)\b|\bwork\s+history\b"
                  r"|\bwalk\s+(?:me|us)\s+through\s+(?:her|jordanne'?s?)\s+(?:career|experience|background)\b")
_ASKS_CAREER_HISTORY = (r"\bwhere\s+(?:did|has|had)\s+(?:she|jordanne)\s+(?:work|worked)\b"
                        r"|\bwhat\s+did\s+(?:she|jordanne)\s+do\s+before\b"
                        r"|\b(?:previous|past|former|prior)\s+(?:employers?|roles?|jobs?|positions?|companies)\b"
                        r"|\b(?:what|which)\s+(?:companies|employers)\s+(?:has|did)\s+(?:she|jordanne)\s+work")
_MARKER = r"[ \t]*\{\{\s*timeline\s*\}\}[ \t]*"
MARKER_RE = re.compile(_MARKER, re.IGNORECASE)
# An answer citing one of these gets the timeline card. Not exp-031, education:
# an education answer would always bring up the whole timeline (Jordanne, 2026-11-04).
CAREER_CHUNKS = {"exp-027", "exp-028", "exp-029", "exp-030", "exp-042"}


def asks_for_timeline(query):
    return bool(re.search(_ASKS_TIMELINE, query, re.IGNORECASE))


def asks_for_overview(query):
    return asks_for_timeline(query) or bool(re.search(_ASKS_OVERVIEW, query, re.IGNORECASE))


def asks_for_career_history(query):
    """asksForCareerHistory(): the card, not FIXED_LEAD, whatever the answer cites."""
    return bool(re.search(_ASKS_CAREER_HISTORY, query, re.IGNORECASE))


def is_timeline_replay(query, timeline_shown):
    """isTimelineReplay(): the page sends timelineShown once it has shown the
    timeline this visit. The runners never send it, so they get FIXED_LEAD."""
    return bool(timeline_shown) and asks_for_timeline(query)


def without_timeline_marker(answer):
    """withoutTimelineMarker(): the answer as it is logged, labelled and judged."""
    return re.sub(_MARKER + r"(?:\r?\n)?", "", answer, flags=re.IGNORECASE)


# ── fixed refusal ────────────────────────────────────────────────────────────
# MUST MATCH EXTRACTION_RE and FIXED_REFUSAL in jordannedyck-ai-web/lib/prompt-request.ts
# (web PR #85, merged 96e2f74, 2026-11-05), character for character. refusal_drift()
# compares. route.ts answers a message that asks for the bot's own instructions with
# FIXED_REFUSAL: no model call, no timeline, no sources, and the chat_events row
# flagged fixed_refusal (decided 2026-11-05, R-20261105-08). A02's shape ("ignore
# your instructions" plus a false claim) does not match: it reaches the model to be
# corrected. SYSTEM_PROMPT's refusal paragraph is unchanged and is the backstop for
# a reworded request the regex misses, so run_eval.py still runs A01 through the
# model to measure it; run_e2e.py checks the route (T2.16).
_EXTRACTION = r"(?:\b(?:output|reveal|print|show|repeat|share|give|tell|display|dump|quote|paste|leak|disclose|recite|write out|copy)\b(?:(?!\b(?:her|she|jordanne|jordanne's|how)\b)[^.?!\n]){0,60}?|\bwhat(?:'s|\s+is|\s+are)\s+)(?:\b(?:your|this\s+(?:bot|chatbot|assistant)'?s?)\s+(?:\w+\s+){0,3}?(?:system\s+prompt|instructions|prompt|rules|guidelines|configuration)|\bthe\s+system\s+prompt|\bthe\s+(?:\w+\s+){0,3}?(?:instructions|prompt|rules|guidelines)\s+you\b)\b(?!\s+(?:for|on|about|around|when|in)\b)(?![^.?!\n]{0,25}\b(?:she|her|jordanne)\b)"
EXTRACTION_RE = re.compile(_EXTRACTION, re.IGNORECASE)
FIXED_REFUSAL = (
    "I don't repeat my instructions in chat, but how I'm built is public, prompt included: "
    "[jordannedyck-ai on GitHub](https://github.com/jordanne-dyck/jordannedyck-ai). "
    "Here, I answer questions about Jordanne's work."
)


def asks_for_instructions(query):
    """asksForInstructions(): the site answers this message with FIXED_REFUSAL."""
    return bool(EXTRACTION_RE.search(query))


# ── which fixed reply, and whether the timeline card shows ────────────────────
# Ported 2026-11-07 from route.ts (web master bbed85c) and lib/timeline.ts.

def cites_career(cited_ids):
    """citesCareer(): the answer cites one of her role-history chunks."""
    return any(i in CAREER_CHUNKS for i in cited_ids)


def leads_with_career(chunks):
    """leadsWithCareer(): the top retrieved chunk is role history. The top one
    only, so an answer that retrieved a career chunk in passing doesn't get
    the timeline."""
    return bool(chunks) and chunks[0].get("chunk_id") in CAREER_CHUNKS


def fixed_reply(query, timeline_shown=False):
    """The reply the site gives without calling the model, or None. An
    instruction request gets the refusal; an overview question gets the lead
    (the replay once the page has shown the timeline this visit). `flags` are
    what the chat_events row records."""
    if asks_for_instructions(query):
        return {"id": "refusal", "answer": FIXED_REFUSAL, "timeline": False,
                "flags": {"fixed_refusal": True}}
    if asks_for_overview(query):
        replay = is_timeline_replay(query, timeline_shown)
        return {"id": "lead", "answer": TIMELINE_REPLAY if replay else FIXED_LEAD, "timeline": True,
                "flags": {"timeline": "explicit" if asks_for_timeline(query) else "overview",
                          "timeline_lead": "replay" if replay else "fixed"}}
    return None


def timeline_card(query, cited_ids, chunks):
    """timelineFor(): the timeline under a generated answer, {reason, explicit},
    or None. `reason` is 'cited' (it cites role history), 'retrieved' (it cites
    nothing and retrieval led with role history) or 'asked' (a career-history
    question). `explicit` when the question asks about her past jobs, for any
    reason: the page then shows the card every time, not once per visit (web
    094684f, 2026-11-08, after "Where did Jordanne work before DECIEM?" showed
    no timeline in v1.16 QA)."""
    explicit = asks_for_career_history(query)
    if cites_career(cited_ids):
        return {"reason": "cited", "explicit": explicit}
    if not cited_ids and leads_with_career(chunks):
        return {"reason": "retrieved", "explicit": explicit}
    if explicit:
        return {"reason": "asked", "explicit": explicit}
    return None
