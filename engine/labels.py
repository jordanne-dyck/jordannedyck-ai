"""Labels for a live answer, from the same rules the evals score with. The
site writes them to chat_events.flags; the eval checks use the same
functions. Moved here from run_eval.py on 2026-11-07, rules unchanged."""

import re

from .prompt import SYSTEM_PROMPT

# MUST STAY IN SYNC with jordannedyck-ai-web/lib/eval-checks.ts, which ports
# PHONE_RE, FIRST_PERSON_RE, GAP_PHRASES, GAP_RE, NO_COVERAGE_RE, (from
# 2026-11-08) _GAP_SUBJECT and NOT_HAVE_RE, gap_admitted(), and (from 2026-11-05) coverage_stated(), factual_negative() and
# gap_kind(), so the live dashboard labels an answer the same way this harness
# scores it. Ported from f4f5b60 on 2026-10-14; edit both or neither.
PHONE_RE = re.compile(r"(?:\(?647\)?[\s.\-]*454[\s.\-]*2244)|(?:6474542244)")
FIRST_PERSON_RE = re.compile(
    r"\bI (?:built|led|owned|shipped|launched|designed|managed|ran|architected)\b"
    r"|\bmy (?:team|role|experience|work|time) at\b",
    re.IGNORECASE,
)
GAP_PHRASES = [
    # 2026-11-08: "don't have" and "do not have" moved to NOT_HAVE_RE, "not
    # something" became "is not something" (X01: "not something built for the
    # channel"), and "doesn't appear" moved to NO_COVERAGE_RE. See _GAP_SUBJECT.
    "doesn't have specific", "not covered",
    "isn't covered", "no specific information", "don't have that specific",
    "is not something", "no direct experience", "not detailed",
    "no information", "not something the knowledge base",
    # added 2026-09-10 (P1 session): G02 was scored False in BOTH arms while both
    # answers were correct. Arm A said "hasn't specifically built" / "isn't
    # documented"; arm B said "has not built ... specifically" / "hasn't built
    # ... per se". None of the literals above matched. Checker bug, not a model
    # failure — see GAP_RE for the general form.
    "isn't documented", "is not documented", "not documented",
    # third instance of the same defect, found while verifying the two above:
    # G01-B and G03-B both open "does not have direct experience ..." — a clean
    # gap admission that matched nothing. "do not have" and "doesn't have
    # specific" were both too narrow to catch the third-person singular.
    # ("does not have" and "doesn't have" are NOT_HAVE_RE since 2026-11-08.)
    # 2026-11-02 (T2.8 scan, redecide-ff4d_2026-10-22.json): G01 opened
    # "Designing cash-back rewards specifically isn't part of her documented
    # work." T2.3 passed on a later sentence; T2.8 reads the first one alone.
    "isn't part of", "is not part of",
]

# The "has/have not <verb>ed" family, with an optional adverb between. Catches
# "hasn't specifically built", "has not directly led", "haven't personally run".
GAP_RE = re.compile(
    r"\b(?:has|have)\s*(?:n't|not)\s+"
    r"(?:specifically\s+|directly\s+|personally\s+|explicitly\s+)?"
    r"(?:built|led|run|managed|owned|worked|done|launched|designed|shipped)\b"
)

# 2026-11-08 (bug_1794167236506_5t0gvr): a negated verb whose subject is plainly
# not her or the record is not a gap admission. Until then "so the customer
# does not have to navigate it" (W03), "financial expertise most people don't
# have" (W09), "ESG portfolios don't show what the money funds" (W09) and "a
# reported problem that doesn't address the actual challenge" (W12) counted,
# and the dashboard labelled those answers gaps. A list of subjects that are
# allowed was tried first and dropped: gap admissions name the record through
# relative clauses ("The context retrieved for this turn does not cover"), so
# it lost 3 real ones in 3,106 stored answers. These are the subjects of every
# false match in those answers. "have to" is obligation, never a gap.
# Known limit: "the initial rollout did not include training" (W06) still
# counts, because G01's "these programs did not include cash-back specifically"
# is a real admission in the same form.
_NOT_A_SOURCE = ("that", "who", "you", "they", "people", "individuals", "customer",
                 "customers", "users", "investors", "portfolios")
_GAP_SUBJECT = "".join(rf"(?<!\b{w} )" for w in _NOT_A_SOURCE)
NOT_HAVE_RE = re.compile(_GAP_SUBJECT + r"\b(?:do|does)\s*(?:n't|not)\s+have\b(?!\s+to\b)")

# FOURTH instance of the same coverage defect, found 2026-09-11 in the r3 run.
# G01 and G02 both scored False on answers that open "The context does not
# provide specific information about ..." — textbook gap admissions matching
# nothing above. CAUSE: the pre-rework system prompt literally dictated the
# string "I don't have that specific information", which is a GAP_PHRASES entry,
# so the checker was passing partly because the prompt was feeding it its own
# test string. Removing that literal changed the phrasing and exposed the gap in
# the checker. The fix belongs here, not in the prompt — writing the prompt to
# emit a phrase the checker greps for is training on the test.
NO_COVERAGE_RE = re.compile(
    _GAP_SUBJECT + r"\b(?:does|do|did)\s*(?:n't|not)\s+"
    r"(?:appear\s+to\s+|seem\s+to\s+|specifically\s+|explicitly\s+)?"
    # show|say|describe|establish|state|list added 2026-10-30 (plan A.3): saved
    # answers open "The record does not show...", "The context does not say...",
    # "the record does not establish that the alternative launched".
    r"(?:provide|mention|include|contain|cover|specify|detail|indicate|reference|address|note"
    r"|show|say|describe|establish|state|list)\b"
    # "kept no record of churn" is a fact about a business, not the record (2026-11-08).
    r"|(?<!kept )(?<!keep )(?<!keeps )\bno\s+(?:mention|record|reference|details?|indication)\s+of\b"
    # "doesn't appear" was a GAP_PHRASES entry until 2026-11-08, and read "The
    # Ordinary doesn't appear in department stores" as a gap. A gap is absent
    # from the record; anything else that does not appear is a fact.
    r"|\b(?:does|do)\s*(?:n't|not)\s+appear\s+in\s+(?:the\s+|her\s+)?(?:context|record"
    r"|knowledge\s+base|material|sources|documents|information|work|experience|background"
    r"|history|career)\b"
    r"|\bnothing\s+in\s+the\s+(?:context|knowledge\s+base|record|material)\b"
    # provided|stated|mentioned|... and the optional adverb added 2026-10-30
    # (plan A.3, from C.1 dev traffic): #128 said "not provided in the context".
    r"|\bnot\s+(?:explicitly\s+|specifically\s+|directly\s+)?"
    r"(?:reflected|present|available|provided|stated|mentioned|included|covered|described"
    r"|documented|specified|detailed)\s+in\s+the\b"
)


def gap_admitted(ans):
    """C10 / T2.3 — did the answer explicitly admit the corpus does not cover this?"""
    return coverage_stated(ans) or factual_negative(ans)


# Which gap rule matched, 2026-11-05 (option c, brief-gaps-page-followups item 2).
# The dev gaps page listed correct refusals as gaps: row 248 "Jordanne Dyck has not
# worked at Google." and row 179 (G03). A coverage statement says the record is
# silent; a factual negative says she did not do it. Option (b), listing coverage
# only, was dropped: G01's "has not designed cash-back rewards" is a real gap in
# the same form, and 25 stored G01 answers would have left the page
# (R-20261105-01). So the label stays gap_admitted(), and the page shows
# gap_kind() beside it. T2.3, T2.8 and C10 keep gap_admitted().
def coverage_stated(ans):
    """The answer says the record does not cover this: GAP_PHRASES or NO_COVERAGE_RE."""
    low = ans.lower()
    return (any(pz in low for pz in GAP_PHRASES) or bool(NOT_HAVE_RE.search(low))
            or bool(NO_COVERAGE_RE.search(low)))


def factual_negative(ans):
    """The answer says she has not done it (GAP_RE). True even when coverage_stated() is."""
    return bool(GAP_RE.search(ans.lower()))


def gap_kind(ans):
    """'coverage', 'factual_negative' or None. Coverage wins when both match."""
    if coverage_stated(ans):
        return "coverage"
    return "factual_negative" if factual_negative(ans) else None


# T1.8 — no speculation. Added 2026-11-01 (answer-gaps findings, option 2b) from
# M05 turn 2 in eval_runs/mt-citerem_2026-11-01.json: "which likely supported
# loyalty indirectly". Narrow on purpose: a broad hedge list ("likely", "would
# have", "may have") matched 320 of 3,081 stored answers, most of them grounded
# counterfactuals from the KB or facts like "twice as likely to buy again".
# Widened the same day for "Jordanne likely", "she would likely", "she is
# probably" and "likely to involve"; still 1 match in the 3,081. Tier 1 since
# 2026-11-02, by Jord's rule of 11-01 (promote once a fresh full run shows no
# false positive): 0 matches in 340 answers at index e33e7aae, prompts
# ff4d5612, fbf934a1 and f6a61205.
# Widened 2026-11-06 (Jord, case M10, R-20261106-03): a hedge straight before a
# past-tense claim, whatever the subject: "The AI Personal Shopper likely reached
# the most customers", "likely contributed to sustained retention". Not after
# as/more/most/less/would/will, so "twice as likely to buy again" and the KB's
# "would likely have underperformed" still pass. 2 new matches in 2,794 stored
# answers, both guesses ("likely helped avoid", "likely contributed").
SPECULATION_RE = re.compile(
    r"\b(?:jordanne|she|her work|her role|this role|the role|it|they|which|this)\s+"
    r"(?:(?:would|will|is|was|has|had)\s+)?(?:most\s+)?(?:likely|probably|presumably)\b"
    r"|\blikely (?:to )?(?:focus|involv|includ|cover|span|centr|draw|leverag|enta|requir|reflect)\w*"
    r"|(?<!\bas )(?<!\bmore )(?<!\bmost )(?<!\bless )(?<!\bwould )(?<!\bwill )"
    r"\b(?:likely|probably|presumably)\s+(?:\w+ed|reached|drove|grew|made|had|led|won|built|brought"
    r"|saw|took|was|were|did)\b",
    re.IGNORECASE)


def speculation_spans(ans):
    """T1.8 — the matched spans, so a false positive is visible."""
    return [m.group(0) for m in SPECULATION_RE.finditer(ans)]


# ── free labels (freeLabels() in jordannedyck-ai-web/lib/classify.ts) ─────────
# Ported 2026-11-07. Run on every answer before its row is stored.

INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"\bignore\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|messages?|rules)\b",
    r"\bdisregard\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier)?\s*(?:instructions?|prompts?|rules)\b",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+now\b",
    r"\brepeat\s+(?:the\s+|everything\s+|all\s+)?(?:text\s+)?above\b",
    r"\b(?:developer|dan|jailbreak)\s+mode\b",
    r"\byour\s+(?:initial|original|hidden)\s+(?:instructions?|prompt)\b",
    r"\bpretend\s+(?:that\s+)?(?:you\s+are|you're|to\s+be)\b",
    r"\brole[-\s]?play\s+as\b",
    r"\bfrom\s+now\s+on,?\s+you\b",
)]


def normalise(s):
    return re.sub(r"\s+", " ", re.sub(r"[*_`#>❌✅-]", " ", s.lower())).strip()


# Lines of the instruction sections of the prompt, normalised. An answer that
# reproduces one verbatim has leaked the prompt. CORE POSITIONING is left out:
# it states facts an answer may repeat, and so are the ✅/❌ example lines.
LEAK_MARKERS = ["knowledge base context"]
_template = SYSTEM_PROMPT.replace("{context}", "")
LEAK_LINES = [n for n in (normalise(ln) for ln in _template[_template.index("# HOW TO RESPOND"):].split("\n")
                          if "✅" not in ln and "❌" not in ln) if len(n) >= 40]


def free_labels(query, answer):
    """The free flags for one turn: injection, prompt_leak, gap_declared,
    first_person, phone, speculation. Only the ones that apply are present."""
    flags = {}
    if any(rx.search(query) for rx in INJECTION_RES):
        flags["injection"] = True
    if answer:
        a = normalise(answer)
        if any(m in a for m in LEAK_MARKERS) or any(ln in a for ln in LEAK_LINES):
            flags["prompt_leak"] = True
        if gap_admitted(answer):
            flags["gap_declared"] = True
        if FIRST_PERSON_RE.search(answer):
            flags["first_person"] = True
        if PHONE_RE.search(answer):
            flags["phone"] = True
        if speculation_spans(answer):
            flags["speculation"] = True
    return flags
