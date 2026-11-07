"""Citation markers in an answer: [ops-009] and [a-1, b-2]. Moved here from
run_eval.py on 2026-11-07."""

import re


# MUST STAY IN SYNC with CITE_RE and stripCitations() in
# jordannedyck-ai-web/lib/citations.ts, which removes the same markers from the
# site's stream. Added 2026-10-15 with the "Cite every claim" instruction.
_CITE_ID = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d+"
CITE_RE = re.compile(rf"[ \t]*\[{_CITE_ID}(?:\s*,\s*{_CITE_ID})*\]")


def strip_citations(ans):
    """The answer as a site visitor sees it: [ops-009] citation markers removed."""
    return CITE_RE.sub("", ans)
