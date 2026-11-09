"""Citation markers in an answer: [ops-009] and [a-1, b-2]. Moved here from
run_eval.py on 2026-11-07."""

import re

# MUST STAY IN SYNC with CITE_RE and stripCitations() in
# jordannedyck-ai-web/lib/citations.ts, which removes the same markers from the
# site's stream. Added 2026-10-15 with the "Cite every claim" instruction.
_CITE_ID = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d+"
# Case-insensitive since 2026-11-08 (bug_1794167236506_5t0gvr): every id in the
# index is lowercase, but "[Exp-001]" was neither stripped nor counted, so a
# visitor would have seen it. 0 of 9,247 stored citations were capitalised.
CITE_RE = re.compile(rf"[ \t]*\[{_CITE_ID}(?:\s*,\s*{_CITE_ID})*\]", re.IGNORECASE)


def strip_citations(ans):
    """The answer as a site visitor sees it: [ops-009] citation markers removed."""
    return CITE_RE.sub("", ans)


def extract_figures(text):
    """Numbers that constitute a factual claim: percentages, money, multiples, counts.
    No longer used by T2.2 (see figure_values). citation_checks.py keeps it because
    it mirrors claimFigures() in the web repo's lib/citations.ts."""
    out = set()
    for m in re.finditer(r"\$?\d[\d,]*\.?\d*\s*(?:%|x|M|B|K|million|billion|\+)?", text):
        tok = m.group(0).strip()
        digits = re.sub(r"[^\d]", "", tok)
        if not digits:
            continue
        if re.fullmatch(r"(19|20)\d\d", digits):   # years are context, not claims
            continue
        if len(digits) <= 1 and "%" not in tok and "x" not in tok.lower():
            continue
        out.add(tok)
    return out


# ── segments and citation labels (lib/citations.ts) ─────────────────────────────
# Moved here from citation_checks.py on 2026-11-07.

ID_RE = re.compile(_CITE_ID, re.IGNORECASE)  # as CITE_RE; ids are lowercased when parsed
FIGURE_RE = re.compile(r"\$?\d[\d,]*\.?\d*\s*(?:%|x|M|B|K|million|billion|\+)?")


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
            _push_segment(segments, line[last:m.start()], [i.lower() for i in ID_RE.findall(m.group(0))])
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


# ── what a visitor sees of the citations (lib/citations.ts, E.2) ──────────────
# Ported 2026-11-07. The source title and the chunk's section heading, never
# chunk ids, filenames or chunk text. Chunks that share a title and heading
# share a number, numbered in the order the answer first cites them. Ids that
# were not retrieved for this turn have no title and are dropped.

def section_of(content, title):
    """sectionOf(): the chunk's first markdown heading, or None when it only
    repeats the title or starts with the title's lead."""
    line = next((ln for ln in (content or "").split("\n") if re.match(r"#{1,6}\s", ln)), None)
    if line is None:
        return None
    heading = re.sub(r"[*_`]", "", re.sub(r"^#+\s*", "", line)).strip()
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    lead = norm(re.split(r"\s[-—]\s", title)[0])
    if not heading or norm(heading) == norm(title) or norm(heading).startswith(lead):
        return None
    return heading


def build_sources(segments, chunks):
    """buildSources(): {sources: [{n, title, section?}], segments: [{text, n}]}."""
    source_of = {}
    for c in chunks:
        if c.get("chunk_id") and c.get("title"):
            source_of[c["chunk_id"]] = (c["title"], section_of(c.get("content"), c["title"]))
    number_of, out = {}, {"sources": [], "segments": []}
    for s in segments:
        n = []
        for cid in s["ids"]:
            source = source_of.get(cid)
            if not source:
                continue
            if source not in number_of:
                number_of[source] = len(number_of) + 1
                title, section = source
                out["sources"].append({"n": number_of[source], "title": title,
                                       **({"section": section} if section else {})})
            if number_of[source] not in n:
                n.append(number_of[source])
        if n:
            out["segments"].append({"text": s["text"], "n": n})
    return out


# ── citations out of a streamed answer (stripCitationsTransform) ──────────────
# Holds back the tail of the text that could still become a citation: an open
# "[" whose text so far fits an id list, and any spaces before it. Everything
# else is safe to send. Ported 2026-11-07.
OPEN_TAIL_RE = re.compile(r"[ \t]*(?:\[[a-z0-9,\s-]*)?\Z", re.IGNORECASE)
MAX_HELD = 200


class CitationStripper:
    """Feed it the model's text deltas; it returns the text a visitor may see
    so far, citations removed. `raw` keeps everything fed, citations included."""

    def __init__(self):
        self.raw = ""
        self._buffer = ""

    def feed(self, text):
        self.raw += text
        self._buffer += text
        return self._flush(False)

    def finish(self):
        """The text still held back, once the stream has ended."""
        return self._flush(True)

    def _flush(self, everything):
        cut = len(self._buffer)
        if not everything:
            held = OPEN_TAIL_RE.search(self._buffer).group(0)
            if len(held) <= MAX_HELD:
                cut -= len(held)
        # Strip before cutting, so a citation completed in this delta goes as one piece.
        out = strip_citations(self._buffer[:cut])
        self._buffer = self._buffer[cut:]
        return out
