"""The project card under an answer: which project an answer is about, from the
chunks it cites. Ported 2026-11-07 from pickProject() and predictProject() in
jordannedyck-ai-web/lib/projects.ts. The cards themselves (copy, images,
links) stay on the site; the engine only picks the slug."""

import re

# The slugs that have a card. MUST MATCH the `slug` values of PROJECTS in
# jordannedyck-ai-web/lib/projects.ts: a file under projects/ with no card
# gets none.
PROJECT_SLUGS = frozenset({
    "ai-resume", "warai-neko", "abnormal-innovation", "agentic-personal-shopper",
    "ai-customer-service-optimization", "brand-incubator-ecosystem",
    "google-gemini-enterprise-deployment", "personalization-roadmap-and-measurement",
    "growth-and-acquisition", "digital-transformation-salesforce", "regimen-builder",
    "tiktok-shop-integration",
})

# Cards for chunks outside projects/, by exact chunk id. growth-002 (referral,
# mostly Loblaw) is left out on purpose: the card is about DECIEM. indie-001
# (overview) and indie-003/004 (her own AI systems) are left out: they are not
# client work.
CHUNK_PROJECTS = {
    "growth-001": "growth-and-acquisition",
    "indie-002": "warai-neko",
    "indie-005": "warai-neko",
    "indie-006": "warai-neko",
}

_PROJECT_FILE_RE = re.compile(r"(?:^|[\\/])projects[\\/]([a-z0-9-]+)\.md\Z")


def slug_of_file(filename):
    """The project a chunk came from, by its knowledge-base file. Matched on the
    file, not the chunk-id prefix: `dtx-` is used outside projects/ too."""
    m = _PROJECT_FILE_RE.search(filename or "")
    return m.group(1) if m and m.group(1) in PROJECT_SLUGS else None


def pick_project(cited_ids, chunks):
    """The project to show for an answer: the one cited most, ties going to the
    one cited first. None when the answer cites no project. One card per answer
    (decided 2026-10-27)."""
    file_of = {c.get("chunk_id"): c.get("filename") for c in chunks}
    counts = {}
    for cid in cited_ids:
        slug = slug_of_file(file_of.get(cid)) or CHUNK_PROJECTS.get(cid)
        if slug:
            counts[slug] = counts.get(slug, 0) + 1
    best = None
    for slug, n in counts.items():
        if best is None or n > counts[best]:
            best = slug
    return best


def predict_project(chunks):
    """The card a pick from the retrieved chunks alone would show, before the
    answer exists. Logged next to the real pick, not shown."""
    return pick_project([c["chunk_id"] for c in chunks if c.get("chunk_id")], chunks)
