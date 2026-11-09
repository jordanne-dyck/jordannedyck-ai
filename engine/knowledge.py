"""Knowledge-base retrieval: the /search call and the context the model sees.
Moved here from run_eval.py on 2026-11-07; the layout is unchanged."""

import json
import urllib.request

SEARCH_URL = "http://localhost:5000/search"

# How each retrieved chunk is laid out in the sources. "labelled" is the
# layout route.ts sends: id and title as labelled lines, with no relevance,
# file path or category. On the same prompt text it cited better than
# "document" (layout tests 2026-10-21/22, nimbalyst-local/prompt-draft-2026-10-21-notes.md).
# "document" is the <document> layout from the ab60e8aa draft, kept for
# comparison runs (--source-format document).
# route.ts took this layout on 2026-10-22 (web branch feat/prompt-ff4d5612),
# with prompt ff4d5612.
SOURCE_FORMAT = "labelled"


def search(query, n_results, search_url=SEARCH_URL):
    """The raw /search results, before they are formatted, and what /search
    reported about itself: {ranker, index_version}, each None from a backend
    that predates it."""
    body = json.dumps({"query": query, "n_results": n_results}).encode()
    req = urllib.request.Request(
        search_url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data.get("results") or [], {"ranker": data.get("ranker"), "index_version": data.get("index_version")}


def format_context(results, truncate=None, source_format=SOURCE_FORMAT):
    """Mirrors searchExperience() in route.ts. `truncate=None` disables the cut.

    Each result carries only id and title, in the `source_format` layout. No
    relevance score, category or file path: they do not help the model and
    the path can leak into answers.
    MUST MATCH route.ts exactly: the same template, joined with "\\n\\n".
    """
    if not results:
        return "No relevant information found.", []

    parts, meta = [], []
    for i, r in enumerate(results):
        content = r["content"]
        shown = content[:truncate] if truncate else content
        tail = "..." if truncate else ""
        if source_format == "labelled":
            parts.append(
                f"### Result {i + 1}\n"
                f"**Id**: {r['metadata'].get('chunk_id')}\n"
                f"**Title**: {r['metadata'].get('title')}\n\n"
                f"{shown}{tail}\n\n---"
            )
        else:
            parts.append(
                f'<document id="{r["metadata"].get("chunk_id")}" '
                f'title="{r["metadata"].get("title")}">\n'
                f"{shown}{tail}\n"
                f"</document>"
            )
        meta.append({
            "chunk_id": r["metadata"].get("chunk_id"),
            "filename": r["metadata"].get("filename"),
            "priority": r["metadata"].get("context_priority"),
            "similarity": round(r["similarity"], 4),
            "score": round(r.get("score", r["similarity"]), 4),
            "chars_total": len(content),
            "chars_shown": len(shown),
            "pct_shown": round(100 * len(shown) / len(content), 1),
        })
    return "\n\n".join(parts), meta
