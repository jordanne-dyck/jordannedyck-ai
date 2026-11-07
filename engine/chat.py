"""One chat turn, as route.ts answers it (web master bbed85c), as a stream of
events: rewrite, retrieval, the fixed reply or the model's answer, then
everything the site shows and logs about it. api_server.py serves it at
POST /chat as newline-delimited JSON. Added 2026-11-07, phase 1 step 3 of the
answer-engine plan.

The site keeps what is about the visitor: input checks, signed turns, the
rate limit, PII redaction, the chat_events row and paid labels. It sends only
turns that passed those checks; this module checks the shape again.

Events, in order:
  meta   once, before any text: prompt and index versions, the rewrite, the
         retrieved chunks, and which fixed reply (if any) this turn gets
  text   {delta}: answer text a visitor may see, citations already removed.
         A fixed overview reply carries the {{timeline}} marker, as on the site.
  done   once, at the end: the cited answer, sources, cards, labels, usage.
         timeline is {explicit, reason} when the page should show the timeline:
         explicit for a fixed overview reply (reason explicit or overview),
         otherwise reason cited, retrieved or asked; None for no timeline.
  error  instead of done: {reason, message}; message is what a visitor sees
"""

import time

from openai import APITimeoutError

from . import (
    answer,
    citations,
    cost,
    fixed,
    knowledge,
    labels,
    projects,
    prompt,
    rewrite,
)

N_RESULTS = 5
# ANSWER_TIMEOUT in route.ts. The first-chunk limit is the read timeout of the
# call, so it also bounds each later gap between chunks.
FIRST_CHUNK_S, TOTAL_S = 20, 60
# lib/chat-input.ts limits.
MAX_USER_TURNS = 20
MAX_USER_MESSAGE_CHARS = 2000
MAX_ASSISTANT_MESSAGE_CHARS = 4 * MAX_USER_MESSAGE_CHARS


class _TotalTimeout(Exception):
    """The answer ran past TOTAL_S."""


def parse_request(body):
    """(messages, timeline_shown) from a /chat body, or ValueError with the reason.
    Messages are user and assistant turns ending on a user turn."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise ValueError("body must be a JSON object with a messages list")  # noqa: TRY004  a bad request, answered 400
    messages = []
    for m in body["messages"]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") \
                or not isinstance(m.get("content"), str):
            raise ValueError("each message needs role user or assistant and string content")
        messages.append({"role": m["role"], "content": m["content"]})
    if not messages or messages[-1]["role"] != "user" or not messages[-1]["content"].strip():
        raise ValueError("the last message must be a non-empty user turn")
    if sum(m["role"] == "user" for m in messages) > MAX_USER_TURNS:
        raise ValueError(f"more than {MAX_USER_TURNS} turns")
    for m in messages:
        cap = MAX_USER_MESSAGE_CHARS if m["role"] == "user" else MAX_ASSISTANT_MESSAGE_CHARS
        if len(m["content"]) > cap:
            raise ValueError(f"{m['role']} message longer than {cap} characters")
    return messages, body.get("timeline_shown") is True


def run_turn(client, search, messages, timeline_shown=False):
    """Yields the events for one turn. `client` is an OpenAI client. `search` is
    (query, n_results) -> (results, info): /search's result list and
    {index_version, ranker}; it raises when retrieval fails."""
    start = time.monotonic()
    ms = lambda since: round((time.monotonic() - since) * 1000)
    query, history = messages[-1]["content"], messages[:-1]

    # Follow-ups are searched on a standalone rewrite; `query` stays the
    # visitor's own words for the labels.
    rw = rewrite.rewrite_query(client, history, query)
    searched = time.monotonic()
    failure, results, info = None, [], {}
    try:
        results, info = search(rw["query"], N_RESULTS)
    except Exception as e:  # noqa: BLE001  any failure is the visitor's error state
        failure = f"search: {type(e).__name__}"
    chunks = [{"chunk_id": r["metadata"].get("chunk_id"), "filename": r["metadata"].get("filename"),
               "title": r["metadata"].get("title"), "similarity": round(r["similarity"], 4),
               "content": r["content"]} for r in results]
    best = max(results, key=lambda r: r["similarity"]) if results else None
    reply = fixed.fixed_reply(query, timeline_shown)
    yield {"type": "meta",
           "prompt_version": prompt.PROMPT_VERSION,
           "index_version": info.get("index_version"),
           "ranker": info.get("ranker"),
           "rewrite": {**rw, "model": rewrite.REWRITE_MODEL if rw["status"] != "skipped" else None},
           "retrieval": {"query": rw["query"], "status": "error" if failure else "ok" if chunks else "empty",
                         "chunks": chunks},
           "retrieval_ms": ms(searched),
           "top_category": best["metadata"].get("category") if best else None,
           "cite_reminder": bool(history) and not reply,
           "fixed": reply["id"] if reply else None,
           "model": None if reply else answer.MODEL}

    # An overview question or an instruction request: the fixed reply, with no
    # model call. Retrieval still ran, so the row records what an answer would
    # have had; a failed search doesn't matter here.
    if reply:
        yield {"type": "text", "delta": reply["answer"]}
        text = fixed.without_timeline_marker(reply["answer"])
        yield {"type": "done", "answer": text, "answer_cited": None, "segments": [], "cited_ids": [],
               "sources": citations.build_sources([], chunks), "project": None, "project_predicted": None,
               "timeline": {"explicit": True, "reason": reply["flags"]["timeline"]} if reply["timeline"] else None,
               "flags": reply["flags"],
               "free_labels": labels.free_labels(query, text), "citation_labels": None,
               "usage": None, "cost_usd": None, "ttft_ms": ms(start), "latency_ms": ms(start)}
        return

    # With no retrieved context the model answers from nothing and invents facts.
    if failure:
        yield {"type": "error", "reason": failure, "message": "Search is unavailable."}
        return

    context, _ = knowledge.format_context(results)
    model_messages = [{"role": "system", "content": prompt.SYSTEM_PROMPT.format(context=context)}, *history,
                      *([{"role": "system", "content": prompt.CITE_REMINDER}] if history else []),
                      {"role": "user", "content": query}]
    stripper, shown, usage, ttft = citations.CitationStripper(), [], None, None
    try:
        stream = answer.open_stream(client, model_messages, timeout=FIRST_CHUNK_S, max_retries=0)
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            for ch in chunk.choices:
                if ch.delta.content:
                    ttft = ttft if ttft is not None else ms(start)
                    out = stripper.feed(ch.delta.content)
                    if out:
                        shown.append(out)
                        yield {"type": "text", "delta": out}
            if time.monotonic() - start > TOTAL_S:
                stream.close()
                raise _TotalTimeout
    except (APITimeoutError, _TotalTimeout):
        yield {"type": "error", "reason": "timeout", "message": "The answer timed out."}
        return
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "reason": f"{type(e).__name__}: {e}"[:500], "message": "The answer failed."}
        return
    tail = stripper.finish()
    if tail:
        shown.append(tail)
        yield {"type": "text", "delta": tail}

    # The timeline marker is for the page only; the model can copy it from an
    # earlier fixed lead in the history.
    text = fixed.without_timeline_marker("".join(shown))
    raw = fixed.without_timeline_marker(stripper.raw)
    segments, cited = citations.parse_citations(raw)
    card = fixed.timeline_card(query, cited, chunks)
    tokens = (usage.prompt_tokens, usage.completion_tokens) if usage else (None, None)
    yield {"type": "done", "answer": text, "answer_cited": raw, "segments": segments, "cited_ids": cited,
           "sources": citations.build_sources(segments, chunks),
           "project": projects.pick_project(cited, chunks),
           "project_predicted": projects.predict_project(chunks),
           "timeline": {"explicit": False, "reason": card} if card else None, "flags": {},
           "free_labels": labels.free_labels(query, text),
           "citation_labels": citations.citation_labels(segments, chunks),
           "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
           "cost_usd": cost.estimate_cost_usd(answer.MODEL, *tokens) if usage else None,
           "ttft_ms": ttft, "latency_ms": ms(start)}
