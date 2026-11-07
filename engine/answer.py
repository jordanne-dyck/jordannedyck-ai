"""The answer model call. Moved here from run_eval.py on 2026-11-07; model,
settings and retries are unchanged."""

import time

from .citations import strip_citations
from .prompt import SYSTEM_PROMPT

MODEL = "gpt-4o"
TEMPERATURE = 0.7
COST_PER_1M_IN, COST_PER_1M_OUT = 2.50, 10.00

# Answer length cap, sent as max_tokens. Mirrors maxTokens: 1000 (maxOutputTokens
# after the AI SDK migration, 2026-10-27; same Chat Completions cap) in route.ts's
# streamText call (security review item 2, merged in web PR #25).
# Longest stored gpt-4o answer, 978 answers checked 2026-10-22: 550 tokens (D02).
ANSWER_MAX_TOKENS = 1000


def generate(client, question, context, history=(), system_prompt=SYSTEM_PROMPT):
    """`history` is the earlier turns as the site resends them: user and
    assistant messages, assistant text with citations already stripped.
    `system_prompt` is a template with {context}.

    Streamed since 2026-10-30 (plan A.6), as route.ts streams, so time to first
    token is measured. The clock starts on each attempt: until then a retried
    answer's latency included the 10-30 s retry waits."""
    system = system_prompt.format(context=context)
    for attempt in range(4):
        try:
            t0, ttft, parts, usage, finish = time.time(), None, [], None, None
            stream = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=ANSWER_MAX_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
                messages=[
                    {"role": "system", "content": system},
                    *history,
                    {"role": "user", "content": question},
                ],
            )
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                for ch in chunk.choices:
                    if ch.delta.content:
                        ttft = ttft or time.time() - t0
                        parts.append(ch.delta.content)
                    finish = ch.finish_reason or finish
            latency = time.time() - t0
            if usage is None:
                raise ValueError("stream ended without usage")
            break
        except Exception as e:
            if attempt == 3:
                raise
            wait = 10 * (attempt + 1)
            print(f"    ! {type(e).__name__}: retrying in {wait}s", flush=True)
            time.sleep(wait)

    text = "".join(parts)
    cost = (usage.prompt_tokens / 1e6) * COST_PER_1M_IN + (usage.completion_tokens / 1e6) * COST_PER_1M_OUT
    return {
        # The checks score what a visitor sees, which has no citations; the
        # cited text is kept so citations can be checked against their chunks.
        "answer": strip_citations(text),
        "answer_cited": text,
        "latency_s": round(latency, 2),
        "ttft_s": round(ttft, 2) if ttft else None,
        # "length" means the answer hit ANSWER_MAX_TOKENS and was cut off.
        "finish_reason": finish,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": round(cost, 5),
    }
