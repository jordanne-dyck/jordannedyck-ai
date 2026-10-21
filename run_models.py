#!/usr/bin/env python
"""
Answer-model comparison for jordbot: gpt-4o against Claude and Gemini models.

Everything except the answering model is held constant:
  - one retrieval per question (arm B: n_results 5, unclipped, as route.ts
    sends today), shared by every model
  - SYSTEM_PROMPT from run_eval.py, byte-identical to lib/system-prompt.ts
  - run_eval.py's automated checks, unchanged
  - citation labels and the per-claim judge (gpt-4o-mini), from citation_checks.py

What is NOT held constant, and cannot be: sampling and thinking. The Claude 5.5
models reject temperature 0.7, Opus 5.5 cannot turn thinking off, and the
Gemini models think by default. Each model's settings are in MODELS below and
are written into every record.

The prompt was tuned against gpt-4o. A result here measures "the site's prompt
on model X", which is the decision being made if the site switches without a
prompt rework. It is not a neutral head-to-head.

Fidelity: answers are streamed so time to first token is measured, but this
calls the providers directly. It does not run route.ts. A switch still needs a
run_e2e.py pass.

Eval-only dependencies, installed into venv and deliberately NOT in
pyproject.toml (uv is not installed here, and the image does not need them):
    venv\\Scripts\\python.exe -m pip install anthropic google-genai

Keys, read from .env: OPENAI_API_KEY (gpt models and the judge),
ANTHROPIC_API_KEY (claude-*), GEMINI_API_KEY (gemini-*).

Usage (from C:\\Users\\Jord\\jordannedyck-ai, api_server.py already running):
    venv\\Scripts\\python.exe run_models.py --estimate                  # no API calls
    venv\\Scripts\\python.exe run_models.py --only W02,G01 --max-cost 1
    venv\\Scripts\\python.exe run_models.py --reps 3 --max-cost 15
    venv\\Scripts\\python.exe run_models.py --report                    # summarise, no calls

Writes eval_results_models.json incrementally (--out to change). Safe to
re-run: completed question/model/run combinations are skipped.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError

from citation_checks import (
    citation_labels,
    judge_grounding,
    leaked_citations,
    parse_citations,
)
from run_eval import (
    EVAL_SET,
    HERE,
    PROMPT_VERSION,
    SEARCH_URL,
    SYSTEM_PROMPT,
    conversation_checks,
    fetch_results,
    lead_chunk,
    run_checks,
    search_experience,
    search_query,
    strip_citations,
)

load_dotenv()

DEFAULT_OUT = HERE / "eval_results_models.json"
ARM = "B"  # the retrieval arm that matches route.ts: n_results 5, no truncation
MAX_TOKENS = 16000

# Per-1M-token prices in USD, standard tier, checked at each provider's pricing
# page 2026-10-15. Output prices include thinking/reasoning tokens.
# gemini-4-argon rises to 4.00 / 20.00 on 2026-12-01; gemini-3.8-flash to
# 1.50 / 7.50 on 2027-01-01.
MODELS = {
    "gpt-4o":            {"provider": "openai",    "price": (2.50, 10.00), "settings": {"temperature": 0.7}},
    "gpt-6.1-sol":       {"provider": "openai",    "price": (2.00, 10.00), "settings": {"reasoning_effort": "low"}},
    "claude-opus-5-5":   {"provider": "anthropic", "price": (4.00, 20.00), "settings": {"effort": "low"}},
    "claude-sonnet-5-5": {"provider": "anthropic", "price": (2.00, 10.00), "settings": {"effort": "low"}},
    "claude-haiku-5-5":  {"provider": "anthropic", "price": (0.10, 0.50),  "settings": {"effort": "low"}},
    "gemini-4-argon":    {"provider": "google",    "price": (2.00, 10.00), "settings": {"thinking_level": "low"}},
    "gemini-3.8-flash":  {"provider": "google",    "price": (0.75, 3.75),  "settings": {"thinking_level": "low"}},
}

KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}

# --estimate assumptions, from the 09-11 arm B run (2,715 in / 196 out, before
# citations) plus headroom. Thinking tokens are a guess until a pilot measures them.
EST_IN, EST_OUT, EST_THINK_HIGH = 3000, 300, 1500
EST_JUDGE = 0.001


# ── provider adapters ─────────────────────────────────────────────────────────
# Each streams one answer and returns the same dict. Thinking/reasoning tokens
# are billed as output by all three providers, so output_tokens includes them.

def _openai(client, model, settings, system, messages):
    # No max_tokens: run_eval.py and route.ts send none, so gpt-4o runs as it does there.
    t0, ttft, parts, usage = time.time(), None, [], None
    stream = client.chat.completions.create(
        model=model, stream=True, stream_options={"include_usage": True},
        messages=[{"role": "system", "content": system}, *messages],
        **settings,
    )
    finish = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        for ch in chunk.choices:
            if ch.delta.content:
                ttft = ttft or time.time() - t0
                parts.append(ch.delta.content)
            finish = ch.finish_reason or finish
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "text": "".join(parts), "t0": t0, "ttft": ttft, "stop": finish,
        "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
        "thinking_tokens": getattr(details, "reasoning_tokens", 0) or 0,
    }


def _anthropic(client, model, settings, system, messages):
    # No fallbacks parameter: a refusal rescued by another model would put a
    # different model's answer under this one's name. A refusal is recorded.
    t0, ttft, parts = time.time(), None, []
    with client.messages.stream(
        model=model, max_tokens=MAX_TOKENS, system=system,
        messages=messages,
        output_config={"effort": settings["effort"]},
    ) as stream:
        for text in stream.text_stream:
            ttft = ttft or time.time() - t0
            parts.append(text)
        msg = stream.get_final_message()
    return {
        "text": "".join(parts), "t0": t0, "ttft": ttft, "stop": msg.stop_reason,
        "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
        # Not reported separately by the Messages API; included in output_tokens.
        "thinking_tokens": None,
    }


def _google(client, model, settings, system, messages):
    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=settings["thinking_level"]),
    )
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in messages]
    t0, ttft, parts, usage, finish = time.time(), None, [], None, None
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.usage_metadata:
            usage = chunk.usage_metadata
        if chunk.candidates and chunk.candidates[0].finish_reason:
            finish = str(chunk.candidates[0].finish_reason)
        if chunk.text:
            ttft = ttft or time.time() - t0
            parts.append(chunk.text)
    thinking = usage.thoughts_token_count or 0
    return {
        "text": "".join(parts), "t0": t0, "ttft": ttft, "stop": finish,
        "input_tokens": usage.prompt_token_count,
        "output_tokens": (usage.candidates_token_count or 0) + thinking,
        "thinking_tokens": thinking,
    }


ADAPTERS = {"openai": _openai, "anthropic": _anthropic, "google": _google}


def make_clients(providers):
    clients = {}
    if "openai" in providers:
        from openai import OpenAI
        clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "anthropic" in providers:
        import anthropic
        clients["anthropic"] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if "google" in providers:
        from google import genai
        clients["google"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return clients


def generate(clients, model, question, context, history=()):
    """`history`: earlier user/assistant turns, assistant text citation-stripped,
    as the site's client resends them."""
    spec = MODELS[model]
    system = SYSTEM_PROMPT.format(context=context)
    messages = [*history, {"role": "user", "content": question}]
    for attempt in range(4):
        try:
            out = ADAPTERS[spec["provider"]](clients[spec["provider"]], model, spec["settings"], system, messages)
            break
        except Exception as e:
            if attempt == 3:
                raise
            wait = 10 * (attempt + 1)
            print(f"    ! {type(e).__name__}: {str(e)[:120]}, retrying in {wait}s", flush=True)
            time.sleep(wait)

    p_in, p_out = spec["price"]
    cost = (out["input_tokens"] / 1e6) * p_in + (out["output_tokens"] / 1e6) * p_out
    return {
        # Same keys as run_eval.generate(), so run_checks() scores it unchanged.
        "answer": strip_citations(out["text"]),
        "answer_cited": out["text"],
        "latency_s": round(time.time() - out["t0"], 2),
        "ttft_s": round(out["ttft"], 2) if out["ttft"] else None,
        "stop_reason": out["stop"],
        "prompt_tokens": out["input_tokens"],
        "completion_tokens": out["output_tokens"],
        "thinking_tokens": out["thinking_tokens"],
        "cost_usd": round(cost, 5),
    }


# ── planning and reporting ────────────────────────────────────────────────────

def plan(questions, models, reps):
    return [(q, m, run) for m in models for q in questions
            for run in range(1, max(q.get("runs", 1), reps) + 1)]


def estimate(calls):
    lo = hi = 0.0
    for _q, m, _r in calls:
        p_in, p_out = MODELS[m]["price"]
        thinks = m != "gpt-4o"
        lo += (EST_IN * p_in + EST_OUT * p_out) / 1e6 + EST_JUDGE
        hi += (EST_IN * 1.35 * p_in + (EST_OUT + (EST_THINK_HIGH if thinks else 0)) * p_out) / 1e6 + EST_JUDGE
    return lo, hi


def _pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def report(results):
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    print(f"\n{'model':<19}{'n':>4}{'fails':>7}{'uncited':>9}{'invent':>8}{'leak':>6}"
          f"{'grounded':>10}{'judge no':>10}{'ttft':>7}{'total':>7}{'p90':>7}{'$/ans':>9}{'$ all':>8}")
    for model, rs in by_model.items():
        n = len(rs)
        fails = sum(1 for r in rs if any(not k.startswith("_") and v is False for k, v in r["checks"].items()))
        segs = sum(r["citations"]["segments"] for r in rs)
        uncited = sum(r["citations"]["uncited"] for r in rs)
        invented = sum(len(r["citations"]["invented"]) for r in rs)
        leaks = sum(1 for r in rs if r["citations"]["leaked"])
        judged = [r for r in rs if r.get("judge") and "verdict" in r["judge"]]
        grounded = sum(1 for r in judged if r["judge"]["verdict"] == "grounded")
        claims = sum(r["judge"]["checked"] for r in judged)
        no = sum(r["judge"]["no"] for r in judged)
        lat = sorted(r["gen"]["latency_s"] for r in rs)
        ttft = [r["gen"]["ttft_s"] for r in rs if r["gen"].get("ttft_s")]
        cost = sum(r["gen"]["cost_usd"] for r in rs)
        print(f"{model:<19}{n:>4}{fails:>7}{_pct(uncited, segs):>9}{invented:>8}{leaks:>6}"
              f"{_pct(grounded, len(judged)):>10}{_pct(no, claims):>10}"
              f"{statistics.mean(ttft) if ttft else 0:>7.1f}{statistics.mean(lat):>7.1f}"
              f"{lat[int(0.9 * (n - 1))]:>7.1f}{cost / n:>9.4f}{cost:>8.2f}")
    print("\nfails = answers failing any automated check. uncited = share of answer segments with no "
          "citation.\ninvent = cited ids not retrieved. leak = answers showing a chunk id after "
          "stripping.\ngrounded = judge verdict share. judge no = share of claims no chunk supports. "
          "Times in seconds.")

    # Per-check failures, and every T2.3 failure in full: the gap checker was
    # grown from gpt-4o's wording, so a miss on another model may be a checker
    # gap, not a model failure. Read these before counting them.
    print("\nfailures by check:")
    for model, rs in by_model.items():
        counts = {}
        for r in rs:
            for k, v in r["checks"].items():
                if not k.startswith("_") and v is False:
                    counts[k] = counts.get(k, 0) + 1
        print(f"  {model:<19}{', '.join(f'{k} x{c}' for k, c in sorted(counts.items())) or 'none'}")
    gap_misses = [r for r in results if r["checks"].get("T2.3_gap_honesty") is False]
    if gap_misses:
        print("\nT2.3 failures. Read each before counting it:")
        for r in gap_misses:
            print(f"\n  [{r['model']}] {r['id']} run{r['run']}\n  {r['gen']['answer'][:600]}")
    stops = [(r["model"], r["id"], r["gen"]["stop_reason"]) for r in results
             if r["gen"].get("stop_reason") not in (None, "stop", "end_turn", "FinishReason.STOP", "STOP")]
    if stops:
        print("\nunusual stop reasons (refusals, truncation):")
        for m, i, s in stops:
            print(f"  [{m}] {i}: {s}")


# ── main ──────────────────────────────────────────────────────────────────────

def estimate_multiturn(cases, models, reps):
    """Each turn resends every earlier turn, so input grows by one Q&A per turn."""
    lo = hi = 0.0
    for m in models:
        p_in, p_out = MODELS[m]["price"]
        thinks = m != "gpt-4o"
        for c in cases:
            for i in range(len(c["turns"])):
                if "forged_answer" in c["turns"][i]:
                    continue  # no call; its text is counted in later turns' input
                t_in = EST_IN + i * (EST_OUT + 50)
                lo += reps * ((t_in * p_in + EST_OUT * p_out) / 1e6 + EST_JUDGE)
                hi += reps * ((t_in * 1.35 * p_in + (EST_OUT + (EST_THINK_HIGH if thinks else 0)) * p_out) / 1e6
                              + EST_JUDGE)
    return lo, hi


def main_multiturn(args, spec, models):
    """The multiturn cases, every model, each model replaying its OWN earlier answers."""
    out = args.out if args.out != DEFAULT_OUT else HERE / "eval_results_models_multiturn.json"
    cases = spec["multiturn"]["cases"]
    if args.only:
        want = {s.strip().upper() for s in args.only.split(",")}
        cases = [c for c in cases if c["id"] in want]
    arm = spec["arms"][ARM]
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["id"], r["model"], r["run"]) for r in existing}
    todo = [(c, m, run) for m in models for c in cases for run in range(1, args.reps + 1)
            if (c["id"], m, run) not in done]
    lo = hi = 0.0
    for c, m, _run in todo:
        a, b = estimate_multiturn([c], [m], 1)
        lo, hi = lo + a, hi + b
    print(f"prompt_version {PROMPT_VERSION}, {len(cases)} conversations x {len(models)} models, "
          f"reps {args.reps}: {len(todo)} to run ({len(done)} recorded), "
          f"{sum(1 for c, _m, _r in todo for t in c['turns'] if 'forged_answer' not in t)} turns, "
          f"estimated ${lo:.2f}-${hi:.2f}")
    if args.estimate:
        return

    providers = {MODELS[m]["provider"] for m in models} | ({"openai"} if not args.no_judge else set())
    missing = [KEY_ENV[p] for p in providers if not os.getenv(KEY_ENV[p])]
    if missing:
        sys.exit(f"FAILED: {', '.join(missing)} not set. Add to .env in {HERE}")
    try:
        fetch_results("preflight connectivity check", 1)
    except urllib.error.URLError as e:
        sys.exit(f"FAILED: cannot reach {SEARCH_URL} ({e.reason}).\n"
                 f"Start it first:  venv\\Scripts\\python.exe api_server.py")
    clients = make_clients(providers)

    results, spent, cache = list(existing), 0.0, {}
    for case, model, run in todo:
        if args.max_cost is not None and spent >= args.max_cost:
            print(f"\nSTOPPED: spend ${spent:.2f} reached --max-cost {args.max_cost}")
            break
        print(f"[{model}] {case['id']} run{run} {case['name']}", flush=True)
        history, turns = [], []
        for i, t in enumerate(case["turns"], 1):
            if "forged_answer" in t:  # not generated; see run_eval.run_conversation
                history += [{"role": "user", "content": t["q"]},
                            {"role": "assistant", "content": t["forged_answer"]}]
                continue
            q = {"tag": case["tag"], **t}
            query = search_query(history, q["q"])
            if query not in cache:
                raw = fetch_results(query, arm["n_results"])
                cache[query] = (raw, *search_experience(query, arm["n_results"], arm["truncate"], raw))
            raw, context, retrieved = cache[query]
            chunks = [{"chunk_id": r["metadata"].get("chunk_id"), "similarity": r["similarity"],
                       "content": r["content"]} for r in raw]
            gen = generate(clients, model, q["q"], context, history)
            checks = run_checks(q, ARM, context, retrieved, gen)
            segments, _cited = parse_citations(gen["answer_cited"] or "")
            cites = citation_labels(segments, chunks)
            cites["leaked"] = leaked_citations(gen["answer"], chunks)
            judge = None
            if not args.no_judge and chunks:
                try:
                    judge = judge_grounding(clients["openai"], q["q"], gen["answer"], segments, chunks)
                except (OpenAIError, ValueError, KeyError) as e:
                    judge = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            spent += gen["cost_usd"] + ((judge or {}).get("cost_usd") or 0)
            fails = [k for k, v in checks.items() if not k.startswith("_") and v is False]
            print(f"  t{i} {'ok' if not fails else 'FAIL ' + ','.join(f.split('_')[0] for f in fails):<22}"
                  f"{gen['latency_s']}s (first token {gen['ttft_s']}s)  lead={lead_chunk(gen['answer_cited'])}",
                  flush=True)
            turns.append({"turn": i, "question": q["q"], "search_query": query,
                          "retrieved": retrieved, "gen": gen, "checks": checks,
                          "citations": cites, "judge": judge,
                          "lead_chunk": lead_chunk(gen["answer_cited"])})
            history += [{"role": "user", "content": q["q"]},
                        {"role": "assistant", "content": gen["answer"]}]

        case_checks = conversation_checks(case, turns)
        print("  " + "  ".join(f"{k} {v}" for k, v in case_checks.items() if not k.startswith("_"))
              + f"  leads={case_checks['_leads']}  uncited={case_checks['_uncited_turns']}")
        results.append({"id": case["id"], "name": case["name"], "tag": case["tag"],
                        "model": model, "provider": MODELS[model]["provider"],
                        "settings": MODELS[model]["settings"], "run": run,
                        "prompt_version": PROMPT_VERSION, "turns": turns,
                        "case_checks": case_checks})
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n${spent:.2f} spent this invocation, {len(results)} conversations in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model ids")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--reps", type=int, default=1, help="runs per question (at least its own 'runs')")
    ap.add_argument("--max-cost", type=float, help="stop once spend reaches this many USD")
    ap.add_argument("--no-judge", action="store_true", help="skip the paid judge")
    ap.add_argument("--estimate", action="store_true", help="print the plan and cost estimate, call nothing")
    ap.add_argument("--report", action="store_true", help="summarise the results file, call nothing")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--multiturn", action="store_true",
                    help="run the multiturn cases (default out: eval_results_models_multiturn.json)")
    args = ap.parse_args()

    if args.multiturn:
        models = [m.strip() for m in args.models.split(",")]
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            sys.exit(f"unknown model(s): {', '.join(unknown)}. Known: {', '.join(MODELS)}")
        return main_multiturn(args, json.loads(EVAL_SET.read_text(encoding="utf-8")), models)

    existing = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else []
    if args.report:
        if not existing:
            sys.exit(f"no results in {args.out}")
        report(existing)
        return

    models = [m.strip() for m in args.models.split(",")]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)}. Known: {', '.join(MODELS)}")

    spec = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        want = {s.strip().upper() for s in args.only.split(",")}
        questions = [q for q in questions if q["id"] in want]
    arm = spec["arms"][ARM]

    done = {(r["id"], r["model"], r["run"]) for r in existing}
    calls = [c for c in plan(questions, models, args.reps) if (c[0]["id"], c[1], c[2]) not in done]
    lo, hi = estimate(calls)
    print(f"prompt_version {PROMPT_VERSION}, arm {ARM} ({arm['name']}), "
          f"{len(questions)} questions x {len(models)} models, reps {args.reps}")
    print(f"{len(calls)} answers to generate ({len(done)} already recorded), "
          f"estimated ${lo:.2f}-${hi:.2f} including the judge")
    if args.estimate:
        return

    providers = {MODELS[m]["provider"] for m in models} | ({"openai"} if not args.no_judge else set())
    missing = [KEY_ENV[p] for p in providers if not os.getenv(KEY_ENV[p])]
    if missing:
        sys.exit(f"FAILED: {', '.join(missing)} not set. Add to .env in {HERE}")
    try:
        fetch_results("preflight connectivity check", 1)
    except urllib.error.URLError as e:
        sys.exit(f"FAILED: cannot reach {SEARCH_URL} ({e.reason}).\n"
                 f"Start it first:  venv\\Scripts\\python.exe api_server.py")
    clients = make_clients(providers)

    results = list(existing)
    spent = 0.0
    retrieved_cache = {}
    for q, model, run in calls:
        if args.max_cost is not None and spent >= args.max_cost:
            print(f"\nSTOPPED: spend ${spent:.2f} reached --max-cost {args.max_cost}")
            break
        if q["id"] not in retrieved_cache:
            raw = fetch_results(q["q"], arm["n_results"])
            retrieved_cache[q["id"]] = (raw, *search_experience(q["q"], arm["n_results"], arm["truncate"], raw))
        raw, context, retrieved = retrieved_cache[q["id"]]
        chunks = [{"chunk_id": r["metadata"].get("chunk_id"), "similarity": r["similarity"],
                   "content": r["content"]} for r in raw]

        print(f"[{model}] {q['id']} run{run:<3} {q['q'][:56]}", flush=True)
        gen = generate(clients, model, q["q"], context)
        checks = run_checks(q, ARM, context, retrieved, gen)
        segments, _cited = parse_citations(gen["answer_cited"] or "")
        cites = citation_labels(segments, chunks)
        cites["leaked"] = leaked_citations(gen["answer"], chunks)
        judge = None
        if not args.no_judge and chunks:
            try:
                judge = judge_grounding(clients["openai"], q["q"], gen["answer"], segments, chunks)
            except (OpenAIError, ValueError, KeyError) as e:
                judge = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        spent += gen["cost_usd"] + ((judge or {}).get("cost_usd") or 0)

        fails = [k for k, v in checks.items() if not k.startswith("_") and v is False]
        print(f"    {'ok' if not fails else 'FAIL ' + ','.join(f.split('_')[0] for f in fails):<22}"
              f"{gen['latency_s']}s (first token {gen['ttft_s']}s)  "
              f"uncited {cites['uncited']}/{cites['segments']}  "
              f"judge {(judge or {}).get('verdict', (judge or {}).get('error', '-'))}  "
              f"${gen['cost_usd']:.4f}", flush=True)

        results.append({
            "id": q["id"], "tag": q["tag"], "req": q.get("req", []),
            "arm": ARM, "model": model, "provider": MODELS[model]["provider"],
            "settings": MODELS[model]["settings"], "run": run, "question": q["q"],
            "expect": q.get("expect", []), "prompt_version": PROMPT_VERSION,
            "retrieved": retrieved, "gen": gen, "checks": checks,
            "citations": cites, "judge": judge, "context_chars": len(context),
        })
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n${spent:.2f} spent this invocation, {len(results)} results in {args.out}")
    report(results)


if __name__ == "__main__":
    main()
