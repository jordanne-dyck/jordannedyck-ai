#!/usr/bin/env python3
"""
run_e2e.py — dual-target END-TO-END eval runner for jordannedyck.com.

WHAT THIS IS
    Drives the REAL Next.js chat route (app/api/chat/route.ts) over HTTP and records
    the answer a visitor would receive, plus client-observed timings. One runner, two
    targets, selected by --base-url:

        --base-url http://localhost:3000   -> source_system "local-route"
        --base-url https://jordannedyck.com -> source_system "prod"

    This satisfies the 2026-09-10 binding rule that evals must exercise the full RAG
    system end to end. run_eval.py does NOT — it calls /search and the model directly
    and never executes route.ts. That harness stays useful as a cheap iteration tool;
    this file is the measurement of record.

WHAT THIS CANNOT DO — read before interpreting any output
    The retrieved context is assembled INSIDE route.ts and never leaves the server.
    So every check that compares the answer against the retrieved chunks is
    structurally unavailable on this path:

        C2  proper-noun fidelity      -> unavailable_no_context
        T2.1 retrieval hit            -> unavailable_no_context
        T2.2 figure fidelity          -> unavailable_no_context
        C9  claim-level reach         -> unavailable_no_context

    They are written into every record with that literal value rather than omitted,
    so the gap is visible in the data instead of being forgotten. Answer-only checks
    (phone, third person, JD echo, emoji, banned names, gap phrasing) DO run here.
    Closing the gap would mean route.ts emitting the chunk ids as a data-stream
    annotation part; that is a frontend change and is not specified from here.

    Retrieval latency is also not separable. route.ts awaits searchExperience() before
    streamText(), so time-to-first-byte bundles retrieval with model connect. Both
    numbers are recorded; neither is labelled as a retrieval time.

STREAM PROTOCOL
    route.ts returns result.toDataStreamResponse() (ai ^4.3.19) — the AI SDK v4 data
    stream protocol: newline-delimited `TYPE:JSON` parts. Text arrives as `0:"..."`,
    finish metadata as `d:`/`e:` (carrying usage when sendUsage is on, which is the
    library default), errors as `3:`.

    This runner's parser is written to that protocol, BUT the protocol has not been
    observed on a real response from this route. Run --probe first. It sends ONE
    question (~$0.01), writes the raw bytes to disk, prints the first lines, and
    reports whether the parse produced text. Do not trust a full run's numbers until
    a probe has confirmed the shape.

PREREQUISITE FOR THE LOCAL TARGET
    route.ts calls checkRateLimit() -> pool.query() before generation and outside the
    try block, and awaits logChatEvent() in onFinish. With no DATABASE_URL, requests
    fail. Three files need the env-guard, not one: lib/db.ts (pool creation),
    lib/rate-limit.ts (checkRateLimit), lib/logging.ts (logChatEvent). That is
    jordannedyck-ai-web territory — frontend session's lane. This runner detects the
    symptom and says so rather than guessing at the cause.

RATE LIMITS
    HOURLY_LIMIT = 20, DAILY_LIMIT = 100, hardcoded in lib/rate-limit.ts, no env
    override, counted per ip_hash. Default --limit is 18 per invocation. 34 questions
    is therefore two sittings against prod. A 429 stops the run cleanly and prints the
    --start value to resume from.

OUTPUT
    eval_e2e_<source_system>_<YYYY-MM-DD>.json. REFUSES to overwrite an existing file.
    Use --resume to append only the question ids not already present. There is no
    --fresh flag; --fresh is what destroyed the 124-chunk run on 2026-09-10.

USAGE
    python run_e2e.py --probe --base-url http://localhost:3000
    python run_e2e.py --base-url https://jordannedyck.com --limit 18
    python run_e2e.py --base-url https://jordannedyck.com --resume --start 18
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import http.cookiejar
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_SET = HERE / "eval_set.json"
INDEX_FILE = HERE / "faiss_db" / "resume.index"
EMBEDDER = HERE / "scripts" / "embed_knowledge_faiss.py"
ROUTE_TS = HERE.parent / "jordannedyck-ai-web" / "app" / "api" / "chat" / "route.ts"

# resume.index is IndexFlatL2 over 1536-dim float32 vectors with a 45-byte header.
INDEX_HEADER_BYTES = 45
INDEX_BYTES_PER_VECTOR = 1536 * 4

BANNED_NAMES = ["slowvember", "blackout friday"]
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F\U00002B00-\U00002BFF]"
)
PART_RE = re.compile(r"^([0-9a-z]):(.*)$", re.DOTALL)


# ----------------------------------------------------------------------------------
# Reused checkers. run_eval.py already holds the validated regexes and n-gram sets;
# duplicating them here would create a fifth copy of shared logic in this project.
# ----------------------------------------------------------------------------------
def load_harness_checkers():
    sys.path.insert(0, str(HERE))
    try:
        import run_eval  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        return None, f"could not import run_eval.py ({exc.__class__.__name__}: {exc})"
    missing = [
        n for n in ("PHONE_RE", "FIRST_PERSON_RE", "GAP_RE", "GAP_PHRASES", "JD_NGRAMS",
                    "ngrams", "SYSTEM_PROMPT", "gap_admitted", "injection_resisted")
        if not hasattr(run_eval, n)
    ]
    if missing:
        return None, f"run_eval.py is missing {', '.join(missing)}"
    return run_eval, None


# ----------------------------------------------------------------------------------
# Preflights. C20 aborts. C22 aborts on the local target, warns on prod.
# ----------------------------------------------------------------------------------
def chunk_count_from_index():
    """C20 — index vintage. Derived from file size, never from a recorded figure."""
    if not INDEX_FILE.exists():
        return None, None, f"{INDEX_FILE} not found"
    size = INDEX_FILE.stat().st_size
    payload = size - INDEX_HEADER_BYTES
    if payload <= 0 or payload % INDEX_BYTES_PER_VECTOR != 0:
        return None, size, (
            f"{size} bytes is not a whole number of 1536-dim float32 vectors plus a "
            f"{INDEX_HEADER_BYTES}-byte header — the index may be a different type"
        )
    return payload // INDEX_BYTES_PER_VECTOR, size, None


def extract_route_prompt(route_text):
    """Pull the systemPrompt template literal out of route.ts."""
    marker = "const systemPrompt = `"
    start = route_text.find(marker)
    if start == -1:
        return None, "no `const systemPrompt = ` template literal found in route.ts"
    start += len(marker)
    end = route_text.find("`;", start)
    if end == -1:
        return None, "systemPrompt template literal is not terminated with a backtick"
    return route_text[start:end], None


def normalise_prompt(text):
    return text.replace("${context}", "{context}").replace("\r\n", "\n").strip()


def prompt_drift_check(harness, route_text, show_diff):
    """
    C22 — prompt drift. The registry in Part 6 of the definition-of-good does not exist
    yet, so this checks the thing the registry is meant to prevent: the two live copies
    of the prompt disagreeing. route.ts is the source of truth; run_eval.py is the copy.
    """
    route_prompt, err = extract_route_prompt(route_text)
    if err:
        return None, None, err
    route_norm = normalise_prompt(route_prompt)
    route_md5 = hashlib.md5(route_norm.encode()).hexdigest()
    if harness is None:
        return route_md5, None, "run_eval.py unavailable — drift not checked"
    harness_norm = normalise_prompt(harness.SYSTEM_PROMPT)
    harness_md5 = hashlib.md5(harness_norm.encode()).hexdigest()
    if route_md5 == harness_md5:
        return route_md5, harness_md5, None
    detail = (
        f"route.ts prompt md5 {route_md5} != run_eval.py SYSTEM_PROMPT md5 {harness_md5} "
        f"({len(route_norm)} vs {len(harness_norm)} chars)"
    )
    if show_diff:
        diff = difflib.unified_diff(
            harness_norm.splitlines(), route_norm.splitlines(),
            fromfile="run_eval.py SYSTEM_PROMPT", tofile="route.ts systemPrompt", lineterm="",
        )
        detail += "\n" + "\n".join(list(diff)[:80])
    return route_md5, harness_md5, detail


def extract_route_config(route_text):
    """Model, temperature, n_results and truncation as route.ts actually sets them."""
    cfg = {"model": None, "temperature": None, "n_results": None, "truncate": None}
    m = re.search(r"const MODEL\s*=\s*['\"]([^'\"]+)['\"]", route_text)
    if m:
        cfg["model"] = m.group(1)
    m = re.search(r"temperature:\s*([0-9.]+)", route_text)
    if m:
        cfg["temperature"] = float(m.group(1))
    m = re.search(r"n_results:\s*(\d+)", route_text)
    if m:
        cfg["n_results"] = int(m.group(1))
    # Truncation was removed 2026-09-10. Detect its return rather than assuming absence.
    # Scoped to searchExperience's body: route.ts also slices error messages to 500 chars
    # in its logging paths, and a whole-file search matches those instead.
    body = re.search(
        r"async function searchExperience\s*\([^)]*\)[^{]*\{(.*?)\n\}", route_text, re.DOTALL
    )
    if body is None:
        cfg["truncate"] = "unknown"
    else:
        m = re.search(r"\.(?:slice|substring)\(\s*0\s*,\s*(\d+)\s*\)", body.group(1))
        cfg["truncate"] = int(m.group(1)) if m else None
    return cfg


def embedding_model_from_embedder():
    if not EMBEDDER.exists():
        return None
    m = re.search(r"text-embedding-[a-z0-9-]+", EMBEDDER.read_text(encoding="utf-8", errors="replace"))
    return m.group(0) if m else None


def build_stamp(args, source_system):
    """C23 — provenance stamp. Attached run-level AND to every record."""
    stamp = {
        "source_system": source_system,
        "base_url": args.base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runner": "run_e2e.py",
        "transport": "http-datastream-v4",
        "turn_shape": "single-turn",  # divergence: the real UI posts full history
        "prompt_version": None,
        "prompt_md5": None,
        "model": None,
        "temperature": None,
        "n_results": None,
        "truncate": None,
        "chunk_count": None,
        "index_bytes": None,
        "embedding_model": None,
        "context_visible": False,
    }
    if source_system == "prod":
        # The deployed bundle is pre-2026-09-10 and its KB provenance is unknown.
        # Nothing on this machine describes what prod is running. Stamped, not guessed.
        # "unknown", not None: a null truncate would read as "no truncation", which is a
        # claim about the deployed bundle that nothing on this machine supports.
        stamp.update({
            "prompt_version": "unknown-prod-bundle",
            "vintage": "pre-2026-09-10 bundle",
            "kb_provenance": "unknown",
            "model": "unknown", "temperature": "unknown", "n_results": "unknown",
            "truncate": "unknown", "embedding_model": "unknown",
            "chunk_count": "unknown", "index_bytes": "unknown",
        })
    return stamp


def run_preflights(args, source_system, harness, stamp):
    """Returns (ok, lines). C20 and C22 abort before the first paid call."""
    lines, fatal = [], False
    local = source_system != "prod"

    count, size, err = chunk_count_from_index()
    if local:
        stamp["chunk_count"], stamp["index_bytes"] = count, size
    if err:
        lines.append(f"  C20 index vintage : {'FAIL' if local else 'warn'} — {err}")
        fatal = fatal or local
    else:
        lines.append(f"  C20 index vintage : ok — {count} chunks ({size} bytes)")
        if local and args.expect_chunks is not None and count != args.expect_chunks:
            lines.append(f"                      FAIL — expected {args.expect_chunks} chunks")
            fatal = True
    if not local:
        lines.append("                      NB prod's index is not on this machine; "
                     "the count above describes the LOCAL index only, and is NOT stamped")

    if not ROUTE_TS.exists():
        lines.append(f"  C22 prompt drift  : {'FAIL' if local else 'warn'} — {ROUTE_TS} not found")
        fatal = fatal or local
    else:
        route_text = ROUTE_TS.read_text(encoding="utf-8", errors="replace")
        route_md5, harness_md5, drift = prompt_drift_check(harness, route_text, args.show_prompt_diff)
        cfg = extract_route_config(route_text)
        if local:
            stamp.update(cfg)
            stamp["prompt_md5"] = route_md5
            stamp["prompt_version"] = f"route-{route_md5[:8]}" if route_md5 else None
            stamp["embedding_model"] = embedding_model_from_embedder()
            missing = [k for k, v in cfg.items() if v is None and k != "truncate"]
            if missing:
                lines.append(f"  route.ts config   : warn — could not extract {', '.join(missing)}")
            else:
                lines.append(f"  route.ts config   : {cfg['model']} @ T={cfg['temperature']}, "
                             f"n_results={cfg['n_results']}, truncate={cfg['truncate']}")
        else:
            lines.append("  route.ts config   : not stamped — the local file does not describe "
                         "the deployed bundle")
        if drift:
            lines.append(f"  C22 prompt drift  : {'FAIL' if local else 'warn'} — {drift}")
            fatal = fatal or local
        else:
            lines.append(f"  C22 prompt drift  : ok — route.ts and run_eval.py agree ({route_md5[:8]})")

    if harness is None:
        lines.append("  checkers          : warn — answer-only checks degraded")
    return (not fatal), lines


# ----------------------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------------------
def make_opener():
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def ask(opener, base_url, question, timeout, raw_sink=None):
    """
    POST one question and read the data stream line by line.

    readline() is deliberate: the AI SDK v4 protocol is newline-delimited, so a line
    boundary is the earliest point at which a text part can be timed. read(n) would
    block until n bytes and destroy the time-to-first-token measurement.
    """
    url = base_url.rstrip("/") + "/api/chat"
    body = json.dumps({"messages": [{"role": "user", "content": question}]}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "*/*",
                 "User-Agent": "jordannedyck-e2e-eval/1.0"},
        method="POST",
    )

    out = {
        "http_status": None, "answer": "", "error": None, "stream_error": None,
        "usage": None, "finish_reason": None,
        "ttfb_ms": None, "ttft_ms": None, "total_ms": None,
        "parts_seen": {}, "unparsed_lines": 0, "raw_bytes": 0,
    }
    t0 = time.perf_counter()
    try:
        resp = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        out["http_status"] = e.code
        payload = e.read().decode("utf-8", errors="replace")[:1000]
        out["error"] = f"HTTP {e.code}: {payload}"
        out["total_ms"] = round((time.perf_counter() - t0) * 1000)
        if e.code == 429:
            out["error"] = f"HTTP 429 rate limited: {payload}"
        return out
    except Exception as e:
        out["error"] = f"{e.__class__.__name__}: {e}"
        out["total_ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    out["http_status"] = resp.status
    out["content_type"] = resp.headers.get("Content-Type")
    out["x_vercel_ai_data_stream"] = resp.headers.get("x-vercel-ai-data-stream")

    with resp:
        while True:
            try:
                line = resp.readline()
            except Exception as e:
                out["stream_error"] = f"read failed: {e.__class__.__name__}: {e}"
                break
            if not line:
                break
            now = time.perf_counter()
            if out["ttfb_ms"] is None:
                out["ttfb_ms"] = round((now - t0) * 1000)
            out["raw_bytes"] += len(line)
            if raw_sink is not None:
                raw_sink.write(line)
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if not text:
                continue
            m = PART_RE.match(text)
            if not m:
                out["unparsed_lines"] += 1
                continue
            kind, payload = m.group(1), m.group(2)
            out["parts_seen"][kind] = out["parts_seen"].get(kind, 0) + 1
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                out["unparsed_lines"] += 1
                continue
            if kind == "0" and isinstance(value, str):
                if out["ttft_ms"] is None:
                    out["ttft_ms"] = round((now - t0) * 1000)
                out["answer"] += value
            elif kind == "3":
                out["stream_error"] = str(value)
            elif kind in ("d", "e") and isinstance(value, dict):
                if value.get("usage"):
                    out["usage"] = value["usage"]
                if value.get("finishReason"):
                    out["finish_reason"] = value["finishReason"]
    out["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


# ----------------------------------------------------------------------------------
# Answer-only checks
# ----------------------------------------------------------------------------------
UNAVAILABLE = "unavailable_no_context"


def check_answer(harness, question, res):
    a = res["answer"]
    checks = {
        "T1.1_response_returned": bool(a.strip()) and not res["error"] and not res["stream_error"],
        "T1.2_under_15s": (res["total_ms"] is not None and res["total_ms"] < 15000),
        "C12_no_emoji": not EMOJI_RE.search(a),
        "C3_no_banned_names": not any(n in a.lower() for n in BANNED_NAMES),
        # Not computable without the retrieved context — see the module docstring.
        "C2_proper_noun_fidelity": UNAVAILABLE,
        "T2.1_retrieval_hit": UNAVAILABLE,
        "T2.2_figure_fidelity": UNAVAILABLE,
    }
    if harness is None:
        for k in ("T1.3_no_phone", "T1.4_third_person", "T1.5_no_jd_echo",
                  "C10_gap_honesty", "C18_injection_resistance"):
            checks[k] = "unavailable_no_checkers"
        return checks
    checks["T1.3_no_phone"] = not harness.PHONE_RE.search(a)
    checks["T1.4_third_person"] = not harness.FIRST_PERSON_RE.search(a)
    overlap = set(harness.ngrams(a)) & set(harness.JD_NGRAMS)
    checks["T1.5_no_jd_echo"] = not overlap
    if overlap:
        checks["T1.5_jd_echo_spans"] = sorted(overlap)[:5]
    # C10 — one implementation, shared with run_eval.py. Previously this called
    # harness.GAP_RE alone on raw text, dropping GAP_PHRASES and the lowercasing
    # run_eval applies; prod G02 scored False here and True under the harness rule.
    if question.get("tag") == "gap":
        checks["C10_gap_honesty"] = harness.gap_admitted(a)
    else:
        checks["C10_gap_honesty"] = "n/a"
    # C18 — Part 4a lists this as a 100% gate; run_e2e.py had no field for it at all.
    if question.get("tag") == "adversarial":
        verdict = harness.injection_resisted(a, question.get("adv_check"))
        checks["C18_injection_resistance"] = "n/a" if verdict is None else verdict
    else:
        checks["C18_injection_resistance"] = "n/a"
    return checks


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def expand_questions(eval_set):
    """One entry per API call, honouring the `runs` field (C01 runs 3x)."""
    out = []
    for q in eval_set["questions"]:
        for i in range(q.get("runs", 1)):
            item = dict(q)
            item["run_index"] = i + 1
            item["record_id"] = q["id"] if q.get("runs", 1) == 1 else f"{q['id']}#{i + 1}"
            out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser(description="End-to-end eval runner against the real chat route.")
    ap.add_argument("--base-url", required=True, help="http://localhost:3000 or https://jordannedyck.com")
    ap.add_argument("--source-system", choices=["local-route", "prod"],
                    help="override the stamp; inferred from --base-url otherwise")
    ap.add_argument("--probe", action="store_true",
                    help="send ONE question, dump the raw stream, verify the protocol, spend ~$0.01")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--start", type=int, default=0, help="skip the first N calls (resume a paced run)")
    ap.add_argument("--limit", type=int, default=18,
                    help="max calls this invocation (default 18, under the hourly 20)")
    ap.add_argument("--force-rate", action="store_true", help="allow --limit above 18")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between calls")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing output file, skipping ids already present")
    ap.add_argument("--out", help="override the output path")
    ap.add_argument("--expect-chunks", type=int,
                    help="abort unless the local index holds exactly this many chunks")
    ap.add_argument("--show-prompt-diff", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run anyway. Every number produced is then uninterpretable. Do not use.")
    args = ap.parse_args()

    if args.limit > 18 and not args.force_rate:
        sys.exit("--limit above 18 risks the hourly cap of 20. Pass --force-rate to mean it.")

    source_system = args.source_system or (
        "local-route" if "localhost" in args.base_url or "127.0.0.1" in args.base_url else "prod"
    )
    harness, harness_err = load_harness_checkers()
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    stamp = build_stamp(args, source_system)

    print(f"target        : {args.base_url}  (source_system: {source_system})")
    print(f"eval set      : {EVAL_SET.name} v{eval_set['meta'].get('version')} "
          f"— {len(eval_set['questions'])} questions")
    if harness_err:
        print(f"WARNING       : {harness_err}")
    print("preflight     :")
    ok, lines = run_preflights(args, source_system, harness, stamp)
    for ln in lines:
        print(ln)
    if not ok and not args.skip_preflight:
        sys.exit("\nPREFLIGHT FAILED — aborted before the first API call. "
                 "A run that cannot identify the system it measures should not spend money.")
    if not ok:
        print("\nWARNING: preflight failed and was skipped. These results are not interpretable.")
        stamp["preflight"] = "FAILED_AND_SKIPPED"

    calls = expand_questions(eval_set)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        calls = [c for c in calls if c["id"] in wanted]

    # UTC for filenames as well as for the C23 stamp, so a sitting that crosses local
    # midnight cannot split across two files and a file's name can never disagree with
    # the timestamp inside it. Probe output is named separately: it is a one-question
    # protocol check, not a run, and must never be mistaken for — or overwrite — one.
    utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = "eval_e2e_probe_" if args.probe else "eval_e2e_"
    out_path = Path(args.out) if args.out else HERE / f"{prefix}{source_system}_{utc_date}.json"
    existing, done = [], set()
    if out_path.exists() and not args.probe:
        if not args.resume:
            sys.exit(f"\n{out_path.name} already exists. Pass --resume to append, or --out to write "
                     f"elsewhere. This runner never overwrites results — --fresh is what cost the "
                     f"124-chunk run on 2026-09-10.")
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        existing = prior.get("results", [])
        done = {r["record_id"] for r in existing}
        print(f"\nresuming      : {len(done)} records already present")

    if args.probe:
        calls, args.limit = calls[:1], 1
        print("\nPROBE MODE — one question, raw stream saved.")

    queue = [c for c in calls[args.start:] if c["record_id"] not in done][:args.limit]
    if not queue:
        sys.exit("\nNothing to run.")
    print(f"\nrunning       : {len(queue)} calls "
          f"({queue[0]['record_id']} -> {queue[-1]['record_id']})\n")

    opener = make_opener()
    results, stopped = list(existing), None
    raw_path = HERE / (
        f"e2e_raw_probe_{source_system}_"
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%SZ')}.txt"
    )
    raw_sink = raw_path.open("wb") if args.probe else None

    try:
        for i, q in enumerate(queue):
            res = ask(opener, args.base_url, q["q"], args.timeout, raw_sink)
            record = {
                "record_id": q["record_id"], "id": q["id"], "run_index": q["run_index"],
                "tag": q.get("tag"), "req": q.get("req", []), "question": q["q"],
                "expect": q.get("expect", []),
                "answer": res["answer"], "error": res["error"], "stream_error": res["stream_error"],
                "http_status": res["http_status"],
                "timings_ms": {
                    "time_to_first_byte": res["ttfb_ms"],
                    "time_to_first_token": res["ttft_ms"],
                    "time_to_complete": res["total_ms"],
                    "note": "client-observed. TTFB bundles retrieval with model connect; "
                            "route.ts awaits searchExperience() before streamText(), so the "
                            "retrieval/generation split is NOT recoverable from this path.",
                },
                "usage": res["usage"],
                "finish_reason": res["finish_reason"],
                "stream": {
                    "parts_seen": res["parts_seen"], "unparsed_lines": res["unparsed_lines"],
                    "raw_bytes": res["raw_bytes"],
                    "content_type": res.get("content_type"),
                    "x_vercel_ai_data_stream": res.get("x_vercel_ai_data_stream"),
                },
                "checks": check_answer(harness, q, res),
                "stamp": stamp,
            }
            results.append(record)

            flag = ""
            if res["error"]:
                flag = f"  !! {res['error'][:90]}"
            elif res["stream_error"]:
                flag = f"  !! stream: {res['stream_error'][:90]}"
            print(f"  [{i + 1}/{len(queue)}] {q['record_id']:<8} "
                  f"ttft={res['ttft_ms'] or '-':>6}ms  total={res['total_ms'] or '-':>6}ms  "
                  f"chars={len(res['answer']):>5}{flag}")

            if res["http_status"] == 429:
                stopped = args.start + i
                print(f"\nRATE LIMITED. Stopping. Resume in an hour with "
                      f"--resume --start {stopped}")
                break
            if res["http_status"] == 500 and source_system == "local-route":
                print("\n  HTTP 500 on the local target. The most likely cause is the Postgres "
                      "dependency, not this runner:\n"
                      "  route.ts calls checkRateLimit() (pool.query) before generation and outside "
                      "the try block,\n  and awaits logChatEvent() in onFinish. Guard lib/db.ts, "
                      "lib/rate-limit.ts and lib/logging.ts\n  on !process.env.DATABASE_URL. "
                      "Check the `npm run dev` console for the actual error before assuming.")
                break
            if i < len(queue) - 1:
                time.sleep(args.sleep)
    finally:
        if raw_sink:
            raw_sink.close()

    payload = {
        "meta": {
            "runner": "run_e2e.py", "stamp": stamp,
            "eval_set_version": eval_set["meta"].get("version"),
            "calls_attempted": len(queue), "records_total": len(results),
            "stopped_at_index": stopped,
            "checks_unavailable_on_this_path": [
                "C2_proper_noun_fidelity", "T2.1_retrieval_hit", "T2.2_figure_fidelity",
                "C9_claim_level_reach",
            ],
            "unavailable_reason": "route.ts assembles the retrieved context server-side and does "
                                 "not emit it; no answer-vs-context check can run end to end.",
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote         : {out_path}")

    if args.probe:
        print(f"raw stream    : {raw_path}")
        r = results[-1]
        print("\nPROBE VERDICT")
        print(f"  http status            : {r['http_status']}")
        print(f"  content-type           : {r['stream']['content_type']}")
        print(f"  x-vercel-ai-data-stream: {r['stream']['x_vercel_ai_data_stream']}")
        print(f"  parts seen             : {r['stream']['parts_seen'] or 'NONE'}")
        print(f"  unparsed lines         : {r['stream']['unparsed_lines']}")
        print(f"  answer chars           : {len(r['answer'])}")
        usage_note = r["usage"] or "ABSENT — token and cost metrics unavailable end to end"
        print(f"  usage in stream        : {usage_note}")
        parts = r["stream"]["parts_seen"] or {}
        if not r["answer"]:
            print("\n  PARSE FAILED or the route errored. Open the raw file above and read the "
                  "first lines\n  before running anything else. Do not trust the parser until "
                  "text comes back.")
        else:
            caveats = []
            if not any(k in parts for k in ("d", "e")):
                caveats.append(
                    "No finish part (d:/e:) arrived. This target streams an OLDER protocol than the "
                    "local\n    bundle's ai ^4.3.19 produces. Consequences: no token counts, no cost, "
                    "and no\n    finishReason — so a truncated stream is indistinguishable from a "
                    "complete answer.\n    Read short answers rather than trusting the character count."
                )
            if r["stream"]["x_vercel_ai_data_stream"] is None:
                caveats.append(
                    "No x-vercel-ai-data-stream header. Same root cause as above; recorded in the "
                    "result\n    so the stream's shape stands as evidence of which bundle answered."
                )
            if r["usage"] is None:
                caveats.append(
                    "Usage absent — per-answer token and cost figures are NOT available on this "
                    "target\n    by any client-side means. They exist only in the chat_events table."
                )
            if caveats:
                print("\n  Parser works on this target — text came back clean. But note:")
                for c in caveats:
                    print(f"  - {c}")
                print("\n  Safe to run the full set, with those fields understood as unavailable.")
            else:
                print("\n  Protocol confirmed, finish part and usage present. Safe to run the full set.")
        head = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()[:6]
        print("\n  first raw lines:")
        for ln in head:
            print(f"    {ln[:160]}")


if __name__ == "__main__":
    main()
