"""
The rubric judge (review plan S1.1, 2026-11-26). Applies the general rules G1 to G4
(eval_set.json `rubric_rules`, G1 judged one part at a time) and the case's `rubric`
items to saved answers, against the record. Report-only: no gate reads its results.
S1.3 validates it against recorded rulings and chooses the model.

Never shown to the judge: a case's `known_failures`, its `allowed` list and its notes.
The input is built from the SHOWN_* field lists below, so a key added to eval_set.json
later stays out until someone adds it here. Every input is also scanned before it is
sent, in --estimate too, for known-failure and allowed text and for the FAILURE
clauses of every case note. A run where any input holds one is refused whole.

Record modes:
  whole   every chunk in faiss_db/documents.pkl, plus CORE POSITIONING.
  ledger  the --ledger file as text, then the answer's own retrieved chunks, plus
          CORE POSITIONING. --ledger has no default: the ledger's home is not decided.
          run_e2e.py files keep each chunk's text; run_eval.py and run_models.py files
          keep ids only, so the text comes from today's faiss_db.

Verdicts: pass, fail or does_not_apply, each with the span of the answer it rests on.
A fail with no span, or with a span not found in the answer, is recorded as invalid,
never as a pass. So is a rule the judge did not return. G4 on a turn with no turn
before it is does_not_apply, set by code, not asked.

Input files: results as run_eval.py and run_models.py write them (single and
--multiturn) and as run_e2e.py writes them (single and --multiturn). Each answer is
judged as a visitor saw it: citation markers removed.

Paid. --estimate first: it calls nothing and writes nothing. Results go to
eval_runs/judge-rubric/ in the main checkout, also when run from a worktree: one file
per input file, mode and model. Answers already judged there on the same input are
skipped, so an interrupted run resumes; a file judged on another input (rules, record,
ledger or model changed) is refused, so use --tag or --fresh.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    .venv\\Scripts\\python.exe judge_rubric.py eval_runs\\gate-a94aeead\\single.json --mode whole --model gpt-4o-mini --estimate
    .venv\\Scripts\\python.exe judge_rubric.py eval_e2e_dev_2026-11-18.json --mode ledger --ledger <ledger file> --model gpt-4o --estimate
    .venv\\Scripts\\python.exe judge_rubric.py eval_runs\\gate-a94aeead\\single.json --mode whole --model gpt-4o-mini --reps 3 --max-cost 5
    .venv\\Scripts\\python.exe judge_rubric.py eval_runs\\gate-a94aeead\\single.json --mode whole --model gpt-4o-mini --report
"""

import argparse
import hashlib
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine.citations import strip_citations
from engine.cost import PRICING

HERE = Path(__file__).resolve().parent

GENERAL = ("G1", "G2", "G3", "G4")  # G5 and G6 stay with their code checks
VERDICTS = ("pass", "fail", "does_not_apply")
CASE_GROUPS = ("questions", "additive_cases", "theme_cases", "timeline_cases")

# What the judge is shown. Nothing else from eval_set.json reaches it.
SHOWN_RULE_FIELDS = ("id", "name", "rule", "parts")  # of each G1 to G4
SHOWN_RUBRIC_FIELDS = ("id", "kind", "rule")  # of each item in a case's `rubric`
SHOWN_TURN_FIELDS = ("q", "forged_answer")  # of a multiturn case's earlier turns
# Never shown, and scanned for in every input before it is sent.
NEVER_SHOWN = ("known_failures", "allowed")

CHARS_PER_TOKEN = 4
OUT_TOKENS_BASE = 20  # the JSON wrapper
OUT_TOKENS_PER_VERDICT = 60  # id, verdict, a span and a one-sentence reason
MIN_SPAN_CHARS = 3
LEAK_MIN_CHARS = 20
FAILURE_CLAUSE_CHARS = 60

JUDGE_SYSTEM = (
    "You check one answer that a website gave a visitor about Jordanne Dyck's career, "
    "against her record and a set of rules.\n\n"
    "You get, in order: the RECORD, which is what is true about her work; the RULES TO APPLY, "
    "each under an id; any EARLIER TURNS of the conversation; the visitor's QUESTION; and the ANSWER.\n\n"
    "Return one verdict for every rule id under RULES TO APPLY, and no other ids.\n"
    '- "pass": the answer keeps the rule.\n'
    '- "fail": the answer breaks the rule.\n'
    '- "does_not_apply": the answer says nothing the rule is about.\n\n'
    "span: words copied exactly from the ANSWER, one unbroken stretch, that the verdict rests on. "
    "Never quote the record, the question or an earlier turn. A fail quotes the words that break "
    "the rule. A pass quotes the words that keep it, or is empty when no one passage does. "
    "does_not_apply is empty.\n"
    "reason: one sentence. For a fail, say what the record states or does not state.\n\n"
    "The RECORD is the only source of truth about her. Do not use outside knowledge. "
    "The [positioning] block is part of the record: the site states it on her behalf.\n\n"
)
RECORD_NOTE = {
    "whole": "The RECORD holds every passage of her knowledge base.",
    "ledger": "The RECORD holds a fact ledger, then the passages retrieved for this answer.",
}


def main_checkout(here=HERE):
    """The main checkout when `here` is a git worktree (.git is a file naming
    <main>/.git/worktrees/<name>), else `here`. Ignored files live there."""
    git = here / ".git"
    if git.is_file():
        m = re.match(r"gitdir:\s*(.+)", git.read_text(encoding="utf-8").strip())
        if m:
            gitdir = Path(m.group(1).strip())
            gitdir = gitdir if gitdir.is_absolute() else (here / gitdir).resolve()
            if gitdir.parent.name == "worktrees":
                return gitdir.parent.parent.parent
    return here


MAIN = main_checkout()


def _positioning():
    """CORE POSITIONING, as the grounding judge reads it. Imported late: citation_checks
    builds it from the system prompt."""
    from citation_checks import POSITIONING
    return POSITIONING


# ── what the judge is shown ───────────────────────────────────────────────────

def general_rules(spec):
    """G1 to G4 from `rubric_rules`, shown fields only."""
    by_id = {r["id"]: r for r in spec["rubric_rules"]}
    missing = [g for g in GENERAL if g not in by_id]
    if missing:
        raise SystemExit(f"STOP: rubric_rules has no {', '.join(missing)}")
    return [{k: by_id[g][k] for k in SHOWN_RULE_FIELDS if k in by_id[g]} for g in GENERAL]


def rule_lines(rules):
    """(id, text) per general verdict: one per G1 part, then G2 to G4."""
    out = []
    for r in rules:
        if r.get("parts"):
            for part, text in r["parts"].items():
                out.append((f"{r['id']}.{part}", f"{r['rule']} {part.capitalize()}: {text}"))
        else:
            out.append((r["id"], r["rule"]))
    return out


def all_cases(spec):
    """Every case by id, multiturn included."""
    out = {}
    for group in CASE_GROUPS:
        for c in spec.get(group, []):
            out[c["id"]] = c
    for c in (spec.get("multiturn") or {}).get("cases", []):
        out[c["id"]] = c
    return out


def shown_rubric(case, turn=None):
    """The case's `rubric` items, shown fields only. An item with a `turn` applies to that turn."""
    items = (case or {}).get("rubric") or []
    return [{k: it[k] for k in SHOWN_RUBRIC_FIELDS} for it in items
            if it.get("turn") is None or it.get("turn") == turn]


def earlier_turns(case, answered, turn):
    """The conversation before `turn`: each earlier question with the reply the visitor's
    client sent back. A forged_answer turn is shown as its forged reply, as the site got it."""
    if not turn or turn <= 1:
        return []
    case_turns = (case or {}).get("turns") or []
    out = []
    for j in range(1, turn):
        ct = {k: case_turns[j - 1][k] for k in SHOWN_TURN_FIELDS if k in case_turns[j - 1]} \
            if j <= len(case_turns) else {}
        if "forged_answer" in ct:
            out.append({"turn": j, "q": ct["q"], "answer": ct["forged_answer"]})
        elif j in answered:
            out.append({"turn": j, "q": answered[j]["question"], "answer": answered[j]["answer"]})
    return out


def record_text(mode, chunks, ledger_text=None):
    """The RECORD block. whole: every chunk. ledger: the ledger, then the retrieved chunks."""
    parts = [f"[positioning]\n{_positioning()}"]
    if mode == "ledger":
        parts.append(f"[ledger]\n{ledger_text}")
    parts += [f"[{c['chunk_id']}]\n{c['content']}" for c in chunks]
    return "\n\n".join(parts)


def build_input(spec, case, job, record):
    """(messages, ids asked, ids set by code). The record comes first so its tokens are
    the same prefix on every call."""
    lines = rule_lines(general_rules(spec))
    earlier = job.get("earlier") or []
    code_set = [] if earlier else ["G4"]
    asked = [(i, t) for i, t in lines if i not in code_set]
    asked += [(it["id"], f"({it['kind'].replace('_', ' ')}) {it['rule']}")
              for it in shown_rubric(case, job.get("turn"))]
    user = f"RECORD:\n{record}\n\nRULES TO APPLY:\n" + "\n".join(f"{i}: {t}" for i, t in asked)
    if earlier:
        user += "\n\nEARLIER TURNS:\n" + "\n\n".join(
            f"Turn {e['turn']} question: {e['q']}\nTurn {e['turn']} reply: {e['answer']}" for e in earlier)
    user += f"\n\nQUESTION:\n{job['question']}\n\nANSWER:\n{job['answer']}"
    system = JUDGE_SYSTEM + RECORD_NOTE[job["mode"]]
    return ([{"role": "system", "content": system}, {"role": "user", "content": user}],
            [i for i, _ in asked], code_set)


def schema(ids):
    return {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "enum": list(ids)},
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "span": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["id", "verdict", "span", "reason"],
            "additionalProperties": False,
        }}},
        "required": ["verdicts"],
        "additionalProperties": False,
    }


# ── leak scan ─────────────────────────────────────────────────────────────────

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                         "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def normalise(s):
    """For matching only: case, curly quotes, dashes, markdown emphasis and spacing ignored."""
    s = (s or "").translate(_QUOTES).replace("*", "").replace("`", "").lower()
    return re.sub(r"\s+", " ", s).strip()


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def leak_texts(spec):
    """(normalised text, where it came from) the judge must never be shown: every string
    of each known_failures and allowed item, and the start of every FAILURE clause in
    any case field (the notes the known failures were taken from)."""
    out = {}
    for cid, case in all_cases(spec).items():
        for field in NEVER_SHOWN:
            for item in case.get(field) or []:
                for s in _strings(item):
                    n = normalise(s)
                    if len(n) >= LEAK_MIN_CHARS:
                        out.setdefault(n, f"{cid} {field} {item.get('id', '') if isinstance(item, dict) else ''}")
        for s in _strings(case):
            for m in re.finditer(r"FAILURE\b[^.]*", s):
                n = normalise(m.group(0))[:FAILURE_CLAUSE_CHARS]
                if len(n) >= LEAK_MIN_CHARS:
                    out.setdefault(n, f"{cid} FAILURE clause")
    return out


def find_leaks(messages, leaks):
    text = normalise("\n".join(m["content"] for m in messages))
    return sorted({src for n, src in leaks.items() if n in text})


# ── verdicts ──────────────────────────────────────────────────────────────────

def span_found(span, answer):
    """True when the span is in the answer. Quotes, spacing, case and bold markers are
    ignored; a span cut with an ellipsis must have every piece in the answer, in order."""
    s = normalise(span).strip(" \"'")
    pieces = [p.strip(" \"'") for p in re.split(r"\.\.\.|\u2026", s)]
    pieces = [p for p in pieces if p]
    if sum(len(p.replace(" ", "")) for p in pieces) < MIN_SPAN_CHARS:
        return False
    text, at = normalise(answer), 0
    for p in pieces:
        at = text.find(p, at)
        if at < 0:
            return False
        at += len(p)
    return True


def validate(content, ids, answer):
    """One record per asked id, from the judge's reply. A fail with no span, or with a span
    not in the answer, an id not returned, a contradicting duplicate and an unreadable
    reply are all `invalid`: never a pass."""
    try:
        got = json.loads(content)["verdicts"]
        assert isinstance(got, list)
    except (ValueError, KeyError, TypeError, AssertionError):
        return [{"id": i, "verdict": None, "span": "", "reason": "", "status": "invalid",
                 "why": "reply not readable"} for i in ids], []
    by_id, unexpected = {}, []
    for v in got:
        if not isinstance(v, dict) or v.get("id") not in ids:
            unexpected.append(v.get("id") if isinstance(v, dict) else v)
        else:
            by_id.setdefault(v["id"], []).append(v)
    out = []
    for i in ids:
        vs = by_id.get(i, [])
        if not vs:
            out.append({"id": i, "verdict": None, "span": "", "reason": "", "status": "invalid",
                        "why": "not returned"})
            continue
        v = vs[0]
        verdict, span = v.get("verdict"), (v.get("span") or "").strip()
        rec = {"id": i, "verdict": verdict, "span": span, "reason": (v.get("reason") or "")[:400],
               "span_in_answer": span_found(span, answer) if span else None, "status": "valid"}
        if len({x.get("verdict") for x in vs}) > 1:
            rec.update(status="invalid", why="returned twice with different verdicts")
        elif verdict not in VERDICTS:
            rec.update(status="invalid", why="unknown verdict")
        elif verdict == "fail" and not span:
            rec.update(status="invalid", why="fail with no span")
        elif verdict == "fail" and not rec["span_in_answer"]:
            rec.update(status="invalid", why="fail span not in the answer")
        out.append(rec)
    return out, unexpected


def code_verdicts(ids):
    return [{"id": i, "verdict": "does_not_apply", "span": "", "reason": "no turn before this one",
             "span_in_answer": None, "status": "valid", "source": "code"} for i in ids]


# ── input files ───────────────────────────────────────────────────────────────

def _visible(text):
    return strip_citations(text or "").strip()


def _chunks(saved_turn):
    """(ids, chunks with text or None) the answer was given. run_e2e keeps the text;
    run_eval and run_models keep ids only."""
    if saved_turn.get("retrieval") is not None:
        cs = (saved_turn["retrieval"] or {}).get("chunks") or []
        return [c["chunk_id"] for c in cs], [{"chunk_id": c["chunk_id"], "content": c["content"]} for c in cs]
    return [c["chunk_id"] for c in saved_turn.get("retrieved") or []], None


def answers_in(path, cases):
    """Every answer in a results file, single-turn rows and multi-turn turns alike, as jobs
    without a mode. Turns with no answer (an error) are skipped and counted."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) else data
    jobs, skipped = [], 0
    for row in rows:
        cid = row.get("id")
        model = row.get("model") or ((row.get("stamp") or {}).get("model")) or ""
        run = row.get("run", row.get("run_index", ""))
        turns = row.get("turns")
        saved = turns if turns else [row]
        answered = {}
        for t in saved:
            answer = _visible((t.get("gen") or {}).get("answer") if "gen" in t else t.get("answer"))
            if not answer:
                skipped += 1
                continue
            turn = t.get("turn") if turns else None
            ids, chunks = _chunks(t)
            job = {"key": f"{cid}|{model}|{run}|{turn or ''}", "id": cid, "turn": turn,
                   "question": t.get("question", ""), "answer": answer, "chunk_ids": ids, "chunks": chunks,
                   "index_version": t.get("index_version") or row.get("index_version")
                   or (row.get("stamp") or {}).get("index_version")}
            if turns:
                job["earlier"] = earlier_turns(cases.get(cid), answered, turn)
                answered[turn] = job
            jobs.append(job)
    return jobs, skipped


def load_index(folder):
    """(chunks in index order, fingerprint) from a faiss_db folder."""
    folder = Path(folder)
    with open(folder / "documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(folder / "metadatas.pkl", "rb") as f:
        metas = pickle.load(f)
    import retrieval
    try:
        version = retrieval.index_fingerprint(folder)
    except OSError:
        version = "no-fingerprint"
    return [{"chunk_id": m.get("chunk_id"), "content": d} for d, m in zip(docs, metas)], version


# ── cost ──────────────────────────────────────────────────────────────────────

def price_of(model, price_arg=None):
    """(USD per 1M input, per 1M output) from engine/cost.py, or --price for a model it lacks."""
    if price_arg:
        a, b = (float(x) for x in price_arg.split(","))
        return a, b
    if model not in PRICING:
        raise SystemExit(f"STOP: no price for {model} in engine/cost.py. Pass --price IN,OUT (USD per 1M tokens).")
    return PRICING[model]


def estimate_usd(sizes, price):
    """sizes: (input characters, verdicts asked) per call. Returns (tokens in, tokens out, USD)
    at 4 characters a token and OUT_TOKENS_* per reply."""
    tin = sum(chars for chars, _ in sizes) / CHARS_PER_TOKEN
    tout = sum(OUT_TOKENS_BASE + OUT_TOKENS_PER_VERDICT * n for _, n in sizes)
    return tin, tout, tin / 1e6 * price[0] + tout / 1e6 * price[1]


# ── run ───────────────────────────────────────────────────────────────────────

def require_key(env):
    if not env.get("OPENAI_API_KEY"):
        raise SystemExit("FAILED: OPENAI_API_KEY not set. Add it to .env in the main checkout.")


def openai_client(env):
    from openai import OpenAI
    return OpenAI(api_key=env["OPENAI_API_KEY"])


def resolve(p):
    """A path as given, else under the main checkout (worktrees hold no results)."""
    p = Path(p)
    return p if p.exists() or p.is_absolute() else MAIN / p


def out_file(args, path):
    p = Path(path).resolve()
    stem = p.stem if p.parent in (HERE, MAIN) else f"{p.parent.name}_{p.stem}"
    return Path(args.out_dir) / f"{stem}_{args.mode}_{args.model}{'-' + args.tag if args.tag else ''}.json"


def index_of(args, out=print):
    """The faiss_db index, loaded once, on first need."""
    if args.index is None:
        chunks, version = load_index(args.faiss_dir)
        args.index = {"chunks": chunks, "by_id": {c["chunk_id"]: c["content"] for c in chunks}, "version": version}
        out(f"index {args.faiss_dir}: {len(chunks)} chunks, {sum(len(c['content']) for c in chunks):,} "
            f"characters, version {version}")
    return args.index


def prepare(args, spec, file, out=print):
    """Jobs for one input file, each with its messages, asked ids and input hash."""
    cases = all_cases(spec)
    jobs, skipped = answers_in(file, cases)
    if args.only:
        want = {s.strip().upper() for s in args.only.split(",")}
        jobs = [j for j in jobs if (j["id"] or "").upper() in want]
    missing = set()
    for j in jobs:
        j["mode"] = args.mode
        if args.mode == "whole":
            chunks = index_of(args, out)["chunks"]
        else:
            chunks = j["chunks"]
            if chunks is None:  # ids only: today's text
                by_id = index_of(args, out)["by_id"]
                missing |= {i for i in j["chunk_ids"] if i not in by_id}
                chunks = [{"chunk_id": i, "content": by_id[i]} for i in j["chunk_ids"] if i in by_id]
        j["messages"], j["ids"], j["code_ids"] = build_input(spec, cases.get(j["id"]), j,
                                                             record_text(args.mode, chunks, args.ledger_text))
        j["has_case"] = j["id"] in cases
        j["input_sha256"] = hashlib.sha256(json.dumps([args.model, j["messages"]]).encode("utf-8")).hexdigest()
    return jobs, skipped, missing


def judge_one(client, model, price, job):
    resp = client.chat.completions.create(
        model=model, temperature=0, messages=job["messages"], timeout=180,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verdicts", "strict": True, "schema": schema(job["ids"])}})
    verdicts, unexpected = validate(resp.choices[0].message.content, job["ids"], job["answer"])
    u = resp.usage
    tin, tout = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
    return {"verdicts": verdicts + code_verdicts(job["code_ids"]), "unexpected": unexpected,
            "usage": {"prompt_tokens": tin, "completion_tokens": tout},
            "cost_usd": round(tin / 1e6 * price[0] + tout / 1e6 * price[1], 6)}


def record_of(args, job, rep, result):
    rec = {k: job[k] for k in ("key", "id", "turn", "question", "answer", "index_version", "has_case")}
    rec.update(rep=rep, mode=args.mode, model=args.model, record=args.record_label,
               asked=job["ids"], input_sha256=job["input_sha256"], **result)
    rec["invalid"] = sum(1 for v in rec.get("verdicts", []) if v["status"] == "invalid")
    rec["fail"] = sum(1 for v in rec.get("verdicts", []) if v["status"] == "valid" and v["verdict"] == "fail")
    return rec


def load_out(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_out(path, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def report(path, done, out=print):
    """Verdicts per rule id, valid ones by verdict, and the invalid count."""
    recs = [r for r in done.values() if "verdicts" in r]
    rows = {}
    for r in recs:
        for v in r["verdicts"]:
            k = v["verdict"] if v["status"] == "valid" else "invalid"
            rows.setdefault(v["id"], {}).setdefault(k, 0)
            rows[v["id"]][k] += 1
    out(f"\n{path.name}: {len(recs)} answers judged, {sum(1 for r in done.values() if 'error' in r)} errors, "
        f"${sum(r.get('cost_usd', 0) for r in recs):.4f} spent")
    general = [i for i in rows if i.split(".")[0] in GENERAL]
    for i in sorted(general) + sorted(set(rows) - set(general)):
        c = rows[i]
        out(f"  {i:<14} pass {c.get('pass', 0):>4}  fail {c.get('fail', 0):>4}  "
            f"n/a {c.get('does_not_apply', 0):>4}  invalid {c.get('invalid', 0):>4}")


def run(args, env=os.environ, make_client=openai_client, out=print):
    """Plans every file, prints the estimate, and judges unless --estimate. Returns the plan totals."""
    spec = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    price = price_of(args.model, args.price)
    leaks = leak_texts(spec)
    args.index, args.ledger_text = None, None
    if args.mode == "ledger":
        raw = Path(args.ledger).read_bytes()
        args.ledger_text = raw.decode("utf-8")
        args.record_label = {"mode": "ledger", "ledger": str(args.ledger),
                             "ledger_sha256": hashlib.sha256(raw).hexdigest()[:12]}
        out(f"ledger {args.ledger}: {len(args.ledger_text):,} characters, sha256 {args.record_label['ledger_sha256']}")
    else:
        args.record_label = {"mode": "whole", "index_version": index_of(args, out)["version"]}

    plan, leaked, totals = [], {}, {"calls": 0, "tokens_in": 0.0, "tokens_out": 0, "usd": 0.0, "answers": 0}
    for f in args.files:
        path = resolve(f)
        jobs, skipped, missing = prepare(args, spec, path, out)
        dest = out_file(args, path)
        done = {} if args.fresh else load_out(dest)
        sha = {f"{j['key']}|rep{rep}": j["input_sha256"] for j in jobs for rep in range(1, args.reps + 1)}
        stale = [k for k, r in done.items() if k in sha and r.get("input_sha256") != sha[k]]
        todo = [(j, rep) for j in jobs for rep in range(1, args.reps + 1)
                if "verdicts" not in done.get(f"{j['key']}|rep{rep}", {})]
        for j in jobs:
            for src in find_leaks(j["messages"], leaks):
                leaked.setdefault(src, []).append(f"{path.name} {j['key']}")
        tin, tout, usd = estimate_usd([(sum(len(m["content"]) for m in j["messages"]), len(j["ids"])) for j, _ in todo], price)
        versions = sorted({j["index_version"] or "?" for j in jobs})
        out(f"{path}: {len(jobs)} answers ({skipped} turns with no answer skipped, "
            f"{sum(1 for j in jobs if not j['has_case'])} with no case), answered on index {', '.join(versions)}; "
            f"{len(todo)} calls to make ({len(jobs) * args.reps - len(todo)} done), "
            f"{tin:,.0f} tokens in, {tout:,} out, about ${usd:.4f} on {args.model}")
        if missing:
            out(f"  {len(missing)} retrieved chunk ids not in today's index, left out: {sorted(missing)}")
        if stale:
            out(f"  {dest.name} holds {len(stale)} verdicts judged on another input (rules, record or answer changed)")
        plan.append((path, dest, done, todo, stale))
        totals["calls"] += len(todo)
        totals["tokens_in"] += tin
        totals["tokens_out"] += tout
        totals["usd"] += usd
        totals["answers"] += len(jobs)
    out(f"total: {totals['answers']} answers x {args.reps} reps, {totals['calls']} calls, "
        f"about ${totals['usd']:.2f} on {args.model} (${price[0]}/${price[1]} per 1M tokens, "
        f"{CHARS_PER_TOKEN} characters a token)")
    if leaked:
        out(f"REFUSED: {len(leaked)} known-failure, allowed or FAILURE-clause texts are in the judge's input:")
        for src, where in sorted(leaked.items()):
            out(f"  {src}: in {len(where)} inputs, e.g. {where[0]}")
    if args.report:
        for path, dest, done, _, _ in plan:
            report(dest, done, out)
        return totals
    if args.estimate:
        out("--estimate: nothing was called or written.")
        return totals
    if leaked:
        raise SystemExit("STOP: the judge would be shown text it must never see. Nothing was called.")
    if any(stale for *_, stale in plan):
        raise SystemExit("STOP: a results file holds verdicts on another input. Use --tag for a new file, "
                         "or --fresh to replace it.")
    require_key(env)
    client = make_client(env)
    spent = 0.0
    for path, dest, done, todo, _ in plan:
        for start in range(0, len(todo), args.workers):
            if args.max_cost is not None and spent >= args.max_cost:
                out(f"STOPPED: spend ${spent:.4f} reached --max-cost {args.max_cost}")
                return totals
            batch = todo[start:start + args.workers]

            def one(item):
                j, rep = item
                try:
                    return j, rep, judge_one(client, args.model, price, j)
                except Exception as e:  # noqa: BLE001  recorded and retried on resume
                    return j, rep, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for j, rep, result in ex.map(one, batch):
                    done[f"{j['key']}|rep{rep}"] = record_of(args, j, rep, result)
                    spent += result.get("cost_usd", 0)
            save_out(dest, done)
            out(f"  {dest.name}: {min(start + args.workers, len(todo))}/{len(todo)}, ${spent:.4f} spent")
        report(dest, done, out)
    out(f"\n${spent:.4f} spent this run")
    return totals


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("files", nargs="+", help="results files from run_eval.py, run_models.py or run_e2e.py")
    ap.add_argument("--mode", choices=["whole", "ledger"], required=True, help="what the judge reads as the record")
    ap.add_argument("--ledger", type=Path, help="the ledger file, read as text (required with --mode ledger)")
    ap.add_argument("--model", required=True, help="the judge model; S1.3 chooses it")
    ap.add_argument("--price", help="IN,OUT USD per 1M tokens, for a model engine/cost.py has no price for")
    ap.add_argument("--reps", type=int, default=1, help="judge each answer this many times")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--estimate", action="store_true", help="print calls and cost, call nothing, write nothing")
    ap.add_argument("--report", action="store_true", help="summarise the saved results, call nothing")
    ap.add_argument("--max-cost", type=float, help="stop once spend reaches this many USD")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", help="suffix for the results file name")
    ap.add_argument("--fresh", action="store_true", help="ignore and replace the saved results")
    ap.add_argument("--eval-set", type=Path, default=HERE / "eval_set.json")
    ap.add_argument("--faiss-dir", type=Path, default=MAIN / "faiss_db")
    ap.add_argument("--out-dir", type=Path, default=MAIN / "eval_runs" / "judge-rubric")
    args = ap.parse_args(argv)
    if args.mode == "ledger" and not args.ledger:
        ap.error("--mode ledger needs --ledger <file>: the ledger has no default home yet")
    return args


def main():
    from dotenv import load_dotenv
    load_dotenv(MAIN / ".env")
    load_dotenv(HERE / ".env")
    run(parse_args())


if __name__ == "__main__":
    main()
