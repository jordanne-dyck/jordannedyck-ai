"""
Checks on the grounding judge (citation_checks.judge_grounding, gpt-4o-mini),
added 2026-10-30 after plan A.8 found it marking non-claims "no": list lead-in
lines, gap admissions, and suggested questions.

Two commands. Both call the OpenAI API and cost money, so run --estimate first.
  cases     Runs the named cases in judge_cases.json and prints each claim's
            verdict against the verdicts it may get. A case marked "open" is a
            known defect: its failure prints OPEN and is not counted.
  rejudge   Re-judges answers already stored in run_models.py results files,
            without generating anything. The stored runs keep chunk ids, not
            chunk text, so the text comes from today's faiss_db/. That is not
            what the judge saw on the day, which is why --setting both judges
            the same answers with lead-ins folded and unfolded against the same
            text: the difference between the two is the folding, not a
            knowledge-base edit.
            Writes eval_runs/rejudge_<results file>_<setting>.json, and skips
            answers already in it, so an interrupted run can be resumed.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    uv run python judge_check.py cases --estimate
    uv run python judge_check.py cases --reps 3
    uv run python judge_check.py rejudge eval_runs/redecide-ff4d_2026-10-22.json --estimate
    uv run python judge_check.py rejudge eval_runs/redecide-ff4d_2026-10-22.json
    uv run python judge_check.py rejudge eval_runs/redecide-ff4d_2026-10-22.json --report
"""

import argparse
import json
import os
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import citation_checks as C
from run_eval import GAP_PHRASES, GAP_RE, NO_COVERAGE_RE

HERE = Path(__file__).resolve().parent
CASES = HERE / "judge_cases.json"
SETTINGS = {"folded": {"fold": True}, "unfolded": {"fold": False}}
OUT_TOKENS_PER_CLAIM = 30  # a verdict plus a one-sentence reason, rounded up

QUESTION_RE = re.compile(r"\?[*_\s]*$|^[-*\s]*(?:ask her|you (?:could|might|can) ask)\b", re.IGNORECASE)


def kind(text):
    """What a judged line is, for the report only: lead-in, question, gap or claim."""
    low = text.lower()
    if C.LEAD_IN_END_RE.search(text):
        return "lead-in"
    if QUESTION_RE.search(text):
        return "question"
    if any(p in low for p in GAP_PHRASES) or GAP_RE.search(low) or NO_COVERAGE_RE.search(low):
        return "gap"
    return "claim"


def estimate(jobs, setting):
    """Calls and dollars, from prompt length (4 characters a token) and claim count."""
    tin = tout = 0
    for query, segments, chunks in jobs:
        judged, prompt = C.judge_prompt(query, "", segments, chunks, SETTINGS[setting]["fold"])
        tin += (len(C.JUDGE_SYSTEM) + len(prompt)) / 4
        tout += 10 + OUT_TOKENS_PER_CLAIM * len(judged)
    return len(jobs), tin / 1e6 * C.JUDGE_COST_PER_1M_IN + tout / 1e6 * C.JUDGE_COST_PER_1M_OUT


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(HERE / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("FAILED: OPENAI_API_KEY not set. Run from the folder containing .env")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── cases ─────────────────────────────────────────────────────────────────────

def run_cases(args):
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    jobs = [(c["question"], c["segments"], c["chunks"]) for c in cases]
    n, usd = estimate(jobs, args.setting)
    print(f"{n} cases x {args.reps} reps = {n * args.reps} calls to {C.JUDGE_MODEL}, lead-ins {args.setting}, "
          f"about ${usd * args.reps:.4f}")
    if args.estimate:
        return
    cl, fails, spent = client(), [], 0.0
    passed = {c["id"]: 0 for c in cases}
    for rep in range(1, args.reps + 1):
        for c in cases:
            j = C.judge_grounding(cl, c["question"], "", c["segments"], c["chunks"], **SETTINGS[args.setting])
            spent += j["cost_usd"]
            got = {v["n"]: v for v in j["claims"]}
            ok_all = len(j["claims"]) == len(c["want"]) and all(
                got.get(k, {}).get("supported") in w for k, w in enumerate(c["want"], 1))
            passed[c["id"]] += ok_all
            if not ok_all and not c.get("open"):
                fails.append(f"{c['id']} rep {rep}")
            for n_, want in enumerate(c["want"], 1):
                v = got.get(n_, {"supported": "(missing)", "reason": "", "text": ""})
                status = "PASS" if v["supported"] in want else "OPEN" if c.get("open") else "FAIL"
                print(f"{status}  {c['id']} ({c['defect']}) claim {n_}: got {v['supported']}, want {'/'.join(want)}")
                print(f"      text:  {v['text']}")
                if v["reason"]:
                    print(f"      judge: {v['reason']}")
            if len(j["claims"]) != len(c["want"]):
                print(f"{'OPEN' if c.get('open') else 'FAIL'}  {c['id']}: judge returned {len(j['claims'])} claims, "
                      f"want {len(c['want'])}")
    print(f"\nlead-ins {args.setting}, passes out of {args.reps}: "
          + "  ".join(f"{c['id']}{'(open)' if c.get('open') else ''} {passed[c['id']]}" for c in cases))
    print(f"spent ${spent:.4f}. {'ALL PASS (open cases not counted)' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


# ── rejudge ───────────────────────────────────────────────────────────────────

def load_chunk_text():
    with open(HERE / "faiss_db" / "documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(HERE / "faiss_db" / "metadatas.pkl", "rb") as f:
        metas = pickle.load(f)
    return {m.get("chunk_id"): d for d, m in zip(docs, metas) if m.get("chunk_id")}


def stored_answers(path, text_by_id):
    """(key, query, answer, segments, chunks) for every judged answer in a
    results file, single-turn rows and multi-turn turns alike, plus the chunk
    ids not in today's index."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out, missing = [], set()
    for r in rows:
        for t in r.get("turns") or [r]:
            if not (t.get("judge") or {}).get("claims") and not (t.get("judge") or {}).get("checked"):
                continue  # not judged on the day: no chunks, or --no-judge
            ids = [c["chunk_id"] for c in t["retrieved"]]
            missing |= {i for i in ids if i not in text_by_id}
            chunks = [{"chunk_id": c["chunk_id"], "similarity": c["similarity"], "content": text_by_id[c["chunk_id"]]}
                      for c in t["retrieved"] if c["chunk_id"] in text_by_id]
            segments, _ = C.parse_citations(t["gen"]["answer_cited"] or "")
            key = f"{r['id']}|{r.get('model', '')}|{r.get('run', '')}|{t.get('turn', '')}"
            out.append((key, t["question"], t["gen"]["answer"], segments, chunks))
    return out, missing


def out_path(results, setting):
    return HERE / "eval_runs" / f"rejudge_{Path(results).stem}_{setting}.json"


def run_rejudge(args):
    text_by_id = load_chunk_text()
    settings = ["unfolded", "folded"] if args.setting == "both" else [args.setting]
    plan = []
    for f in args.files:
        answers, missing = stored_answers(f, text_by_id)
        if missing:
            print(f"{f}: {len(missing)} chunk ids not in today's index, judged without them: {sorted(missing)}")
        for s in settings:
            done = _load_out(out_path(f, s))
            todo = [a for a in answers if a[0] not in done]
            n, usd = estimate([(q, segs, c) for _, q, _, segs, c in todo], s)
            print(f"{f} lead-ins {s}: {len(answers)} judged answers, {n} still to judge, about ${usd:.4f}")
            plan.append((f, s, todo, done))
    if args.estimate:
        return
    if not args.report:
        cl = client()
        for f, s, todo, done in plan:
            def one(a, s=s):
                key, q, ans, segs, chunks = a
                return key, C.judge_grounding(cl, q, ans, segs, chunks, **SETTINGS[s])
            with ThreadPoolExecutor(max_workers=8) as ex:
                for i, (key, j) in enumerate(ex.map(one, todo), 1):
                    done[key] = j
                    if i % 25 == 0 or i == len(todo):
                        out_path(f, s).write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
                        print(f"  {f} {s}: {i}/{len(todo)}", flush=True)
    for f, s, _, done in plan:
        report(f, s, done)
    if len(settings) == 2:
        for f in args.files:
            compare(f, _load_out(out_path(f, "unfolded")), _load_out(out_path(f, "folded")))


def _load_out(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def report(f, s, done):
    nos = [(k, c) for k, j in done.items() for c in j["claims"] if c["supported"] == "no"]
    by_kind = {}
    for k, c in nos:
        by_kind.setdefault(kind(c["text"]), []).append((k, c))
    flagged = sum(1 for j in done.values() if j["no"])
    spent = sum(j["cost_usd"] for j in done.values())
    print(f"\n{f}, lead-ins {s}: {len(done)} answers, {flagged} flagged (any claim \"no\"), "
          f"{len(nos)} claims \"no\", lead-ins folded {sum(j.get('folded', 0) for j in done.values())}, "
          f"spent ${spent:.4f}")
    for kd in ("lead-in", "gap", "question", "claim"):
        print(f"  \"no\" on {kd}: {len(by_kind.get(kd, []))}")
        if kd != "claim":
            for k, c in by_kind.get(kd, []):
                print(f"    [{k}] cites={','.join(c['ids']) or 'none'} | {c['text']}")
                print(f"      judge: {c['reason']}")


def compare(f, before, after):
    """Answers whose flag changed between unfolded and folded, with the claims behind it."""
    keys = sorted(set(before) & set(after))
    cleared = [k for k in keys if before[k]["no"] and not after[k]["no"]]
    raised = [k for k in keys if not before[k]["no"] and after[k]["no"]]
    print(f"\n{f}: {len(keys)} answers judged both ways. Flagged unfolded "
          f"{sum(1 for k in keys if before[k]['no'])}, folded {sum(1 for k in keys if after[k]['no'])}; "
          f"cleared {len(cleared)}, newly flagged {len(raised)}")
    for label, ks, side in (("cleared (unfolded \"no\" claims)", cleared, before),
                            ("newly flagged (folded \"no\" claims)", raised, after)):
        print(f"  {label}:")
        for k in ks:
            for c in side[k]["claims"]:
                if c["supported"] == "no":
                    print(f"    [{k}] {kind(c['text'])} | {c['text']}")
                    print(f"      judge: {c['reason']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cases", help="run judge_cases.json")
    a.add_argument("--reps", type=int, default=1)
    a.add_argument("--setting", choices=list(SETTINGS), default="folded")
    a.add_argument("--estimate", action="store_true", help="print calls and cost, call nothing")
    b = sub.add_parser("rejudge", help="re-judge stored run_models.py answers")
    b.add_argument("files", nargs="+", type=Path)
    b.add_argument("--setting", choices=[*SETTINGS, "both"], default="both")
    b.add_argument("--estimate", action="store_true", help="print calls and cost, call nothing")
    b.add_argument("--report", action="store_true", help="print the saved re-judge results, call nothing")
    args = ap.parse_args()
    (run_cases if args.cmd == "cases" else run_rejudge)(args)


if __name__ == "__main__":
    main()
