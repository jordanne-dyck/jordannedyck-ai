#!/usr/bin/env python
"""
Release gate (2026-11-05). Writes the evidence file a prod change needs before
CI in jordannedyck-ai-web lets its PR merge.

A prod change is one of: a jordbot-web tag, a jordbot tag, or prod's FAISS_DIR.
Each needs, for the prompt and index it will run with:
  - a full single-turn run_eval.py run          (history kind "single", scope "full")
  - a full multi-turn run_eval.py run           (kind "multiturn", scope "full")
  - a full run_e2e.py run against jordbot-dev   (kind "e2e", scope "full"), whose stamp
    says dev served that prompt, taken after the tested commit, and (2026-11-09) whose
    answers cover every case run_e2e.py's full set holds today, the timeline cases
    TL01-TL06 included; a run labelled full before they joined it does not count
  - the same for run_e2e.py --multiturn         (kind "e2e_multiturn"): every
    multiturn case (M01-M11) replayed through the route with signed answers. Its
    failures are listed under its own kind; like the e2e run's, they cannot be
    signed off as "not served"
  - (2026-11-09) each of those four answered from the index going out, by the
    index_version its answers recorded from /search, never by filename, and its
    results file still holding the answers its history line was written from (same
    run_id). An e2e run proves its index only from a site that sends the
    data-index part (web, 2026-11-09); an older run recorded none and does not count.
    A multi-turn e2e run needs it on every turn of every conversation.
  - every failing check signed off by Jord, by case, in the evidence file; a tier-1
    failure only as "checker miss: <why>", as "not served: <why>" when the e2e run
    passed that case on dev, or (Jord, 2026-11-17) as "known: <why>" naming the
    results-log entry that measured its rate (R-YYYYMMDD-NN)
  - (2026-11-18) a verdict, true or false, from every tier-1 check on every answer
    that carries it, in all four runs. None, "n/a" or "unavailable_*" there is a
    problem no sign-off clears: the check did not run
  - every QA item ticked, in the evidence file
  - (2026-11-07) the public docs brought up to date and approved by Jord: each doc
    in DOCS, as it stands on origin/master, has "approved": true in the evidence
    file; COVERAGE.md on origin/master is what coverage_report.py generates; and
    no results-log entry is still "reviewed: no"

The report also gives rates (2026-11-26), under "rates" in the evidence file. They
are a report only: no rate adds a problem or clears one. Per case in each of the
four runs, the share of its repetitions that passed every check, and each failing
check's share of the case's answers, with 95% Wilson ranges. Then, apart, the
checks that did not fail once in the history over the period (--since, default
the last 30 days): a check that cannot fail is an invariant, not evidence of
quality. --rates FILE... prints the same report for saved results files, such as
a rate run, without running the gate.

Reads eval_history/history.jsonl, the results files it names, the results log and
the docs on origin/master (it fetches both repos first). Calls no paid API and
spends nothing; the runs above are separate, paid, and need Jord's go.

Re-running keeps the sign-offs, ticks and approvals already in the file, so the
loop is: run the gate, read the failures it prints, write Jord's ruling into each
"signoff" (her words, not a summary), tick each QA item she did, set "approved"
on each doc she approved, run it again. An approval holds for the commit it was
given on: a doc that changes on master afterwards needs approving again. A sign-off
holds for the run and count it was given on: a new run needs a new ruling.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    uv run python release_gate.py --image jordbot-web --tag v1.16 --sha <web commit> \
        --prompt dd326013 --index <index_version>
    uv run python release_gate.py --faiss-dir /faiss/kb-762b34e --prompt dd326013 --index <index_version>
    uv run python release_gate.py --rates eval_runs/rate-2026-11-17/m10-a94aeead.json --since 2026-11-01
Exit 0 only when the evidence file says "status": "pass".
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import bootstrap_ci as B
import run_e2e as E2E
import run_metrics as RM

# A "known:" tier-1 sign-off names the results-log entry that measured the rate.
KNOWN_RATE_RE = re.compile(r"\bR-\d{8}-\d{2}\b")

HERE = Path(__file__).resolve().parent
HISTORY = HERE / "eval_history" / "history.jsonl"
E2E_KINDS = ("e2e", "e2e_multiturn")
WEB = HERE.parent / "jordannedyck-ai-web"

QA = [
    "Overview question and a past-jobs question on a phone: the timeline plays, scrolls and exits",
    "An answer that cites a project shows its project card, image legible",
    "Fixed replies: the career lead and the instructions refusal, asked once each",
    "One conversation of three follow-ups keeps context and cites every answer",
    "/dashboard/admin loads; /api/health reports ok, db and the expected prompt_version",
    "Five answers read end to end for voice, not just for a response",
    # 2026-11-10 (Jord): prod #112 turned the lead form on before lead email had worked
    # on dev once (Mailgun 401 until a sending key went in). Dev first, every release.
    "Lead email on dev: scripts/lead-check.mjs (web repo) ends OK, and one real lead from the dev site reaches Jordanne's inbox",
    # 2026-11-18 (Jord): jordbot-web v1.18 carried the traffic tab and its page-view
    # beacon (web 1f22beb) with no QA item covering them.
    "Traffic tab on dev: browse two pages and ask one question from a browser not left out, then /dashboard/admin/traffic shows that visit and the chat joined to it",
]

RESULTS_LOG = HERE / "nimbalyst-local" / "results-log.md"
# Docs a release brings up to date (Jord, 2026-11-07). Claude drafts the changes
# from reviewed results, Jord reads what was added and approves each doc, and the
# approved text is merged to master before the prod PR. "ai" is this repo.
DOCS = [("ai", "README.md"), ("ai", "EVALS.md"), ("ai", "DECISIONS.md"),
        ("ai", "COVERAGE.md"), ("web", "README.md")]


def results_path(source):
    for p in (HERE / "eval_runs" / source, HERE / source):
        if p.exists():
            return p
    return None


def load_run(row):
    """(data, None) while the row's results file still holds the answers its run_id
    was computed from, else (None, why). As analyse.py checks: a re-run that reuses
    a filename changes the file, not the history line (bug, 2026-11-07)."""
    p = results_path(row["source"])
    if p is None:
        return None, f"results file {row['source']} not found (eval_runs/ is not in git)"
    data = json.loads(p.read_text(encoding="utf-8"))
    ids = [ln["run_id"] for ln in RM.history_lines(
        data, runner=row["runner"], source=row["source"], label=row.get("label"),
        date=row.get("date"), scope=row.get("scope"))]
    if row["run_id"] not in ids:
        return None, f"{row['source']} no longer holds run {row['run_id']} (it holds {', '.join(ids)})"
    return data, None


def find_runs(rows, prompt, index, since, skipped=None, e2e_ids=None, mt_ids=None):
    """Latest full run of each kind for this prompt and index. None where absent.
    single and multiturn match on the index_version the run recorded, never on the
    filename (bug, 2026-11-07); runs from before 2026-11-09 recorded none and do not
    count. e2e and e2e_multiturn match on prompt and time; main() checks their index.
    An e2e line whose file is gone or changed, or whose answers miss any of `e2e_ids`
    (the record ids of run_e2e.py's full set), is passed over, its reason appended to
    `skipped[kind]`; an e2e_multiturn line likewise for `mt_ids` (the multiturn case
    ids)."""
    evals = {}
    for kind in ("single", "multiturn"):
        hits = [r for r in rows if r.get("runner") == "run_eval" and r.get("kind") == kind
                and r.get("scope") == "full" and r.get("prompt_version") == prompt
                # Arm B is the site's path; arm A truncates chunks at 500 chars.
                and r.get("arm") in ("B", None)
                and r.get("index_version") == index]
        evals[kind] = hits[-1] if hits else None
    for kind in E2E_KINDS:
        evals[kind] = None
        want = (e2e_ids if kind == "e2e" else mt_ids) or ()
        for r in reversed(rows):
            if r.get("runner") != "run_e2e" or r.get("kind") != kind or r.get("scope") != "full":
                continue
            data, why = load_run(r)
            if why:
                if skipped is not None:
                    skipped.setdefault(kind, []).append(why)
                continue
            stamp = data.get("meta", {}).get("stamp", {})
            ts = stamp.get("timestamp")
            if not (stamp.get("source_system") == "dev" and stamp.get("site_prompt_version") == prompt
                    and ts and (since is None or datetime.fromisoformat(ts) >= since)):
                continue
            missing = sorted(set(want) - {x.get("record_id") for x in data["results"]})
            if missing:
                if skipped is not None:
                    skipped.setdefault(kind, []).append(f"{r['source']} (run {r['run_id']}) does not cover today's full "
                                   f"{kind} set: no {', '.join(missing)}")
                continue
            evals[kind] = r
            break
    return evals


def clean(rec):
    """No check False in a record: its own, any turn's, or the conversation's."""
    groups = [rec.get("checks"), rec.get("case_checks")] + [t.get("checks") for t in rec.get("turns") or []]
    return not any(v is False for g in groups for k, v in (g or {}).items() if not k.startswith("_"))


def failures(data):
    """(case, check) -> [failed, total, questions failed] from a run's results, as
    load_run() returns them."""
    records = data["results"] if isinstance(data, dict) else data
    out = {}

    def add(case, checks, question=None):
        for k, v in (checks or {}).items():
            if k.startswith("_") or not isinstance(v, bool):
                continue
            n = out.setdefault((case, k), [0, 0, set()])
            n[1] += 1
            n[0] += v is False
            if v is False and question:
                n[2].add(question)

    for rec in records:
        case = rec.get("id") or rec.get("record_id")
        for t in rec.get("turns") or []:
            add(case, t.get("checks"), t.get("question"))
        add(case, rec.get("case_checks"))
        add(case, rec.get("checks"), rec.get("question"))
    return {k: v for k, v in out.items() if v[0]}


def no_verdict(data, tier1):
    """check -> [answers without a verdict, answers carrying the check, where] for
    each tier-1 check that holds neither True nor False on some answer of a run, as
    load_run() returns it. `tier1` is the tier-1 ids from eval_set.json ("T1.7").
    None, "n/a" and "unavailable_*" are no verdict: a must-pass check that did not
    run is not a pass (2026-11-18: T1.7 scored None or n/a on every answer of four
    gate runs, R-20261118-02, and the gate counted only False). A list or dict under
    a tier-1 key is a check's detail (T1.6_voice_spans), not its verdict. Tier-2
    checks are not read: n/a there means the check does not apply."""
    records = data["results"] if isinstance(data, dict) else data
    prefixes = tuple(f"{k}_" for k in tier1)
    out = {}

    def add(where, checks):
        for k, v in (checks or {}).items():
            if not k.startswith(prefixes) or isinstance(v, (list, dict)):
                continue
            n = out.setdefault(k, [0, 0, []])
            n[1] += 1
            if not isinstance(v, bool):
                n[0] += 1
                n[2].append(where)

    for rec in records:
        case = rec.get("id") or rec.get("record_id")
        for t in rec.get("turns") or []:
            add(f"{case} t{t.get('turn')}", t.get("checks"))
        add(f"{case} (conversation)", rec.get("case_checks"))
        add(case, rec.get("checks"))
    return {k: v for k, v in out.items() if v[0]}


def no_verdict_problems(kind, run_id, data, tier1):
    """One gate problem per tier-1 check without a verdict in a run. Cases are named
    when only some answers lack it, the first 10 of them."""
    return [f"{kind} run {run_id}: tier-1 check {check} has no verdict (neither true nor "
            f"false) on {n} of {total} answers"
            + ("" if n == total else ": " + ", ".join(where[:10])
               + (f" and {n - 10} more" if n > 10 else ""))
            + "; a must-pass check that did not run blocks the release"
            for check, (n, total, where) in sorted(no_verdict(data, tier1).items())]


# ── rates (2026-11-26): report only ───────────────────────────────────────────
# Nothing here adds a problem, clears one or reads a sign-off. Whether gate runs
# default to repetitions, and how a tier-1 check treats a failure that happens at a
# rate, are Jord's decisions, not this code's.

RATE_DAYS = 30  # default period of the no-failure block, back from today


def _range(passed, n):
    lo, hi = B.wilson(passed, n)
    return {"passed": passed, "of": n,
            "lo": None if lo is None else round(lo, 4), "hi": None if hi is None else round(hi, 4)}


def case_rates(data):
    """case -> {"passed", "of", "lo", "hi", "checks"} for a run, as load_run()
    returns it. A record is one repetition of its case (one conversation, for
    multi-turn) and passes when no check on it is False, as clean() reads it; "of"
    counts the repetitions with at least one verdict. "checks" gives every check's
    passed and of over the case's answers, the counts failures() takes, and on
    multi-turn runs per turn as well ("turns": {"2": [passed, of]}). lo and hi are
    the 95% Wilson range."""
    records = data["results"] if isinstance(data, dict) else data
    tally = {}
    for rec in records:
        c = tally.setdefault(rec.get("id") or rec.get("record_id"), [0, 0, {}])
        groups = [(t.get("turn"), t.get("checks")) for t in rec.get("turns") or []]
        groups += [(None, rec.get("case_checks")), (None, rec.get("checks"))]
        seen = False
        for turn, checks in groups:
            for k, v in (checks or {}).items():
                if k.startswith("_") or not isinstance(v, bool):
                    continue
                seen = True
                n = c[2].setdefault(k, [0, 0, {}])
                n[0] += v
                n[1] += 1
                if turn is not None:
                    t = n[2].setdefault(turn, [0, 0])
                    t[0] += v
                    t[1] += 1
        if seen:
            c[0] += clean(rec)
            c[1] += 1
    return {case: {**_range(p, n), "checks": {
        k: {**_range(cp, cn), **({"turns": {str(t): v for t, v in sorted(turns.items())}} if turns else {})}
        for k, (cp, cn, turns) in sorted(checks.items())}}
        for case, (p, n, checks) in tally.items()}


def no_failure(rows, since, tier1):
    """The checks that did not fail once in the history lines dated `since`
    (YYYY-MM-DD) or later, each with the answers it ran on and the 95% Wilson upper
    bound on its failure rate. Counts come from each line's "pass" ({check: [passed,
    applicable]}); retrieval and judge-agreement lines have none and are skipped.
    Lines are runs with distinct answers (run_id), so no answer counts twice unless
    two results files hold the same answer. Tier-1 checks with no verdict on any
    answer in the period are listed apart: a check that never ran has not passed."""
    lines = [h for h in rows if str(h.get("date") or "") >= since and isinstance(h.get("pass"), dict)]
    agg = {}
    for h in lines:
        for k, (p, n) in h["pass"].items():
            a = agg.setdefault(k, [0, 0])
            a[0] += n - p
            a[1] += n
    return {"since": since, "runs": len(lines), "answers": sum(h.get("answers") or 0 for h in lines),
            "checks": [{"check": k, "of": n, "failure_rate_up_to": round(B.wilson(0, n)[1], 4)}
                       for k, (f, n) in sorted(agg.items()) if f == 0 and n],
            "tier1_no_verdict": [t for t in tier1 if not any(k.startswith(f"{t}_") for k in agg)]}


def rates_evidence(runs, invariants):
    """The evidence file's "rates": per kind, every case's rate over its repetitions
    and only the checks that failed on it (the others passed every answer), then
    the no-failure block. `runs` is kind -> case_rates()."""
    return {"range": "95% Wilson",
            "cases": {kind: [{"case": case, **{k: v for k, v in r.items() if k != "checks"},
                              "failing_checks": {k: c for k, c in r["checks"].items()
                                                 if c["passed"] < c["of"]}}
                             for case, r in cases.items()]
                      for kind, cases in runs.items()},
            "no_failure": invariants}


def _pc(v):
    return "n/a" if v is None else f"{v:.0%}"


def _span(r):
    return f"({_pc(r['lo'])} to {_pc(r['hi'])})"


def _rate(r):
    return f"{r['passed']}/{r['of']}  {_pc(r['passed'] / r['of'] if r['of'] else None)}  {_span(r)}"


def print_rates(name, cases, every_check=False, out=print):
    """Cases with a failure in full; the rest in one line per repetition count, or
    in full with every_check (for a rate run of a few cases)."""
    out(f"\n  {name}: {len(cases)} case(s). Rate = repetitions passing every check, 95% Wilson range")
    shown = {c: r for c, r in cases.items() if every_check or r["passed"] < r["of"]}
    for case, r in shown.items():
        out(f"    {case:<8} {_rate(r)}")
        for k, c in r["checks"].items():
            if not every_check and c["passed"] == c["of"]:
                continue
            turns = ", ".join(f"t{t} {p}/{n}" + ("" if p == n else f" {_span(_range(p, n))}")
                              for t, (p, n) in c.get("turns", {}).items())
            out(f"        {k:<32} {_rate(c)}" + (f"   {turns}" if turns else ""))
    rest = {}
    for case, r in cases.items():
        if case not in shown:
            rest.setdefault(r["of"], []).append(case)
    for n, ids in sorted(rest.items()):
        out(f"    {len(ids)} other case(s) with {n} repetition(s) each passed every check: "
            f"{_rate(_range(n, n))} each")


def print_no_failure(nf, out=print):
    out(f"\n  checks with no failure since {nf['since']}: {nf['runs']} history line(s), "
        f"{nf['answers']:,} answers. Kept apart from the rates: a check that has not failed "
        "is an invariant, not evidence of quality")
    for c in nf["checks"]:
        out(f"    {c['check']:<32} 0 of {c['of']:,}  (failure rate up to {c['failure_rate_up_to']:.1%})")
    if nf["tier1_no_verdict"]:
        out(f"    tier 1 with no verdict on any answer in the period (never ran, not never failed): "
            f"{', '.join(nf['tier1_no_verdict'])}")


def period_start(since):
    """The no-failure block's first day: --since, or RATE_DAYS back from today."""
    return (since or date.today() - timedelta(days=RATE_DAYS)).isoformat()


def resolve(sha):
    """(full sha, commit time) in the web repo, or (None, None)."""
    try:
        out = subprocess.run(["git", "-C", str(WEB), "show", "-s", "--format=%H %cI", sha],
                             capture_output=True, text=True, check=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return out[0], datetime.fromisoformat(out[1]).astimezone(UTC)


def git_out(repo, *args):
    """stdout of a git command in `repo`, or None when it fails."""
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                           encoding="utf-8", check=False)
    except FileNotFoundError:
        return None
    return r.stdout if r.returncode == 0 else None


def unreviewed(text):
    """Results-log ids whose entry still says `- reviewed: no`."""
    out, cur = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = None
        m = re.match(r"### (R-\d{8}-\d+)\b", line)
        if m:
            cur = m.group(1)
        elif cur and re.match(r"- reviewed:\s*no\b", line):
            out.append(cur)
    return out


def doc_entries(old_docs, last_commit):
    """One entry per DOCS item. `last_commit(repo, path)` is the newest commit on
    origin/master that touched the doc, or None. Jord's approval and note carry over
    only while that commit is the one she approved; a newer commit resets it."""
    prev = {(d.get("repo"), d.get("path")): d for d in old_docs}
    out = []
    for repo, path in DOCS:
        p = prev.get((repo, path), {})
        commit = last_commit(repo, path)
        e = {"repo": repo, "path": path, "commit": commit,
             "approved": commit is not None and p.get("commit") == commit
             and p.get("approved") is True}
        if p.get("note"):
            e["note"] = p["note"]
        out.append(e)
    return out


def carry_signoffs(old_fails, fails):
    """Set each failure's "signoff" from the last evidence file. A ruling holds for
    the run and count it was given on (2026-11-08): on a new run_id, or a different
    failed/of, it is cleared and kept only as "previous_signoff", which no check
    reads, so Jord rules on the new failure herself."""
    prev = {(p.get("kind"), p.get("case"), p.get("check")): p for p in old_fails}
    for f in fails:
        f["signoff"] = None
        p = prev.get((f["kind"], f["case"], f["check"]))
        if p is None:
            continue
        if all(p.get(x) == f[x] for x in ("run_id", "failed", "of")):
            f["signoff"] = p.get("signoff")
            if p.get("previous_signoff"):
                f["previous_signoff"] = p["previous_signoff"]
        elif (p.get("signoff") or "").strip():
            f["previous_signoff"] = {x: p.get(x) for x in ("signoff", "run_id", "failed", "of")}
        elif p.get("previous_signoff"):
            f["previous_signoff"] = p["previous_signoff"]
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--image", choices=["jordbot-web", "jordbot"])
    ap.add_argument("--tag", help="vX.Y the release PR sets")
    ap.add_argument("--sha", help="the commit tested in dev and built as the tag")
    ap.add_argument("--faiss-dir", help="/faiss/kb-<commit>, for a FAISS_DIR promotion")
    ap.add_argument("--prompt", help="prompt_version prod will serve (required for the gate)")
    ap.add_argument("--index", help="index_version /search reports for the index going out (8 hex chars; "
                    "required for the gate)")
    ap.add_argument("--out-dir", default=str(WEB / "releases"))
    ap.add_argument("--since", type=date.fromisoformat,
                    help=f"first day (YYYY-MM-DD) of the no-failure block; default {RATE_DAYS} days ago")
    ap.add_argument("--rates", nargs="+", type=Path, metavar="FILE",
                    help="print the rate report for these results files and stop: no gate, no evidence file")
    args = ap.parse_args()
    if args.rates:
        rows = RM.read_history(HISTORY)
        tier1 = list(json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))["checks"]["tier1"])
        print("rates, report only (release_gate.py --rates)")
        for f in args.rates:
            print_rates(f.name, case_rates(json.loads(f.read_text(encoding="utf-8"))), every_check=True)
        print_no_failure(no_failure(rows, period_start(args.since), tier1))
        return
    if not (args.prompt and args.index):
        ap.error("the following arguments are required: --prompt, --index")
    if bool(args.image) == bool(args.faiss_dir):
        ap.error("give --image with --tag and --sha, or --faiss-dir")
    if args.image and not (args.tag and args.sha):
        ap.error("--image needs --tag and --sha")
    if args.tag and not re.fullmatch(r"v\d+\.\d+(\.\d+)?", args.tag):
        ap.error("--tag must be vX.Y")
    # Git Bash rewrites a leading /faiss/ into C:/Program Files/Git/faiss/ when it
    # starts a Windows python (2026-11-07: CI caught it in the kb-762b34e evidence).
    # Set MSYS_NO_PATHCONV=1 there.
    if args.faiss_dir and not re.fullmatch(r"/faiss/kb-[0-9a-f]{7,40}", args.faiss_dir):
        ap.error(f"--faiss-dir must look like /faiss/kb-<commit>, got {args.faiss_dir!r} "
                 "(Git Bash: set MSYS_NO_PATHCONV=1)")

    name = (f"{args.image}-{args.tag}" if args.image
            else "faiss-" + args.faiss_dir.rstrip("/").rsplit("/", 1)[-1])
    out = Path(args.out_dir) / f"{name}.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    ticked = {q["item"]: q.get("done") for q in old.get("qa", [])}
    noted = {q["item"]: q.get("note") for q in old.get("qa", []) if q.get("note")}

    since = None
    if args.image == "jordbot-web":
        args.sha, since = resolve(args.sha) if args.sha else (None, None)
        if since is None:
            sys.exit(f"not a commit in {WEB}; fetch it first")

    rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    skipped = {}  # kind -> why each e2e line of that kind was passed over
    # Until 2026-11-09 the e2e run the gate took asked the 34 questions only, so no
    # timeline case ran end to end before a release (task_1794247869717_m72eoh). A
    # run counts only if it holds every case the full set holds now.
    # 2026-11-09: and the multi-turn e2e run must hold every multiturn case, M10
    # and M11 included.
    eval_set = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
    e2e_ids = [c["record_id"] for c in E2E.expand_questions(E2E.full_cases(eval_set))]
    mt_ids = [c["id"] for c in E2E.multiturn_cases(eval_set)]
    evals = find_runs(rows, args.prompt, args.index, since, skipped, e2e_ids, mt_ids)
    loaded = {}  # kind -> (results, why)
    for kind, row in evals.items():
        if row is not None:
            loaded[kind] = load_run(row)

    # Cases, and question texts, the e2e run passed outright: what the site served
    # for them was clean. By text too, because the site's fixed replies key on the
    # visitor's words: M04 turn 3 sends A01's text and gets A01's fixed reply, but
    # the e2e set has no M04 (single turns only).
    # 2026-11-08: the e2e_multiturn run now carries M04 itself; the text match stays
    # for single-turn questions asked inside a conversation elsewhere.
    e2e_clean, e2e_clean_q = set(), set()
    for kind in E2E_KINDS:
        data = loaded.get(kind, (None,))[0]
        for rec in (data or {}).get("results", []):
            if clean(rec):
                e2e_clean.add(rec.get("id") or rec.get("record_id"))
                if rec.get("question"):
                    e2e_clean_q.add(rec["question"].strip())

    problems, fails, index_read = [], [], {}
    for kind, row in evals.items():
        if row is None:
            problems.append(f"no full {kind} run for prompt {args.prompt}"
                            + (f", index_version {args.index}" if kind not in E2E_KINDS else
                               " served by jordbot-dev" + (f" after {since:%Y-%m-%d %H:%M}Z" if since else "")
                               + "".join(f"; passed over: {w}" for w in skipped.get(kind, []))))
            continue
        data, why = loaded[kind]
        if why:
            problems.append(f"{kind} run {row['run_id']}: {why}")
            continue
        # Read from the answers themselves, not the history line. An e2e run has
        # it only from a site that sends data-index (bug_1794186971153_gec1c5); the
        # gate blocks any e2e run that cannot prove its index (Jord, 2026-11-08).
        # A multi-turn e2e run is read turn by turn: one turn with none, or on
        # another index, blocks it (2026-11-09).
        index_read[kind] = RM.index_version_of(data)
        if index_read[kind] != args.index:
            problems.append(f"{kind} run {row['run_id']}: answered from index_version "
                            f"{index_read[kind]}, not {args.index}"
                            + (" (a run from before the site sent data-index, or one with a sitting"
                               + (" or turn" if kind == "e2e_multiturn" else "") + " on another index)"
                               if kind in E2E_KINDS else ""))
        # Each failure keeps its own run's kind, so the web repo's CI check finds
        # its run_id under evals[kind] (2026-11-09: listing the multi-turn e2e run's
        # failures as "e2e" failed that check). A failure the site itself served is
        # never "passed end to end", so "not served" cannot clear it.
        for (case, check), (n, total, qs) in sorted(failures(data).items()):
            passed = kind not in E2E_KINDS and (
                case in e2e_clean or (bool(qs) and all(q.strip() in e2e_clean_q for q in qs)))
            fails.append({"kind": kind, "case": case, "check": check, "failed": n, "of": total,
                          "run_id": row["run_id"], "e2e_case_passed": passed})
        # 2026-11-18: a tier-1 check with no verdict is a problem no sign-off clears.
        problems += no_verdict_problems(kind, row["run_id"], data, eval_set["checks"]["tier1"])
    carry_signoffs(old.get("failures", []), fails)
    unsigned = [f for f in fails if not (f["signoff"] or "").strip()]
    # README: "a release needs all [tier-1 checks] to pass". A tier-1 failure can
    # be signed off only as a checker miss, or (Jord, 2026-11-06) as "not served:"
    # text the site never sends, which counts only when the e2e run passed that
    # case on dev. First use: A01, whose model refusal the site replaces with
    # FIXED_REFUSAL (web #85).
    # 2026-11-17 (Jord): a known defect is accepted as "known: <why>" when the
    # sign-off names the results-log entry that measured its rate. The a94aeead gate
    # runs failed A02 T1.6 ("feel free to ask") and M11 t3 T1.8 ("likely contributed"),
    # draws from nonzero rates (R-20261116-09, R-20261117-06) that no ruling covered.
    t1 = tuple(f"{k}_" for k in eval_set["checks"]["tier1"])

    def tier1_ok(f):
        so = f["signoff"].strip().lower()
        return so.startswith("checker miss") or (
            so.startswith("not served") and f["kind"] not in E2E_KINDS and f["e2e_case_passed"]) or (
            so.startswith("known:") and bool(KNOWN_RATE_RE.search(f["signoff"])))

    tier1 = [f for f in fails if f["check"].startswith(t1) and (f["signoff"] or "").strip()
             and not tier1_ok(f)]
    # "note": what Jord found doing the item, kept across re-runs. A tick with a
    # note is checked-with-findings, not a clean pass.
    qa = [{"item": i, "done": ticked.get(i) is True, **({"note": noted[i]} if i in noted else {})}
          for i in QA]
    undone = [q for q in qa if not q["done"]]
    if unsigned:
        problems.append(f"{len(unsigned)} failing check(s) without Jord's sign-off")
    if tier1:
        problems.append(f"{len(tier1)} tier-1 failure(s) not signed off as 'checker miss: ...', "
                        "as 'not served: ...' with the e2e run passing that case, or as "
                        "'known: ...' naming the results-log entry (R-YYYYMMDD-NN) that measured "
                        "its rate; a tier-1 failure blocks the release")
    if undone:
        problems.append(f"{len(undone)} QA item(s) not ticked")

    # Docs: judged as they stand on origin/master, so fetch both repos first.
    repos = {"ai": HERE, "web": WEB}
    for repo in repos.values():
        if git_out(repo, "fetch", "-q", "origin", "master") is None:
            problems.append(f"could not fetch origin/master in {repo}; docs not checked against it")

    def last_commit(key, path):
        sha = (git_out(repos[key], "log", "-1", "--format=%H", "origin/master", "--", path) or "").strip()
        return sha or None

    docs = doc_entries(old.get("docs", []), last_commit)
    for d in docs:
        if d["commit"] is None:
            problems.append(f"{d['repo']}:{d['path']} not found on origin/master")
    unapproved = [d for d in docs if d["commit"] and not d["approved"]]
    if unapproved:
        problems.append(f"{len(unapproved)} doc(s) not approved by Jord at their master commit: "
                        + ", ".join(f"{d['repo']}:{d['path']}" for d in unapproved))

    import coverage_report  # only the gate needs it
    on_master = git_out(HERE, "show", "origin/master:COVERAGE.md")
    spec_text = git_out(HERE, "show", "origin/master:eval_set.json")
    coverage_ok = False
    if on_master is not None and spec_text is not None:
        spec = json.loads(spec_text)
        cases = [c for g in coverage_report.CASE_GROUPS for c in spec.get(g, [])]
        want = coverage_report.markdown(spec, cases, spec.get("multiturn", {}).get("cases", []))
        coverage_ok = on_master.replace("\r\n", "\n") == want
    if not coverage_ok:
        problems.append("COVERAGE.md on origin/master is not what coverage_report.py generates "
                        "from eval_set.json; run it with --write and merge the result")

    pending = unreviewed(RESULTS_LOG.read_text(encoding="utf-8")) if RESULTS_LOG.exists() else None
    if pending is None:
        problems.append(f"results log not found at {RESULTS_LOG}")
    elif pending:
        problems.append(f"{len(pending)} results-log entr{'y' if len(pending) == 1 else 'ies'} "
                        "still reviewed: no")

    # 2026-11-26: rates, a report only. Taken after every problem is counted, and a
    # fault in it is printed and recorded, never a gate result.
    rate_lines = ["", "  rates, report only: no rate passes or fails the gate"]
    try:
        run_rates = {kind: case_rates(data) for kind, (data, _why) in loaded.items() if data is not None}
        invariants = no_failure(rows, period_start(args.since), list(eval_set["checks"]["tier1"]))
        rates = rates_evidence(run_rates, invariants)
        for kind, cases in run_rates.items():
            print_rates(kind, cases, out=rate_lines.append)
        print_no_failure(invariants, out=rate_lines.append)
    except Exception as e:  # noqa: BLE001  a report fault must not decide the gate
        rates = {"error": f"{type(e).__name__}: {e}"}
        rate_lines = ["", f"  rates not reported (no effect on the gate): {rates['error']}"]

    evidence = {
        "generated_by": "jordannedyck-ai/release_gate.py",
        "generated": date.today().isoformat(),
        "image": args.image, "tag": args.tag, "sha": args.sha, "faiss_dir": args.faiss_dir,
        "prompt_version": args.prompt, "index": args.index,
        "evals": {k: None if r is None else {
            **{x: r.get(x) for x in ("run_id", "date", "source", "answers", "conversations",
                                     "all_checks_pass_rate")},
            "index_version": index_read.get(k)} for k, r in evals.items()},
        "failures": fails,
        "rates": rates,
        "qa": qa,
        "docs": docs,
        "coverage_generated": coverage_ok,
        "results_log": {"unreviewed": pending},
        "problems": problems,
        "status": "fail" if problems else "pass",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for kind, r in evals.items():
        print(f"{kind:13} " + (f"{r['run_id']}  {r['source']}  all-checks {r['all_checks_pass_rate']}  "
                              f"index_version {index_read.get(kind)} (want {args.index})"
                              if r else "MISSING"))
    for f in fails:
        was = f.get("previous_signoff")
        state = ("ok " if (f["signoff"] or "").strip()
                 else "NEEDS NEW RULING" if was else "UNSIGNED")
        print(f"  {state} {f['kind']:9} {f['case']:6} {f['check']:32} {f['failed']}/{f['of']}")
        if was and not (f["signoff"] or "").strip():
            print(f"      was: run {was['run_id']} {was['failed']}/{was['of']}, {was['signoff']!r}")
    print("\n".join(rate_lines) + "\n")
    for q in undone:
        print(f"  QA not ticked: {q['item']}")
    for d in docs:
        print(f"  doc {'approved' if d['approved'] else 'NOT APPROVED'} {d['repo']}:{d['path']} "
              f"{(d['commit'] or 'missing')[:7]}")
    if pending:
        print(f"  results log, reviewed: no: {', '.join(pending)}")
    print(f"\n{out}\nstatus: {evidence['status']}")
    for p in problems:
        print(f"  - {p}")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
