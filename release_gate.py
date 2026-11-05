#!/usr/bin/env python
"""
Release gate (2026-11-05). Writes the evidence file a prod change needs before
CI in jordannedyck-ai-web lets its PR merge.

A prod change is one of: a jordbot-web tag, a jordbot tag, or prod's FAISS_DIR.
Each needs, for the prompt and index it will run with:
  - a full single-turn run_eval.py run          (history kind "single", scope "full")
  - a full multi-turn run_eval.py run           (kind "multiturn", scope "full")
  - a full run_e2e.py run against jordbot-dev   (kind "e2e", scope "full"), whose stamp
    says dev served that prompt, taken after the tested commit
  - every failing check signed off by Jord, by case, in the evidence file; a tier-1
    failure only as "checker miss: <why>", since a release needs every tier-1 check
  - every QA item ticked, in the evidence file

Pure: reads eval_history/history.jsonl and the results files it names. Calls no
API and spends nothing; the runs above are separate, paid, and need Jord's go.

Re-running keeps the sign-offs and ticks already in the file, so the loop is:
run the gate, read the failures it prints, write Jord's ruling into each
"signoff" (her words, not a summary) and tick each QA item she did, run it again.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    uv run python release_gate.py --image jordbot-web --tag v1.16 --sha <web commit> \
        --prompt dd326013 --index fc423314
    uv run python release_gate.py --faiss-dir /faiss/kb-762b34e --prompt dd326013 --index fc423314
Exit 0 only when the evidence file says "status": "pass".
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORY = HERE / "eval_history" / "history.jsonl"
WEB = HERE.parent / "jordannedyck-ai-web"

QA = [
    "Overview question and a past-jobs question on a phone: the timeline plays, scrolls and exits",
    "An answer that cites a project shows its project card, image legible",
    "Fixed replies: the career lead and the instructions refusal, asked once each",
    "One conversation of three follow-ups keeps context and cites every answer",
    "/dashboard/admin loads; /api/health reports ok, db and the expected prompt_version",
    "Five answers read end to end for voice, not just for a response",
]


def results_path(source):
    for p in (HERE / "eval_runs" / source, HERE / source):
        if p.exists():
            return p
    return None


def latest(rows, **want):
    hits = [r for r in rows if all(r.get(k) == v for k, v in want.items())]
    return hits[-1] if hits else None


def find_runs(rows, prompt, index, since):
    """Latest full run of each kind for this prompt and index. None where absent."""
    evals = {}
    for kind in ("single", "multiturn"):
        hits = [r for r in rows if r.get("runner") == "run_eval" and r.get("kind") == kind
                and r.get("scope") == "full" and r.get("prompt_version") == prompt
                and f"_{index}_" in (r.get("source") or "")]
        evals[kind] = hits[-1] if hits else None
    evals["e2e"] = None
    for r in reversed(rows):
        if r.get("runner") != "run_e2e" or r.get("scope") != "full":
            continue
        p = results_path(r["source"])
        if not p:
            continue
        stamp = json.loads(p.read_text(encoding="utf-8")).get("meta", {}).get("stamp", {})
        ts = stamp.get("timestamp")
        if (stamp.get("source_system") == "dev" and stamp.get("site_prompt_version") == prompt
                and ts and (since is None or datetime.fromisoformat(ts) >= since)):
            evals["e2e"] = r
            break
    return evals


def failures(kind, row):
    """(case, check) -> [failed, total] from the results file the history row names."""
    data = json.loads(results_path(row["source"]).read_text(encoding="utf-8"))
    records = data["results"] if isinstance(data, dict) else data
    out = {}

    def add(case, checks):
        for k, v in (checks or {}).items():
            if k.startswith("_") or not isinstance(v, bool):
                continue
            n = out.setdefault((case, k), [0, 0])
            n[1] += 1
            n[0] += v is False

    for rec in records:
        case = rec.get("id") or rec.get("record_id")
        for t in rec.get("turns") or []:
            add(case, t.get("checks"))
        add(case, rec.get("case_checks"))
        add(case, rec.get("checks"))
    return {k: v for k, v in out.items() if v[0]}


def resolve(sha):
    """(full sha, commit time) in the web repo, or (None, None)."""
    try:
        out = subprocess.run(["git", "-C", str(WEB), "show", "-s", "--format=%H %cI", sha],
                             capture_output=True, text=True, check=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return out[0], datetime.fromisoformat(out[1]).astimezone(UTC)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--image", choices=["jordbot-web", "jordbot"])
    ap.add_argument("--tag", help="vX.Y the release PR sets")
    ap.add_argument("--sha", help="the commit tested in dev and built as the tag")
    ap.add_argument("--faiss-dir", help="/faiss/kb-<commit>, for a FAISS_DIR promotion")
    ap.add_argument("--prompt", required=True, help="prompt_version prod will serve")
    ap.add_argument("--index", required=True, help="index id the evals ran on, e.g. fc423314")
    ap.add_argument("--out-dir", default=str(WEB / "releases"))
    args = ap.parse_args()
    if bool(args.image) == bool(args.faiss_dir):
        ap.error("give --image with --tag and --sha, or --faiss-dir")
    if args.image and not (args.tag and args.sha):
        ap.error("--image needs --tag and --sha")
    if args.tag and not re.fullmatch(r"v\d+\.\d+(\.\d+)?", args.tag):
        ap.error("--tag must be vX.Y")

    name = (f"{args.image}-{args.tag}" if args.image
            else "faiss-" + args.faiss_dir.rstrip("/").rsplit("/", 1)[-1])
    out = Path(args.out_dir) / f"{name}.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    signed = {f"{f['kind']}|{f['case']}|{f['check']}": f.get("signoff")
              for f in old.get("failures", [])}
    ticked = {q["item"]: q.get("done") for q in old.get("qa", [])}

    since = None
    if args.image == "jordbot-web":
        args.sha, since = resolve(args.sha) if args.sha else (None, None)
        if since is None:
            sys.exit(f"not a commit in {WEB}; fetch it first")

    rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    evals = find_runs(rows, args.prompt, args.index, since)

    problems, fails = [], []
    for kind, row in evals.items():
        if row is None:
            problems.append(f"no full {kind} run for prompt {args.prompt}"
                            + (f", index {args.index}" if kind != "e2e" else
                               " served by jordbot-dev" + (f" after {since:%Y-%m-%d %H:%M}Z" if since else "")))
            continue
        for (case, check), (n, total) in sorted(failures(kind, row).items()):
            key = f"{kind}|{case}|{check}"
            fails.append({"kind": kind, "case": case, "check": check, "failed": n, "of": total,
                          "run_id": row["run_id"], "signoff": signed.get(key)})
    unsigned = [f for f in fails if not (f["signoff"] or "").strip()]
    # README: "a release needs all [tier-1 checks] to pass". A tier-1 failure can
    # only be signed off as a checker miss, never accepted as a known defect.
    t1 = tuple(f"{k}_" for k in json.loads((HERE / "eval_set.json").read_text(
        encoding="utf-8"))["checks"]["tier1"])
    tier1 = [f for f in fails if f["check"].startswith(t1) and (f["signoff"] or "").strip()
             and not f["signoff"].strip().lower().startswith("checker miss")]
    qa = [{"item": i, "done": ticked.get(i) is True} for i in QA]
    undone = [q for q in qa if not q["done"]]
    if unsigned:
        problems.append(f"{len(unsigned)} failing check(s) without Jord's sign-off")
    if tier1:
        problems.append(f"{len(tier1)} tier-1 failure(s) signed off as something other than "
                        "'checker miss: ...'; a tier-1 failure blocks the release")
    if undone:
        problems.append(f"{len(undone)} QA item(s) not ticked")

    evidence = {
        "generated_by": "jordannedyck-ai/release_gate.py",
        "generated": date.today().isoformat(),
        "image": args.image, "tag": args.tag, "sha": args.sha, "faiss_dir": args.faiss_dir,
        "prompt_version": args.prompt, "index": args.index,
        "evals": {k: None if r is None else {
            x: r.get(x) for x in ("run_id", "date", "source", "answers", "conversations",
                                  "all_checks_pass_rate")} for k, r in evals.items()},
        "failures": fails,
        "qa": qa,
        "problems": problems,
        "status": "fail" if problems else "pass",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for kind, r in evals.items():
        print(f"{kind:9} " + (f"{r['run_id']}  {r['source']}  all-checks {r['all_checks_pass_rate']}"
                              if r else "MISSING"))
    for f in fails:
        print(f"  {'ok ' if (f['signoff'] or '').strip() else 'UNSIGNED'} {f['kind']:9} {f['case']:6} "
              f"{f['check']:32} {f['failed']}/{f['of']}")
    for q in undone:
        print(f"  QA not ticked: {q['item']}")
    print(f"\n{out}\nstatus: {evidence['status']}")
    for p in problems:
        print(f"  - {p}")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
