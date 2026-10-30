"""
Bootstrap confidence ranges on pass rates and on blind-sample means, and whether
a gap between two runs or models sits inside the noise (plan A.7, 2026-10-30).
Free and read-only: reads results files, calls no API, writes nothing.

The resampling unit is the question. A resample draws questions with
replacement and takes every answer each question has: its repeat runs, and
every turn of a multi-turn conversation. Repeats of one question share their
retrieval and their source content, so they pass or fail together; resampling
single answers would treat them as independent and make the ranges too narrow.

A gap is paired: both sides are resampled on the same questions, counting only
questions both sides answered. "Inside the noise" means the range of the gap
includes zero. With 8 questions (both blind samples) or 7 conversations
(multi-turn) a range is rough: it can show a gap is noise, not that a small gap
is real.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    uv run python bootstrap_ci.py eval_runs/FILE.json                  # ranges per model/arm
    uv run python bootstrap_ci.py FILE --models gpt-4o claude-opus-5-5 # gap, two models in one file
    uv run python bootstrap_ci.py FILE_A FILE_B                        # gap, run B minus run A
    uv run python bootstrap_ci.py --blind eval_runs/blind-sample-2026-10-22_SCORES.json
"""

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import run_metrics as RM

SEED = 20261030
RESAMPLES = 10_000
LEVEL = 0.95
ALL = "all checks pass"
ROUGH = 10  # fewer questions than this: the range is flagged as rough
DIMS = ("D1", "D2", "D3", "D4", "D5")


# ── cells: one (numerator, denominator) per question ──────────────────────────

def question_cells(rows, convs=(), key=lambda r: r["id"]):
    """{question: {check: [passed, applicable]}}, plus ALL: [answers passing
    every check, answers scored]. Counts come from run_metrics.pass_counts, so a
    check that does not apply to an answer is out of the denominator, as in the
    headline. key is the resampling unit; the question id unless a test says
    otherwise."""
    by, cv = {}, {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    for c in convs:
        cv.setdefault(c["id"], []).append(c)
    out = {}
    for q, rs in by.items():
        pc = RM.pass_counts(rs)
        pc.update(RM.pass_counts(cv.get(q, [])))
        scored = [r for r in rs if r["checks"]]
        if scored:
            pc[ALL] = [sum(all(r["checks"].values()) for r in scored), len(scored)]
        out[q] = pc
    return out


def for_check(cells, check):
    """{question: (passed, applicable)} for the questions the check applies to."""
    return {q: tuple(pc[check]) for q, pc in cells.items() if check in pc}


def blind_cells(scores):
    """{model: {"score": {question: (sum of D1-D5 means, items)}, "ship": {...}}}.
    The 10-16 file keys the question as "q" and ship as 0/1; the 10-22 file uses
    "qid" and true/false."""
    out = {}
    for s in scores:
        q = s.get("q") or s.get("qid")
        m = out.setdefault(s["model"], {"score": {}, "ship": {}})
        for name, v in (("score", sum(s[d] for d in DIMS) / len(DIMS)), ("ship", float(bool(s["ship"])))):
            c = m[name].setdefault(q, [0.0, 0])
            c[0] += v
            c[1] += 1
    return out


# ── the bootstrap ─────────────────────────────────────────────────────────────

def _ratio(cells):
    den = sum(c[1] for c in cells)
    return sum(c[0] for c in cells) / den if den else None


def _draws(n, resamples, seed):
    rng = random.Random(seed)
    for _ in range(resamples):
        yield [rng.randrange(n) for _ in range(n)]


def interval(values, level=LEVEL):
    """Percentile interval, using run_metrics.percentile."""
    tail = (1 - level) / 2 * 100
    return RM.percentile(values, tail), RM.percentile(values, 100 - tail)


def bootstrap(cells, *, resamples=RESAMPLES, seed=SEED, level=LEVEL):
    """Range on sum(numerators) / sum(denominators) over the questions in cells.
    For pass rates that is passed / applicable; for blind scores, the mean."""
    ids = sorted(cells)
    vals = [cells[q] for q in ids]
    stats = [_ratio([vals[j] for j in d]) for d in _draws(len(ids), resamples, seed)]
    lo, hi = interval(stats, level)
    return {"n": len(ids), "value": _ratio(vals), "lo": lo, "hi": hi}


def paired_gap(a, b, *, resamples=RESAMPLES, seed=SEED, level=LEVEL):
    """Range on B minus A, both resampled on the same questions, over the
    questions both have."""
    ids = sorted(set(a) & set(b))
    va, vb = [a[q] for q in ids], [b[q] for q in ids]
    stats = []
    for d in _draws(len(ids), resamples, seed):
        ra, rb = _ratio([va[j] for j in d]), _ratio([vb[j] for j in d])
        if ra is not None and rb is not None:
            stats.append(rb - ra)
    lo, hi = interval(stats, level)
    ra, rb = _ratio(va), _ratio(vb)
    gap = None if ra is None or rb is None else rb - ra
    return {"n": len(ids), "a": ra, "b": rb, "gap": gap, "lo": lo, "hi": hi,
            "verdict": verdict(lo, hi)}


def verdict(lo, hi):
    if lo is None:
        return "no shared questions"
    # Sums of fractional scores leave float dust (8.9e-16 where the gap is 0).
    lo, hi = round(lo, 9), round(hi, 9)
    if lo == hi == 0:
        return "no gap"
    return "inside the noise" if lo <= 0 <= hi else "outside the noise"


# ── reading runs ──────────────────────────────────────────────────────────────

def groups(data):
    """{(model, arm): (answer rows, conversation rows)} for one results file."""
    out = {}
    for r in RM.answer_rows(data):
        out.setdefault((r["model"], r["arm"]), ([], []))[0].append(r)
    convs = RM.conversation_rows(data)
    for (model, _arm), (_rows, cs) in out.items():
        cs.extend(c for c in convs if c["model"] == model)
    return out


def label(key):
    model, arm = key
    return " ".join(x for x in (model or "model not recorded", arm and f"arm {arm}") if x)


def pick(gs, sel, source):
    """The one group whose model or arm letter is sel."""
    hits = [k for k in gs if sel in k]
    if len(hits) != 1:
        sys.exit(f"{source}: '{sel}' matches {len(hits)} groups; groups are: "
                 + ", ".join(label(k) for k in gs))
    return hits[0]


def only(gs, source):
    if len(gs) != 1:
        sys.exit(f"{source} holds {len(gs)} groups ({', '.join(label(k) for k in gs)}); "
                 "choose with --models A B")
    return next(iter(gs))


# ── printing ──────────────────────────────────────────────────────────────────

def _pct(v):
    return "  n/a" if v is None else f"{v:5.0%}"


def _pts(v):
    return "   n/a" if v is None else f"{v * 100:+5.0f}"


def _rough(n):
    return "  rough" if n < ROUGH else ""


def no_spread(r):
    """Every question passed (or every one failed): resampling cannot move the
    rate, so the range collapses to a point. The rule of three gives the 95%
    bound the bootstrap cannot: with n questions and no failures, the true
    failure rate could still be up to 3/n."""
    if r["n"] and r["lo"] == r["hi"] == r["value"] in (0, 1):
        if r["value"] == 1:
            return f"  no spread; rule of three: could be as low as {max(0, 1 - 3 / r['n']):.0%}"
        return f"  no spread; rule of three: could be as high as {min(1, 3 / r['n']):.0%}"
    return ""


def setup_line(resamples, seed):
    return (f"95% ranges, {resamples:,} resamples by question, seed {seed}. "
            f"'rough' = fewer than {ROUGH} questions.")


def report_ranges(gs, *, resamples, seed, out=print):
    for key, (rows, convs) in gs.items():
        cells = question_cells(rows, convs)
        checks = sorted({c for pc in cells.values() for c in pc} - {ALL})
        out(f"\n{label(key)}: {len(rows)} answers from {len(cells)} questions")
        out(f"  {'check':<30} {'rate':>5}  {'range':<13} questions")
        for c in [ALL] + checks:
            r = bootstrap(for_check(cells, c), resamples=resamples, seed=seed)
            out(f"  {c:<30} {_pct(r['value'])}  {_pct(r['lo'])} – {_pct(r['hi']).strip():<5} "
                f"{r['n']:>3}{_rough(r['n'])}{no_spread(r)}")


def report_gap(a, b, name_a, name_b, *, resamples, seed, out=print):
    ca, cb = question_cells(*a), question_cells(*b)
    checks = sorted(({c for pc in ca.values() for c in pc} & {c for pc in cb.values() for c in pc}) - {ALL})
    out("\ngap = B minus A, in percentage points, on the questions both answered")
    out(f"  A: {name_a}\n  B: {name_b}")
    out(f"  {'check':<30} {'A':>5} {'B':>5} {'gap':>6}  {'range':<13} {'qs':>3}  verdict")
    for c in [ALL] + checks:
        g = paired_gap(for_check(ca, c), for_check(cb, c), resamples=resamples, seed=seed)
        out(f"  {c:<30} {_pct(g['a'])} {_pct(g['b'])} {_pts(g['gap'])}  "
            f"{_pts(g['lo']).strip():>4} to {_pts(g['hi']).strip():<5} {g['n']:>3}  {g['verdict']}"
            f"{_rough(g['n'])}")


def report_blind(scores, source, *, resamples, seed, out=print):
    bc = blind_cells(scores)
    out(f"\nblind sample {source}: {len(scores)} items, {len(bc)} models")
    res = {m: {k: bootstrap(c[k], resamples=resamples, seed=seed) for k in ("score", "ship")}
           for m, c in bc.items()}
    order = sorted(res, key=lambda m: -res[m]["score"]["value"])
    out(f"  {'model':<20} {'mean D1-D5':>10}  {'range':<11}  {'ship':>4}  {'range':<11} questions")
    for m in order:
        s, p = res[m]["score"], res[m]["ship"]
        out(f"  {m:<20} {s['value']:>10.2f}  {s['lo']:.2f} – {s['hi']:.2f}  {p['value']:>4.0%}  "
            f"{p['lo']:>3.0%} – {p['hi']:<4.0%}  {s['n']:>3}{_rough(s['n'])}{no_spread(p)}")
    pairs = list(combinations(order, 2))
    out(f"\n  gaps in mean D1-D5, higher minus lower, paired by question. {len(pairs)} gaps: at 95%,"
        "\n  about 1 in 20 lands outside the noise by chance alone")
    for hi_m, lo_m in pairs:
        g = paired_gap(bc[lo_m]["score"], bc[hi_m]["score"], resamples=resamples, seed=seed)
        out(f"  {hi_m:<18} − {lo_m:<18} {g['gap']:+.2f}  range {g['lo']:+.2f} to {g['hi']:+.2f}  "
            f"{g['verdict']}{_rough(g['n'])}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("files", nargs="*", type=Path, help="one or two results files from any runner")
    ap.add_argument("--models", nargs=2, metavar=("A", "B"),
                    help="model names or arm letters to compare (A from the first file, B from the "
                         "second when two are given)")
    ap.add_argument("--blind", type=Path, help="a blind-sample *_SCORES.json file")
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    kw = {"resamples": args.resamples, "seed": args.seed}
    if not args.blind and not args.files:
        sys.exit("give one or two results files, or --blind SCORES.json")
    print(setup_line(args.resamples, args.seed))
    if args.blind:
        report_blind(_load(args.blind), args.blind.name, **kw)
    if len(args.files) > 2:
        sys.exit("give at most two results files")
    if len(args.files) == 1 and not args.models:
        report_ranges(groups(_load(args.files[0])), **kw)
    elif args.files:
        fa, fb = args.files[0], args.files[-1]
        ga, gb = groups(_load(fa)), groups(_load(fb))
        if args.models:
            ka, kb = pick(ga, args.models[0], fa.name), pick(gb, args.models[1], fb.name)
        else:
            ka, kb = only(ga, fa.name), only(gb, fb.name)
        report_gap(ga[ka], gb[kb], f"{fa.name}, {label(ka)}", f"{fb.name}, {label(kb)}", **kw)


if __name__ == "__main__":
    main()
