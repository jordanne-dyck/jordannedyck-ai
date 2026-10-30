"""
How far the grounding judge agrees with Jord's blind scores (plan A.8,
2026-10-30). Free: reads the blind-sample KEY and SCORES files and the results
files the answers came from, and calls no API. Since G.2 it adds one line per
sample to eval_history/history.jsonl (numbers only; a repeat run adds nothing).

The two are not asked the same question. The judge (gpt-4o-mini,
citation_checks.JUDGE_MODEL) asks whether the retrieved chunks say each claim.
D1 asks whether every fact is true to her record. A true claim the chunks did
not carry is flagged by the judge and passed by Jord, so agreement cannot be
perfect and is not meant to be. D1 is the main comparison; ship is secondary,
because it also turns answers down for voice.

Judge score per answer: the share of its claims the judge marked "no". Lead-in
lines ("Her approach breaks down into four steps:") are not claims, but the
stored verdicts mark them "no"; they are left out, and the numbers with them
kept are printed too. Since 2026-10-30 the judge folds lead-ins into the claim
after them (citation_checks.fold_lead_ins), so verdicts made since then, and
--rejudged folded, have none to leave out. The filter stays for the verdicts
stored before that, which include both blind samples.

--rejudged reads the verdicts judge_check.py rejudge wrote for the same
answers instead of the stored ones. Usage as below, adding e.g.
    uv run python judge_agreement.py eval_runs/blind-sample-2026-10-22_KEY.json --rejudged folded

Measures, each with a 95% range by question (bootstrap_ci: 10,000 resamples,
seed 20261030; 8 questions, so every range is rough):
  - rank correlation (Spearman) of the judge score against D1, sign set so that
    positive means "the judge flags more where Jord scored lower";
  - the flag table against the rubric's D1 threshold (4.5): how many low-D1
    answers the judge flagged (caught), and how many acceptable ones it flagged
    anyway (false alarms); Cohen's kappa on the same table.
Each sample is reported on its own: the two were answered under different
prompts, so pooling them raw would mix a level shift into the agreement. The
combined line ranks answers within each sample first.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    uv run python judge_agreement.py                 # both blind samples
    uv run python judge_agreement.py eval_runs/blind-sample-2026-10-22_KEY.json
"""

import argparse
import json
import re
from pathlib import Path

import bootstrap_ci as B
import run_metrics as RM

HERE = Path(__file__).resolve().parent
KEYS = [HERE / "eval_runs" / "blind-sample-2026-10-16_KEY.json",
        HERE / "eval_runs" / "blind-sample-2026-10-22_KEY.json"]
# The 10-16 KEY predates the "results" field. Its source was confirmed by text
# on 2026-10-30: each sheet answer's word overlap with its own stored answer
# was at least 0.68, and with any other answer to the same question at most 0.59.
RESULTS_FALLBACK = {"blind-sample-2026-10-16_KEY.json": "pass1_combined_2026-10-16.json"}
D1_OK = 4.5  # eval-rubric-d1-d5.md: D1 threshold >= 4.5

# A short line ending in a colon, with no figures, introduces a list; it states
# nothing to check. A figure keeps it a claim ("grew email from 1M to 4M by:").
LEAD_IN_RE = re.compile(r"(#+\s*)?([-*]\s*)?[^.!?\d]{0,80}:[*\s]*")


def is_lead_in(text):
    return bool(LEAD_IN_RE.fullmatch(text.strip()))


# ── joining ───────────────────────────────────────────────────────────────────

def join(key, scores, results, sample):
    """One row per blind item: the human scores and the stored judge verdict of
    the answer the item showed, matched on (question, model, run)."""
    by_item = {s["item"]: s for s in scores}
    by_answer = {(r["id"], r["model"], r["run"]): r for r in results}
    rows = []
    for it in key["items"]:
        s = by_item[it["item"]]
        if (s.get("q") or s.get("qid")) != it["question_id"] or s["model"] != it["model"]:
            raise ValueError(f"{sample} item {it['item']}: SCORES and KEY disagree")
        r = by_answer.get((it["question_id"], it["model"], it["run"]))
        if r is None or not (r.get("judge") or {}).get("claims"):
            raise ValueError(f"{sample} item {it['item']}: no stored judge verdict")
        rows.append({"sample": sample, "item": it["item"], "q": it["question_id"],
                     "model": it["model"], "d1": s["D1"], "ship": bool(s["ship"]),
                     "note": s.get("notes", ""), "claims": r["judge"]["claims"]})
    return rows


def judge_score(row, keep_lead_ins=False):
    """Share of the answer's claims marked "no"; 0 when nothing is left to judge."""
    cl = [c for c in row["claims"] if keep_lead_ins or not is_lead_in(c["text"])]
    return sum(c["supported"] == "no" for c in cl) / len(cl) if cl else 0.0


# ── measures ──────────────────────────────────────────────────────────────────

def ranks(xs):
    """Average ranks, ties sharing the mean of their positions."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if not va or not vb:
        return None  # one side is constant: no ranking to compare
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def spearman(a, b):
    return _pearson(ranks(a), ranks(b)) if len(a) > 1 else None


def flag_table(judge, low):
    """(flagged and low, flagged and ok, clean and low, clean and ok)."""
    t = [0, 0, 0, 0]
    for f, lo in zip(judge, low):
        t[(0 if f else 2) + (0 if lo else 1)] += 1
    return tuple(t)


def kappa(t):
    a, b, c, d = t
    n = a + b + c + d
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return None if pe == 1 else ((a + d) / n - pe) / (1 - pe)


def _share(num, den):
    return num / den if den else None


def measures(rows, human, keep_lead_ins=False):
    """human: "d1" or "ship". Higher judge score means more claims marked "no",
    so the correlation is taken against the negated score."""
    if not rows:
        return {}
    js = [judge_score(r, keep_lead_ins) for r in rows]
    hv = [float(r[human]) for r in rows]
    low = [r["d1"] < D1_OK if human == "d1" else not r["ship"] for r in rows]
    t = flag_table([j > 0 for j in js], low)
    return {"rho": spearman([-j for j in js], hv), "table": t, "kappa": kappa(t),
            "caught": _share(t[0], t[0] + t[2]), "false_alarm": _share(t[1], t[1] + t[3])}


def within_sample_rho(rows, human, keep_lead_ins=False):
    """Spearman on ranks taken within each sample, as fractions of that sample's
    size, so a level shift between samples does not count as agreement."""
    js, hv = [], []
    for s in sorted({r["sample"] for r in rows}):
        rs = [r for r in rows if r["sample"] == s]
        n = len(rs)
        js += [x / n for x in ranks([-judge_score(r, keep_lead_ins) for r in rs])]
        hv += [x / n for x in ranks([float(r[human]) for r in rs])]
    return _pearson(js, hv) if len(js) > 1 else None


def by_question_range(rows, stat, *, resamples=B.RESAMPLES, seed=B.SEED):
    """95% range on stat(rows), resampling questions with every item they hold,
    via bootstrap_ci's draws and percentile interval."""
    qs = sorted({r["q"] for r in rows})
    by_q = {q: [r for r in rows if r["q"] == q] for q in qs}
    vals = []
    for d in B._draws(len(qs), resamples, seed):
        v = stat([r for j in d for r in by_q[qs[j]]])
        if v is not None:
            vals.append(v)
    lo, hi = B.interval(vals) if vals else (None, None)
    return {"n": len(qs), "value": stat(rows), "lo": lo, "hi": hi, "verdict": B.verdict(lo, hi)}


# ── printing ──────────────────────────────────────────────────────────────────

def _num(v, fmt="+.2f"):
    return "n/a" if v is None else format(v, fmt)


def _rng(r, fmt="+.2f"):
    return f"{_num(r['lo'], fmt)} to {_num(r['hi'], fmt)}"


def _zero_word(r):
    """bootstrap_ci.verdict, read for an agreement measure: does the range
    include no agreement at all?"""
    return {"inside the noise": "could be no link", "outside the noise": "above zero",
            "no gap": "no spread"}.get(r["verdict"], r["verdict"])


def report(rows, name, *, resamples, seed, out=print):
    kw = {"resamples": resamples, "seed": seed}
    nq = len({r["q"] for r in rows})
    out(f"\n{name}: {len(rows)} answers, {nq} questions{B._rough(nq)}")
    for human, what in (("d1", "D1"), ("ship", "ship")):
        m = measures(rows, human)
        rho = by_question_range(rows, lambda rs, h=human: measures(rs, h).get("rho"), **kw)
        kap = by_question_range(rows, lambda rs, h=human: measures(rs, h).get("kappa"), **kw)
        cau = by_question_range(rows, lambda rs, h=human: measures(rs, h).get("caught"), **kw)
        fal = by_question_range(rows, lambda rs, h=human: measures(rs, h).get("false_alarm"), **kw)
        low = f"D1 < {D1_OK}" if human == "d1" else "not shipped"
        a, b, c, d = m["table"]
        out(f"  vs {what}:")
        out(f"    rank correlation {_num(m['rho'])}  range {_rng(rho)}  {_zero_word(rho)}")
        out(f"    {low}: {a + c} answers; judge flagged {a} ({_num(m['caught'], '.0%')}, "
            f"range {_rng(cau, '.0%')})")
        out(f"    acceptable: {b + d} answers; judge flagged {b} anyway ({_num(m['false_alarm'], '.0%')}, "
            f"range {_rng(fal, '.0%')})")
        out(f"    kappa {_num(m['kappa'])}  range {_rng(kap)}  {_zero_word(kap)}")
    k = measures(rows, "d1", keep_lead_ins=True)
    a, b, c, d = k["table"]
    out(f"  with lead-in lines kept as claims, vs D1: rank correlation {_num(k['rho'])}, "
        f"caught {a} of {a + c}, false alarms {b} of {b + d}, kappa {_num(k['kappa'])}")


def report_combined(rows, *, resamples, seed, out=print):
    kw = {"resamples": resamples, "seed": seed}
    nq = len({r["q"] for r in rows})
    out(f"\nboth samples, ranked within each sample: {len(rows)} answers, {nq} questions{B._rough(nq)}")
    for human, what in (("d1", "D1"), ("ship", "ship")):
        r = by_question_range(rows, lambda rs, h=human: within_sample_rho(rs, h), **kw)
        out(f"  rank correlation vs {what} {_num(r['value'])}  range {_rng(r)}  {_zero_word(r)}")


def _short(t, n=160):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n - 1] + "…"


def report_disagreements(rows, out=print):
    missed = [r for r in rows if r["d1"] <= 3 and judge_score(r) == 0]
    out(f"\nmissed by the judge: D1 3 or lower, no claim marked \"no\" ({len(missed)})")
    for r in missed:
        out(f"  {r['sample']} item {r['item']} {r['q']} {r['model']}  D1 {r['d1']:g}")
        out(f"    Jord: {_short(r['note']) or '(no note)'}")
        for c in r["claims"]:
            if c["supported"] == "other":
                out(f"    judge 'other': {_short(c['text'], 100)}")
    alarms = [r for r in rows if r["d1"] == 5 and judge_score(r) > 0]
    out(f"\nflagged by the judge, D1 5 from Jord ({len(alarms)})")
    for r in alarms:
        nos = [c for c in r["claims"] if c["supported"] == "no" and not is_lead_in(c["text"])]
        out(f"  {r['sample']} item {r['item']} {r['q']} {r['model']}  {len(nos)} of "
            f"{len(r['claims'])} claims \"no\"")
        for c in nos[:2]:
            out(f"    claim: {_short(c['text'], 120)}")
            out(f"    judge: {_short(c['reason'], 120)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rejudged(results_rows, path):
    """The results rows with each judge verdict swapped for the one judge_check.py
    rejudge wrote for the same answer. Keys there are "id|model|run|turn"."""
    by_key = _load(path)
    return [dict(r, judge=by_key[f"{r['id']}|{r.get('model', '')}|{r.get('run', '')}|"]) for r in results_rows
            if f"{r['id']}|{r.get('model', '')}|{r.get('run', '')}|" in by_key]


def load_sample(key_path, setting=None):
    """setting "folded" or "unfolded" reads the verdicts judge_check.py rejudge wrote,
    instead of the ones stored on the day; a sample with none is skipped."""
    key_path = Path(key_path)
    key = _load(key_path)
    results = key.get("results") or RESULTS_FALLBACK[key_path.name]
    sample = key_path.name.removeprefix("blind-sample-").removesuffix("_KEY.json")
    rows = _load(key_path.parent / results)
    if setting:
        path = key_path.parent / f"rejudge_{Path(results).stem}_{setting}.json"
        if not path.exists():
            return None, results
        rows, results = rejudged(rows, path), path.name
    return join(key, _load(key_path.with_name(key_path.name.replace("_KEY", "_SCORES"))),
                rows, sample), results


# ── history (plan G.2) ────────────────────────────────────────────────────────

def _r3(v):
    return None if v is None else round(v, 3)


def history_run_id(rows, verdicts, resamples=B.RESAMPLES, seed=B.SEED):
    """Same scores, verdicts and bootstrap setup, same id. Cheap, so a repeat
    can be skipped before the ranges are computed."""
    ident = {"verdicts": verdicts or "stored", "resamples": resamples, "seed": seed,
             "rows": [[r["item"], r["q"], r["model"], r["d1"], r["ship"],
                       [[c["text"], c["supported"]] for c in r["claims"]]] for r in rows]}
    return RM.hashlib.sha256(RM.json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def history_line(key_path, verdicts=None, *, date, resamples=B.RESAMPLES, seed=B.SEED, rows=None, results=None):
    """One eval_history line for one blind sample and one set of verdicts:
    the measures report() prints, against D1 and against ship, with their
    ranges by question. Ids and numbers only. None when the sample has no
    verdicts in that setting."""
    if rows is None:
        rows, results = load_sample(key_path, verdicts)
        if rows is None:
            return None
    kw = {"resamples": resamples, "seed": seed}

    def block(human):
        m = measures(rows, human)
        out = {"table": list(m["table"])}
        for name in ("rho", "kappa", "caught", "false_alarm"):
            r = by_question_range(rows, lambda rs, h=human, n=name: measures(rs, h).get(n), **kw)
            out[name], out[f"{name}_range"] = _r3(m[name]), [_r3(r["lo"]), _r3(r["hi"])]
        return out

    ln = {"run_id": history_run_id(rows, verdicts, resamples, seed),
          "date": date, "runner": "judge_agreement", "kind": "judge_agreement",
          "label": f"judge vs blind {rows[0]['sample']}, {verdicts or 'stored'} verdicts",
          "source": f"{Path(key_path).name} + {results}", "model": None, "arm": None,
          "prompt_version": None, "scope": "full", "sample": rows[0]["sample"],
          "verdicts": verdicts or "stored", "answers": len(rows), "questions": len({r["q"] for r in rows}),
          "d1_threshold": D1_OK, "d1": block("d1"), "ship": block("ship"),
          "resamples": resamples, "seed": seed}
    ln["missing"] = [f"{h}.{n}" for h in ("d1", "ship") for n in ("rho", "kappa") if ln[h][n] is None]
    return ln


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("keys", nargs="*", type=Path, default=KEYS, help="blind-sample *_KEY.json files")
    ap.add_argument("--resamples", type=int, default=B.RESAMPLES)
    ap.add_argument("--seed", type=int, default=B.SEED)
    ap.add_argument("--rejudged", choices=["folded", "unfolded", "gap-wording-rejected"],
                    help="use judge_check.py rejudge verdicts (lead-ins folded, unfolded, or the "
                         "rejected gap wording), not the stored ones")
    args = ap.parse_args()
    kw = {"resamples": args.resamples, "seed": args.seed}
    print(B.setup_line(args.resamples, args.seed))
    print(f"Judge score: share of claims marked \"no\", lead-in lines left out. "
          f"Flagged: any claim \"no\". Low: D1 < {D1_OK}.")
    everything, lines = [], []
    for k in args.keys:
        rows, results = load_sample(k, args.rejudged)
        if rows is None:
            print(f"\n{k.name}: no re-judged verdicts ({results}), skipped")
            continue
        everything += rows
        report(rows, f"blind sample {rows[0]['sample']} (answers and verdicts from {results})", **kw)
        lines.append(history_line(k, args.rejudged, date=RM.today(), rows=rows, results=results, **kw))
    if len({r["sample"] for r in everything}) > 1:
        report_combined(everything, **kw)
    report_disagreements(everything)
    if lines:
        RM.record_lines(lines)


if __name__ == "__main__":
    main()
