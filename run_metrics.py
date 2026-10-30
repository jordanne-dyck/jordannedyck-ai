#!/usr/bin/env python
"""
Headline metrics for an eval run, and the committed results history
(plan A.6 and A.9, 2026-10-30; retrieval and judge-agreement lines, G.2).
Pure: reads results files, calls no API.

Reads every results shape the runners write:
  - run_eval.py and run_models.py, single-turn: a list of answer records;
  - the same two with --multiturn: a list of conversations with "turns";
  - run_e2e.py: {"meta": ..., "results": [...]};
  - retrieval checks and run_eval.py --probe files, scored as retrieval_scores.py
    scores them (kind "retrieval": recall@5, MRR@5, index, ranker).
judge_agreement.py writes its own lines (kind "judge_agreement") through
record_lines. History lines of every kind share run_id, date, runner, kind,
label, source, scope and missing.

Per answer it keeps latency, time to first token (TTFT), cost and every check
result, so A.7 (bootstrap ranges) can resample answers rather than aggregates.
A value a run did not record stays None. Nothing is estimated.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    uv run python run_metrics.py eval_results.json            # print the headline block
    uv run python run_metrics.py FILE --append --label NAME   # also add it to the history
    uv run python run_metrics.py CHECK.json [CHECK2.json] --append --label NAME   # a retrieval check
    uv run python run_metrics.py --backfill                   # history from eval_runs/
    uv run python run_metrics.py --chart                      # redraw eval_history/trend.svg
"""

import argparse
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORY = HERE / "eval_history" / "history.jsonl"
TREND = HERE / "eval_history" / "trend.svg"
EVAL_RUNS = HERE / "eval_runs"


# ── per-answer rows ───────────────────────────────────────────────────────────

def _checks(d):
    """The pass/fail results only. Skips "_" detail keys, None ("does not apply")
    and run_e2e.py's string markers such as "n/a" and "unavailable_no_context"."""
    return {k: v for k, v in (d or {}).items() if not k.startswith("_") and isinstance(v, bool)}


def _ms(v):
    return round(v / 1000, 3) if isinstance(v, (int, float)) else None


def answer_rows(data):
    """One row per generated answer: id, model, arm, run, turn, latency_s,
    ttft_s, cost_usd and checks. Turn is 1 for single-turn answers.

    Cost is what answering cost: the answer, plus the follow-up query rewrite on
    multi-turn turns. The run_models.py judge is a measuring cost and is left out.
    run_e2e.py records have no cost: the stream carries no prices, and the UI
    stream carries no usage either."""
    if isinstance(data, dict) and "results" in data:  # run_e2e.py
        rows = []
        for r in data["results"]:
            t = r.get("timings_ms") or {}
            rows.append({"id": r["id"], "model": (r.get("stamp") or {}).get("model"), "arm": None,
                         "run": r.get("run_index"), "turn": 1,
                         "latency_s": _ms(t.get("time_to_complete")),
                         "ttft_s": _ms(t.get("time_to_first_token")),
                         "cost_usd": None, "checks": _checks(r.get("checks"))})
        return rows
    rows = []
    for r in data:
        if "turns" in r:
            for t in r["turns"]:
                g = t["gen"]
                cost = g.get("cost_usd")
                if cost is not None and t.get("rewrite_cost_usd") is not None:
                    cost = round(cost + t["rewrite_cost_usd"], 6)
                rows.append({"id": r["id"], "model": r.get("model"), "arm": "B", "run": r.get("run", 1),
                             "turn": t["turn"], "latency_s": g.get("latency_s"),
                             "ttft_s": g.get("ttft_s"), "cost_usd": cost,
                             "checks": _checks(t.get("checks"))})
        else:
            g = r["gen"]
            rows.append({"id": r["id"], "model": r.get("model"), "arm": r.get("arm"),
                         "run": r.get("run", 1), "turn": 1, "latency_s": g.get("latency_s"),
                         "ttft_s": g.get("ttft_s"), "cost_usd": g.get("cost_usd"),
                         "checks": _checks(r.get("checks"))})
    return rows


def conversation_rows(data):
    """Case-level checks (T2.6, T2.7), one row per conversation."""
    if not isinstance(data, list):
        return []
    return [{"id": r["id"], "model": r.get("model"), "run": r.get("run", 1),
             "checks": _checks(r.get("case_checks"))} for r in data if "turns" in r]


def kind_of(data):
    if isinstance(data, dict) and "results" in data:
        return "e2e"
    return "multiturn" if data and "turns" in data[0] else "single"


# ── maths ─────────────────────────────────────────────────────────────────────

def percentile(values, p):
    """Linear interpolation between closest ranks (numpy's default). None values
    are dropped; an empty list gives None."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pass_counts(rows):
    """{check: [passed, applicable]}. A check that does not apply to a row is
    absent from it, so it is not in the denominator."""
    out = {}
    for r in rows:
        for k, v in r["checks"].items():
            c = out.setdefault(k, [0, 0])
            c[0] += v
            c[1] += 1
    return dict(sorted(out.items()))


def _r(v, nd):
    return None if v is None else round(v, nd)


def summarise(rows, convs=()):
    """Headline metrics over answer rows (and conversation rows, for T2.6/T2.7)."""
    lat = [r["latency_s"] for r in rows]
    ttft = [r["ttft_s"] for r in rows]
    ttft_fu = [r["ttft_s"] for r in rows if r["turn"] > 1]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    passes = pass_counts(rows)
    passes.update(pass_counts(convs))
    all_pass = sum(1 for r in rows if r["checks"] and all(r["checks"].values()))
    scored = sum(1 for r in rows if r["checks"])
    s = {
        "answers": len(rows),
        "conversations": len(convs) or None,
        "all_checks_pass_rate": _r(all_pass / scored, 3) if scored else None,
        "pass": passes,
        "latency_p50_s": _r(percentile(lat, 50), 2),
        "latency_p95_s": _r(percentile(lat, 95), 2),
        "ttft_p50_s": _r(percentile(ttft, 50), 2),
        "ttft_p95_s": _r(percentile(ttft, 95), 2),
        "followups": sum(1 for r in rows if r["turn"] > 1) or None,
        "ttft_followup_p50_s": _r(percentile(ttft_fu, 50), 2),
        "ttft_followup_p95_s": _r(percentile(ttft_fu, 95), 2),
        # Mean over the answers that recorded a cost; None when none did.
        "cost_per_answer_usd": _r(sum(costs) / len(costs), 5) if costs else None,
        "cost_total_usd": _r(sum(costs), 4) if costs else None,
    }
    # Missing, not zero: the run did not record it. Follow-up TTFT is only
    # missing when the run had follow-ups.
    s["missing"] = [k for k in ("latency_p50_s", "ttft_p50_s", "cost_per_answer_usd") if s[k] is None]
    if s["followups"] and s["ttft_followup_p50_s"] is None:
        s["missing"].append("ttft_followup_p50_s")
    return s


def headline(s, title=""):
    """The block the runners print at the end of a run."""
    def sec(a, b):
        return "not recorded" if a is None else f"p50 {a:.2f}s  p95 {b:.2f}s"
    lines = [f"\nheadline metrics{' — ' + title if title else ''}",
             f"  answers          {s['answers']}"
             + (f" in {s['conversations']} conversations" if s["conversations"] else ""),
             f"  latency          {sec(s['latency_p50_s'], s['latency_p95_s'])}",
             f"  first token      {sec(s['ttft_p50_s'], s['ttft_p95_s'])}"]
    if s["followups"]:
        lines.append(f"  first token, follow-ups ({s['followups']})  "
                     f"{sec(s['ttft_followup_p50_s'], s['ttft_followup_p95_s'])}")
    lines.append("  cost per answer  " + ("not recorded" if s["cost_per_answer_usd"] is None else
                 f"${s['cost_per_answer_usd']:.4f}  (total ${s['cost_total_usd']:.2f})"))
    if s["all_checks_pass_rate"] is not None:
        lines.append(f"  all checks pass  {s['all_checks_pass_rate']:.0%} of answers")
    lines.append("  pass rate per check:")
    for k, (p, t) in s["pass"].items():
        lines.append(f"    {k:<30} {p:>3}/{t:<3} {p / t:>5.0%}")
    return "\n".join(lines)


# ── history ───────────────────────────────────────────────────────────────────

def run_id(rows):
    """Same answers, same id: re-summarising a run never adds a second line."""
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]


def history_lines(data, *, runner, source, label, date, scope, prompt_version=None):
    """One history line per model and arm in a results file. Lines hold ids and
    numbers only, never answer text: the history is committed."""
    rows, convs, kind = answer_rows(data), conversation_rows(data), kind_of(data)
    if kind == "e2e" and data["results"]:
        stamp = data["results"][0].get("stamp") or {}
        prompt_version = prompt_version or stamp.get("site_prompt_version")
    groups = {}
    for r in rows:
        # run_eval.py records had no "model" field until 2026-10-30. Its MODEL has
        # been "gpt-4o" since the file was added (874a852; `git log -S` finds no change).
        if r["model"] is None and runner == "run_eval":
            r["model"] = "gpt-4o"
        groups.setdefault((r["model"], r["arm"]), []).append(r)
    out = []
    for (model, arm), rs in groups.items():
        cs = [c for c in convs if (c["model"] or model) == model]
        pv = prompt_version
        if pv is None and isinstance(data, list):
            pv = Counter(d.get("prompt_version") for d in data
                         if (d.get("model") or model) == model).most_common(1)[0][0]
        out.append({"run_id": run_id(rs), "date": date, "runner": runner, "kind": kind,
                    "label": label, "source": source, "model": model, "arm": arm,
                    "prompt_version": pv, "scope": scope, **summarise(rs, cs)})
    return out


def read_history(path=HISTORY):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def append_history(lines, path=HISTORY):
    """Appends the lines whose run_id is new. Returns how many were added."""
    seen = {h["run_id"] for h in read_history(path)}
    new = [ln for ln in lines if ln["run_id"] not in seen]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            for ln in new:
                f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    return len(new)


def today():
    return datetime.now(UTC).strftime("%Y-%m-%d")


def record_lines(lines, *, path=HISTORY, chart=TREND, out=print):
    """Appends lines of any kind and redraws the chart. Returns how many were added."""
    added = append_history(lines, path)
    if chart:
        write_trend(read_history(path), chart)
    off = sorted({ln["scope"] for ln in lines if ln["scope"] != "full"} |
                 {ln["kind"] for ln in lines if ln["kind"] not in CHART_KINDS})
    out(f"\nhistory: {added} line(s) added to {path.relative_to(HERE) if path.is_relative_to(HERE) else path}"
        + (f" ({', '.join(off)}: not on the trend chart)" if off else ""))
    return added


def record_run(data, *, runner, source, label, scope, path=HISTORY, chart=TREND, out=print):
    """What a runner calls at the end: print the headline block per model and
    arm, append to the history, redraw the chart."""
    lines = history_lines(data, runner=runner, source=source, label=label, date=today(), scope=scope)
    for ln in lines:
        title = " ".join(str(x) for x in (ln["model"], f"arm {ln['arm']}" if ln["arm"] else None) if x)
        out(headline(ln, title))
    record_lines(lines, path=path, chart=chart, out=out)
    return lines


# ── retrieval records (plan G.2) ──────────────────────────────────────────────
# One line per retrieval check, scored by retrieval_scores.py. Scores depend on
# the expect lists as well as the check, and those lists change, so each line
# carries a fingerprint of them: compare retrieval lines only when it matches.

def is_retrieval_check(data):
    """The three shapes retrieval_scores.load_check reads."""
    if isinstance(data, list):
        return bool(data) and isinstance(data[0], dict) and "ids" in data[0] and "sims" in data[0]
    if isinstance(data, dict) and data and "results" not in data:
        v = next(iter(data.values()))
        return isinstance(v, list) or (isinstance(v, dict) and "results" in v)
    return False


def expect_version(spec):
    """Fingerprint of the expect lists retrieval_scores.py scores against."""
    import retrieval_scores as RS
    ex = {c["id"]: c.get("expect") or [] for c in RS.cases(spec)}
    return hashlib.sha256(json.dumps(ex, sort_keys=True).encode()).hexdigest()[:8]


def check_ranker(data):
    """The ranker a check file records (saved since 2026-10-30, G.2), or None."""
    rankers = {v.get("ranker") for v in data.values() if isinstance(v, dict)} if isinstance(data, dict) else set()
    rankers.discard(None)
    if len(rankers) > 1:
        raise ValueError(f"more than one ranker in one check: {sorted(rankers)}")
    return rankers.pop() if rankers else None


def retrieval_line(paths, *, spec, runner, label, date, scope=None, arm=None, cost_usd=None):
    """One history line for a retrieval check (one file, or several taken on
    the same index and scored together). Values the files do not hold, such as
    the ranker before G.2 or the probe's rewrite cost, stay None."""
    import retrieval_scores as RS
    check, versions, rankers = {}, set(), set()
    for p in paths:
        part, v = RS.load_check(p)
        overlap = set(check) & set(part)
        if overlap:
            raise ValueError(f"question ids in more than one check file: {sorted(overlap)}")
        check.update(part)
        versions.add(v)
        rankers.add(check_ranker(json.loads(Path(p).read_text(encoding="utf-8"))))
    if len(versions) > 1:
        raise ValueError(f"check files come from different indexes: {sorted(map(str, versions))}")
    if len(rankers) > 1:
        raise ValueError(f"check files come from different rankers: {sorted(map(str, rankers))}")
    agg = RS.report(spec, check, out=lambda *a: None)
    ev = expect_version(spec)
    if scope is None:
        scope = "partial" if agg["not_in_check"] else "full"
    ln = {"run_id": hashlib.sha256(json.dumps({"check": check, "expect": ev}, sort_keys=True)
                                   .encode()).hexdigest()[:12],
          "date": date, "runner": runner, "kind": "retrieval", "label": label,
          "source": " + ".join(Path(p).name for p in paths), "model": None, "arm": arm,
          "prompt_version": None, "scope": scope,
          "index_version": versions.pop(), "ranker": rankers.pop(), "expect_version": ev,
          "k": agg["k"], "questions": len(check), "cases_scored": agg["n_scored"],
          "mean_recall_at_5": _r(agg["mean_recall"], 3), "micro_recall_at_5": _r(agg["micro_recall"], 3),
          "mrr_at_5": _r(agg["mrr"], 3), "zero_hit": agg["zero_hit"],
          "not_in_check": len(agg["not_in_check"]), "cost_usd": cost_usd}
    ln["missing"] = [k for k in ("index_version", "ranker", "mean_recall_at_5") if ln[k] is None]
    return ln


def retrieval_headline(ln):
    def f(v):
        return "n/a" if v is None else f"{v:.3f}"
    return (f"\nretrieval — {ln['label']}  (index {ln['index_version'] or 'not recorded'}, "
            f"ranker {ln['ranker'] or 'not recorded'}, expect lists {ln['expect_version']})\n"
            f"  {ln['cases_scored']} cases scored of {ln['questions']} questions: mean recall@5 "
            f"{f(ln['mean_recall_at_5'])}  micro {f(ln['micro_recall_at_5'])}  MRR@5 {f(ln['mrr_at_5'])}")


def record_retrieval(paths, *, runner, label, scope=None, arm=None, cost_usd=None, spec=None,
                     path=HISTORY, chart=TREND, out=print):
    """What a retrieval check or probe calls once its check file is saved."""
    if spec is None:
        spec = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
    ln = retrieval_line(paths, spec=spec, runner=runner, label=label, date=today(), scope=scope,
                        arm=arm, cost_usd=cost_usd)
    out(retrieval_headline(ln))
    record_lines([ln], path=path, chart=chart, out=out)
    return ln


# ── backfill ──────────────────────────────────────────────────────────────────
# Saved runs in eval_runs/, read 2026-10-30. Left out: blind-sample scores,
# retrieval checks and probes (not eval runs), pass1_gemini-* (inside
# pass1_combined), the flash-overload backup, gemini_check (2 answers), the
# 09-10 e2e files (1 record each), and the narrative-X* partials.
# (file, runner, label, scope)
BACKFILL = [
    ("eval_e2e_local-route_2026-09-11.json", "run_e2e", "e2e local-route", "full"),
    ("eval_e2e_local-route_2026-09-11_r2.json", "run_e2e", "e2e local-route r2", "full"),
    ("pass1_combined_2026-10-16.json", "run_models", "model pass 1", "full"),
    ("baseline_2026-10-19.json", "run_models", "baseline", "full"),
    ("baseline_multiturn_2026-10-19.json", "run_models", "baseline", "full"),
    ("kb-r1_2026-10-20.json", "run_models", "kb-r1", "full"),
    ("kb-r1_multiturn_2026-10-20.json", "run_models", "kb-r1", "full"),
    ("kb-r2_2026-10-21.json", "run_eval", "kb-r2", "full"),
    ("kb-r2_multiturn_2026-10-21.json", "run_eval", "kb-r2", "full"),
    ("kb-r3_2026-10-21.json", "run_eval", "kb-r3", "full"),
    ("kb-r3_multiturn_2026-10-21.json", "run_eval", "kb-r3", "full"),
    ("labelled-ab60_2026-10-21.json", "run_eval", "labelled-ab60", "full"),
    ("labelled-ab60_multiturn_2026-10-21.json", "run_eval", "labelled-ab60", "full"),
    ("oldprompt-newkb_2026-10-21.json", "run_eval", "oldprompt-newkb", "full"),
    ("oldprompt-newkb_multiturn_2026-10-21.json", "run_eval", "oldprompt-newkb", "full"),
    ("labelled-f138_2026-10-22.json", "run_eval", "labelled-f138", "full"),
    ("labelled-f138_multiturn_2026-10-22.json", "run_eval", "labelled-f138", "full"),
    ("variant-A_2026-10-22.json", "run_eval", "variant-A", "full"),
    ("variant-C_2026-10-22.json", "run_eval", "variant-C", "full"),
    ("hybrid-ff4d_2026-10-22.json", "run_eval", "hybrid-ff4d", "full"),
    ("hybrid-ff4d_multiturn_2026-10-22.json", "run_eval", "hybrid-ff4d", "full"),
    ("hybrid-ff4d-r2_2026-10-22.json", "run_eval", "hybrid-ff4d r2", "full"),
    ("redecide-ff4d_2026-10-22.json", "run_models", "re-decide ff4d", "full"),
    ("redecide-ff4d_multiturn_2026-10-22.json", "run_models", "re-decide ff4d", "full"),
    ("e2e-dev-e24a3627_2026-10-23.json", "run_e2e", "e2e dev e24a3627", "partial"),
    # G.2, 2026-10-30: runs from 2026-10-23 to 2026-10-30. A name starting "../"
    # is in the repo root. The spot-check and hybrid-d1 files are copies of
    # eval_results*.json files the next run would overwrite; the spot checks hold
    # no date, so theirs is the file's modified time. The e2e probes are one
    # protocol-check answer each. The e2e _parts/ folder is left out: its
    # answers are the attempts behind the merged file.
    ("../eval_e2e_probe_dev_2026-10-23.json", "run_e2e", "e2e probe dev", "probe"),
    ("e2e-dev-e24a3627_2026-10-23_additive-X16-X17.json", "run_e2e", "e2e dev e24a3627 X16-X17", "partial"),
    ("spot-check-before_2026-10-23.json", "run_eval", "spot check before (date: file time)", "partial"),
    ("spot-check_2026-10-23.json", "run_eval", "spot check (date: file time)", "partial"),
    ("../eval_e2e_probe_local-route_2026-10-28.json", "run_e2e", "e2e probe local-route", "probe"),
    ("hybrid-d1_2026-10-30.json", "run_eval", "ff4d5612 labelled", "full"),
    ("hybrid-d1_multiturn_2026-10-30.json", "run_eval", "multiturn ff4d5612", "full"),
]

# Retrieval checks and probes, 2026-10-23 to 2026-10-30, scored against the
# expect lists in eval_set.json now. None of these files records the ranker.
# D.1's vector/hybrid checks were in a removed worktree and are not here.
# The 10-23 checks hold 36 of the 50 cases the eval set had then: partial.
# (files, label, scope, arm)
RETRIEVAL_BACKFILL = [
    (("retrieval_check_2026-10-23_e24a3627.json",), "retrieval dev e24a3627", "partial", None),
    (("retrieval_check_2026-10-23_1124701d.json",), "retrieval dev 1124701d", "partial", None),
    (("retrieval_check_2026-10-30_75c257fb.json", "retrieval_check_2026-10-30_75c257fb_X19.json"),
     "retrieval baseline (A.1)", "full", None),
    (("retrieval_check_2026-10-30_f5784094_X19.json",), "retrieval X19", "partial", None),
    (("retrieval_check_2026-10-30_f5784094_X20-X23.json",), "retrieval X20-X23", "partial", None),
    (("retrieval_check_2026-10-30_848efc4e_X21.json",), "retrieval X21", "partial", None),
] + [((f"corrective_probe{v}_2026-10-30_848efc4e{n}_{arm}.json",), f"corrective probe {name}", "full", arm)
     for v, n, name in (("", "", "v1"), ("", "_noname", "v1 name stripped"), ("_v2", "", "v2"))
     for arm in ("orig", "rewrite", "merged")]

# judge_agreement.py runs from 2026-10-30 (results log R-20261030-06 and -07):
# the stored verdicts of both blind samples, and the 10-22 sample re-judged.
# The 10-16 sample has no re-judge files. (KEY file, verdicts, date)
JUDGE_BACKFILL = [
    ("blind-sample-2026-10-16_KEY.json", None, "2026-10-30"),
    ("blind-sample-2026-10-22_KEY.json", None, "2026-10-30"),
    ("blind-sample-2026-10-22_KEY.json", "unfolded", "2026-10-30"),
    ("blind-sample-2026-10-22_KEY.json", "folded", "2026-10-30"),
    ("blind-sample-2026-10-22_KEY.json", "gap-wording-rejected", "2026-10-30"),
]


def backfill(path=HISTORY, out=print, runs=None, retrieval=None, judge=None, eval_runs=EVAL_RUNS, spec=None):
    """Adds every saved run, retrieval check and judge-agreement run listed
    above. Lines already in the history (same run_id) are skipped, so it can be
    re-run."""
    added = 0
    for name, runner, label, scope in BACKFILL if runs is None else runs:
        src = (eval_runs / name).resolve()
        if not src.exists():
            out(f"  missing  {name}")
            continue
        date = re.search(r"\d{4}-\d{2}-\d{2}", name).group(0)
        data = json.loads(src.read_text(encoding="utf-8"))
        source = src.relative_to(HERE).as_posix() if src.is_relative_to(HERE) else src.name
        lines = history_lines(data, runner=runner, source=source, label=label, date=date, scope=scope)
        n = append_history(lines, path)
        added += n
        for ln in lines:
            out(f"  {date} {label:<20} {ln['kind']:<9} {ln['model']!s:<18} arm {ln['arm'] or '-'}  "
                f"n={ln['answers']:<4} missing: {', '.join(ln['missing']) or 'none'}")
    if spec is None:
        spec = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
    for names, label, scope, arm in RETRIEVAL_BACKFILL if retrieval is None else retrieval:
        srcs = [eval_runs / n for n in names]
        if not all(s.exists() for s in srcs):
            out(f"  missing  {' + '.join(names)}")
            continue
        date = re.search(r"\d{4}-\d{2}-\d{2}", names[0]).group(0)
        ln = retrieval_line(srcs, spec=spec, runner="retrieval_scores", label=label, date=date,
                            scope=scope, arm=arm)
        added += append_history([ln], path)
        out(f"  {date} {label:<30} retrieval index {ln['index_version']}  n={ln['cases_scored']:<3} "
            f"recall@5 {ln['mean_recall_at_5']}  MRR {ln['mrr_at_5']}")
    import judge_agreement as JA  # imports bootstrap_ci, which imports this module
    for key, verdicts, date in JUDGE_BACKFILL if judge is None else judge:
        if not (eval_runs / key).exists():
            out(f"  missing  {key}")
            continue
        rows, results = JA.load_sample(eval_runs / key, verdicts)
        if rows is None:
            out(f"  missing  re-judge files for {key} {verdicts}")
            continue
        if JA.history_run_id(rows, verdicts) in {h["run_id"] for h in read_history(path)}:
            out(f"  {date} {key} {verdicts or 'stored'}: already in the history")
            continue
        ln = JA.history_line(eval_runs / key, verdicts, date=date, rows=rows, results=results)
        added += append_history([ln], path)
        out(f"  {date} {ln['label']:<40} rho vs D1 {ln['d1']['rho']}  kappa {ln['d1']['kappa']}")
    out(f"{added} line(s) added to {path.relative_to(HERE) if path.is_relative_to(HERE) else path}")
    return added


# ── trend chart ───────────────────────────────────────────────────────────────
# Hand-written SVG, so the chart needs no plotting dependency. Three panels,
# one measure each, never two scales on one axis. Colours: the dataviz skill's
# reference categorical slots 1-3, in fixed order.

SERIES = [("single", "gpt-4o, single-turn", "#2a78d6", "#3987e5"),
          ("multiturn", "gpt-4o, multi-turn", "#eb6834", "#d95926"),
          ("e2e", "site, end to end", "#1baf7a", "#199e70")]
CHART_KINDS = {k for k, *_ in SERIES}  # retrieval and judge_agreement lines are not answer runs
PANELS = [("all_checks_pass_rate", "Answers passing every check", lambda v: f"{v:.0%}"),
          ("latency_p95_s", "Latency, p95 (seconds)", lambda v: f"{v:.1f}"),
          ("cost_per_answer_usd", "Cost per answer (USD)", lambda v: f"${v:.4f}")]


def trend_points(history):
    """Full-scope lines for the answering model (gpt-4o) and e2e runs, in history order."""
    pts = [h for h in history if h["kind"] in CHART_KINDS and h["scope"] == "full" and h["arm"] in ("B", None)
           and (h["kind"] == "e2e" or h["model"] in ("gpt-4o", None))]
    return sorted(pts, key=lambda h: h["date"])  # stable: same-day runs keep their order


def write_trend(history, path=TREND):
    pts = trend_points(history)
    W, PH, L, R, T = 1000, 170, 70, 120, 64
    H = T + len(PANELS) * (PH + 46) + 120
    xs = {id(p): L + (i + 0.5) * (W - L - R) / max(len(pts), 1) for i, p in enumerate(pts)}
    e = html.escape
    o = [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="system-ui, sans-serif" font-size="12" role="img" '
         f'aria-label="Eval history: pass rate, p95 latency and cost per answer by run">'),
         "<style>.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e4e3df}"
         + "".join(f".s{i}{{stroke:{lc};fill:{lc}}}" for i, (_k, _n, lc, _d) in enumerate(SERIES))
         + "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}"
         ".grid{stroke:#383835}"
         + "".join(f".s{i}{{stroke:{dc};fill:{dc}}}" for i, (_k, _n, _l, dc) in enumerate(SERIES))
         + "}</style>",
         f'<rect class="bg" width="{W}" height="{H}"/>',
         f'<text class="t1" x="{L}" y="24" font-size="16" font-weight="600">jordbot eval history</text>',
         (f'<text class="t2" x="{L}" y="44">Full runs only, arm B. Each point is one run; hover for '
         f'the run. Checks were added over time, so pass rates are not like for like across all dates.</text>')]
    for pi, (key, title, fmt) in enumerate(PANELS):
        top = T + pi * (PH + 46) + 26
        vals = [p[key] for p in pts if p.get(key) is not None]
        o.append(f'<text class="t1" x="{L}" y="{top - 8}" font-weight="600">{e(title)}</text>')
        if not vals:
            o.append(f'<text class="t2" x="{L}" y="{top + 20}">no data</text>')
            continue
        lo = 0.0
        hi = max(vals) * 1.1 or 1.0
        if key == "all_checks_pass_rate":
            hi = 1.0
        def y(v, top=top, lo=lo, hi=hi):
            return top + PH - (v - lo) / (hi - lo) * PH
        for g in range(5):
            gv = lo + (hi - lo) * g / 4
            o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(gv):.1f}" y2="{y(gv):.1f}"/>')
            o.append(f'<text class="t2" x="{L - 6}" y="{y(gv) + 4:.1f}" text-anchor="end">{e(fmt(gv))}</text>')
        for si, (kind, _name, _l, _d) in enumerate(SERIES):
            sp = [p for p in pts if p["kind"] == kind and p.get(key) is not None]
            if len(sp) > 1:
                d = " ".join(f"{xs[id(p)]:.1f},{y(p[key]):.1f}" for p in sp)
                o.append(f'<polyline class="s{si}" fill="none" stroke-width="2" points="{d}" style="fill:none"/>')
            for p in sp:
                tip = f"{p['date']} {p['label']} ({p['kind']}, {p['answers']} answers): {fmt(p[key])}"
                o.append(f'<circle class="s{si}" cx="{xs[id(p)]:.1f}" cy="{y(p[key]):.1f}" r="4.5" '
                         f'stroke="#fcfcfb" stroke-width="2"><title>{e(tip)}</title></circle>')
    base = T + len(PANELS) * (PH + 46) + 10
    for p in pts:
        x = xs[id(p)]
        o.append(f'<text class="t2" font-size="10" transform="translate({x:.1f},{base}) rotate(45)">'
                 f'{e(p["date"][5:] + " " + p["label"])}</text>')
    lx = L
    for si, (_k, name, _l, _d) in enumerate(SERIES):
        o.append(f'<circle class="s{si}" cx="{lx + 5}" cy="{H - 14}" r="5"/>'
                 f'<text class="t1" x="{lx + 15}" y="{H - 10}">{e(name)}</text>')
        lx += 190
    o.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("files", nargs="*", type=Path,
                    help="a results file from any runner, or one or more retrieval check files "
                         "taken on the same index (scored together as one line)")
    ap.add_argument("--append", action="store_true", help="add the file's lines to the history")
    ap.add_argument("--label", help="history label (default: the file name)")
    ap.add_argument("--runner", default="manual")
    ap.add_argument("--scope", choices=["full", "partial", "probe"],
                    help="default: full for runs; for retrieval checks, partial when the check "
                         "misses a case in eval_set.json")
    ap.add_argument("--arm", help="retrieval checks: the probe arm (orig, rewrite, merged)")
    ap.add_argument("--backfill", action="store_true",
                    help="add the saved runs, retrieval checks and judge-agreement runs listed in BACKFILL")
    ap.add_argument("--chart", action="store_true", help="redraw the trend chart from the history")
    args = ap.parse_args()
    if args.backfill:
        backfill()
        write_trend(read_history())
        print(f"chart written to {TREND.relative_to(HERE)}")
        return
    if args.chart:
        write_trend(read_history())
        print(f"chart written to {TREND.relative_to(HERE)}")
        return
    if not args.files:
        sys.exit("give a results file, --backfill or --chart")
    datas = [json.loads(f.read_text(encoding="utf-8")) for f in args.files]
    label = args.label or args.files[0].stem
    if all(is_retrieval_check(d) for d in datas):
        if args.append:
            record_retrieval(args.files, runner=args.runner, label=label, scope=args.scope, arm=args.arm)
        else:
            spec = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
            print(retrieval_headline(retrieval_line(args.files, spec=spec, runner=args.runner, label=label,
                                                    date="-", scope=args.scope, arm=args.arm)))
        return
    if len(args.files) > 1:
        sys.exit("give one results file at a time (several files only for retrieval checks)")
    data, file, scope = datas[0], args.files[0], args.scope or "full"
    if args.append:
        record_run(data, runner=args.runner, source=file.name, label=label, scope=scope)
    else:
        for ln in history_lines(data, runner=args.runner, source=file.name, label=label,
                                date="-", scope=scope):
            print(headline(ln, " ".join(str(x) for x in (ln["model"], ln["arm"] and f"arm {ln['arm']}") if x)))


if __name__ == "__main__":
    main()
