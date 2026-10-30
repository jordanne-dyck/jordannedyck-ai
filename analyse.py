#!/usr/bin/env python
"""
Analyses one history line against the last comparable run (plan G.3,
2026-10-30). Reads eval_history/history.jsonl and the run files it names.

The free part calls no API: the gap on every check against the last
comparable run, with a 95% range paired by question (bootstrap_ci.py), the
weakest checks, and cost and latency outliers. "Comparable" means the same
kind, model, arm, prompt version, scope and check set; for retrieval, the same
expect lists, arm and scope; for judge agreement, the same blind sample. A line
with no prompt version is compared with nothing: the prompt cannot be shown to
be the same.

The paid part sends that report (ids and numbers only, never answer text) and
the closed decisions to gpt-4o, which writes a narrative and ranked
recommendations. A recommendation is kept only if every number it cites is in
the report; one the model says would reopen a closed decision is listed apart.
The decisions are read from CLAUDE.local.md (park's mirror of memory, only as
fresh as the last park) and the plan's Decided and Rejected sections.

Paid runs need --go until Jord sets STANDING_BUDGET_USD.

Usage (from C:\\Users\\Jord\\jordannedyck-ai):
    uv run python analyse.py                     # the latest history line, free part only
    uv run python analyse.py --run LABEL_OR_ID   # another line (run id prefix or label)
    uv run python analyse.py --estimate          # price the narrative, call nothing
    uv run python analyse.py --go                # free part plus the narrative (paid)
    uv run python analyse.py --go --log          # also add a "proposed" entry to the results log
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import bootstrap_ci as B
import retrieval_scores as RS
import run_metrics as RM

HERE = RM.HERE
LOG = HERE / "nimbalyst-local" / "results-log.md"
ANALYSES = HERE / "nimbalyst-local" / "analysis"
PLAN = HERE / "nimbalyst-local" / "plans" / "eval-harness-broaden-plan.md"
STATE = HERE / "CLAUDE.local.md"

NARRATIVE_MODEL = "gpt-4o"
COST_PER_1M_IN, COST_PER_1M_OUT = 2.50, 10.00  # same as run_eval.py (selftest checks)
MAX_OUT = 1500
# Paid runs at or under this estimate need no --go. None until Jord sets one.
STANDING_BUDGET_USD = None

ANSWER_KINDS = ("single", "multiturn", "e2e")
JUDGE_MEASURES = ("rho", "kappa", "caught", "false_alarm")
WEAKEST = 5
FENCE = 1.5  # Tukey: an answer is an outlier above Q3 + 1.5 x IQR
MIN_RUNS = 3  # earlier runs needed before a run-level value is called out of range


# ── which line, compared with which ───────────────────────────────────────────

def comparable_key(h):
    """Lines with the same key may be compared. None: compare with nothing."""
    if h["kind"] in ANSWER_KINDS:
        if h.get("prompt_version") in (None, "unknown"):
            return None
        return (h["kind"], h["model"], h["arm"], h["prompt_version"], h["scope"], tuple(sorted(h["pass"])))
    if h["kind"] == "retrieval":
        return (h["kind"], h["expect_version"], h["arm"], h["scope"])
    if h["kind"] == "judge_agreement":
        return (h["kind"], h["sample"])
    return None


def pick_line(history, sel=None):
    """(index, line): the last line, or the last whose run id starts with sel
    or whose label contains it."""
    if sel is None:
        return len(history) - 1, history[-1]
    hits = [(i, h) for i, h in enumerate(history) if h["run_id"].startswith(sel) or sel in h["label"]]
    if not hits:
        sys.exit(f"no history line matches '{sel}'")
    return hits[-1]


def candidates(history, i):
    """Comparable lines before line i, newest first (by date, then history order)."""
    key = comparable_key(history[i])
    if key is None:
        return []
    here = (history[i]["date"], i)
    earlier = sorted(((h["date"], j) for j, h in enumerate(history)
                      if (h["date"], j) < here and comparable_key(h) == key), reverse=True)
    return [history[j] for _, j in earlier]


def previous(history, i):
    """The last comparable line before line i."""
    return next(iter(candidates(history, i)), None)


# ── per-question cells, read back from the run files ──────────────────────────

def resolve(source):
    """Paths for a history source ("a.json" or "a.json + b.json"); None if any is gone."""
    out = []
    for name in source.split(" + "):
        p = next((d / name for d in (HERE, RM.EVAL_RUNS) if (d / name).exists()), None)
        if p is None:
            return None
        out.append(p)
    return out


def answer_run(h):
    """(question cells, answer rows) for an answer-run line, or (None, why)."""
    paths = resolve(h["source"])
    if paths is None:
        return None, f"run file not found: {h['source']}"
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    for (model, arm), (rows, convs) in B.groups(data).items():
        if model is None and h["runner"] == "run_eval":
            model = "gpt-4o"  # as run_metrics.history_lines fills it
            for r in rows:
                r["model"] = model
        if (model, arm) != (h["model"], h["arm"]):
            continue
        if RM.run_id(rows) != h["run_id"]:
            return None, f"{paths[0].name} no longer holds the answers this line was written from"
        return B.question_cells(rows, convs), rows
    return None, f"{paths[0].name} holds no {h['model']} arm {h['arm']} answers"


def retrieval_run(h, spec):
    """({"recall@5": cells, "MRR@5": cells}, None) or (None, why). Cells are
    (score, 1) per question, so bootstrap_ci's ratio is the mean."""
    if h["expect_version"] != RM.expect_version(spec):
        return None, f"expect lists changed since this line ({h['expect_version']}); stored numbers only"
    paths = resolve(h["source"])
    if paths is None:
        return None, f"check file not found: {h['source']}"
    check = {}
    for p in paths:
        check.update(RS.load_check(p)[0])
    expect = {c["id"]: c.get("expect") or [] for c in RS.cases(spec)}
    out = {"recall@5": {}, "MRR@5": {}}
    for q, retrieved in check.items():
        if expect.get(q):
            s = RS.score(expect[q], retrieved)
            out["recall@5"][q] = (s["recall"], 1)
            out["MRR@5"][q] = (s["rr"], 1)
    mean = sum(v for v, _ in out["recall@5"].values()) / len(out["recall@5"]) if out["recall@5"] else None
    if h["mean_recall_at_5"] is not None and (mean is None or round(mean, 3) != h["mean_recall_at_5"]):
        return None, f"{h['source']} no longer scores as the line says ({mean} vs {h['mean_recall_at_5']})"
    return out, None


# ── the free analysis ─────────────────────────────────────────────────────────

def _gap(a, b, *, resamples, seed):
    g = B.paired_gap(a, b, resamples=resamples, seed=seed)
    return {k: (None if g[k] is None else round(g[k], 4)) for k in ("a", "b", "gap", "lo", "hi")} | \
        {"n": g["n"], "verdict": g["verdict"], "rough": g["n"] < B.ROUGH}


def gaps(cur, prev, measures, *, resamples, seed):
    """{measure: gap} for measures both runs have, previous = A, this run = B."""
    return {m: _gap(prev[m], cur[m], resamples=resamples, seed=seed) for m in measures if m in cur and m in prev}


def answer_measures(cells):
    return {c: B.for_check(cells, c) for c in sorted({c for pc in cells.values() for c in pc})}


def weakest(measures, *, resamples, seed, n=WEAKEST):
    """The n lowest-scoring measures below 100%, with their ranges."""
    out = []
    for m, cells in measures.items():
        if m == B.ALL or not cells:
            continue
        r = B.bootstrap(cells, resamples=resamples, seed=seed)
        if r["value"] is not None and r["value"] < 1:
            out.append({"measure": m, "value": round(r["value"], 4), "lo": round(r["lo"], 4),
                        "hi": round(r["hi"], 4), "n": r["n"], "rough": r["n"] < B.ROUGH})
    return sorted(out, key=lambda w: (w["value"], w["measure"]))[:n]


def answer_outliers(rows):
    """Answers above the Tukey fence on latency or cost. Ids and numbers only."""
    out = []
    for field in ("latency_s", "cost_usd"):
        vals = [r[field] for r in rows if r[field] is not None]
        if len(vals) < 4:
            continue
        q1, q3 = RM.percentile(vals, 25), RM.percentile(vals, 75)
        fence = q3 + FENCE * (q3 - q1)
        for r in rows:
            if r[field] is not None and r[field] > fence:
                out.append({"measure": field, "id": r["id"], "run": r["run"], "turn": r["turn"],
                            "value": r[field], "fence": round(fence, 4)})
    return out


def run_level(history, i):
    """This run's latency, first token and cost against every earlier full run
    of the same kind, model and arm (any prompt). Called out of range only with
    MIN_RUNS earlier values."""
    h = history[i]
    peers = [p for j, p in enumerate(history) if j != i and p["kind"] == h["kind"] and p["model"] == h["model"]
             and p["arm"] == h["arm"] and p["scope"] == "full" and (p["date"], j) < (h["date"], i)]
    out = []
    for m in ("latency_p50_s", "latency_p95_s", "ttft_p50_s", "cost_per_answer_usd"):
        vals = [p[m] for p in peers if p.get(m) is not None]
        if h.get(m) is None or not vals:
            continue
        lo, hi = min(vals), max(vals)
        side = None if len(vals) < MIN_RUNS else "below" if h[m] < lo else "above" if h[m] > hi else None
        out.append({"measure": m, "value": h[m], "earlier_min": lo, "earlier_max": hi,
                    "earlier_runs": len(vals), "out_of_range": side})
    return out


def judge_changes(cur, prev):
    """Each agreement measure, then and now, with the stored ranges. The two
    runs score the same answers, but the ranges are not paired, so overlap is
    shown, not a verdict."""
    out = {}
    for basis in ("d1", "ship"):
        for m in JUDGE_MEASURES:
            a, b = prev[basis][m], cur[basis][m]
            ra, rb = prev[basis][f"{m}_range"], cur[basis][f"{m}_range"]
            out[f"{basis} {m}"] = {"a": a, "b": b, "change": None if a is None or b is None else round(b - a, 3),
                                  "a_range": ra, "b_range": rb,
                                  "ranges_overlap": ra[0] <= rb[1] and rb[0] <= ra[1]}
    return out


def _line(h):
    return {k: h.get(k) for k in ("run_id", "date", "label", "kind", "model", "arm", "prompt_version", "scope",
                                  "source", "index_version", "ranker", "expect_version")}


def analyse(history, i, *, spec=None, resamples=B.RESAMPLES, seed=B.SEED):
    """The free report for history line i. Ids and numbers only."""
    h = history[i]
    prev = previous(history, i)
    rep = {"run": _line(h), "previous": _line(prev) if prev else None, "notes": [],
           "setup": {"level": B.LEVEL, "resamples": resamples, "seed": seed, "unit": "question"}}
    if comparable_key(h) is None:
        rep["notes"].append("compared with nothing: " + ("no prompt version recorded"
                            if h["kind"] in ANSWER_KINDS else f"kind {h['kind']} has no comparison"))
    elif prev is None:
        rep["notes"].append("no earlier comparable run in the history")

    if h["kind"] in ANSWER_KINDS:
        cells, rows = answer_run(h)
        rep["headline"] = {k: h.get(k) for k in ("answers", "conversations", "all_checks_pass_rate",
                                                 "latency_p50_s", "latency_p95_s", "ttft_p50_s",
                                                 "cost_per_answer_usd")}
        rep["run_level"] = run_level(history, i)
        if cells is None:
            rep["notes"].append(rows)
            return rep
        cur = answer_measures(cells)
        rep["weakest"] = weakest(cur, resamples=resamples, seed=seed)
        rep["outliers"] = answer_outliers(rows)
        if prev:
            pcells, prows = answer_run(prev)
            if pcells is None:
                rep["notes"].append("previous run: " + prows)
            else:
                rep["gaps"] = gaps(cur, answer_measures(pcells), [B.ALL] + sorted(set(cur) - {B.ALL}),
                                   resamples=resamples, seed=seed)
    elif h["kind"] == "retrieval":
        if spec is None:
            spec = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
        rep["headline"] = {k: h.get(k) for k in ("cases_scored", "mean_recall_at_5", "micro_recall_at_5",
                                                 "mrr_at_5", "zero_hit")}
        cur, why = retrieval_run(h, spec)
        if cur is None:
            rep["notes"].append(why)
            return rep
        rep["weakest"] = [{"measure": "RR@5", "id": q, "value": round(v, 4)} for q, (v, _) in
                          sorted(cur["MRR@5"].items(), key=lambda kv: (kv[1][0], kv[0]))[:WEAKEST]]
        # A partial check covers a few questions: walk back to the newest
        # comparable check that shares at least one of them.
        rep["previous"] = None
        for cand in candidates(history, i):
            pc, why = retrieval_run(cand, spec)
            if pc is None:
                rep["notes"].append(f"skipped {cand['date']} {cand['label']}: {why}")
            elif set(pc["recall@5"]) & set(cur["recall@5"]):
                rep["previous"] = _line(cand)
                rep["gaps"] = gaps(cur, pc, ("recall@5", "MRR@5"), resamples=resamples, seed=seed)
                break
        if prev and rep["previous"] is None:
            rep["notes"].append("no earlier comparable check shares a scored question with this one")
    elif h["kind"] == "judge_agreement":
        rep["headline"] = {"answers": h["answers"], "questions": h["questions"],
                           **{b: {m: h[b][m] for m in JUDGE_MEASURES} for b in ("d1", "ship")}}
        if prev:
            rep["judge_changes"] = judge_changes(h, prev)
    if "gaps" in rep:
        # Fewer than ROUGH questions cannot carry a verdict: with one question the
        # range collapses to the gap itself.
        out = {m: g for m, g in rep["gaps"].items() if g["verdict"] == "outside the noise"}
        rep["regressions"] = [m for m, g in out.items() if g["gap"] < 0 and not g["rough"]]
        rep["improvements"] = [m for m, g in out.items() if g["gap"] > 0 and not g["rough"]]
        rep["rough_not_counted"] = [m for m, g in out.items() if g["rough"]]
    return rep


# ── printing ──────────────────────────────────────────────────────────────────

def _v(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}" if 0 < abs(v) < 0.1 else f"{v:.3f}"  # costs are under a cent
    return str(v)


def render(rep):
    """The report as text. The narrative is checked against this text."""
    r, p = rep["run"], rep["previous"]
    o = [f"analysis of {r['date']} {r['label']} ({r['kind']}, {r['model'] or '-'}, arm {r['arm'] or '-'}, "
         f"prompt {r['prompt_version'] or 'not recorded'}, {r['scope']}; run {r['run_id']})"]
    o.append(f"compared with: {p['date']} {p['label']} (run {p['run_id']})" if p else "compared with: nothing")
    o += [f"note: {n}" for n in rep["notes"]]
    o.append("headline: " + ", ".join(f"{k} {_v(v)}" for k, v in (rep.get("headline") or {}).items()))
    s = rep["setup"]
    if rep.get("gaps"):
        o.append(f"\ngap = this run minus the previous, on questions both have; {s['level']:.0%} range, "
                 f"{s['resamples']:,} resamples by question, seed {s['seed']}")
        for m, g in rep["gaps"].items():
            o.append(f"  {m:<30} {_v(g['a'])} -> {_v(g['b'])}  gap {_v(g['gap'])}  range {_v(g['lo'])} to "
                     f"{_v(g['hi'])}  n={g['n']}  {g['verdict']}{'  rough' if g['rough'] else ''}")
        o.append("regressions outside the noise: " + (", ".join(rep["regressions"]) or "none"))
        o.append("improvements outside the noise: " + (", ".join(rep["improvements"]) or "none"))
        if rep["rough_not_counted"]:
            o.append(f"outside the noise but under {B.ROUGH} questions, not counted: "
                     + ", ".join(rep["rough_not_counted"]))
    if rep.get("judge_changes"):
        o.append("\njudge agreement, previous -> this run (stored 95% ranges; not paired)")
        for m, c in rep["judge_changes"].items():
            o.append(f"  {m:<18} {_v(c['a'])} -> {_v(c['b'])}  change {_v(c['change'])}  ranges {c['a_range']} "
                     f"and {c['b_range']}  {'overlap' if c['ranges_overlap'] else 'do not overlap'}")
    if rep.get("weakest"):
        o.append("\nweakest in this run:")
        for w in rep["weakest"]:
            if "lo" in w:
                o.append(f"  {w['measure']:<30} {_v(w['value'])}  range {_v(w['lo'])} to {_v(w['hi'])}  "
                         f"n={w['n']}{'  rough' if w['rough'] else ''}")
            else:
                o.append(f"  {w['id']:<8} {w['measure']} {_v(w['value'])}")
    if rep.get("outliers"):
        o.append(f"\nanswers above the outlier fence (Q3 + {FENCE} x IQR):")
        for x in rep["outliers"]:
            o.append(f"  {x['measure']:<10} {x['id']} run {x['run']} turn {x['turn']}: {_v(x['value'])} "
                     f"(fence {_v(x['fence'])})")
    if rep.get("run_level"):
        o.append("\nrun level, against earlier full runs of the same kind, model and arm (any prompt):")
        for x in rep["run_level"]:
            o.append(f"  {x['measure']:<20} {_v(x['value'])}  earlier {_v(x['earlier_min'])} to "
                     f"{_v(x['earlier_max'])} over {x['earlier_runs']} runs"
                     f"{'  ' + x['out_of_range'].upper() + ' THE EARLIER RANGE' if x['out_of_range'] else ''}")
    return "\n".join(o)


# ── closed decisions ──────────────────────────────────────────────────────────

def _bullets(text):
    """Top-level "- " bullets, with their continuation and nested lines folded in."""
    items, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("- "):
            cur = [ln[2:].strip()]
            items.append(cur)
        elif cur is not None and ln.startswith(" ") and ln.strip():
            cur.append(ln.strip().removeprefix("- "))
        else:
            cur = None
    return [" ".join(i) for i in items]


def decided_items(state=STATE, plan=PLAN):
    """[(where, text)]: the STATE decided field, then the plan's Decided and
    Rejected sections. A missing file gives nothing from it."""
    out = []
    if state.exists():
        m = re.search(r"^decided:\n(.*?)^[a-z-]+:", state.read_text(encoding="utf-8"), re.S | re.M)
        out += [("STATE decided", t) for t in _bullets(m.group(1))] if m else []
    if plan.exists():
        text = plan.read_text(encoding="utf-8")
        for head in ("Decided", "Rejected"):
            m = re.search(rf"^## {head}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
            out += [(f"plan {head}", t) for t in _bullets(m.group(1))] if m else []
    return out


# ── the paid narrative ────────────────────────────────────────────────────────

SYSTEM = """You analyse eval results for jordbot, a RAG chatbot that answers recruiters' questions about one
person's career. You get a report of numbers and a list of closed decisions.

Rules:
- Use only numbers that appear in the report. Copy each one exactly as the report prints it.
- A gap "inside the noise", or a change whose two ranges overlap, is not a change. Do not
  call it a drop or a rise, and do not base a recommendation on it, except to say more
  questions are needed to tell.
- A measure marked "rough" rests on fewer than 10 questions. Say so when you cite it.
- Do not recommend anything a closed decision rules out. If a move the report's numbers point
  to would reopen one, put it in "blocked" with the decision quoted. Leave "blocked" empty
  when no such move arises; do not list decisions the report does not touch.
- Rank recommendations by expected gain on the weakest measures, then by cost.
- Plain words, short sentences. No praise, no adjectives without a number.

Reply with JSON only:
{"narrative": "3 to 6 sentences on what moved and what did not",
 "recommendations": [{"text": "...", "evidence": ["exact figures from the report", ...]}],
 "blocked": [{"text": "...", "decision": "the closed decision, quoted"}]}"""


def prompt(report_text, decided):
    dec = "\n".join(f"- [{w}] {t}" for w, t in decided) or "(none found)"
    return f"REPORT\n{report_text}\n\nCLOSED DECISIONS\n{dec}"


def estimate(user):
    """(low, high) USD. Input is about 4 characters a token; output 300 to MAX_OUT tokens."""
    t_in = (len(SYSTEM) + len(user)) / 4
    return ((t_in * 0.8 * COST_PER_1M_IN + 300 * COST_PER_1M_OUT) / 1e6,
            (t_in * 1.3 * COST_PER_1M_IN + MAX_OUT * COST_PER_1M_OUT) / 1e6)


NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*%?")


def _norm(n):
    return n.lstrip("+").rstrip(".").replace(",", "")


def cited_ok(evidence, report_text):
    """True when the evidence cites at least one number and every number it
    cites appears in the report."""
    have = {_norm(n) for n in NUM.findall(report_text)}
    nums = [_norm(n) for e in evidence for n in NUM.findall(e)]
    return bool(nums) and all(n in have for n in nums)


def narrative(report_text, decided, client):
    """Calls the model once. Returns the checked narrative and its cost."""
    user = prompt(report_text, decided)
    resp = client.chat.completions.create(
        model=NARRATIVE_MODEL, temperature=0, max_tokens=MAX_OUT, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    u = resp.usage
    cost = round((u.prompt_tokens * COST_PER_1M_IN + u.completion_tokens * COST_PER_1M_OUT) / 1e6, 6)
    out = json.loads(resp.choices[0].message.content)
    recs = out.get("recommendations") or []
    return {"model": NARRATIVE_MODEL, "cost_usd": cost, "narrative": out.get("narrative", ""),
            "recommendations": [r for r in recs if cited_ok(r.get("evidence") or [], report_text)],
            "uncited": [r for r in recs if not cited_ok(r.get("evidence") or [], report_text)],
            "blocked": out.get("blocked") or []}


def render_narrative(n):
    o = [f"\nnarrative ({n['model']}, ${n['cost_usd']:.4f})", n["narrative"], "\nrecommendations, ranked:"]
    o += [f"  {i}. {r['text']}  [{'; '.join(r['evidence'])}]" for i, r in enumerate(n["recommendations"], 1)] \
        or ["  none with checked numbers"]
    if n["uncited"]:
        o.append("dropped, cites a number not in the report:")
        o += [f"  - {r.get('text')}  [{'; '.join(r.get('evidence') or [])}]" for r in n["uncited"]]
    if n["blocked"]:
        o.append("not proposed, would reopen a closed decision:")
        o += [f"  - {b.get('text')}  (closed: {b.get('decision')})" for b in n["blocked"]]
    return "\n".join(o)


# ── the results log ───────────────────────────────────────────────────────────

def next_id(log_text, date):
    d = date.replace("-", "")
    nums = [int(n) for n in re.findall(rf"^### R-{d}-(\d\d)\b", log_text, re.M)]
    return f"R-{d}-{max(nums, default=0) + 1:02d}"


def _rel(p):
    return p.relative_to(HERE).as_posix() if p.is_relative_to(HERE) else str(p)


def log_entry(rep, report_text, narr, entry_id, date, analysis_file):
    r, p = rep["run"], rep["previous"]
    title = f"Analyser: {r['label']}" + (f" vs {p['label']}" if p else "")
    srcs = [r["source"]] + ([p["source"]] if p else [])
    files = list(dict.fromkeys(f"`ai:{_rel(f)}`" for s in srcs for f in (resolve(s) or [])))
    numbers = [("    - " if ln.startswith("  ") else "  - ") + ln.strip()
               for ln in report_text.splitlines()[1:] if ln.strip()]
    if narr:
        decision = (f"Not decided. {narr['model']} recommendations (${narr['cost_usd']:.4f}), ranked:\n"
                    + "\n".join(f"  {i}. {x['text']} [{'; '.join(x['evidence'])}]"
                                for i, x in enumerate(narr["recommendations"], 1)))
        method_n = f" Narrative: {narr['model']}; recommendations kept only when every number they cite is in the report."
    else:
        decision, method_n = "Not decided. Narrative not run.", ""
    return "\n".join([
        f"\n### {entry_id} · {title}\n",
        f"- date: {date}",
        "- repo: ai",
        f"- question: What moved in \"{r['label']}\" ({r['date']}) against the last comparable run, and what "
        "should be optimised next?",
        f"- method: `analyse.py` (G.3). Paired bootstrap by question, 95% ranges, "
        f"{rep['setup']['resamples']:,} resamples, seed {rep['setup']['seed']}.{method_n}",
        "- numbers: all of them **[read]** from the run files by `analyse.py`.",
        *numbers,
        f"- decision: {decision}",
        "- source: " + ", ".join(["`ai:eval_history/history.jsonl`", *files,
                                  f"`ai:{_rel(analysis_file)}`"]),
        "- promotion: none",
        "- status: proposed",
    ]) + "\n"


def write_log(rep, report_text, narr, *, log=LOG, analyses=ANALYSES, date=None):
    date = date or RM.today()
    text = log.read_text(encoding="utf-8")
    entry_id = next_id(text, date)
    analyses.mkdir(parents=True, exist_ok=True)
    af = analyses / f"analysis_{date}_{rep['run']['run_id']}.json"
    af.write_text(json.dumps({"id": entry_id, "report": rep, "narrative": narr}, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    with log.open("a", encoding="utf-8", newline="\n") as f:
        f.write(log_entry(rep, report_text, narr, entry_id, date, af))
    return entry_id, af


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--run", help="run id prefix or label text (default: the last history line)")
    ap.add_argument("--estimate", action="store_true", help="price the narrative and call nothing")
    ap.add_argument("--go", action="store_true", help="call the model for the narrative (paid)")
    ap.add_argument("--log", action="store_true", help="add a proposed entry to nimbalyst-local/results-log.md")
    args = ap.parse_args()

    history = RM.read_history()
    i, _ = pick_line(history, args.run)
    rep = analyse(history, i)
    text = render(rep)
    print(text)

    decided = decided_items()
    lo, hi = estimate(prompt(text, decided))
    paid = args.go or (STANDING_BUDGET_USD is not None and hi <= STANDING_BUDGET_USD)
    print(f"\nnarrative: {len(decided)} closed decisions read; {NARRATIVE_MODEL} estimated ${lo:.4f}-${hi:.4f}")
    narr = None
    if args.estimate or not paid:
        print("narrative not run" + ("" if args.estimate else ": add --go to run it"))
    else:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        narr = narrative(text, decided, OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        print(render_narrative(narr))
    if args.log and not args.estimate:
        entry_id, af = write_log(rep, text, narr)
        print(f"\nresults log: {entry_id} added (status proposed); report saved to {_rel(af)}")


if __name__ == "__main__":
    main()
