"""Does engine/ give the same results as the site's TypeScript, on real answers?

Runs the same inputs through both and lists every difference. The site's side
is jordannedyck-ai-web's lib/*.ts at a git ref (default origin/master, what is
live), run by Node (scripts/engine_parity.mjs). The inputs:

  - every cited answer saved in eval_runs/ (question, answer_cited, the chunk
    ids it retrieved), with each chunk's text, title and file read from the
    local index (faiss_db/). The index can be newer than the run; both sides
    get the same chunks, which is all a parity test needs.
  - every eval question, for the fixed-reply and timeline triggers.
  - every chunk in the index, for its section heading.

Compared: segments and cited ids, the sources list, the citation labels, the
project card and its prediction, the timeline-card rules, the free labels, the
streaming citation strip (each answer cut into deltas of 1 to 9 characters),
the fixed-reply triggers and the cost table. Free: no model is called.

    venv\\Scripts\\python.exe scripts\\engine_parity.py
    venv\\Scripts\\python.exe scripts\\engine_parity.py --web-ref feat/x --runs-dir ..\\eval_runs
"""

import argparse
import io
import json
import pickle
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from engine import citations, cost, fixed, labels, projects  # noqa: E402


def load_chunks(faiss_dir):
    metas = pickle.loads((faiss_dir / "metadatas.pkl").read_bytes())
    docs = pickle.loads((faiss_dir / "documents.pkl").read_bytes())
    return {m["chunk_id"]: {**m, "content": d} for m, d in zip(metas, docs)}


def load_answers(runs_dir, index):
    """(query, answer_cited, chunks) for every distinct saved answer with citations."""
    seen, out = set(), []
    for f in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rows = data if isinstance(data, list) else []
        turns = [t for r in rows if isinstance(r, dict) for t in (r.get("turns") or [r])]
        for t in turns:
            gen, query = t.get("gen") or {}, t.get("question") or t.get("q")
            raw = gen.get("answer_cited") if isinstance(gen, dict) else None
            if not raw or not query or (query, raw) in seen:
                continue
            seen.add((query, raw))
            chunks = []
            for r in t.get("retrieved") or []:
                c = index.get(r.get("chunk_id"))
                if c:
                    chunks.append({"chunk_id": c["chunk_id"], "filename": c["filename"], "title": c["title"],
                                   "content": c["content"], "similarity": r.get("similarity", 0)})
            out.append({"query": query, "answer_cited": raw, "chunks": chunks, "source": f.name})
    return out


def deltas_of(text, rng):
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 9)
        out.append(text[i:i + n])
        i += n
    return out


def python_side(fx):
    res = {"answers": [], "questions": [], "sections": [], "costs": []}
    for a in fx["answers"]:
        segments, cited = citations.parse_citations(a["answer_cited"])
        visible = fixed.without_timeline_marker(citations.strip_citations(a["answer_cited"]))
        s = citations.CitationStripper()
        out = [t for t in (s.feed(d) for d in a["deltas"]) if t]
        tail = s.finish()
        res["answers"].append({
            "segments": segments, "citedIds": cited,
            "sources": citations.build_sources(segments, a["chunks"]),
            "labels": citations.citation_labels(segments, a["chunks"]),
            "project": projects.pick_project(cited, a["chunks"]),
            "predicted": projects.predict_project(a["chunks"]),
            "citesCareer": fixed.cites_career(cited),
            "leadsWithCareer": fixed.leads_with_career(a["chunks"]),
            "freeLabels": labels.free_labels(a["query"], visible),
            "stream": {"out": out + ([tail] if tail else []), "raw": s.raw},
        })
    for q in fx["questions"]:
        res["questions"].append({
            "instructions": fixed.asks_for_instructions(q), "overview": fixed.asks_for_overview(q),
            "timeline": fixed.asks_for_timeline(q), "careerHistory": fixed.asks_for_career_history(q),
            "replay": fixed.is_timeline_replay(q, True), "freeLabelsQueryOnly": labels.free_labels(q, None),
        })
    res["sections"] = [citations.section_of(s["content"], s["title"]) for s in fx["sections"]]
    res["costs"] = [cost.estimate_cost_usd(c["model"], c["prompt"], c["completion"]) for c in fx["costs"]]
    return res


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-12
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default=str(HERE.parent / "jordannedyck-ai-web"), help="web repo checkout")
    ap.add_argument("--web-ref", default="origin/master", help="git ref of the web repo to test against")
    ap.add_argument("--runs-dir", default=str(HERE / "eval_runs"))
    ap.add_argument("--faiss-dir", default=str(HERE / "faiss_db"))
    ap.add_argument("--show", type=int, default=3, help="examples to print per differing field")
    args = ap.parse_args()

    index = load_chunks(Path(args.faiss_dir))
    rng = random.Random(20261107)
    answers = load_answers(Path(args.runs_dir), index)
    for a in answers:
        a["deltas"] = deltas_of(a["answer_cited"], rng)
    spec = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
    questions = sorted({q["q"] if "q" in q else q["question"] for q in spec["questions"] + spec["timeline_cases"]}
                       | {t["q"] for c in spec["multiturn"]["cases"] for t in c["turns"]}
                       | {a["query"] for a in answers})
    fx = {"answers": answers, "questions": questions,
          "sections": [{"content": c["content"], "title": c["title"]} for c in index.values()],
          "costs": [{"model": m, "prompt": p, "completion": c} for m in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "nope")
                    for p, c in ((0, 0), (3214, 412), (1_000_000, 1))]}
    print(f"inputs: {len(answers)} answers from {args.runs_dir}, {len(questions)} questions, "
          f"{len(index)} chunks; web {args.web_ref}")

    tmp = Path(tempfile.mkdtemp(prefix="engine-parity-"))
    try:
        tar = subprocess.run(["git", "-C", args.web, "archive", args.web_ref, "lib"],
                             capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(tmp, filter="data")
        (tmp / "fixtures.json").write_text(json.dumps(fx), encoding="utf-8")
        subprocess.run(["node", str(HERE / "scripts" / "engine_parity.mjs"), str(tmp / "lib"),
                        str(tmp / "fixtures.json"), str(tmp / "out.json")], check=True)
        site = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ours = python_side(fx)

    differ = 0
    for group, key_of in (("answers", lambda i: f"{answers[i]['source']} | {answers[i]['query'][:60]}"),
                          ("questions", lambda i: questions[i])):
        fields = list(site[group][0]) if site[group] else []
        for f in fields:
            bad = [i for i, (s, o) in enumerate(zip(site[group], ours[group])) if not same(s[f], o[f])]
            print(f"{group}.{f}: {len(site[group]) - len(bad)}/{len(site[group])} identical")
            differ += len(bad)
            for i in bad[:args.show]:
                print(f"    {key_of(i)}\n      site:   {json.dumps(site[group][i][f])[:400]}"
                      f"\n      engine: {json.dumps(ours[group][i][f])[:400]}")
    for group in ("sections", "costs"):
        bad = [i for i, (s, o) in enumerate(zip(site[group], ours[group])) if not same(s, o)]
        print(f"{group}: {len(site[group]) - len(bad)}/{len(site[group])} identical")
        differ += len(bad)
        for i in bad[:args.show]:
            print(f"    site: {site[group][i]!r}  engine: {ours[group][i]!r}")
    print("ALL IDENTICAL" if differ == 0 else f"{differ} DIFFERENCES")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
