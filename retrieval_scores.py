"""Recall@5 and MRR per question, from a saved retrieval check. Scoring is free:
no /search call, no embedding call, no API key. Plan item A.1, the baseline every retrieval
change in Phase D is measured against.

    python retrieval_scores.py eval_runs/retrieval_check_2026-10-23_1124701d.json
    python retrieval_scores.py A.json B.json     # e.g. core set + a P01-P15 check
    python retrieval_scores.py --estimate        # what --check would call; calls nothing
    python retrieval_scores.py --check [OUT]     # PAID: fresh check via /search, then scores it

--check is the only part that costs anything (one embedding call per case on
the running api_server). Run --estimate first; --check only on Jord's go.

Expected sources are the `expect` lists in eval_set.json, read as they are. This
script never writes to eval_set.json. Write content to the questions, never to
the expect lists.

Ranks are positions in what /search returned (similarity order, at most 3 per
file), i.e. what route.ts puts in front of the model. Both scores are cut at
k=5 so a check saved at depth 10 scores the same as one saved at depth 5.

Three saved shapes are read:
  r3:      {"W01": [[chunk_id, similarity, content], ...], ...}
  10-23:   [{"id": "W01", "ids": [...], "sims": [...]}, ...]
  P01-P15: {"P01": {"q": ..., "index_version": ..., "results": [{"rank", "chunk_id", "similarity", ...}]}}
"""
import json
import pickle
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import retrieval

HERE = Path(__file__).resolve().parent
EVAL_SET = HERE / "eval_set.json"
FAISS_DB = HERE / "faiss_db"
BASE_URL = "http://localhost:5000"
CHECK_DEPTH = 10
K = 5
CASE_GROUPS = ("questions", "additive_cases", "theme_cases")


def load_check(path):
    """Saved check -> ({qid: [(chunk_id, similarity), ...] in rank order}, index_version or None)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    versions = set()
    out = {}
    if isinstance(data, list):
        for row in data:
            out[row["id"]] = list(zip(row["ids"], row["sims"]))
            if row.get("index_version"):
                versions.add(row["index_version"])
    else:
        for qid, v in data.items():
            if isinstance(v, dict):
                rows = sorted(v["results"], key=lambda r: r["rank"])
                out[qid] = [(r["chunk_id"], r["similarity"]) for r in rows]
                if v.get("index_version"):
                    versions.add(v["index_version"])
            else:
                out[qid] = [(r[0], r[1]) for r in v]
    if len(versions) > 1:
        raise ValueError(f"{path.name}: more than one index_version inside: {sorted(versions)}")
    if not versions:
        m = re.search(r"_([0-9a-f]{8})\.json$", path.name)
        if m:
            versions.add(m.group(1))
    return out, (versions.pop() if versions else None)


def score(expected, retrieved, k=K):
    """expected: chunk ids. retrieved: [(chunk_id, sim), ...] in rank order.
    Returns hits (sorted by rank), recall@k and reciprocal rank@k. recall is
    None when there is nothing expected (a gap case): it is not a zero."""
    top = [cid for cid, _ in retrieved[:k]]
    hits = [cid for cid in top if cid in set(expected)]
    if not expected:
        return {"hits": hits, "recall": None, "rr": None, "first_rank": None}
    first = next((i + 1 for i, cid in enumerate(top) if cid in set(expected)), None)
    return {"hits": hits, "recall": len(set(hits)) / len(set(expected)),
            "rr": (1 / first) if first else 0.0, "first_rank": first}


def disk_index():
    """(index_version, chunk_ids) of faiss_db/ on disk, same fingerprint as
    api_server.py. (None, None) when there is no local index, e.g. in CI."""
    if not all((FAISS_DB / n).exists() for n in retrieval.INDEX_FILES):
        return None, None
    with open(FAISS_DB / "metadatas.pkl", "rb") as f:
        ids = {m.get("chunk_id") for m in pickle.load(f)}
    return retrieval.index_fingerprint(FAISS_DB), ids


def cases(spec):
    for group in CASE_GROUPS:
        yield from spec[group]


def report(spec, check, index_ids=None, k=K, out=print):
    """Prints the per-question table and the aggregate. Returns the aggregate."""
    scored, gaps, missing = [], [], []
    for c in cases(spec):
        qid, expected = c["id"], c.get("expect") or []
        if qid not in check:
            missing.append(qid)
            continue
        retrieved = check[qid]
        s = score(expected, retrieved, k)
        rank_of = {cid: i + 1 for i, cid in enumerate(cid for cid, _ in retrieved)}
        out(f"\n{qid}  [{c.get('tag', '')}]")
        if expected:
            parts = []
            for cid in expected:
                where = f"rank {rank_of[cid]}" if cid in rank_of else "not retrieved"
                if index_ids is not None and cid not in index_ids:
                    where += ", NOT IN THE INDEX ON DISK"
                parts.append(f"{cid} ({where})")
            out("  expected:  " + ", ".join(parts))
        else:
            out("  expected:  (none: gap case, not scored)")
        out("  retrieved: " + "  ".join(
            f"{i + 1}. {cid}{'*' if cid in expected else ''} {sim:.4f}" for i, (cid, sim) in enumerate(retrieved)))
        if expected:
            n_hit, n_exp = len(set(s["hits"])), len(set(expected))
            out(f"  recall@{k} {n_hit}/{n_exp} = {s['recall']:.3f}   "
                f"RR {s['rr']:.3f}" + (f" (first hit rank {s['first_rank']})" if s["first_rank"] else " (no hit in top 5)"))
            scored.append((qid, s, n_hit, n_exp))
        else:
            gaps.append(qid)
    n = len(scored)
    agg = {
        "k": k, "n_scored": n,
        "mean_recall": sum(s["recall"] for _, s, _, _ in scored) / n if n else None,
        "mrr": sum(s["rr"] for _, s, _, _ in scored) / n if n else None,
        "micro_recall": (sum(h for *_, h, _ in scored) / sum(e for *_, e in scored)) if n else None,
        "zero_hit": [qid for qid, s, _, _ in scored if s["first_rank"] is None],
        "gap_cases": gaps, "not_in_check": missing,
    }
    out(f"\n=== aggregate over {n} cases with an expect list ===")
    if n:
        out(f"  mean recall@{k}   {agg['mean_recall']:.3f}")
        out(f"  micro recall@{k}  {agg['micro_recall']:.3f}  "
            f"({sum(h for *_, h, _ in scored)} of {sum(e for *_, e in scored)} expected chunks in a top {k})")
        out(f"  MRR@{k}           {agg['mrr']:.3f}")
    out(f"  no expected chunk in top {k} ({len(agg['zero_hit'])}): {', '.join(agg['zero_hit']) or '-'}")
    out(f"  gap cases, not scored ({len(gaps)}): {', '.join(gaps) or '-'}")
    out(f"  in eval_set.json but not in this check ({len(missing)}): {', '.join(missing) or '-'}")
    return agg


def write_check(spec, path, base_url=BASE_URL, depth=CHECK_DEPTH, disk_version=None, out=print):
    """Takes a fresh check: one /search call per single-turn case, saved in the
    P01-P15 shape with index_version and full content. Each call is one
    embedding call on the running api_server, so this costs money: run
    --estimate first and only on Jord's go. Refuses if the running server holds
    a different index than faiss_db/ on disk (a reindex is not a deployment)."""
    health = json.load(urllib.request.urlopen(f"{base_url}/health", timeout=30))
    out(f"/health: index_version {health.get('index_version')}, chunks {health.get('chunks')}")
    if disk_version and health.get("index_version") != disk_version:
        raise SystemExit(f"STOP: api_server holds index {health.get('index_version')}, faiss_db on disk is "
                         f"{disk_version}. Restart api_server (Jord) before taking a check.")
    saved = {}
    for c in cases(spec):
        req = urllib.request.Request(f"{base_url}/search",
                                     data=json.dumps({"query": c["q"], "n_results": depth}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=60))
        saved[c["id"]] = {"q": c["q"], "index_version": r["index_version"], "results": [
            {"rank": i + 1, "chunk_id": x["metadata"].get("chunk_id"), "file": x["metadata"].get("filename"),
             "similarity": round(x["similarity"], 4), "content": x["content"]}
            for i, x in enumerate(r["results"])]}
    Path(path).write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
    out(f"wrote {path}: {len(saved)} questions at depth {depth}")
    return saved


def main(args):
    sys.stdout.reconfigure(encoding="utf-8")
    if args[:1] in (["--estimate"], ["--check"]):
        spec = json.loads(EVAL_SET.read_text(encoding="utf-8"))
        n = sum(1 for _ in cases(spec))
        disk_version, _ = disk_index()
        if args[0] == "--estimate":
            print(f"--check would make {n} /search calls ({', '.join(CASE_GROUPS)}), i.e. {n} embedding "
                  f"calls on api_server, depth {CHECK_DEPTH}, against faiss_db {disk_version}. "
                  f"Multi-turn cases are not included (their rewrite is a paid LLM call). Nothing was called.")
            return
        out = args[1] if len(args) > 1 else HERE / "eval_runs" / f"retrieval_check_{date.today()}_{disk_version}.json"
        write_check(spec, out, disk_version=disk_version)
        args = [str(out)]
    paths = args
    if not paths:
        sys.exit(__doc__)
    check, versions = {}, set()
    for p in paths:
        part, v = load_check(p)
        print(f"check: {p}  index_version {v or 'not recorded'}  ({len(part)} questions)")
        versions.add(v)
        overlap = set(check) & set(part)
        if overlap:
            sys.exit(f"question ids appear in more than one check file: {sorted(overlap)}")
        check.update(part)
    if len(versions) > 1:
        sys.exit(f"check files come from different indexes: {sorted(map(str, versions))}")
    check_version = versions.pop()
    disk_version, index_ids = disk_index()
    print(f"faiss_db on disk: index_version {disk_version or 'no local index'}")
    if disk_version and check_version != disk_version:
        print(f"WARNING: this check was taken on index {check_version or '(not recorded)'}, not the "
              f"index on disk ({disk_version}). These scores describe that older index; they are not a "
              f"baseline for the current one. A fresh check costs embedding calls: Jord's go.")
    spec = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    report(spec, check, index_ids)


if __name__ == "__main__":
    main(sys.argv[1:])
