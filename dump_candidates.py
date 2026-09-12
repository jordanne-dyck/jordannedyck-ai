"""
dump_candidates.py - P0 retrieval-reach diagnostic for jordannedyck.com

WHY THIS EXISTS
---------------
eval_results.json records only the chunks that were RETURNED (top 5 / top 8).
It cannot answer the question the P0 fix depends on: where do the 82 dark chunks
actually sit in the ranking? Rank 9, or rank 90?

That distinction picks the fix:
  - dark chunks clustered just outside the window  -> a ranking fix (drop the
    priority boost, loosen max_per_source) reaches them.
  - dark chunks scattered deep in the tail         -> ranking cannot reach them;
    the semantic signal itself is the problem (embedding model, or hybrid
    lexical retrieval), and no amount of reordering the top 8 will help.

This script embeds each eval question ONCE and searches the FULL index, so every
chunk gets a rank on every question. No generation, no gpt-4o.

COST: 34 short embedding calls on text-embedding-ada-002. Roughly $0.0001 total.

SCORING IS REPLICATED FROM api_server.py VERBATIM (as of 2026-09-10):
  base_similarity = 1 / (1 + distance)      # FAISS IndexFlatL2 -> SQUARED L2
  boosted_score   = base_similarity * PRIORITY_BOOST[context_priority]
  selection       = sort by score desc, max 3 per source file, take n_results
If api_server.py changes, change it here too or the diagnostic lies.

USAGE (from the repo root, C:\\Users\\Jord\\jordannedyck-ai):
    venv\\Scripts\\python.exe dump_candidates.py

OUTPUT:
    candidate_dump.json   - every candidate for every question, full ranking
    stdout                - the summary that answers the fix question
"""

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# --- config, kept identical to api_server.py -------------------------------

EMBED_MODEL = "text-embedding-ada-002"
PRIORITY_BOOST = {"critical": 1.06, "high": 1.03, "medium": 1.0, "low": 0.97}
MAX_PER_SOURCE = 3
N_RESULTS = 5

INDEX_PATH = Path("faiss_db/resume.index")
DOCS_PATH = Path("faiss_db/documents.pkl")
METAS_PATH = Path("faiss_db/metadatas.pkl")
EVAL_SET_PATH = Path("eval_set.json")
OUT_PATH = Path("candidate_dump.json")

# --- preflight -------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    sys.exit(
        "FAIL: OPENAI_API_KEY not set.\n"
        "  Cause: .env not found, which means you are not in the repo root.\n"
        "  Fix:   cd C:\\Users\\Jord\\jordannedyck-ai  and run again."
    )

for p in (INDEX_PATH, DOCS_PATH, METAS_PATH, EVAL_SET_PATH):
    if not p.exists():
        sys.exit(
            f"FAIL: {p} not found.\n"
            "  Cause: wrong working directory. All paths here are relative to the repo root,\n"
            "         exactly as api_server.py expects them.\n"
            "  Fix:   cd C:\\Users\\Jord\\jordannedyck-ai  and run again."
        )

index = faiss.read_index(str(INDEX_PATH))
with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)
with open(METAS_PATH, "rb") as f:
    metadatas = pickle.load(f)

# Index vintage, readable from the file itself rather than from a remembered log line.
index_bytes = INDEX_PATH.stat().st_size
implied_vectors = (index_bytes - 45) / (1536 * 4)
print("INDEX CHECK")
print(f"  resume.index      {index_bytes:,} bytes -> implies {implied_vectors:.1f} vectors")
print(f"  index.ntotal      {index.ntotal}")
print(f"  documents.pkl     {len(documents)} chunks")
print(f"  metadatas.pkl     {len(metadatas)} chunks")
if not (index.ntotal == len(documents) == len(metadatas)):
    sys.exit(
        "FAIL: index / documents / metadatas disagree on chunk count.\n"
        "  Cause: a partial or interrupted reindex left faiss_db/ inconsistent.\n"
        "  Fix:   re-run scripts/embed_knowledge_faiss.py to completion."
    )
if abs(implied_vectors - index.ntotal) > 0.5:
    print(f"  NOTE: file size implies {implied_vectors:.1f}, index reports {index.ntotal}. "
          f"Dimension may not be 1536.")
print()

with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
    eval_set = json.load(f)
questions = eval_set["questions"]
print(f"EVAL SET      {len(questions)} questions loaded from {EVAL_SET_PATH}\n")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed(text):
    r = client.embeddings.create(model=EMBED_MODEL, input=text, timeout=30)
    return r.data[0].embedding


def short(fn):
    return str(fn).replace("\\", "/").split("/")[-1]


def label(meta):
    return f"{short(meta.get('filename',''))}::{meta.get('chunk_id','?')}"


# Every chunk in the corpus, so we can name the ones that never surface.
all_labels = [label(m) for m in metadatas]

# --- run -------------------------------------------------------------------

dump = []
embed_calls = 0

for qi, q in enumerate(questions, 1):
    qid = q["id"]
    qtext = q["q"]
    expect = q.get("expect", [])

    vec = np.array([embed(qtext)]).astype("float32")
    embed_calls += 1

    # FULL index search - every chunk gets a rank on every question.
    distances, indices = index.search(vec, len(documents))

    cands = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= len(documents):
            continue
        meta = metadatas[idx]
        sim = float(1 / (1 + dist))
        prio = meta.get("context_priority", "medium")
        cands.append(
            {
                "vec_idx": int(idx),
                "label": label(meta),
                "chunk_id": meta.get("chunk_id", "?"),
                "file": short(meta.get("filename", "")),
                "priority": prio,
                "similarity": round(sim, 6),
                "score": round(sim * PRIORITY_BOOST.get(prio, 1.0), 6),
                "chars_total": len(documents[idx]),
            }
        )

    by_sim = sorted(cands, key=lambda c: -c["similarity"])
    by_score = sorted(cands, key=lambda c: -c["score"])
    for r, c in enumerate(by_sim, 1):
        c["rank_by_sim"] = r
    # keyed by vec_idx, not label: two marker-less files can both emit "section_4"
    rank_by_score = {c["vec_idx"]: r for r, c in enumerate(by_score, 1)}
    for c in cands:
        c["rank_by_score"] = rank_by_score[c["vec_idx"]]

    def select(ordered, max_per_source, n):
        out, counts = [], Counter()
        for c in ordered:
            counts[c["file"]] += 1
            if counts[c["file"]] <= max_per_source:
                out.append(c["label"])
            if len(out) >= n:
                break
        return out

    # What the route returns today, and three counterfactuals.
    sel_current = select(by_score, MAX_PER_SOURCE, N_RESULTS)          # live behaviour
    sel_nosboost = select(by_sim, MAX_PER_SOURCE, N_RESULTS)           # boost removed
    sel_nocap = select(by_score, 999, N_RESULTS)                       # cap removed
    sel_cap2 = select(by_score, 2, N_RESULTS)                          # cap tightened
    sel_nosboost_cap2 = select(by_sim, 2, N_RESULTS)                   # both

    # Where do the chunks the question was WRITTEN to retrieve actually sit?
    expect_ranks = []
    for e in expect:
        hits = [c for c in cands if c["chunk_id"] == e]
        if not hits:
            expect_ranks.append({"expect": e, "found": False})
            continue
        for h in hits:
            expect_ranks.append(
                {
                    "expect": e,
                    "found": True,
                    "label": h["label"],
                    "rank_by_sim": h["rank_by_sim"],
                    "rank_by_score": h["rank_by_score"],
                    "similarity": h["similarity"],
                    "priority": h["priority"],
                }
            )

    # The fetch window api_server actually uses today.
    fetch_count = min(max(N_RESULTS * 5, 30), len(documents))

    dump.append(
        {
            "id": qid,
            "tag": q.get("tag"),
            "question": qtext,
            "expect": expect,
            "expect_ranks": expect_ranks,
            "fetch_count_today": fetch_count,
            "selected_current": sel_current,
            "selected_no_boost": sel_nosboost,
            "selected_no_cap": sel_nocap,
            "selected_cap2": sel_cap2,
            "selected_no_boost_cap2": sel_nosboost_cap2,
            "candidates": cands,
        }
    )
    print(f"  [{qi:2d}/{len(questions)}] {qid}  ranked {len(cands)} chunks")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dump, f, indent=1)
print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)\n")

# --- summary: the questions the fix decision turns on ----------------------

line = "=" * 78

print(line)
print("1. EXPECTED-CHUNK DEPTH  - can reranking even reach them?")
print(line)
print("   For each question, where the chunks it was written to retrieve actually rank.")
print("   Ranks are over the FULL corpus by pure similarity (rank_by_sim).\n")

depth_buckets = Counter()
never_reachable = []
for d in dump:
    parts = []
    for er in d["expect_ranks"]:
        if not er["found"]:
            parts.append(f"{er['expect']}=ABSENT")
            depth_buckets["not in index"] += 1
            continue
        r = er["rank_by_sim"]
        parts.append(f"{er['expect']}=#{r}")
        if r <= 5:
            depth_buckets["1-5 (already in)"] += 1
        elif r <= 8:
            depth_buckets["6-8 (arm B reach)"] += 1
        elif r <= 30:
            depth_buckets["9-30 (in fetch window)"] += 1
        else:
            depth_buckets["31+ (outside window)"] += 1
            never_reachable.append((d["id"], er["expect"], r))
    print(f"   {d['id']:5s} {'  '.join(parts)}")

print(f"\n   DEPTH DISTRIBUTION of every expected chunk:")
for k in ["1-5 (already in)", "6-8 (arm B reach)", "9-30 (in fetch window)",
          "31+ (outside window)", "not in index"]:
    if depth_buckets[k]:
        print(f"     {k:26s} {depth_buckets[k]}")

print("\n   READ THIS AS:")
print("     mostly 9-30  -> a RANKING fix reaches them. Drop the boost / loosen the cap.")
print("     mostly 31+   -> ranking CANNOT reach them. The semantic signal is the problem:")
print("                     embedding model upgrade, or hybrid lexical retrieval.")

print()
print(line)
print("2. WHAT EACH COUNTERFACTUAL CHANGES  - vs the live top-5")
print(line)
variants = [
    ("no_boost", "selected_no_boost", "score = similarity, cap stays 3"),
    ("no_cap", "selected_no_cap", "boost stays, cap removed"),
    ("cap2", "selected_cap2", "boost stays, cap 3 -> 2"),
    ("no_boost_cap2", "selected_no_boost_cap2", "both"),
]
seen_now = set()
for d in dump:
    seen_now.update(d["selected_current"])

for name, key, desc in variants:
    changed = 0
    new_chunks = set()
    swaps = 0
    for d in dump:
        cur, alt = d["selected_current"], d[key]
        if cur != alt:
            changed += 1
        swaps += len(set(alt) - set(cur))
        new_chunks.update(set(alt) - seen_now)
    print(f"\n   {name:15s} ({desc})")
    print(f"     reorders/changes {changed}/{len(dump)} questions, {swaps} slot swaps")
    print(f"     surfaces {len(new_chunks)} chunks that are DARK today")
    if new_chunks:
        for c in sorted(new_chunks):
            print(f"       + {c}")

print()
print(line)
print("3. DARK CHUNKS  - best rank each one ever achieves")
print(line)
best = {}
for d in dump:
    for c in d["candidates"]:
        prev = best.get(c["label"])
        if prev is None or c["rank_by_sim"] < prev[0]:
            best[c["label"]] = (c["rank_by_sim"], d["id"])

dark = sorted(
    [(lbl, best[lbl][0], best[lbl][1]) for lbl in all_labels if lbl not in seen_now],
    key=lambda t: t[1],
)
print(f"   {len(dark)} of {len(all_labels)} chunks never enter the live top-5.\n")
print(f"   {'chunk':52s} {'best rank':>9s}  on")
for lbl, r, qid in dark:
    print(f"   {lbl:52s} {r:9d}  {qid}")

reachable = sum(1 for _, r, _ in dark if r <= 30)
print(f"\n   {reachable} of {len(dark)} dark chunks reach the top 30 on at least one question.")
print(f"   {len(dark)-reachable} never do - those are unreachable by ANY reranking of the")
print(f"   current fetch window, and need vocabulary or a better embedding model.")

print()
print(line)
print(f"DONE. {embed_calls} embedding calls (~${embed_calls * 25 * 0.10 / 1_000_000:.5f}).")
print(f"Full per-candidate data in {OUT_PATH}.")
print(line)
