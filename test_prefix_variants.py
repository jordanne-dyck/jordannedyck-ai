"""
test_prefix_variants.py - option 3a offline test for jordannedyck.com

WHAT THIS TESTS
---------------
scripts/embed_knowledge_faiss.py prepends a context header to every chunk before
embedding it (line 87):

    [projects] [Agentic Personal Shopper - AI Agent System] [shopper-003]

Measured over the 34-question eval set, that [category] token tracks retrieval
rank almost perfectly, and two of the four categories rank WORSE THAN RANDOM
(random = 62 of 123):

    personality   8 chunks   mean rank 28.5
    experience   39 chunks   mean rank 40.1
    projects     63 chunks   mean rank 74.8
    skills       13 chunks   mean rank 86.2

Hypothesis: every eval question asks about a PERSON. Chunks whose header says
"projects" or "skills" embed as being about an artifact, so they lose to
person-shaped chunks regardless of what words are in the body.

This script tests that hypothesis WITHOUT touching the live index.

WHY IT DOESN'T RE-CHUNK
-----------------------
documents.pkl already holds the exact text that was embedded, header included.
This script strips the header line and substitutes a new one, so chunking,
YAML stripping, weight filtering and body text are all held constant and the
PREFIX IS THE ONLY VARIABLE. Replicating the embedder would risk introducing a
second difference and invalidating the comparison.

NON-DESTRUCTIVE
---------------
Reads faiss_db/ and never writes to it. Scratch indices are built in memory.
Embeddings are cached to prefix_test_cache.json so a re-run costs nothing.
No reindex, no restart, no effect on the live site or the MCP server.

SELECTION LOGIC matches api_server.py AS PATCHED 2026-09-10:
rank by raw similarity over the full corpus, max 3 per source file, take 5.

COST: 3 variants x 123 chunks + 34 questions on text-embedding-ada-002.
Roughly $0.015 on the first run, $0 on re-runs.

USAGE (from the repo root, C:\\Users\\Jord\\jordannedyck-ai):
    venv\\Scripts\\python.exe test_prefix_variants.py
"""

import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-ada-002"
MAX_PER_SOURCE = 3
N_RESULTS = 5
CACHE_PATH = Path("prefix_test_cache.json")

# --- variants --------------------------------------------------------------
# Each takes the parsed header fields and returns the replacement header line.
# "baseline" is not re-embedded - the live index IS the baseline.

VARIANTS = {
    "B_no_prefix": lambda cat, title, cid: "",
    "C_person": lambda cat, title, cid: f"Jordanne Dyck - {title}",
    "D_person_evidence": lambda cat, title, cid: (
        f"Jordanne Dyck - {title}. What this shows about how she works."
    ),
}

# --- preflight -------------------------------------------------------------

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit(
        "FAIL: OPENAI_API_KEY not set.\n"
        "  Cause: .env not found, so you are not in the repo root.\n"
        "  Fix:   cd C:\\Users\\Jord\\jordannedyck-ai  and run again."
    )

for p in ["faiss_db/resume.index", "faiss_db/documents.pkl",
          "faiss_db/metadatas.pkl", "eval_set.json"]:
    if not Path(p).exists():
        sys.exit(
            f"FAIL: {p} not found.\n"
            "  Cause: wrong working directory.\n"
            "  Fix:   cd C:\\Users\\Jord\\jordannedyck-ai  and run again."
        )

index_live = faiss.read_index("faiss_db/resume.index")
documents = pickle.load(open("faiss_db/documents.pkl", "rb"))
metadatas = pickle.load(open("faiss_db/metadatas.pkl", "rb"))
questions = json.load(open("eval_set.json", encoding="utf-8"))["questions"]

if not (index_live.ntotal == len(documents) == len(metadatas)):
    sys.exit("FAIL: faiss_db/ is internally inconsistent. Re-run the embedder.")

print(f"corpus {len(documents)} chunks   eval set {len(questions)} questions\n")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
_new_calls = 0


def embed(text):
    """Embed with an on-disk cache so re-runs are free."""
    global _new_calls
    # md5, not hash() - Python's str hash is randomised per process, so hash()
    # would silently miss the cache on every re-run and re-bill the whole corpus.
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]
    for attempt in range(4):
        try:
            v = client.embeddings.create(
                model=EMBED_MODEL, input=text, timeout=30
            ).data[0].embedding
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"    retry {attempt+1} after {type(e).__name__}", flush=True)
            time.sleep(2 ** attempt)
    cache[key] = v
    _new_calls += 1
    return v


def short(fn):
    return str(fn).replace("\\", "/").split("/")[-1]


# --- split every document into (header, body) ------------------------------
# Header shapes produced by the embedder:
#   "[category] [title] [chunk_id]"   (files with chunk markers)
#   "[category] [title]"              (marker-less fallback, e.g. technical-skills)

HDR = re.compile(r"^\[([^\]]*)\]\s*\[([^\]]*)\](?:\s*\[([^\]]*)\])?\s*$")

bodies, headers = [], []
unparsed = 0
for d in documents:
    first, sep, rest = d.partition("\n\n")
    m = HDR.match(first.strip())
    if m:
        headers.append((m.group(1), m.group(2), m.group(3) or ""))
        bodies.append(rest)
    else:
        # No recognisable header - keep the whole doc as body, note it.
        headers.append(("", "", ""))
        bodies.append(d)
        unparsed += 1

print(f"parsed headers on {len(documents)-unparsed}/{len(documents)} chunks")
if unparsed:
    print(f"  NOTE: {unparsed} chunks had no parseable header and are passed through "
          f"unchanged in every variant (they cannot move the comparison).")
print()

# --- query embeddings, shared across all variants --------------------------

print("embedding questions (shared across variants)...")
qvecs = {}
for q in questions:
    qvecs[q["id"]] = np.array([embed(q["q"])]).astype("float32")
print(f"  {len(qvecs)} questions ready\n")


# --- evaluation ------------------------------------------------------------

def evaluate(index, label):
    """Run all questions against one index. Returns a stats dict."""
    depth = Counter()
    hits = 0
    expected_total = 0
    cat_ranks = defaultdict(list)
    per_q = {}

    for q in questions:
        distances, indices = index.search(qvecs[q["id"]], len(documents))
        cands = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(documents):
                cands.append((int(idx), float(1 / (1 + dist))))
        cands.sort(key=lambda c: -c[1])
        rank_of = {idx: r for r, (idx, _) in enumerate(cands, 1)}

        for idx, r in rank_of.items():
            cat_ranks[metadatas[idx].get("category", "?")].append(r)

        # selection, matching patched api_server.py
        sel, counts = [], Counter()
        for idx, _ in cands:
            f = short(metadatas[idx].get("filename", ""))
            counts[f] += 1
            if counts[f] <= MAX_PER_SOURCE:
                sel.append(metadatas[idx].get("chunk_id"))
            if len(sel) >= N_RESULTS:
                break

        expect = q.get("expect", [])
        expected_total += len(expect)
        q_hits = 0
        q_ranks = []
        for e in expect:
            match = [i for i in range(len(documents))
                     if metadatas[i].get("chunk_id") == e]
            if not match:
                depth["not in index"] += 1
                continue
            r = min(rank_of[i] for i in match)
            q_ranks.append(r)
            if r <= 5:
                depth["1-5"] += 1
            elif r <= 8:
                depth["6-8"] += 1
            elif r <= 30:
                depth["9-30"] += 1
            else:
                depth["31+"] += 1
            if e in sel:
                q_hits += 1
        hits += q_hits
        if expect:
            per_q[q["id"]] = (q_hits, len(expect), q_ranks)

    return {
        "label": label,
        "depth": depth,
        "hits": hits,
        "expected_total": expected_total,
        "cat_ranks": {k: mean(v) for k, v in cat_ranks.items()},
        "per_q": per_q,
    }


results = [evaluate(index_live, "A_baseline (live)")]

for name, fn in VARIANTS.items():
    print(f"embedding variant {name} ({len(documents)} chunks)...")
    vecs = []
    for (cat, title, cid), body in zip(headers, bodies):
        if not cat and not title:
            text = body                      # unparsed passthrough
        else:
            new_hdr = fn(cat, title, cid)
            text = f"{new_hdr}\n\n{body}" if new_hdr else body
        vecs.append(embed(text))
    arr = np.array(vecs).astype("float32")
    idx = faiss.IndexFlatL2(arr.shape[1])
    idx.add(arr)
    results.append(evaluate(idx, name))
    CACHE_PATH.write_text(json.dumps(cache))
    print(f"  done\n")

CACHE_PATH.write_text(json.dumps(cache))

# --- report ----------------------------------------------------------------

line = "=" * 84
print(line)
print("EXPECTED-CHUNK DEPTH BY VARIANT   (89 expected chunks; lower buckets are better)")
print(line)
print(f"{'variant':24s} {'1-5':>6s} {'6-8':>6s} {'9-30':>6s} {'31+':>6s} {'in top-5':>12s}")
for r in results:
    d = r["depth"]
    print(f"{r['label']:24s} {d['1-5']:6d} {d['6-8']:6d} {d['9-30']:6d} {d['31+']:6d} "
          f"{r['hits']:5d}/{r['expected_total']:<3d} {100*r['hits']/r['expected_total']:5.1f}%")

print()
print(line)
print("MEAN RANK BY CATEGORY   (123 chunks, random = 62.0; lower is better)")
print(line)
cats = sorted({c for r in results for c in r["cat_ranks"]})
print(f"{'variant':24s} " + " ".join(f"{c:>13s}" for c in cats))
for r in results:
    print(f"{r['label']:24s} " +
          " ".join(f"{r['cat_ranks'].get(c, float('nan')):13.1f}" for c in cats))

base = results[0]
print()
print(line)
print("PER-QUESTION MOVEMENT vs baseline   (expected chunks landing in the top 5)")
print(line)
for r in results[1:]:
    ups = [(q, base["per_q"][q][0], r["per_q"][q][0], r["per_q"][q][1])
           for q in r["per_q"] if r["per_q"][q][0] > base["per_q"][q][0]]
    downs = [(q, base["per_q"][q][0], r["per_q"][q][0], r["per_q"][q][1])
             for q in r["per_q"] if r["per_q"][q][0] < base["per_q"][q][0]]
    print(f"\n  {r['label']}   +{len(ups)} improved, -{len(downs)} regressed")
    for q, b, a, n in ups:
        print(f"     UP   {q:5s} {b} -> {a} of {n}")
    for q, b, a, n in downs:
        print(f"     DOWN {q:5s} {b} -> {a} of {n}")

print()
print(line)
print(f"{_new_calls} new embedding calls this run "
      f"(~${_new_calls * 350 * 0.10 / 1_000_000:.4f}). Cache: {CACHE_PATH}")
print("faiss_db/ was NOT modified. No reindex or restart has happened.")
print(line)
print()
print("HOW TO READ THIS:")
print("  A variant is worth a real reindex only if it moves BOTH the depth")
print("  distribution (chunks out of the 31+ bucket) AND the projects/skills")
print("  category means toward the experience/personality ones. A variant that")
print("  improves the category means while the top-5 hit rate falls is moving")
print("  the wrong chunks - reject it.")
