"""Ranking for /search (api_server.py) and the MCP tool (mcp_server/server_faiss.py).
One copy, imported by both, so the two readers cannot drift apart. Plan item D.1.

Hybrid search: the FAISS vector ranking and a BM25 keyword ranking over the same
chunks, merged with reciprocal rank fusion (RRF). The top 3 slots stay the vector
ranking's; fusion fills the rest. The per-file cap applies across both.

Why: some facts are in the KB but never reach the top 5 on vectors alone
($150M+, TikTok Shop $30M+, mystery shopping, PC Express subscriptions). An exact
word or figure in the question is evidence the vectors blur; BM25 keeps it.

What the scores mean:
  - 'similarity' is still the raw vector similarity, 1 / (1 + squared L2), for
    every chunk. route.ts logs the highest one as top_similarity and the
    dashboard thresholds read it, so its meaning must not change.
  - 'score' is the fused RRF score, the sort key. The list is no longer in
    similarity order: rank 1 is not always the most similar chunk.
  - RRF uses ranks only, never raw BM25 or similarity values, so neither scale
    has to be calibrated against the other. RRF_K = 60 is the standard default
    (Cormack et al. 2009). It is not tuned on the eval set: tuning it on the
    questions it is scored on would be training on the test.

No API calls in this module. Pure Python plus numpy, so selftest runs it in CI.
It also records which embedding model an index was built with (D.3, at the end).
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

RANKER = "hybrid-lock3-bm25-rrf60"
RRF_K = 60
# The top VECTOR_LOCK slots are the vector ranking's; fusion fills the rest.
# Measured 2026-10-30 (plan D.1): with plain equal-weight RRF, a chunk sharing
# no keyword with the question could not compete, and strong vector rank-1 hits
# on abstract questions dropped out (X06, P03, W06, W07). Locking the top 3
# keeps them; BM25 can still lift an exact-figure chunk into ranks 4 and 5.
# Chosen after seeing that data, so its own result is provisional.
VECTOR_LOCK = 3
BM25_K1 = 1.5
BM25_B = 0.75

# Source diversity: at most 3 chunks per file, applied AFTER fusion.
#
# max_per_source stays at 3 on evidence, not inertia. Measured over the eval
# set (vector ranking, 2026-09): removing the cap surfaced 0 chunks that are
# otherwise never retrieved, tightening it to 2 surfaced 1. Neither is worth the
# churn, and the earlier max_per_source=2 proposal is rejected on this data.
max_per_source = 3

# Question words that carry no topic. Without this list, "How does Jordanne..."
# matches every chunk that says "how" or "does". Small on purpose.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "her", "hers", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "so", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "whom", "why", "will", "with", "would", "you", "your", "about",
    "also", "any", "just", "more", "most", "not", "only", "other", "really", "some",
    "such", "very", "jordanne", "jordanne's", "jord",
})

# Figures stay whole: "$150M+" and "150M" both become "150m", "86%" becomes
# "86%". Everything else is runs of letters and digits.
TOKEN_RE = re.compile(r"\$?(\d[\d.,]*\d|\d)\s?([mkb%])?\+?(?![a-z])|[a-z0-9]+")


def tokenize(text):
    """Lowercase tokens: figures normalised, possessive 's dropped, stopwords
    dropped, and a plain trailing -s stripped so "recommendations" matches
    "recommendation". No other stemming."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    out = []
    for m in TOKEN_RE.finditer(text):
        if m.group(1):
            tok = m.group(1).replace(",", "") + (m.group(2) or "")
        else:
            tok = m.group(0)
            if tok in STOPWORDS:
                continue
            if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
                tok = tok[:-1]
        out.append(tok)
    return out


class BM25:
    """Okapi BM25 over the chunk texts, built once at startup. The idf is the
    non-negative variant log(1 + (N - df + 0.5) / (df + 0.5)), so a term in
    most chunks counts for little but never against a chunk."""

    def __init__(self, texts, k1=BM25_K1, b=BM25_B):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(texts) else 0.0
        n = len(texts)
        df = Counter(term for c in self.tf for term in c)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def scores(self, query):
        """BM25 score of every chunk for the query, in index order."""
        out = np.zeros(len(self.tf))
        terms = set(tokenize(query))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
            s = 0.0
            for term in terms:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


def rank(index, metadatas, bm25, query, query_vector, n_results, hybrid=True):
    """Ranks the FULL corpus and returns the top n_results after the per-file
    cap: [{'idx', 'similarity', 'score', 'vector_rank', 'bm25_rank',
    'bm25_score'}, ...]. hybrid=False is the old vector-only ranking (score ==
    similarity), kept so a measurement can compare the two on one embedding.

    The whole corpus, not the top 30: at ~150 chunks ranking everything costs
    microseconds, and a cap of 30 once left 28 of 89 expected chunks
    unreachable. Revisit only if the index grows by an order of magnitude."""
    n = index.ntotal
    distances, indices = index.search(np.asarray(query_vector, dtype="float32").reshape(1, -1), n)
    similarity, vector_rank = {}, {}
    for r, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        idx = int(idx)
        if 0 <= idx < len(metadatas):
            similarity[idx] = float(1 / (1 + distance))
            vector_rank[idx] = r

    bm = bm25.scores(query)
    # Only chunks that share a term with the question get a BM25 rank. Ranking
    # the zero-score chunks would hand out RRF credit for arbitrary tie order.
    bm25_order = sorted((i for i in similarity if bm[i] > 0), key=lambda i: (-bm[i], vector_rank[i]))
    bm25_rank = {idx: r for r, idx in enumerate(bm25_order, start=1)}

    def fused(idx):
        if not hybrid:
            return similarity[idx]
        s = 1 / (RRF_K + vector_rank[idx])
        if idx in bm25_rank:
            s += 1 / (RRF_K + bm25_rank[idx])
        return s

    # Two passes, one shared per-file cap. Pass 1 takes the first VECTOR_LOCK
    # slots in vector order; pass 2 fills the rest in fused order. Ties go to the
    # better vector rank, so the order is fixed for a given input.
    by_vector = sorted(similarity, key=lambda i: vector_rank[i])
    by_fused = sorted(similarity, key=lambda i: (-fused(i), vector_rank[i]))
    results, taken, source_counts = [], set(), {}

    def take(order, limit):
        for idx in order:
            if len(results) >= limit:
                return
            source = metadatas[idx].get("filename", "")
            if idx in taken or source_counts.get(source, 0) >= max_per_source:
                continue
            source_counts[source] = source_counts.get(source, 0) + 1
            taken.add(idx)
            results.append({
                "idx": idx,
                "similarity": similarity[idx],
                "score": fused(idx),
                "vector_rank": vector_rank[idx],
                "bm25_rank": bm25_rank.get(idx),
                "bm25_score": float(bm[idx]),
            })

    if hybrid:
        take(by_vector, min(VECTOR_LOCK, n_results))
    take(by_fused, n_results)
    return results


# ─── Which embedding model an index was built with (plan D.3) ───
#
# The query must be embedded with the model that built the index. A mismatch
# does not fail: FAISS still answers, with rankings that mean nothing. So the
# indexer records the model in EMBEDDING_FILE next to the three index files,
# and both readers take the model from there instead of from a constant of
# their own. An index without the file predates D.3 and was built with
# LEGACY_EMBED_MODEL (the cluster volumes' kb-* folders, faiss_db/ before D.3).
EMBEDDING_FILE = "embedding.json"
LEGACY_EMBED_MODEL = "text-embedding-ada-002"
# What a plain reindex (scripts/reindex.ps1, no --model) builds with. README.md
# states it; selftest checks that. Changing it is a reindex and both restarts.
DEFAULT_EMBED_MODEL = LEGACY_EMBED_MODEL
# Default output size per model, from developers.openai.com/api/docs/guides/embeddings
# (checked 2026-10-30). ada-002 is fixed at 1536 and takes no `dimensions`.
DEFAULT_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
INDEX_FILES = ("resume.index", "documents.pkl", "metadatas.pkl")


def embedding_spec(model, dimensions=None):
    """{'model', 'dimensions'} for a model, checked. dimensions=None is the model's default."""
    if model not in DEFAULT_DIMENSIONS:
        raise ValueError(f"unknown embedding model {model!r}; known: {sorted(DEFAULT_DIMENSIONS)}")
    full = DEFAULT_DIMENSIONS[model]
    dimensions = dimensions or full
    if model == LEGACY_EMBED_MODEL and dimensions != full:
        raise ValueError(f"{model} cannot be shortened; it is always {full} dimensions")
    if not 1 <= dimensions <= full:
        raise ValueError(f"{model} gives at most {full} dimensions, not {dimensions}")
    return {"model": model, "dimensions": dimensions}


def embedding_request(spec):
    """Keyword arguments for openai embeddings.create. `dimensions` is sent only
    to the -3 models; ada-002 rejects the parameter."""
    if spec["model"] == LEGACY_EMBED_MODEL:
        return {"model": spec["model"]}
    return {"model": spec["model"], "dimensions": spec["dimensions"]}


def read_embedding_spec(folder):
    """The spec an index folder was built with. No file means a pre-D.3 ada-002 index."""
    path = Path(folder) / EMBEDDING_FILE
    if not path.exists():
        return embedding_spec(LEGACY_EMBED_MODEL)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return embedding_spec(raw["model"], raw.get("dimensions"))


def index_fingerprint(folder):
    """First 8 hex chars of a sha256 over the index files: the index_version that
    /search, /health and chat_events carry. EMBEDDING_FILE is hashed too when it
    exists, so the model is part of the version. Pre-D.3 folders have no such
    file and keep the fingerprint they always had."""
    digest = hashlib.sha256()
    for name in INDEX_FILES + (EMBEDDING_FILE,):
        path = Path(folder) / name
        if name == EMBEDDING_FILE and not path.exists():
            continue
        digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def check_index_matches(index_dimensions, spec, folder):
    """Refuses an index whose vectors are not the size its recorded model gives.
    Catches an index rebuilt with another model but an old or missing EMBEDDING_FILE."""
    if index_dimensions != spec["dimensions"]:
        raise SystemExit(
            f"STOP: {folder} holds {index_dimensions}-dimension vectors, but its "
            f"{EMBEDDING_FILE} says {spec['model']} at {spec['dimensions']}. The index and "
            f"the query model disagree; rebuild the index or fix {EMBEDDING_FILE}.")
