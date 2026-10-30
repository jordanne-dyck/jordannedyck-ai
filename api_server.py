import hashlib
import os
import pickle
import threading
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
)

import retrieval

load_dotenv()

app = Flask(__name__)
# No CORS. Every caller of /search is server-side (route.ts, the eval runners,
# eval-candidates.mjs), so no browser needs cross-origin access. CORS(app) was
# open to all origins. Security review 2026-10-20, item 8.

# Request bounds for /search. MAX_QUERY_CHARS matches MAX_USER_MESSAGE_CHARS in
# jordannedyck-ai-web/lib/chat-input.ts. MAX_N_RESULTS leaves headroom over the
# largest caller (eval-candidates.mjs asks for 10). Each query is one paid
# embedding call, so an unbounded query is a cost path.
MAX_QUERY_CHARS = 2000
MAX_N_RESULTS = 20

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index.
#
# Paths resolve against THIS FILE, not the working directory. 2026-09-11: the
# relative form silently loaded a different faiss_db whenever the server was
# launched from anywhere but the repo root, and NOTHING in the response
# distinguishes the two - metadata['filename'] is a relative path baked in by
# the embedder, so it is byte-identical in every copy of the index. The startup
# line below is the only cheap way to see which index a running server holds.
#
# FAISS_DIR overrides the location. On the cluster it points at one immutable
# version folder on the shared index volume, e.g. /faiss/kb-bb98431, and is
# set in the deployment, so promoting an index is a PR. Unset, it is the repo's
# own faiss_db/, as before.
PROJECT_ROOT = Path(__file__).resolve().parent
FAISS_DB = Path(os.getenv("FAISS_DIR") or PROJECT_ROOT / "faiss_db")
index = faiss.read_index(str(FAISS_DB / "resume.index"))
with open(FAISS_DB / "documents.pkl", "rb") as f:
    documents = pickle.load(f)
with open(FAISS_DB / "metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)
print(f"[api_server] loaded {len(documents)} chunks from {FAISS_DB}", flush=True)
# Keyword side of the hybrid ranking, built from the same chunk texts. Rebuilt
# only on restart, like everything else loaded here.
bm25 = retrieval.BM25(documents)
# Size of the index this process loaded, not whatever is on disk now. A reindex
# is not a deployment: this only changes on restart.
INDEX_BYTES = (FAISS_DB / "resume.index").stat().st_size

# Short fingerprint of the three index files this process loaded, returned by
# /health and /search. The web app stores it on every chat_events row as
# index_version, so a quality change can be traced to a reindex. Same shape as
# the web side's PROMPT_VERSION: first 8 hex chars of a sha256.
def index_fingerprint():
    digest = hashlib.sha256()
    for name in ("resume.index", "documents.pkl", "metadatas.pkl"):
        digest.update((FAISS_DB / name).read_bytes())
    return digest.hexdigest()[:8]

INDEX_VERSION = index_fingerprint()
print(f"[api_server] index_version {INDEX_VERSION}", flush=True)

# The models this backend calls. 'model_missing' means the key works but one of
# them is no longer listed (retired, or blocked for this project). Mirrors
# REQUIRED_MODELS in jordannedyck-ai-web/lib/monitor.ts.
REQUIRED_MODELS = ("text-embedding-ada-002",)

# /health key check. GET /v1/models is a list call, so it costs no tokens.
# Cached for 60 seconds so a monitor polling /health cannot hammer OpenAI.
OPENAI_CHECK_TTL = 60
_openai_check = {"status": None, "at": 0.0}
_openai_check_lock = threading.Lock()

def check_openai():
    with _openai_check_lock:
        now = time.monotonic()
        if _openai_check["status"] and now - _openai_check["at"] < OPENAI_CHECK_TTL:
            return _openai_check["status"]
        try:
            models = openai_client.with_options(timeout=5, max_retries=0).models.list()
            listed = {m.id for m in models}
            status = "ok" if all(m in listed for m in REQUIRED_MODELS) else "model_missing"
        except (AuthenticationError, PermissionDeniedError):
            status = "auth_failed"
        except (APIConnectionError, APIStatusError):
            # Network error, timeout (a subclass of APIConnectionError), or any
            # other non-auth HTTP failure such as 429 or 5xx.
            status = "unreachable"
        _openai_check.update(status=status, at=now)
        return status

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        timeout=30
    )
    return response.data[0].embedding

# RETIRED 2026-09-10 (second revision). Kept here as documentation of what was
# removed and why - it is no longer applied to the sort.
#
# History: {1.4, 1.2, 1.0, 0.8} x embedding_weight was compressed to
# {1.06, 1.03, 1.00, 0.97} earlier the same day, on the belief that within-list
# similarities span ~6-9% and a ~3% tier gap would act as a tiebreak.
#
# Measured, not assumed (dump_candidates.py over the 34-question eval set):
#   - actual within-list similarity spread is 3.6% mean, rank 1 to rank 5
#   - so a 6% critical-vs-medium gap EXCEEDS the entire semantic range of a
#     result list, and the tag remained the sort key, not a tiebreak
#   - the compressed boost still flipped rank 1 on 41% of questions and
#     reordered the returned list on 71%
#   - it also concentrated slots in the two most critical-heavy files
#     (how-i-operate, professional-overview took 60% of all slots)
#
# Any future reintroduction must be sized against a MEASURED within-list spread,
# not a remembered one, and re-verified with dump_candidates.py.
PRIORITY_BOOST_RETIRED = {"critical": 1.06, "high": 1.03, "medium": 1.0, "low": 0.97}

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'body must be a JSON object'}), 400
    query = data.get('query', '')
    n_results = data.get('n_results', 5)
    if not isinstance(query, str) or not query.strip():
        return jsonify({'error': 'query must be a non-empty string'}), 400
    if len(query) > MAX_QUERY_CHARS:
        return jsonify({'error': f'query longer than {MAX_QUERY_CHARS} characters'}), 400
    # bool is a subclass of int, so True would otherwise pass as 1.
    if isinstance(n_results, bool) or not isinstance(n_results, int) or not 1 <= n_results <= MAX_N_RESULTS:
        return jsonify({'error': f'n_results must be an integer from 1 to {MAX_N_RESULTS}'}), 400

    # Hybrid ranking over the FULL corpus: the vector top 3, then vectors plus
    # BM25 merged with reciprocal rank fusion, at most 3 chunks per file. The logic lives in
    # retrieval.py, shared with mcp_server/server_faiss.py.
    #
    # 'similarity' is still the raw vector similarity, so route.ts's
    # top_similarity and the dashboard thresholds keep their meaning. 'score' is
    # the fused score and is the sort key, so the list is no longer in
    # similarity order. No priority boost and no embedding_weight: see the
    # PRIORITY_BOOST_RETIRED note. Both are still returned in metadata.
    query_vector = np.array([get_embedding(query)], dtype='float32')
    ranked = retrieval.rank(index, metadatas, bm25, query, query_vector, n_results)
    results = [{
        'content': documents[r['idx']],
        'metadata': metadatas[r['idx']],
        'similarity': r['similarity'],
        'score': r['score'],
        'vector_rank': r['vector_rank'],
        'bm25_rank': r['bm25_rank'],
    } for r in ranked]

    return jsonify({'results': results, 'index_version': INDEX_VERSION, 'ranker': retrieval.RANKER})

@app.route('/health', methods=['GET'])
def health():
    # Always 200 while the process is up; the web side reads 'ok', and a 404
    # means this endpoint does not exist yet. 'ok' is false whenever /search
    # would fail: an empty index or an OpenAI key that does not work.
    openai_status = check_openai()
    return jsonify({
        'ok': len(documents) > 0 and openai_status == 'ok',
        'chunks': len(documents),
        'index_bytes': INDEX_BYTES,
        'index_version': INDEX_VERSION,
        'openai': openai_status,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
