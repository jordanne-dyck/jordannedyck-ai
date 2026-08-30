from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import time
from collections import defaultdict, deque
from threading import Lock
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Restrict cross-origin browser access. The frontend calls this API
# server-to-server (not from browser JS), so the default is to allow no
# origins; set ALLOWED_ORIGINS (comma-separated) to opt specific origins in.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
CORS(app, resources={r"/search": {"origins": ALLOWED_ORIGINS}})

# Optional shared-secret auth. Unset by default so existing deployments keep
# working; set BACKEND_API_KEY to require a matching X-API-Key header.
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

# In-process per-IP rate limit (sliding window) to cap billed OpenAI calls.
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_lock = Lock()
_request_log = defaultdict(deque)


def _is_authorized(req):
    if not BACKEND_API_KEY:
        return True
    return req.headers.get('X-API-Key') == BACKEND_API_KEY


def _is_rate_limited(client_ip):
    now = time.monotonic()
    with _rate_limit_lock:
        window = _request_log[client_ip]
        while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= RATE_LIMIT_MAX_REQUESTS:
            return True
        window.append(now)
        return False

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index
index = faiss.read_index("faiss_db/resume.index")
with open("faiss_db/documents.pkl", "rb") as f:
    documents = pickle.load(f)
with open("faiss_db/metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        timeout=30
    )
    return response.data[0].embedding

PRIORITY_BOOST = {"critical": 1.4, "high": 1.2, "medium": 1.0, "low": 0.8}

@app.route('/search', methods=['POST'])
def search():
    if not _is_authorized(request):
        return jsonify({'error': 'Unauthorized'}), 401

    if _is_rate_limited(request.remote_addr or 'unknown'):
        return jsonify({'error': 'Too many requests'}), 429

    data = request.json
    query = data.get('query', '')
    n_results = data.get('n_results', 5)

    # Get embedding and search - over-fetch for re-ranking (min 30 candidates)
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype('float32')
    fetch_count = min(max(n_results * 5, 30), len(documents))

    distances, indices = index.search(query_vector, fetch_count)

    # Score with metadata boost: similarity * priority_boost * embedding_weight
    candidates = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(documents):
            meta = metadatas[idx]
            base_similarity = float(1 / (1 + distance))
            priority = meta.get("context_priority", "medium")
            weight = meta.get("embedding_weight", 1.0)
            boosted_score = base_similarity * PRIORITY_BOOST.get(priority, 1.0) * weight
            candidates.append({
                'content': documents[idx],
                'metadata': meta,
                'similarity': base_similarity,
                'score': boosted_score,
            })

    # Re-rank by boosted score with source diversity (max 3 per file, ensures mix)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    results = []
    source_counts = {}
    max_per_source = 3
    for c in candidates:
        source = c['metadata'].get('filename', '')
        source_counts[source] = source_counts.get(source, 0) + 1
        if source_counts[source] <= max_per_source:
            results.append(c)
        if len(results) >= n_results:
            break

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)