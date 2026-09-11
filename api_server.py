from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
from pathlib import Path
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

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
PROJECT_ROOT = Path(__file__).resolve().parent
FAISS_DB = PROJECT_ROOT / "faiss_db"
index = faiss.read_index(str(FAISS_DB / "resume.index"))
with open(FAISS_DB / "documents.pkl", "rb") as f:
    documents = pickle.load(f)
with open(FAISS_DB / "metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)
print(f"[api_server] loaded {len(documents)} chunks from {FAISS_DB}", flush=True)

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
    data = request.json
    query = data.get('query', '')
    n_results = data.get('n_results', 5)

    # Get embedding and rank the FULL corpus.
    #
    # fetch_count was min(max(n_results*5, 30), len(documents)), i.e. 30. Over a
    # 123-vector IndexFlatL2 that cap saved microseconds and cost reach: measured,
    # 28 of 89 chunks the eval set expects sat at rank 31+ and were therefore
    # structurally unreachable by any re-ranking. At this corpus size, rank
    # everything. Revisit only if the index grows by an order of magnitude.
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype('float32')
    fetch_count = len(documents)

    distances, indices = index.search(query_vector, fetch_count)

    # Score = raw similarity. No priority boost, no embedding_weight.
    #
    # embedding_weight stays out for the reason it always did: it is already
    # applied as a <0.3 drop filter in scripts/embed_knowledge_faiss.py, and
    # stacking it here was a second prior on top of the first.
    # context_priority now stays out too - see the PRIORITY_BOOST_RETIRED note.
    # Both are still returned in metadata, so callers can see them without the
    # sort being decided by them.
    candidates = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(documents):
            meta = metadatas[idx]
            base_similarity = float(1 / (1 + distance))
            candidates.append({
                'content': documents[idx],
                'metadata': meta,
                'similarity': base_similarity,
                # 'score' retained as a key so downstream callers do not break;
                # it is now identical to 'similarity' by design.
                'score': base_similarity,
            })

    # Rank by similarity, with source diversity (max 3 per file).
    #
    # max_per_source stays at 3 on evidence, not inertia. Measured over the eval
    # set: removing the cap surfaced 0 chunks that are otherwise never retrieved,
    # tightening it to 2 surfaced 1. Neither is worth the churn, and the earlier
    # max_per_source=2 proposal is rejected on this data.
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
    app.run(port=5000, debug=False)