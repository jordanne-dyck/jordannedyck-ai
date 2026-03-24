from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

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
    app.run(port=5000, debug=True)