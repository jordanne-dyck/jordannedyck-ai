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

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    n_results = data.get('n_results', 5)
    
    # Get embedding and search
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype('float32')
    
    distances, indices = index.search(query_vector, min(n_results, 10))
    
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(documents):
            results.append({
                'content': documents[idx],
                'metadata': metadatas[idx],
                'similarity': float(1 / (1 + distance))
            })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(port=5000, debug=True)