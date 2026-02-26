import os
import pickle
from pathlib import Path
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """Get embedding using OpenAI directly"""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        timeout=30
    )
    return response.data[0].embedding

# Read all markdown files from knowledge-base
knowledge_base_path = Path("knowledge-base")
documents = []
metadatas = []
embeddings = []

doc_id = 0
for md_file in knowledge_base_path.rglob("*.md"):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip empty files
    if not content.strip():
        print(f"Skipped (empty): {md_file}")
        continue
    
    # Chunk large documents
    max_length = 8000
    if len(content) > max_length:
        chunks = [content[i:i+max_length] for i in range(0, len(content), max_length)]
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            embeddings.append(get_embedding(chunk))
            metadatas.append({
                "filename": str(md_file),
                "category": md_file.parent.name,
                "chunk": idx
            })
        print(f"Added (chunked): {md_file} ({len(chunks)} chunks)")
    else:
        documents.append(content)
        embeddings.append(get_embedding(content))
        metadatas.append({
            "filename": str(md_file),
            "category": md_file.parent.name
        })
        print(f"Added: {md_file}")
    
    doc_id += 1

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# Save everything
os.makedirs("faiss_db", exist_ok=True)
faiss.write_index(index, "faiss_db/resume.index")

with open("faiss_db/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open("faiss_db/metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print(f"\n✅ Successfully embedded {len(documents)} document chunks from {doc_id} files!")
print(f"Index saved to faiss_db/")