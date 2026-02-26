import os
from pathlib import Path
import chromadb
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

# Initialize ChromaDB client with persistence (no embedding function)
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create collection (without embedding function)
try:
    collection = client.get_collection("resume_knowledge")
    client.delete_collection("resume_knowledge")
    print("Deleted existing collection")
except:
    pass

collection = client.create_collection("resume_knowledge")

# Read all markdown files from knowledge-base
knowledge_base_path = Path("knowledge-base")
documents = []
metadatas = []
ids = []
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
            ids.append(f"doc_{doc_id}_chunk_{idx}")
        print(f"Added (chunked): {md_file} ({len(chunks)} chunks)")
    else:
        documents.append(content)
        embeddings.append(get_embedding(content))
        metadatas.append({
            "filename": str(md_file),
            "category": md_file.parent.name
        })
        ids.append(f"doc_{doc_id}")
        print(f"Added: {md_file}")
    
    doc_id += 1

# Add to collection in batches
if documents:
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=batch_embeddings
        )
    print(f"\n✅ Successfully embedded {len(documents)} document chunks from {doc_id} files!")
else:
    print("⚠️ No markdown files found in knowledge-base/")