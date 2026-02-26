import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = chromadb.PersistentClient(path="./chroma_db")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        timeout=30
    )
    return response.data[0].embedding

collection = client.get_collection("resume_knowledge")
print("Getting embedding...")
query_embedding = get_embedding("AI projects")
print("Got embedding, querying collection...")
results = collection.query(query_embeddings=[query_embedding], n_results=3)
print(f"Found {len(results['documents'][0])} results!")
for i, doc in enumerate(results['documents'][0]):
    print(f"\n--- Result {i+1} ---")
    print(doc[:200])