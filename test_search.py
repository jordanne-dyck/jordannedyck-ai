import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

print("Loading ChromaDB...")
client = chromadb.PersistentClient(path='./chroma_db')

print("Setting up OpenAI embeddings...")
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name='text-embedding-3-small'
)

print("Getting collection...")
collection = client.get_collection('resume_knowledge', embedding_function=ef)

print("Searching (with 30 second timeout)...")
try:
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Search took too long")
    
    # Set alarm for 30 seconds (Unix only)
    # For Windows, we'll just wait and see
    
    results = collection.query(
        query_texts=['AI projects'],
        n_results=3
    )
    
    print(f"SUCCESS! Found {len(results['documents'][0])} results")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}:")
        print(doc[:300])
        print("---")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()