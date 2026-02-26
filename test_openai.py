from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("Testing OpenAI API...")
try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test query",
        timeout=10
    )
    print("SUCCESS! API key works")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"ERROR: {e}")