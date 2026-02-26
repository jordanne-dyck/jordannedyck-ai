import asyncio
import sys
import os
import pickle
import faiss
import numpy as np
from openai import OpenAI
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# Load FAISS index and documents
try:
    index = faiss.read_index(os.path.join(PROJECT_ROOT, "faiss_db", "resume.index"))
    
    with open(os.path.join(PROJECT_ROOT, "faiss_db", "documents.pkl"), "rb") as f:
        documents = pickle.load(f)
    
    with open(os.path.join(PROJECT_ROOT, "faiss_db", "metadatas.pkl"), "rb") as f:
        metadatas = pickle.load(f)
    
    print(f"Loaded {len(documents)} documents", file=sys.stderr)
except Exception as e:
    print(f"Error loading FAISS index: {e}", file=sys.stderr)
    print("Run embed_knowledge_faiss.py first.", file=sys.stderr)
    exit(1)

# Create MCP server
server = Server("jordan-resume")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="search_experience",
            description="Search through Jordan's professional experience, skills, projects, and personality. Use this to answer questions about Jordan's background, capabilities, work style, or specific projects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'AI projects', 'Python experience', 'work style')"
                    },
                    "n_results": {
                        "type": "number",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests."""
    
    try:
        print(f"=== Starting tool call ===", file=sys.stderr)
        sys.stderr.flush()
        
        if name != "search_experience":
            raise ValueError(f"Unknown tool: {name}")
        
        if not arguments or "query" not in arguments:
            raise ValueError("Missing query argument")
        
        query = arguments["query"]
        n_results = arguments.get("n_results", 5)
        
        print(f"Searching for: {query}", file=sys.stderr)
        sys.stderr.flush()
        
        # Get embedding for query
        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        print(f"Got embedding, searching index...", file=sys.stderr)
        sys.stderr.flush()
        
        # Search FAISS index
        k = min(int(n_results), 10)
        distances, indices = index.search(query_vector, k)
        
        print(f"Found {len(indices[0])} results", file=sys.stderr)
        sys.stderr.flush()
        
        # Format results
        formatted_results = []
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(documents):
                doc = documents[idx]
                metadata = metadatas[idx]
                
                # Calculate similarity score (FAISS returns L2 distance, lower is better)
                similarity = 1 / (1 + distance)
                
                result_text = f"""
### Result {i+1} (Relevance: {similarity:.2f})
**Source**: {metadata.get('filename', 'Unknown')}
**Category**: {metadata.get('category', 'Unknown')}

{doc[:500]}...

---
"""
                formatted_results.append(result_text)
        
        combined_text = "\n".join(formatted_results) if formatted_results else "No relevant information found."
        
        print(f"Returning {len(combined_text)} characters", file=sys.stderr)
        sys.stderr.flush()
        
        return [
            types.TextContent(
                type="text",
                text=combined_text
            )
        ]
    
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="jordan-resume",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())