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

# The ranking is shared with api_server.py through retrieval.py in the repo root,
# so the MCP tool stays a valid diagnostic for what the site returns.
sys.path.insert(0, PROJECT_ROOT)
import retrieval  # noqa: E402

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """Embeds the query with the model the loaded index was built with (EMBEDDING)."""
    response = openai_client.embeddings.create(
        **retrieval.embedding_request(EMBEDDING),
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
    
    bm25 = retrieval.BM25(documents)

    # Same rule as api_server.py: the query model comes from the index folder.
    EMBEDDING = retrieval.read_embedding_spec(os.path.join(PROJECT_ROOT, "faiss_db"))
    retrieval.check_index_matches(index.d, EMBEDDING, os.path.join(PROJECT_ROOT, "faiss_db"))

    print(f"Loaded {len(documents)} documents, {EMBEDDING['model']}", file=sys.stderr)
except Exception as e:
    print(f"Error loading FAISS index: {e}", file=sys.stderr)
    print("Run embed_knowledge_faiss.py first.", file=sys.stderr)
    exit(1)

# Create MCP server
server = Server("jordanne-resume")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="search_experience",
            description="Search through Jordanne's professional experience, skills, projects, and personality. Use this to answer questions about Jordanne's background, capabilities, work style, or specific projects.",
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
        print("=== Starting tool call ===", file=sys.stderr)
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
        query_vector = np.array([get_embedding(query)], dtype='float32')

        print("Got embedding, ranking...", file=sys.stderr)
        sys.stderr.flush()

        # Hybrid ranking (vector top 3, then vectors plus BM25 by RRF, max 3 per file) from
        # retrieval.py, the same code api_server.py's /search runs.
        top = retrieval.rank(index, metadatas, bm25, query, query_vector, int(n_results))

        # Format results
        formatted_results = []

        for i, r in enumerate(top):
            idx, similarity = r["idx"], r["similarity"]
            doc = documents[idx]
            metadata = metadatas[idx]

            result_text = f"""
### Result {i+1} (Relevance: {similarity:.2f})
**Source**: {metadata.get('filename', 'Unknown')}
**Category**: {metadata.get('category', 'Unknown')}

{doc}

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
                server_name="jordanne-resume",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())