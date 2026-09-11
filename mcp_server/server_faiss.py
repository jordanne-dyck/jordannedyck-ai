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
        
        # Rank the FULL corpus. The old cap was min(max(n_results*5, 30), N) = 30;
        # over a 123-vector IndexFlatL2 that saved microseconds and cost reach.
        # See the note in api_server.py.
        fetch_count = len(documents)
        distances, indices = index.search(query_vector, fetch_count)

        print(f"Found {len(indices[0])} candidates, re-ranking...", file=sys.stderr)
        sys.stderr.flush()

        # Rank by raw similarity. No priority boost.
        # MUST STAY IN SYNC WITH api_server.py - this is a second, independent copy
        # of the same scorer. api_server.py serves the website's chat route; this
        # serves the jordan-resume MCP tool. If they diverge, the MCP tool stops
        # being a valid diagnostic for what the site actually returns.
        #
        # The priority boost was retired 2026-09-10 after measurement showed the
        # tag was still the sort key (rank 1 flipped on 41% of eval questions),
        # because the real within-list similarity spread is 3.6%, not the 6-9%
        # the compression had been sized against. Full note in api_server.py.
        candidates = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(documents):
                similarity = 1 / (1 + distance)
                candidates.append((idx, similarity, similarity))

        candidates.sort(key=lambda x: x[2], reverse=True)
        # Source diversity: max 3 chunks per file. Kept at 3 on evidence -
        # removing it surfaced 0 otherwise-dark chunks, tightening to 2 surfaced 1.
        top = []
        source_counts = {}
        for c in candidates:
            source = metadatas[c[0]].get('filename', '')
            source_counts[source] = source_counts.get(source, 0) + 1
            if source_counts[source] <= 3:
                top.append(c)
            if len(top) >= int(n_results):
                break

        # Format results
        formatted_results = []

        for i, (idx, similarity, score) in enumerate(top):
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