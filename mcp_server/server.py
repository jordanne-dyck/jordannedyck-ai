import asyncio
import sys
import os
from typing import Any
import chromadb
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

# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, "chroma_db"))

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

# Get the collection
try:
    collection = client.get_collection("resume_knowledge")
    print(f"Collection loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"Error: resume_knowledge collection not found: {e}", file=sys.stderr)
    print("Run embed_knowledge.py first.", file=sys.stderr)
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
        
        if name != "search_experience":
            raise ValueError(f"Unknown tool: {name}")
        
        if not arguments or "query" not in arguments:
            raise ValueError("Missing query argument")
        
        query = arguments["query"]
        n_results = arguments.get("n_results", 5)
        
        print(f"Searching for: {query}", file=sys.stderr)
        print(f"Getting embedding...", file=sys.stderr)
        sys.stderr.flush()
        
        # Get embedding for query
        query_embedding = get_embedding(query)
        
        print(f"Got embedding, querying collection...", file=sys.stderr)
        sys.stderr.flush()
        
        # Search the collection  
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(int(n_results), 10)
            )
        except Exception as query_error:
            print(f"Query error: {query_error}", file=sys.stderr)
            sys.stderr.flush()
            raise
        
        print(f"Found {len(results['documents'][0]) if results['documents'] else 0} results", file=sys.stderr)
        print(f"Formatting results...", file=sys.stderr)
        sys.stderr.flush()
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                result_text = f"""
### Result {i+1} (Relevance: {1 - distance:.2f})
**Source**: {metadata.get('filename', 'Unknown')}
**Category**: {metadata.get('category', 'Unknown')}

{doc[:500]}...

---
"""
                formatted_results.append(result_text)
            
            combined_text = "\n".join(formatted_results)
        else:
            combined_text = "No relevant information found in Jordan's knowledge base."
        
        print(f"Returning {len(combined_text)} characters", file=sys.stderr)
        sys.stderr.flush()
        
        return [
            types.TextContent(
                type="text",
                text=combined_text
            )
        ]
    
    except Exception as e:
        print(f"ERROR in handle_call_tool: {str(e)}", file=sys.stderr)
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