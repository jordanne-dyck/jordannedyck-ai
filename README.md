# jordannedyck-ai

RAG-powered AI backend that lets Claude (or any LLM) answer detailed questions about my professional experience. It embeds a personal knowledge base into a FAISS vector index and exposes it via a REST API and a Claude MCP server.

## How It Works

1. **Embed** — Markdown knowledge base files are parsed, chunked (with priority metadata), and embedded using OpenAI's `text-embedding-ada-002`
2. **Search** — Queries are embedded and matched against the FAISS index with priority-boosted re-ranking and source diversity
3. **Serve** — Results are returned via a Flask API (`/search`) or directly through Claude's Model Context Protocol

## Architecture

```
knowledge-base/          # Markdown files (experience, projects, skills, personality)
        |
  embed_knowledge_faiss.py   # Chunk, embed, build FAISS index
        |
    faiss_db/            # Vector index + serialized docs
        |
   ┌────┴────┐
   |         |
api_server.py   mcp_server/server_faiss.py
 (Flask REST)     (Claude MCP tool)
```

## Tech Stack

- **Python** / **Flask** — API server
- **FAISS** — Vector similarity search
- **OpenAI API** — Embeddings (`text-embedding-ada-002`)
- **MCP** — Claude Model Context Protocol integration
- **ChromaDB** — Alternative vector store (development)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Build the FAISS index
python scripts/embed_knowledge_faiss.py

# Start the API server
python api_server.py
```

The API runs on `http://localhost:5000` with a single endpoint:

```
POST /search
Body: { "query": "string", "n_results": 5 }
```

## MCP Integration

To use with Claude, add the MCP server to your Claude config:

```json
{
  "mcpServers": {
    "jordanne-resume": {
      "command": "python",
      "args": ["mcp_server/server_faiss.py"]
    }
  }
}
```

This exposes a `search_experience` tool that Claude can call to look up information about my background.

## Evaluation

The `eval/` directory contains an evaluation suite with 20+ test cases that validate retrieval quality and response accuracy:

```bash
python eval/run_eval.py
```

## Related

- [jordannedyck-ai-web](https://github.com/jordanne-dyck/jordannedyck-ai-web) — The frontend chat interface that consumes this API
