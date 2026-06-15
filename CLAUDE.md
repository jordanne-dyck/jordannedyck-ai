# jordannedyck-ai — Agent Instructions

Flask backend for jordbot. Serves FAISS-powered RAG queries at `/search` (port 5000).

**Full development and deployment instructions are in the companion repo:**
`/home/peter/workdir/jordannedyck-ai-web/CLAUDE.md`

Read that file before taking any action involving builds, cluster resources, or deployment.

## This repo's role

- `api_server.py` — Flask app, the only file copied into the container image
- `Containerfile` — multi-stage UBI9/python-311 build; `faiss_db/` is **not** copied in (mounted from PVC at runtime)
- `requirements.txt` — Python dependencies
- `scripts/embed_knowledge_faiss.py` — rebuilds the FAISS index from the knowledge base (run locally, then re-seed the cluster PVC)

## Key rules

- Do not commit `.env` or any file containing `OPENAI_API_KEY`. Run `gitleaks git --no-banner` before every push.
- Changes to `api_server.py` or `requirements.txt` require a Shipwright rebuild in `jordbot-dev` before testing on the cluster. See the companion CLAUDE.md for the build loop.
- The FAISS index (`faiss_db/`) is **not** in git and **not** in the image — it lives on the cluster PVC. Rebuilding the index locally and re-seeding the PVC is a separate manual step.
