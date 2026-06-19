# jordannedyck-ai — Agent Instructions

Flask backend for jordbot. Serves FAISS-powered RAG queries at `/search` (port 5000).

**Full development and deployment instructions are in the companion repo:**
`/home/peter/workdir/jordannedyck-ai-web/CLAUDE.md`

Read that file before taking any action involving builds, cluster resources, or deployment.

## This repo's role

- `api_server.py` — Flask app, the only file copied into the container image
- `Containerfile` — multi-stage UBI9/python-311 build using `uv sync --frozen`; `faiss_db/` is **not** copied in (mounted from PVC at runtime)
- `pyproject.toml` — direct Python dependencies
- `uv.lock` — full pinned dependency tree (31 packages); used by Trivy for CVE scanning and by the container build
- `requirements.txt` — kept for reference; the container build now uses `uv.lock` via `pyproject.toml`
- `scripts/embed_knowledge_faiss.py` — rebuilds the FAISS index from the knowledge base (run locally, then re-seed the cluster PVC)

## Key rules

- Do not commit `.env` or any file containing `OPENAI_API_KEY`. Run `gitleaks git --no-banner` before every push.
- After changing `pyproject.toml`, run `uv lock` to regenerate `uv.lock` and commit both files together.
- Changes to `api_server.py` or `pyproject.toml` require a Shipwright rebuild in `jordbot-dev` before testing on the cluster. **Check CI passes first** (`gh run list --repo jordanne-dyck/jordannedyck-ai --limit 1`) then see the companion CLAUDE.md for the build loop.
- The FAISS index (`faiss_db/`) is **not** in git and **not** in the image — it lives on the cluster PVC. Rebuilding the index locally and re-seeding the PVC is a separate manual step.
