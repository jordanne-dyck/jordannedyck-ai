# jordannedyck-ai — Agent Instructions

Flask backend for jordbot. Serves FAISS-powered RAG queries at `/search` (port 5000).

**Full development and deployment instructions are in the companion repo:**
`../jordannedyck-ai-web/CLAUDE.md` — the sibling checkout, next to this repo on every machine (`C:\Users\Jord\jordannedyck-ai-web\CLAUDE.md` on Jord's, `/home/peter/workdir/jordannedyck-ai-web/CLAUDE.md` on Peter's)

Read that file before taking any action involving builds, cluster resources, or deployment.

## This repo's role

- `api_server.py` — Flask app, the only file copied into the container image
- `Containerfile` — multi-stage UBI9/python-311 build using `uv sync --frozen`; `faiss_db/` is **not** copied in (mounted from PVC at runtime)
- `pyproject.toml` — direct Python dependencies
- `uv.lock` — full pinned dependency tree (30 packages); used by Trivy for CVE scanning and by the container build
- `requirements.txt` — kept for reference; the container build now uses `uv.lock` via `pyproject.toml`
- `engine/` — the answer pipeline (system prompt, fixed replies, follow-up rewrite, context layout, answer model call, citations and sources, project and timeline cards, live-answer labels, cost). The eval runners import it; `run_eval.py` and `citation_checks.py` re-export its names. The site still runs its own TypeScript copy, so the MUST STAY IN SYNC rules and the drift checks still apply. Not in the container image yet.
- `scripts/engine_parity.py` — runs `engine/` and the web repo's `lib/*.ts` (Node 22.18+) on every saved eval answer and lists every difference. Free. Run it after changing either side.
- `scripts/embed_knowledge_faiss.py` — rebuilds the FAISS index from the knowledge base (run locally, then re-seed the cluster PVC)

## Key rules

- Do not commit `.env` or any file containing `OPENAI_API_KEY`. Run `gitleaks git --no-banner` before every push.
- After changing `pyproject.toml`, run `uv lock` to regenerate `uv.lock` and commit both files together.
- Changes to `api_server.py` or `pyproject.toml` require a Shipwright rebuild in `jordbot-dev` before testing on the cluster. **Check CI passes first** (`gh run list --repo jordanne-dyck/jordannedyck-ai --limit 1`) then see the companion CLAUDE.md for the build loop.
- `README.md` is the public, hiring-manager-facing description of the build. When a change alters anything it states (eval case counts, checks, models, timeouts, ranking, stack, decisions), update the README in the same commit. `selftest.py` fails in CI when the README's case counts, models, timeouts or ranking limits no longer match the code; facts it cannot check (blind-score results, the decisions list) must be updated by hand.
- Worktrees you create go inside this repo at `.worktrees\<name>` (gitignored), never as a sibling folder in `C:\Users\Jord`: `git worktree add .worktrees\<name> -b <branch>`. Remove them with `git worktree remove` once the branch is pushed or merged. Nimbalyst's own `jordannedyck-ai_worktrees\` folder is outside this rule.
- The FAISS index (`faiss_db/`) is **not** in git and **not** in the image — it lives on the cluster PVC. Rebuilding the index locally and re-seeding the PVC is a separate manual step.
