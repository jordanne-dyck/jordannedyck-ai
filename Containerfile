# Stage 1: builder — installs gcc and compiles Python packages into a venv
FROM registry.access.redhat.com/ubi9/python-311 AS builder

USER root
RUN dnf install -y gcc python3-devel && dnf clean all

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN UV_PROJECT_ENVIRONMENT=/venv uv sync --frozen --no-dev --no-install-project \
    --python python3.11 --compile-bytecode


# Stage 2: runtime — lean image with no build tools
FROM registry.access.redhat.com/ubi9/python-311

WORKDIR /opt/app-root/src

# Copy the entire venv from builder
COPY --from=builder /venv /venv

# Copy only the application entrypoint, the ranking module and the answer engine it imports
COPY api_server.py retrieval.py ./
COPY engine ./engine

# faiss_db/ is NOT copied here — it is mounted from a PVC at /opt/app-root/src/faiss_db at runtime
# api_server.py uses relative path "faiss_db/resume.index" so WORKDIR must stay /opt/app-root/src

EXPOSE 5000

# gunicorn, not the Flask development server: /chat streams each answer for up to
# 60 s, so requests must run side by side. One worker holds one copy of the index;
# its threads serve requests concurrently. --timeout is the worker heartbeat, not
# a per-request limit.
CMD ["/venv/bin/gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "gthread", "--threads", "8", "--timeout", "120", "api_server:app"]
