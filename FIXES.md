# PAEKA v0.11.1 — Fix Notes

This document explains every problem found in v0.11.1, the root cause, and what was changed.

---

## Problem 1 — `paeka-llamacpp` marked unhealthy / never starts

**Root cause (primary): missing `.gguf` file**

The healthcheck polls `GET /health` on the llama.cpp container.  That endpoint only returns `{"status":"ok"}` **after** the model file has been fully mmap'd into memory.  If the model path in `.env` points to a file that does not exist in `./models/`, the server starts but immediately exits — Docker marks it as unhealthy and `paeka-api` (which depended on `service_healthy`) never starts either.

**Fix:**
1. Download the model first: `uv run python scripts/download_model.py` or place the `.gguf` manually under `./models/qwen/`.
2. `docker-compose.yml`: changed `paeka-api` dependency to `service_started` (not `service_healthy`) so the API comes up even while the model is loading.  The API already handles an unreachable LLM gracefully with a warning log.
3. Raised `start_period` from 60s → 120s.  A 20+ GB Q4_K_M GGUF can take 60–90 s to mmap on a cold HDD.

**Root cause (secondary): CUDA-only Docker image on a machine without GPU**

The compose file pinned `ghcr.io/ggml-org/llama.cpp:server-cuda`.  On a machine without the NVIDIA Container Toolkit (or without a GPU), Docker will refuse to create the container entirely — it never becomes unhealthy, it simply does not start.

**Fix:** The default `paeka-llamacpp` service now uses `ghcr.io/ggml-org/llama.cpp:server` (CPU build, no GPU requirement).  A separate `paeka-llamacpp-gpu` service with the CUDA image is activated via `docker compose --profile gpu up`.

---

## Problem 2 — Terax cannot connect (no OpenAI-compatible endpoint)

**Root cause:** PAEKA's chat endpoint was `POST /api/conversations/{id}/chat` using Server-Sent Events with a custom JSON schema.  The openai Python library (which Terax uses) expects `POST /v1/chat/completions` with the OpenAI wire format.  These are completely different APIs.

**Fix:** Added `backend/api/routes/openai_compat.py` which implements:

- `GET  /v1/models` — returns available models in OpenAI list format
- `POST /v1/chat/completions` — accepts OpenAI request format, supports both `stream=true` (SSE chunks) and `stream=false` (full JSON)

Mounted at `/v1` in `app.py`.  Security scanning is applied identically to the regular chat route.

**Terax configuration:**
```
Base URL: https://<your-paeka-host>/v1
API Key:  <value of PAEKA_SECURITY__TOKEN, or "paeka-local" if auth disabled>
Model:    (any value — PAEKA ignores it and uses the configured GGUF)
```

---

## Problem 3 — `docker build` fails: `uv sync --frozen` with no `uv.lock`

**Root cause:** The Dockerfile ran `uv sync --frozen` but `uv.lock` was not committed to the repo (only `uv.lock*` glob with silent skip).  `--frozen` strictly requires a lock file and aborts if absent.

**Fix:** Removed `--frozen` flag.  uv will resolve fresh on first build (slower) and use the lock file when present.  **To get fast reproducible builds, run `uv lock` locally and commit `uv.lock`.**

---

## Problem 4 — SearXNG unreachable from Caddy (`/search/` returns 502)

**Root cause:** `paeka-searxng` was only attached to the `backend` network (which has `internal: true` — no external routing).  Caddy lives on the `frontend` network.  The two containers could not communicate.

**Fix:** `paeka-searxng` now joins both `frontend` and `backend` networks.  The Caddyfile also gained an explicit `handle /search/*` block that proxies to `paeka-searxng:8888`.

---

## Problem 5 — Caddyfile missing `/v1` proxy block

**Root cause:** Even after the OpenAI endpoint was added to `app.py`, Caddy had no `handle /v1/*` directive.  Requests from Terax arrived at Caddy, matched no handler, and returned a 404 before ever reaching `paeka-api`.

**Fix:** Added `handle /v1/*` before `handle /api/*` in the Caddyfile (order matters — Caddy matches top-to-bottom within a site block).

---

## Problem 6 — Python version mismatch (3.14 vs available wheels)

**Root cause:** `pyproject.toml` stated `requires-python = ">=3.12"` and you want to use Python 3.14.2.  However, as of June 2026, **no pre-built binary wheels exist for Python 3.14** for these dependencies:
- `torch` / `torchvision` (PyTorch)
- `sentence-transformers`
- `FlagEmbedding`
- `docling`

Docker build would attempt to compile them from C source, fail on missing system headers, and abort.

**Fix:** `pyproject.toml` pins `requires-python = ">=3.12,<3.13"` and the Dockerfile uses `python:3.12-slim-bookworm`.  When PyTorch and the ML stack publish 3.14 wheels, bump both files back to `>=3.14`.

---

## Problem 7 — `SEARXNG_SECRET_KEY` never injected into container

**Root cause:** `infra/searxng/settings.yml` hardcoded a placeholder key.  SearXNG refuses to start if `secret_key` is the literal string `"ultrasecretkey"` (its default); the placeholder in settings.yml was close but not exactly that string, however it was still a static value that could not be rotated without editing a tracked file.

**Fix:** Docker Compose now passes `SEARXNG_SECRET_KEY` from `.env` as an environment variable.  SearXNG reads `SEARXNG_SECRET_KEY` from the environment and uses it to override `settings.yml`.

---

## Migration checklist

```bash
# 1. Replace the five modified files:
#    docker-compose.yml
#    Dockerfile
#    pyproject.toml
#    infra/caddy/Caddyfile
#    .env.example → copy to .env if you haven't already

# 2. Add the new file:
#    backend/api/routes/openai_compat.py

# 3. Update backend/api/app.py (adds openai_compat router import + include_router)

# 4. Generate the lock file (do this once, commit it):
uv lock

# 5. Download your model if not already present:
uv run python scripts/download_model.py

# 6. Rebuild and start (CPU mode — no GPU required):
docker compose build --no-cache
docker compose up -d

# 7. GPU mode (if NVIDIA Container Toolkit is installed):
docker compose --profile gpu up -d

# 8. Verify:
curl -k https://localhost/api/health
curl -k https://localhost/v1/models
```
