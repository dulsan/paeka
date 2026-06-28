# =============================================================================
# PAEKA — Dockerfile  (v0.11.10)
# =============================================================================
# [FIX] CMD previously launched `uvicorn main:app --http httptools ...`
#       directly, bypassing main.py entirely. That predates this project's
#       migration to Granian (see main.py / pyproject.toml) and was never
#       updated to match -- uvicorn is no longer even a direct dependency
#       (httptools, wsproto, and websockets were all removed alongside it),
#       so the image failed at startup with ModuleNotFoundError: httptools.
#       Never caught natively because native runs always go through
#       `uv run python main.py`, which correctly uses Granian.
#
#       Fix: CMD now runs the same entrypoint native invocations use
#       (python main.py) instead of a parallel, drift-prone server launch
#       command. The original concerns that CMD was trying to address --
#       large multipart uploads not getting silently dropped, idle
#       connections surviving long ingestion, uploads visible in logs --
#       are exactly what main.py's _run_server() already needs to handle
#       correctly today, since native deployments depend on it daily; no
#       separate flags are needed here to re-solve a problem specific to
#       uvicorn's h11 implementation, which this image no longer uses.
# =============================================================================


FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml README.md ./
COPY uv.lock* ./

RUN uv sync \
    --no-dev \
    --no-install-project \
    --compile-bytecode

COPY backend/   ./backend/
COPY config/    ./config/
COPY main.py    ./

RUN uv sync \
    --no-dev \
    --compile-bytecode


FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      wget \
    && rm -rf /var/lib/apt/lists/*

# [FIX] execute_code (backend/agent/sandbox.py) shells out to `docker run`/
# `docker info`/`docker kill` -- those didn't exist in this image at all,
# so the tool only ever worked for native (non-Docker) deployments. The
# official docker:cli image ships exactly the client binary, nothing else
# (no dockerd) -- copying it in is far lighter than adding Docker's apt
# repo + GPG key just for this. paeka-api talks to the daemon over
# DOCKER_HOST (set in docker-compose.yml, pointed at the docker-socket-
# proxy sidecar, not the raw socket -- see SETUP_DOCKER.md section 4),
# so no socket file or group membership is needed here either.
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

RUN groupadd -r paeka && useradd -r -g paeka -d /app -s /sbin/nologin paeka

COPY --from=builder /app/.venv /app/.venv

COPY backend/   ./backend/
COPY config/    ./config/
COPY main.py    ./

RUN mkdir -p \
      data/uploads \
      data/documents \
      data/images \
      data/exports \
      data/hf_cache \
      database/sqlite \
      sandbox_scratch \
      models \
    && chown -R paeka:paeka /app

USER paeka

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --start-period=180s --retries=10 \
  CMD python3 -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Run the same entrypoint native invocations use -- see header above.
CMD ["python3", "main.py"]
