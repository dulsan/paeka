# =============================================================================
# PAEKA — Dockerfile  (v0.11.9)
# =============================================================================
# Changes from v0.11.4:
#
# [FIX-A] Switched uvicorn HTTP implementation from h11 to httptools.
#
#         h11 (uvicorn's default) has a hard 65536-byte (64 KB) limit on
#         individual HTTP header values, and more critically imposes a
#         DEFAULT_MAX_INCOMPLETE_EVENT_SIZE of 16 KB on the request body
#         buffer. For multipart file uploads this causes uvicorn to silently
#         drop the connection mid-upload when the form body exceeds the buffer,
#         with zero log output — the route handler is never called.
#
#         httptools has no such limit and handles large multipart bodies
#         correctly. It is already installed as part of uvicorn[standard].
#
# [FIX-B] Added --timeout-keep-alive 75 to prevent idle connection drops
#         during long ingestion operations where the client is waiting.
#
# [FIX-C] Removed --no-access-log flag so uploads are visible in logs.
#
# Everything else unchanged from v0.11.4.
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
      database/weaviate \
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

# FIX-A: --http httptools — handles large multipart uploads correctly
# FIX-B: --timeout-keep-alive 75 — keeps connection alive during ingestion
CMD ["uvicorn", "main:app", \
     "--host",               "0.0.0.0", \
     "--port",               "8000", \
     "--workers",            "1", \
     "--http",               "httptools", \
     "--timeout-keep-alive", "75", \
     "--log-level",          "info"]
