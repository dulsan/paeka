"""
backend/api/routes/openai_compat.py
=====================================
OpenAI-compatible /v1/chat/completions and /v1/models endpoints.

Changes from v0.11.3:
  [FIX-A] Tool calling now works end-to-end with Terax.

          The previous implementation ignored the `tools` and `tool_choice`
          fields in the request, so the model would generate tool calls as
          raw text (e.g. {"path": "."}) instead of structured tool_calls.

          Now:
          1. `tools`, `tool_choice`, `parallel_tool_calls` are forwarded
             directly to llama.cpp which handles the function-calling grammar.
          2. Non-streaming responses parse `tool_calls` from the llama.cpp
             response and return them in proper OpenAI format.
          3. Streaming responses detect tool call deltas and forward them
             correctly so Terax can accumulate the tool call arguments.

          The tool EXECUTION still happens in Terax — PAEKA's role here is
          purely as a pass-through that formats the protocol correctly.

  [FIX-B] response_model=None on the route (FastAPI Union type fix, retained).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai-compat"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class OAIFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None
    strict: bool | None = None


class OAITool(BaseModel):
    type: str = "function"
    function: OAIFunction


class OAIToolChoice(BaseModel):
    type: str = "function"
    function: dict | None = None


class OAIMessage(BaseModel):
    role: str
    content: str | list | None = None
    name: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


class OAIChatRequest(BaseModel):
    model: str = "paeka-model"
    messages: list[OAIMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    n: int = 1
    user: str | None = None
    # Tool calling fields — forwarded directly to llama.cpp
    tools: list[OAITool] | None = None
    tool_choice: str | OAIToolChoice | None = None
    parallel_tool_calls: bool | None = None


class OAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OAIFunctionCall(BaseModel):
    name: str
    arguments: str


class OAIToolCall(BaseModel):
    id: str
    type: str = "function"
    function: OAIFunctionCall


class OAIChoice(BaseModel):
    index: int
    message: OAIMessage
    finish_reason: str = "stop"


class OAIStreamDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list | None = None


class OAIStreamChoice(BaseModel):
    index: int
    delta: OAIStreamDelta
    finish_reason: str | None = None


class OAIChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage = Field(default_factory=OAIUsage)


class OAIStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[OAIStreamChoice]


class OAIModelEntry(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "paeka"


class OAIModelList(BaseModel):
    object: str = "list"
    data: list[OAIModelEntry]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_provider_messages(messages: list[OAIMessage]) -> list[dict]:
    """Convert OAI message objects to plain dicts for llama.cpp."""
    out = []
    for m in messages:
        msg: dict = {"role": m.role}
        if m.content is not None:
            msg["content"] = m.content
        if m.tool_calls:
            msg["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        out.append(msg)
    return out


def _build_payload(body: OAIChatRequest, messages: list[dict],
                   stream: bool, settings) -> dict:
    """Build the JSON payload for llama.cpp /v1/chat/completions."""
    payload: dict[str, Any] = {
        "model":    body.model or settings.llm.model or "paeka-model",
        "messages": messages,
        "stream":   stream,
    }
    if body.temperature is not None:
        payload["temperature"] = body.temperature
    if body.top_p is not None:
        payload["top_p"] = body.top_p
    if body.max_tokens is not None:
        payload["max_tokens"] = body.max_tokens
    if body.stop is not None:
        payload["stop"] = body.stop
    # FIX-A: forward tool calling fields
    if body.tools:
        payload["tools"] = [t.model_dump(exclude_none=True) for t in body.tools]
    if body.tool_choice is not None:
        payload["tool_choice"] = (
            body.tool_choice if isinstance(body.tool_choice, str)
            else body.tool_choice.model_dump(exclude_none=True)
        )
    if body.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = body.parallel_tool_calls
    return payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/models", response_model=OAIModelList)
async def list_models(request: Request) -> OAIModelList:
    from backend.shared.config import get_settings
    settings = get_settings()
    llm = request.app.state.llm
    try:
        backend_models = await llm.list_models()
    except Exception:  # noqa: BLE001
        backend_models = []
    if not backend_models:
        backend_models = [settings.llm.model or "paeka-model"]
    return OAIModelList(data=[OAIModelEntry(id=m) for m in backend_models])


@router.post("/chat/completions", response_model=None)
async def chat_completions(body: OAIChatRequest, request: Request) -> Response:
    """
    OpenAI-compatible chat completions with full tool calling support.

    Forwards requests directly to llama.cpp which handles:
    - Function calling grammar / constrained generation
    - tool_calls in the response
    - parallel tool calls

    Terax (or any openai-library client) connects here and the tool
    EXECUTION happens in Terax. PAEKA just formats the protocol.
    """
    from backend.shared.config import get_settings
    settings = get_settings()

    scanner = request.app.state.scanner
    messages = _to_provider_messages(body.messages)

    if not messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Scan last user message
    user_text = next(
        (m.get("content", "") for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        ""
    )
    if user_text:
        scan = scanner.scan_input(user_text, source="user")
        if scan.is_blocked:
            raise HTTPException(
                status_code=400,
                detail=f"Message blocked: {scan.findings[0]}",
            )

    model_id      = body.model or settings.llm.model or "paeka-model"
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_ts    = int(time.time())

    # Build URL directly to llama.cpp — bypass our LLMProvider wrapper since
    # we need the raw response to extract tool_calls
    llama_url = settings.llm.base_url.rstrip("/") + "/chat/completions"

    # [FIX] Previously always sent Authorization: Bearer {settings.llm.api_key},
    # which becomes the literal string "Bearer None" when no key is configured
    # (the common case for a local Ollama setup -- Ollama doesn't require one).
    # Harmless against Ollama in practice (it ignores Authorization entirely),
    # but a real bug waiting to surface against any backend that actually
    # checks the header. Only send it if a real key is configured.
    headers = {"Content-Type": "application/json"}
    if settings.llm.api_key:
        headers["Authorization"] = f"Bearer {settings.llm.api_key}"

    # ── Streaming ─────────────────────────────────────────────────────────
    if body.stream:
        payload = _build_payload(body, messages, stream=True, settings=settings)

        async def _stream():
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(
                    connect=10.0, read=float(settings.llm.request_timeout),
                    write=30.0, pool=5.0
                )) as client:
                    async with client.stream("POST", llama_url,
                                             json=payload, headers=headers) as resp:
                        # [FIX] raise_for_status()'s default exception string
                        # is just "Client error '404 Not Found' for url
                        # '...'" -- it does not include the response body.
                        # For a streaming response the body is exactly what
                        # tells us WHY the backend rejected the request (e.g.
                        # an Ollama/llama.cpp error explaining the model
                        # doesn't support the requested feature). Read and
                        # log it explicitly before raising, since
                        # raise_for_status() consumes the response state.
                        if resp.is_error:
                            error_body = await resp.aread()
                            logger.error(
                                "openai_compat stream error: backend returned %d for %s -- body: %s",
                                resp.status_code, llama_url, error_body.decode(errors="replace")[:1000],
                            )
                            resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                    break
                                # Pass chunks through unchanged — llama.cpp
                                # already formats them as OpenAI stream chunks
                                # including tool_call deltas
                                yield f"data: {data_str}\n\n"
            except httpx.HTTPStatusError as exc:
                # Body and status already logged with full detail above,
                # right before raise_for_status() raised this. Don't log
                # the same failure twice with less detail the second time.
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.error("openai_compat stream error: %s", exc)
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering":"no",
                "Connection":       "keep-alive",
            },
        )

    # ── Non-streaming ─────────────────────────────────────────────────────
    payload = _build_payload(body, messages, stream=False, settings=settings)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(
            connect=10.0, read=float(settings.llm.request_timeout),
            write=30.0, pool=5.0
        )) as client:
            resp = await client.post(llama_url, json=payload, headers=headers)
            # [FIX] Same reasoning as the streaming path above -- capture the
            # actual response body before raise_for_status() raises, since
            # the exception's default string form discards it entirely.
            if resp.is_error:
                logger.error(
                    "openai_compat HTTP error: backend returned %d for %s -- body: %s",
                    resp.status_code, llama_url, resp.text[:1000],
                )
            resp.raise_for_status()
            llama_resp = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("openai_compat HTTP error: %s", exc)
        # Surface the backend's actual error body to the caller (Terax) too,
        # not just the bare status line -- this is the detail that actually
        # explains *why* the request was rejected.
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("openai_compat error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # FIX-A: Pass the llama.cpp response through with tool_calls intact.
    # Override id/created for consistency but keep all other fields.
    llama_resp["id"]      = completion_id
    llama_resp["created"] = created_ts

    # Scan text content of the response (not tool call arguments)
    for choice in llama_resp.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        if content:
            out_scan = scanner.scan_output(content)
            msg["content"] = out_scan.sanitised_text

    return JSONResponse(content=llama_resp)
