"""
backend/observability/logfire_setup.py
=========================================
Phase 1, Step 1: local-only Logfire tracing.

send_to_logfire=False means no account, no network call, no data leaves
the machine -- traces are visible only in the local Logfire console/UI
(`logfire` CLI) or printed to the configured log sink. This is the
"diagnostic black box" requested before pushing real traffic through the
ReAct loop: every prompt sent, every raw completion received, every tool
call and its latency, all visible in one place.

What gets instrumented:
  - httpx: every outbound HTTP call (Ollama API, MCP client calls, SearXNG)
    automatically traced with method/URL/status/duration. This is the
    lowest-level net: even calls we didn't explicitly wrap show up.
    Requires the optional 'httpx' extra (opentelemetry-instrumentation-httpx)
    -- see pyproject.toml: logfire[httpx]>=3.0.0.
    [FIX-2] excluded_urls is NOT a parameter of instrument_httpx() at all --
    confirmed directly against opentelemetry-instrumentation-httpx's source.
    A previous round passed excluded_urls= as a kwarg; logfire's wrapper
    silently accepted it without error, but it had no effect, which is why
    the huggingface.co traffic kept appearing despite the "fix". The actual
    mechanism is the OTEL_PYTHON_HTTPX_EXCLUDED_URLS environment variable,
    read by the instrumentor at instrument()-time. Set in .env, with a
    os.environ.setdefault() here as a defensive fallback.
  - Pydantic: every model_validate() call/failure across the app.
    [FIX] record="failure" instead of the default "all". The stated purpose
    of this instrumentation is measuring tool-argument validation FAILURES
    (the concrete metric for whether native function calling reduced
    malformed tool calls vs. the old JSON-parsing approach) -- "all" was
    never the right setting for that goal. In practice "all" was also
    tracing every successful validation of LiteLLM's own internal settings
    models (KeyManagementSettings, HiddenParams, CallbackOnUI,
    RouterGeneralSettings) on every startup, which have nothing to do with
    PAEKA's tool calls and were drowning out the actual signal.
    "failure" still records metrics for all validations (so volume/rate is
    still visible) but only produces trace spans for actual failures.
  - LangGraph node execution: manual spans wrapped around agent_node and
    tool_node in react_graph.py (logfire has no built-in LangGraph
    integration, so this is done explicitly at the call sites).
  - LiteLLM completions: manual spans wrapped around acompletion_raw() in
    litellm_provider.py, capturing model, message count, tool count,
    finish_reason, and tool_calls returned.

Each instrument_*() call is wrapped in its own try/except so a missing
optional package for ONE integration can't crash the whole app on startup
(this bit a previous round: opentelemetry-instrumentation-httpx was missing
and took the entire FastAPI lifespan down with it). Each integration now
degrades independently.

Also configures the `transformers` library's own logger to ERROR level.
This is a deliberate, scoped logging-level configuration, not a
warnings.filterwarnings() suppression -- distinguishing the two matters.
transformers emits a large volume of "Accessing X from Y, returning Z
instead" lazy-import-alias notices on import, from its own internal
module-aliasing system scanning its full model registry. This is not a
deprecation warning about anything PAEKA's own code does (PAEKA never
touches those code paths directly -- they fire purely from
sentence-transformers/FlagEmbedding importing transformers at all), so
there's no root cause in our code to fix here, unlike e.g. the websockets
deprecation warning that granian's adoption genuinely fixed at the source.
Setting a third-party library's own logger level is a normal, standard
practice (the same pattern already used for LiteLLM's logger in
litellm_provider.py), not warning suppression.

Call configure_observability() exactly once, as early as possible in the
FastAPI lifespan -- before any Pydantic models are imported/used if
instrument_pydantic() is to catch validation events from app startup
onward. In practice this means calling it as step 0, before step 1 (database).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_configured = False


def configure_observability() -> bool:
    """
    Configure Logfire in local-only mode.

    Returns True if Logfire itself is active (logfire.configure() ran
    successfully), even if one or more individual instrumentations were
    skipped due to missing optional packages. Returns False only if the
    logfire package itself is not installed.
    """
    global _configured
    if _configured:
        return True

    # Scoped third-party logger level -- see module docstring for why this
    # is a normal logging-config decision, not warning suppression.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # [FIX] Same reasoning as the OTEL_PYTHON_HTTPX_EXCLUDED_URLS
    # setdefault() elsewhere in this function: primarily set in .env, but
    # set here too as a fallback in case this module runs before .env is
    # loaded for any reason. Must happen before sentence-transformers
    # imports torch/transformers (this function is step 0 of the
    # lifespan, well before the embedder loads at step 6) -- these
    # libraries read the env var at import/call time, not continuously.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    try:
        import logfire
    except ImportError:
        logger.info("logfire not installed -- observability disabled. "
                    "Install with: uv add logfire")
        return False

    send_to_cloud = bool(os.environ.get("LOGFIRE_TOKEN"))

    logfire.configure(
        send_to_logfire=send_to_cloud,
        service_name="paeka",
        console=logfire.ConsoleOptions(
            min_log_level="info",
            show_project_link=False,
        ) if not send_to_cloud else None,
    )

    # [FIX] excluded_urls= as a kwarg to instrument_httpx() does nothing --
    # confirmed directly against opentelemetry-instrumentation-httpx's source.
    # HTTPXClientInstrumentor.instrument() has no such parameter; it reads
    # exclusions from the OTEL_PYTHON_HTTPX_EXCLUDED_URLS environment
    # variable instead, checked at instrument-time. setdefault() here is
    # defense-in-depth in case .env wasn't loaded (e.g. running main.py
    # directly without start_fixed.ps1) -- the primary place this is set
    # is .env itself.
    os.environ.setdefault(
        "OTEL_PYTHON_HTTPX_EXCLUDED_URLS",
        "huggingface\\.co,raw\\.githubusercontent\\.com",
    )

    # Lowest-level net: every outbound HTTP call (Ollama, MCP, SearXNG, Qdrant
    # via qdrant-client's httpx transport where applicable).
    # Requires logfire[httpx] (opentelemetry-instrumentation-httpx). If that
    # extra wasn't installed, skip this one integration rather than crashing
    # the whole app -- everything else still works without it.
    try:
        logfire.instrument_httpx(capture_headers=False)
    except Exception as exc:
        logger.warning(
            "logfire.instrument_httpx() unavailable, skipping HTTP tracing: %s. "
            "Install with: uv sync (after adding logfire[httpx] to pyproject.toml)",
            exc,
        )

    # [FIX] record="failure": only trace actual validation failures, not
    # every successful validation across the whole process (including
    # LiteLLM's own internal settings models). Metrics are still recorded
    # for all validations -- only the trace-span volume is reduced.
    try:
        logfire.instrument_pydantic(record="failure")
    except Exception as exc:
        logger.warning("logfire.instrument_pydantic() unavailable, skipping: %s", exc)

    _configured = True
    logger.info(
        "Logfire observability active (local-only=%s, send_to_logfire=%s)",
        not send_to_cloud, send_to_cloud,
    )
    return True


def is_active() -> bool:
    return _configured
