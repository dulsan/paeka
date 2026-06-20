"""
backend/agent/guardrails.py
=============================
Phase 1, Step 2: orchestration guardrails against inefficient tool use.

This is deliberately separate from backend/tools/schemas.py's Pydantic
validation. Pydantic answers "is this call well-formed?" -- it has no
concept of call history and cannot detect a perfectly valid call being
repeated pointlessly. That is an orchestration concern, not a structural
validation one, so it lives here instead of inside a Pydantic model.

Two protections, both scoped to a single ReActGraph.run() (one guard
instance per conversation turn, threaded through LangGraph config so it
persists across rounds within that run):

  1. Call memoization
     First call with a given (tool, args) pair executes normally.
     The exact same (tool, args) pair seen again returns the cached result
     immediately -- no network round-trip, no wasted Ollama generation,
     and removes the model's reason to "retry" something that already
     succeeded. A third identical attempt is treated as a stuck loop and
     blocked outright with a directive message.

  2. Circuit breaker
     If a tool fails 3 times in a row (consecutive, not cumulative -- any
     success resets the counter), it is disabled for the remainder of the
     run. Further attempts get an immediate blocked response instead of
     hitting the tool again, and the LLM is told explicitly why.

Every trip (memoized-loop-blocked, circuit-breaker-tripped) fires a
deliberate logfire event so these show up as first-class entries in the
trace, not buried inside a generic tool-call span.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CONSECUTIVE_FAILURE_LIMIT = 3
DUPLICATE_CALL_LIMIT      = 2   # 1st call executes, 2nd is memoized, 3rd+ is blocked


def _call_hash(tool_name: str, args: dict) -> str:
    canonical = tool_name + "|" + json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class GuardDecision:
    allowed: bool
    result: str | None = None       # populated when returning a cached/memoized result
    reason: str | None = None       # populated when blocked


@dataclass
class ToolCallGuard:
    """
    One instance per ReActGraph.run() call. Threaded through LangGraph's
    config["configurable"]["guard"] so state persists across agent<->tool
    rounds within a single run, then discarded when the run ends.
    """
    _call_counts:    dict[str, int] = field(default_factory=dict)
    _call_cache:     dict[str, str] = field(default_factory=dict)
    _consec_fail:    dict[str, int] = field(default_factory=dict)
    _disabled_tools: set[str]       = field(default_factory=set)

    def check(self, tool_name: str, args: dict) -> GuardDecision:
        """Call before dispatching to MCP. Decides whether the call may proceed."""
        if tool_name in self._disabled_tools:
            reason = (
                f"Tool '{tool_name}' has been disabled after "
                f"{CONSECUTIVE_FAILURE_LIMIT} consecutive failures in this run. "
                f"Do not retry it -- use a different tool or approach."
            )
            self._fire_event("circuit_breaker_blocked", tool_name=tool_name)
            return GuardDecision(allowed=False, reason=reason)

        h     = _call_hash(tool_name, args)
        count = self._call_counts.get(h, 0)

        if count == 0:
            return GuardDecision(allowed=True)

        if count == 1:
            # Exact duplicate -- return the cached result, skip re-execution.
            self._fire_event("duplicate_call_memoized", tool_name=tool_name)
            cached = self._call_cache.get(h, "(no cached result available)")
            return GuardDecision(
                allowed=True,
                result=f"[memoized -- identical call already made] {cached}",
            )

        # count >= DUPLICATE_CALL_LIMIT: this is a stuck loop, not a retry.
        reason = (
            f"You have already called '{tool_name}' with these exact arguments "
            f"{count} times in this run. Repeating it will not produce a different "
            f"result. Change your arguments or try a different tool."
        )
        self._fire_event("loop_detected_blocked", tool_name=tool_name, repeat_count=count)
        return GuardDecision(allowed=False, reason=reason)

    def record_result(self, tool_name: str, args: dict, success: bool, result: str) -> None:
        """Call after a real (non-blocked, non-memoized) tool execution completes."""
        h = _call_hash(tool_name, args)
        self._call_counts[h] = self._call_counts.get(h, 0) + 1
        if h not in self._call_cache:
            self._call_cache[h] = result

        if success:
            self._consec_fail[tool_name] = 0
            return

        fails = self._consec_fail.get(tool_name, 0) + 1
        self._consec_fail[tool_name] = fails
        if fails >= CONSECUTIVE_FAILURE_LIMIT and tool_name not in self._disabled_tools:
            self._disabled_tools.add(tool_name)
            self._fire_event(
                "circuit_breaker_tripped",
                tool_name=tool_name,
                consecutive_failures=fails,
            )

    def _fire_event(self, event_name: str, **attrs) -> None:
        """Deliberate, structured log event -- visible as a distinct entry in
        the Logfire trace rather than buried inside a generic span."""
        logger.warning("guardrail.%s: %s", event_name, attrs)
        try:
            import logfire
            logfire.info(f"guardrail.{event_name}", **attrs)
        except ImportError:
            pass
