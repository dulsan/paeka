"""
backend/agent/tool_graph.py
============================
Self-Healing Tool Calling pipeline built with LangGraph.

Graph topology:
                ┌──────────────────┐
                │   TOOL SELECTOR  │  LLM decides which tools to call
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  TOOL EXECUTOR   │  Runs tools, captures output + errors
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  TOOL EVALUATOR  │  Did every call succeed? Is output useful?
                └────────┬─────────┘
                         │
          has_failures? ─┼─ yes ──► REFLECTOR ──► TOOL SELECTOR (retry)
                         │ no
                ┌────────▼─────────┐
                │   SYNTHESISER    │  Compose final response from tool results
                └──────────────────┘

Self-healing behaviour:
  - Failed tool calls → Reflector analyses the error in natural language
  - Reflector produces corrected arguments or selects an alternative tool
  - Loop capped at max_retries to prevent infinite loops
  - Permanent failures (auth errors, missing tools) are not retried

Inspiration: Unsloth Studio autonomous iteration pattern + LangGraph Reflexion.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from langgraph.graph import StateGraph, END

from backend.agent.state import ToolCallingState, ToolCall, ToolResult
from backend.llm.base import LLMProvider, Message

logger = logging.getLogger(__name__)

# Errors that should NOT be retried — they won't improve with different arguments
_PERMANENT_ERROR_SIGNALS = frozenset({
    "authentication", "unauthorized", "forbidden", "not found",
    "does not exist", "permission denied", "api key",
})

_SELECTOR_PROMPT = """\
You are a tool-calling agent. Given the user request and available tools,
decide which tools to call and with what arguments.

User request: {request}

Available tools:
{tools}

{reflection_context}

Respond ONLY with a JSON array of tool calls (no markdown, no preamble):
[
  {{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}},
  ...
]

Rules:
- Call only tools listed in Available tools.
- If no tools are needed, return an empty array: []
- Arguments must match the tool's schema exactly.
- Do not repeat calls that already succeeded in this session.
"""

_REFLECTOR_PROMPT = """\
You are a self-healing agent analysing tool call failures.

User request: {request}

Failed tool calls:
{failures}

Previous reflections:
{previous_reflections}

Diagnose what went wrong and suggest corrections.
Be specific: wrong argument type, missing required field, wrong tool chosen, etc.

Respond ONLY with a JSON object:
{{
  "diagnosis": "what went wrong",
  "corrections": [
    {{"tool": "tool_name", "arguments": {{"corrected_arg": "value"}}}},
    ...
  ],
  "give_up": false
}}

Set "give_up": true if the failure is permanent (auth error, tool doesn't exist, etc.)
"""

_SYNTHESISER_PROMPT = """\
You are synthesising a final response from tool execution results.

User request: {request}

Tool results:
{results}

Write a clear, direct response that answers the user's request using the tool outputs.
If some tools failed, acknowledge it but focus on what succeeded.
"""


class SelfHealingToolGraph:
    """
    LangGraph-based self-healing tool calling pipeline.

    Parameters
    ----------
    llm:
        LLMProvider instance.
    tools:
        Dict mapping tool_name → async callable(arguments) → str
    max_retries:
        Maximum Reflector → Selector retry loops (default 3).
    max_iterations:
        Maximum total tool execution rounds (default 5).
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: dict[str, Any],
        max_retries: int = 3,
        max_iterations: int = 5,
    ) -> None:
        self._llm            = llm
        self._tools          = tools
        self._max_retries    = max_retries
        self._max_iterations = max_iterations
        self._graph          = self._build()

    async def run(self, request: str, system_prompt: str = "") -> dict[str, Any]:
        """
        Execute the self-healing tool calling loop.

        Returns
        -------
        dict with:
          final_response: str
          tool_results:   list[ToolResult]
          succeeded:      bool
          iterations:     int
          reflections:    list[str]
        """
        tool_schemas = [
            {
                "name": name,
                "description": getattr(fn, "__doc__", "").strip().split("\n")[0],
            }
            for name, fn in self._tools.items()
        ]

        initial: ToolCallingState = {
            "user_request":    request,
            "system_prompt":   system_prompt,
            "available_tools": tool_schemas,
            "tool_calls":      [],
            "tool_results":    [],
            "iteration":       0,
            "max_iterations":  self._max_iterations,
            "failed_calls":    [],
            "reflections":     [],
            "retry_count":     0,
            "max_retries":     self._max_retries,
            "final_response":  "",
            "succeeded":       False,
            "error":           None,
        }

        try:
            final: ToolCallingState = await self._graph.ainvoke(initial)
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool graph error: %s", exc)
            return {
                "final_response": f"Tool execution error: {exc}",
                "tool_results":   [],
                "succeeded":      False,
                "iterations":     0,
                "reflections":    [],
            }

        return {
            "final_response": final.get("final_response", ""),
            "tool_results":   final.get("tool_results", []),
            "succeeded":      final.get("succeeded", False),
            "iterations":     final.get("iteration", 0),
            "reflections":    final.get("reflections", []),
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self) -> StateGraph:
        llm   = self._llm
        tools = self._tools

        async def _selector(s: ToolCallingState) -> ToolCallingState:
            return await _tool_selector(s, llm)

        async def _executor(s: ToolCallingState) -> ToolCallingState:
            return await _tool_executor(s, tools)

        async def _evaluator(s: ToolCallingState) -> ToolCallingState:
            return await _tool_evaluator(s)

        async def _reflector(s: ToolCallingState) -> ToolCallingState:
            return await _tool_reflector(s, llm)

        async def _synth(s: ToolCallingState) -> ToolCallingState:
            return await _tool_synthesiser(s, llm)

        def _route_after_eval(s: ToolCallingState) -> str:
            if s.get("failed_calls") and s.get("retry_count", 0) < s.get("max_retries", 3):
                return "reflector"
            return "synthesiser"

        def _route_after_reflect(s: ToolCallingState) -> str:
            # give_up flag set by reflector on permanent failures
            if s.get("error") or s.get("iteration", 0) >= s.get("max_iterations", 5):
                return "synthesiser"
            return "selector"

        b = StateGraph(ToolCallingState)
        b.add_node("selector",    _selector)
        b.add_node("executor",    _executor)
        b.add_node("evaluator",   _evaluator)
        b.add_node("reflector",   _reflector)
        b.add_node("synthesiser", _synth)

        b.set_entry_point("selector")
        b.add_edge("selector",  "executor")
        b.add_edge("executor",  "evaluator")
        b.add_conditional_edges("evaluator", _route_after_eval,
                                {"reflector": "reflector", "synthesiser": "synthesiser"})
        b.add_conditional_edges("reflector", _route_after_reflect,
                                {"selector": "selector", "synthesiser": "synthesiser"})
        b.add_edge("synthesiser", END)

        return b.compile()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


async def _tool_selector(s: ToolCallingState, llm: LLMProvider) -> ToolCallingState:
    """LLM decides which tools to call (with correction context on retries)."""
    tools_text = "\n".join(
        f"  {t['name']}: {t.get('description', '')}"
        for t in s["available_tools"]
    )

    # Build reflection context for retries
    reflection_ctx = ""
    if s["reflections"]:
        reflection_ctx = (
            "Previous failures and corrections:\n"
            + "\n".join(f"  - {r}" for r in s["reflections"][-3:])
        )

    prompt = _SELECTOR_PROMPT.format(
        request=s["user_request"],
        tools=tools_text,
        reflection_context=reflection_ctx,
    )

    try:
        raw = await llm.complete([{"role": "user", "content": prompt}],
                                  max_tokens=512, temperature=0.1)
        calls_data = _parse_json_list(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool selector LLM error: %s", exc)
        calls_data = []

    tool_calls: list[ToolCall] = [
        ToolCall(
            tool_name=str(c.get("tool", "")),
            arguments=dict(c.get("arguments", {})),
            call_id=str(uuid.uuid4())[:8],
        )
        for c in calls_data
        if isinstance(c, dict) and c.get("tool")
    ]

    logger.info("Tool selector: %d calls planned %s",
                len(tool_calls), [c["tool_name"] for c in tool_calls])

    return {**s, "tool_calls": tool_calls, "failed_calls": []}


async def _tool_executor(s: ToolCallingState, tools: dict[str, Any]) -> ToolCallingState:
    """Execute all planned tool calls, capturing outputs and errors."""
    results: list[ToolResult] = list(s.get("tool_results", []))

    for call in s["tool_calls"]:
        name = call["tool_name"]
        tool_fn = tools.get(name)

        if tool_fn is None:
            results.append(ToolResult(
                call_id=call["call_id"],
                tool_name=name,
                output="",
                success=False,
                error=f"Tool '{name}' does not exist.",
            ))
            continue

        try:
            output = await tool_fn(**call["arguments"])
            results.append(ToolResult(
                call_id=call["call_id"],
                tool_name=name,
                output=str(output),
                success=True,
                error="",
            ))
            logger.debug("Tool '%s' succeeded: %d chars output", name, len(str(output)))
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            results.append(ToolResult(
                call_id=call["call_id"],
                tool_name=name,
                output="",
                success=False,
                error=error_msg,
            ))
            logger.warning("Tool '%s' failed: %s", name, error_msg)

    return {**s,
            "tool_results": results,
            "iteration":    s.get("iteration", 0) + 1}


async def _tool_evaluator(s: ToolCallingState) -> ToolCallingState:
    """Classify tool results as succeeded or failed."""
    all_results = s.get("tool_results", [])
    # Only evaluate results from this iteration's calls
    this_call_ids = {c["call_id"] for c in s.get("tool_calls", [])}
    failed = [
        r for r in all_results
        if not r["success"] and r["call_id"] in this_call_ids
    ]
    return {**s, "failed_calls": failed}


async def _tool_reflector(s: ToolCallingState, llm: LLMProvider) -> ToolCallingState:
    """LLM analyses failures and suggests corrections."""
    failures_text = "\n".join(
        f"  Tool: {f['tool_name']}\n  Error: {f['error']}"
        for f in s["failed_calls"]
    )
    prev_reflections = "\n".join(s.get("reflections", [])[-3:]) or "None"

    prompt = _REFLECTOR_PROMPT.format(
        request=s["user_request"],
        failures=failures_text,
        previous_reflections=prev_reflections,
    )

    try:
        raw  = await llm.complete([{"role": "user", "content": prompt}],
                                   max_tokens=512, temperature=0.1)
        data = _parse_json_obj(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reflector LLM error: %s", exc)
        data = {"diagnosis": str(exc), "corrections": [], "give_up": False}

    diagnosis   = str(data.get("diagnosis", ""))
    give_up     = bool(data.get("give_up", False))
    corrections = data.get("corrections", [])

    # Detect permanent errors without LLM explicitly saying give_up
    if any(sig in failures_text.lower() for sig in _PERMANENT_ERROR_SIGNALS):
        give_up = True

    reflections = list(s.get("reflections", []))
    reflections.append(f"[Retry {s.get('retry_count', 0) + 1}] {diagnosis}")

    # Build corrected tool calls
    corrected_calls: list[ToolCall] = [
        ToolCall(
            tool_name=str(c.get("tool", "")),
            arguments=dict(c.get("arguments", {})),
            call_id=str(uuid.uuid4())[:8],
        )
        for c in corrections
        if isinstance(c, dict) and c.get("tool")
    ]

    logger.info(
        "Reflector: give_up=%s corrections=%d diagnosis=%s",
        give_up, len(corrected_calls), diagnosis[:60],
    )

    return {
        **s,
        "reflections":  reflections,
        "tool_calls":   corrected_calls,
        "retry_count":  s.get("retry_count", 0) + 1,
        "error":        "permanent_failure" if give_up else None,
    }


async def _tool_synthesiser(s: ToolCallingState, llm: LLMProvider) -> ToolCallingState:
    """Compose final response from all tool results."""
    results = s.get("tool_results", [])
    if not results:
        return {**s, "final_response": "No tools were executed.", "succeeded": False}

    results_text = "\n".join(
        f"[{r['tool_name']}] {'✓' if r['success'] else '✗'}\n{r['output'] or r['error']}"
        for r in results
    )
    prompt = _SYNTHESISER_PROMPT.format(
        request=s["user_request"],
        results=results_text,
    )

    try:
        response = await llm.complete([{"role": "user", "content": prompt}],
                                       max_tokens=1024)
    except Exception as exc:  # noqa: BLE001
        response = f"Synthesis error: {exc}\n\nRaw results:\n{results_text}"

    succeeded = any(r["success"] for r in results)
    return {**s, "final_response": response, "succeeded": succeeded}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str) -> list:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _parse_json_obj(raw: str) -> dict:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
