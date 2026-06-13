"""
tests/unit/test_tool_graph.py
==============================
Unit tests for the self-healing tool calling graph.
All LLM calls and tool executions are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agent.state import ToolCallingState


def _base_state(**overrides) -> ToolCallingState:
    s: ToolCallingState = {
        "user_request":    "Get the weather in London",
        "system_prompt":   "You are PAEKA.",
        "available_tools": [{"name": "get_weather", "description": "Get weather for a city"}],
        "tool_calls":      [],
        "tool_results":    [],
        "iteration":       0,
        "max_iterations":  5,
        "failed_calls":    [],
        "reflections":     [],
        "retry_count":     0,
        "max_retries":     3,
        "final_response":  "",
        "succeeded":       False,
        "error":           None,
    }
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# Tool selector
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_selector_plans_tool_calls():
    from backend.agent.tool_graph import _tool_selector

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='[{"tool": "get_weather", "arguments": {"city": "London"}}]')

    state = _base_state()
    result = await _tool_selector(state, llm)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool_name"] == "get_weather"
    assert result["tool_calls"][0]["arguments"]["city"] == "London"


@pytest.mark.anyio
async def test_selector_handles_empty_response():
    from backend.agent.tool_graph import _tool_selector

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="[]")

    state = _base_state()
    result = await _tool_selector(state, llm)
    assert result["tool_calls"] == []


@pytest.mark.anyio
async def test_selector_handles_malformed_json():
    from backend.agent.tool_graph import _tool_selector

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="I cannot determine which tools to use.")

    state = _base_state()
    result = await _tool_selector(state, llm)
    assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_executor_success():
    from backend.agent.tool_graph import _tool_executor
    from backend.agent.state import ToolCall

    async def mock_weather(city: str) -> str:
        return f"Weather in {city}: 15°C, cloudy"

    tools = {"get_weather": mock_weather}
    call = ToolCall(tool_name="get_weather", arguments={"city": "London"}, call_id="abc1")
    state = _base_state(tool_calls=[call])

    result = await _tool_executor(state, tools)

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["success"] is True
    assert "London" in result["tool_results"][0]["output"]


@pytest.mark.anyio
async def test_executor_tool_raises_exception():
    from backend.agent.tool_graph import _tool_executor
    from backend.agent.state import ToolCall

    async def failing_tool(query: str) -> str:
        raise ValueError("API rate limit exceeded")

    tools = {"search": failing_tool}
    call = ToolCall(tool_name="search", arguments={"query": "test"}, call_id="abc2")
    state = _base_state(tool_calls=[call])

    result = await _tool_executor(state, tools)

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["success"] is False
    assert "rate limit" in result["tool_results"][0]["error"]


@pytest.mark.anyio
async def test_executor_unknown_tool():
    from backend.agent.tool_graph import _tool_executor
    from backend.agent.state import ToolCall

    tools = {}  # empty — no tools registered
    call = ToolCall(tool_name="nonexistent", arguments={}, call_id="abc3")
    state = _base_state(tool_calls=[call])

    result = await _tool_executor(state, tools)

    assert result["tool_results"][0]["success"] is False
    assert "does not exist" in result["tool_results"][0]["error"]


# ---------------------------------------------------------------------------
# Tool evaluator
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_evaluator_identifies_failures():
    from backend.agent.tool_graph import _tool_evaluator
    from backend.agent.state import ToolResult, ToolCall

    call = ToolCall(tool_name="get_weather", arguments={}, call_id="xyz")
    failed_result = ToolResult(
        call_id="xyz", tool_name="get_weather",
        output="", success=False, error="Timeout"
    )
    state = _base_state(tool_calls=[call], tool_results=[failed_result])
    result = await _tool_evaluator(state)

    assert len(result["failed_calls"]) == 1


@pytest.mark.anyio
async def test_evaluator_passes_clean_results():
    from backend.agent.tool_graph import _tool_evaluator
    from backend.agent.state import ToolResult, ToolCall

    call = ToolCall(tool_name="get_weather", arguments={}, call_id="xyz")
    good_result = ToolResult(
        call_id="xyz", tool_name="get_weather",
        output="15°C", success=True, error=""
    )
    state = _base_state(tool_calls=[call], tool_results=[good_result])
    result = await _tool_evaluator(state)

    assert result["failed_calls"] == []


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reflector_produces_corrections():
    from backend.agent.tool_graph import _tool_reflector
    from backend.agent.state import ToolResult

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='''{
        "diagnosis": "Wrong argument name 'location' should be 'city'",
        "corrections": [{"tool": "get_weather", "arguments": {"city": "London"}}],
        "give_up": false
    }''')

    failed = ToolResult(
        call_id="x", tool_name="get_weather",
        output="", success=False,
        error="TypeError: got unexpected keyword argument 'location'"
    )
    state = _base_state(failed_calls=[failed], retry_count=0)
    result = await _tool_reflector(state, llm)

    assert len(result["reflections"]) == 1
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["arguments"]["city"] == "London"
    assert result["error"] is None


@pytest.mark.anyio
async def test_reflector_gives_up_on_permanent_error():
    from backend.agent.tool_graph import _tool_reflector
    from backend.agent.state import ToolResult

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='''{
        "diagnosis": "Authentication failed — invalid API key",
        "corrections": [],
        "give_up": true
    }''')

    failed = ToolResult(
        call_id="x", tool_name="get_weather",
        output="", success=False, error="Unauthorized: invalid API key"
    )
    state = _base_state(failed_calls=[failed])
    result = await _tool_reflector(state, llm)

    assert result["error"] == "permanent_failure"


# ---------------------------------------------------------------------------
# Full pipeline integration (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_full_pipeline_success():
    from backend.agent.tool_graph import SelfHealingToolGraph

    llm = MagicMock()
    # Selector returns a call
    # Synthesiser composes response
    call_responses = [
        '[{"tool": "add", "arguments": {"a": 1, "b": 2}}]',  # selector
        "The result of 1 + 2 is 3.",                          # synthesiser
    ]
    llm.complete = AsyncMock(side_effect=call_responses)

    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    graph = SelfHealingToolGraph(llm=llm, tools={"add": add}, max_retries=1)
    result = await graph.run("What is 1 + 2?")

    assert result["succeeded"] is True
    assert "3" in result["final_response"]


@pytest.mark.anyio
async def test_full_pipeline_self_heals():
    """Pipeline should retry and succeed after one failure."""
    from backend.agent.tool_graph import SelfHealingToolGraph

    llm = MagicMock()
    responses = [
        # Round 1: wrong arg name
        '[{"tool": "divide", "arguments": {"numerator": 10, "denominator": 2}}]',
        # Reflector diagnoses and corrects
        '{"diagnosis": "wrong args", "corrections": [{"tool": "divide", "arguments": {"a": 10, "b": 2}}], "give_up": false}',
        # Round 2: correct args
        '[{"tool": "divide", "arguments": {"a": 10, "b": 2}}]',
        # Synthesiser
        "10 divided by 2 is 5.",
    ]
    llm.complete = AsyncMock(side_effect=responses)

    call_count = {"n": 0}

    async def divide(a: float, b: float) -> str:
        """Divide a by b."""
        call_count["n"] += 1
        return str(a / b)

    graph = SelfHealingToolGraph(llm=llm, tools={"divide": divide}, max_retries=3)
    result = await graph.run("What is 10 / 2?")

    assert result["succeeded"] is True
