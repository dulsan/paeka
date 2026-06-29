"""
tests/unit/test_react_graph.py
================================
Unit tests for ReActGraph.run(): history threading into initial_messages,
and the tool-call trace it now returns alongside the final text instead
of discarding. All LLM calls and MCP tool calls are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage


class _FakeLLM:
    """Returns the next AIMessage from a fixed queue on each ainvoke(),
    mirroring ChatOllama.bind_tools(...).ainvoke(messages) closely enough
    for _agent_node's purposes (it only ever uses these two calls)."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.bind_tools_calls: list[list[dict]] = []
        self.ainvoke_calls: list[list] = []

    def bind_tools(self, tools):
        self.bind_tools_calls.append(tools)
        return self

    async def ainvoke(self, messages):
        self.ainvoke_calls.append(messages)
        return self._responses.pop(0)


def _tool_call_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture(autouse=True)
def _no_real_tool_schemas(monkeypatch):
    # Every test sets its own get_tool_schemas() behaviour explicitly via
    # monkeypatch where it matters; this fixture just guarantees no test
    # accidentally reaches the network if it forgets to.
    monkeypatch.setattr(
        "backend.agent.react_graph.get_tool_schemas", AsyncMock(return_value=[])
    )


async def test_run_returns_response_with_no_tool_calls():
    from backend.agent.react_graph import ReActGraph

    llm = _FakeLLM([AIMessage(content="Hello there!")])
    graph = ReActGraph(llm=llm, max_rounds=3)

    result = await graph.run(user_message="hi")

    assert result["response"] == "Hello there!"
    assert result["tool_calls"] == []


async def test_run_executes_a_tool_and_returns_its_trace(monkeypatch):
    from backend.agent.react_graph import ReActGraph

    monkeypatch.setattr(
        "backend.agent.react_graph.get_tool_schemas",
        AsyncMock(return_value=[{"name": "calculator", "description": "adds numbers"}]),
    )
    monkeypatch.setattr(
        "backend.agent.react_graph.call_tool", AsyncMock(return_value="4")
    )

    llm = _FakeLLM(
        [
            _tool_call_message("calculator", {"expr": "2+2"}, "call_1"),
            AIMessage(content="The answer is 4."),
        ]
    )
    graph = ReActGraph(llm=llm, max_rounds=3)

    result = await graph.run(user_message="what is 2+2?")

    assert result["response"] == "The answer is 4."
    assert result["tool_calls"] == [
        {"id": "call_1", "name": "calculator", "args": {"expr": "2+2"}, "result": "4", "ok": True}
    ]
    # bind_tools() should have been handed exactly the schemas get_tool_schemas() returned
    assert llm.bind_tools_calls[0] == [{"name": "calculator", "description": "adds numbers"}]


async def test_run_marks_a_failed_tool_call_as_not_ok(monkeypatch):
    from backend.agent.react_graph import ReActGraph

    monkeypatch.setattr(
        "backend.agent.react_graph.get_tool_schemas",
        AsyncMock(return_value=[{"name": "web_search", "description": "search the web"}]),
    )
    monkeypatch.setattr(
        "backend.agent.react_graph.call_tool",
        AsyncMock(return_value="[MCP ERROR] timeout contacting search backend"),
    )

    llm = _FakeLLM(
        [
            _tool_call_message("web_search", {"query": "paeka"}, "call_2"),
            AIMessage(content="I couldn't find anything."),
        ]
    )
    graph = ReActGraph(llm=llm, max_rounds=3)

    result = await graph.run(user_message="look this up")

    assert result["tool_calls"][0]["ok"] is False
    assert result["tool_calls"][0]["result"].startswith("[MCP ERROR]")


async def test_run_threads_prior_history_into_initial_messages():
    from backend.agent.react_graph import ReActGraph

    llm = _FakeLLM([AIMessage(content="ok")])
    graph = ReActGraph(llm=llm, max_rounds=3)

    await graph.run(
        user_message="and then?",
        history=[
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "first reply"},
        ],
    )

    contents = [m.content for m in llm.ainvoke_calls[0]]
    # First message is always the system prompt -- everything after it
    # should be history in order, then the new turn last.
    assert contents[1:] == ["first turn", "first reply", "and then?"]


async def test_run_with_no_history_only_has_system_and_new_message():
    from backend.agent.react_graph import ReActGraph

    llm = _FakeLLM([AIMessage(content="ok")])
    graph = ReActGraph(llm=llm, max_rounds=3)

    await graph.run(user_message="hi")

    contents = [m.content for m in llm.ainvoke_calls[0]]
    assert contents[1:] == ["hi"]


def test_extract_tool_call_trace_pairs_calls_with_results_by_id():
    from backend.agent.react_graph import _extract_tool_call_trace
    from langchain_core.messages import ToolMessage

    messages = [
        _tool_call_message("a", {"x": 1}, "id1"),
        ToolMessage(content="result-a", tool_call_id="id1", name="a"),
        _tool_call_message("b", {"y": 2}, "id2"),
        ToolMessage(content="[BLOCKED] repeated call", tool_call_id="id2", name="b"),
    ]

    trace = _extract_tool_call_trace(messages)

    assert trace == [
        {"id": "id1", "name": "a", "args": {"x": 1}, "result": "result-a", "ok": True},
        {"id": "id2", "name": "b", "args": {"y": 2}, "result": "[BLOCKED] repeated call", "ok": False},
    ]


def test_extract_tool_call_trace_handles_a_call_with_no_matching_result():
    # Shouldn't happen in practice (every tool_call gets a ToolMessage),
    # but a partially-failed graph run is exactly the case where being
    # defensive here actually matters.
    from backend.agent.react_graph import _extract_tool_call_trace

    messages = [_tool_call_message("a", {}, "id1")]
    trace = _extract_tool_call_trace(messages)

    assert trace == [{"id": "id1", "name": "a", "args": {}, "result": "", "ok": True}]
