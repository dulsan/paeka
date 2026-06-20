"""
backend/agent/react_graph.py
=============================
ReAct tool-calling loop using langchain-ollama's ChatOllama directly.

Two nodes: agent_node <-> tool_node, looping until no tool_calls are
returned or max_rounds is hit.

[MIGRATION] LiteLLM removed. Confirmed before making this change:
  - ChatOllama.bind_tools() accepts OpenAI-format tool dicts directly
    (the exact shape backend/mcp/client.py's get_tool_schemas() already
    produces) -- LangChain's internal convert_to_openai_tool() utility
    used by all bind_tools() implementations passes through dicts already
    in that shape, no conversion needed on our end.
  - ChatOllama.ainvoke() returns a LangChain AIMessage whose .tool_calls
    is ALREADY a list of {"name", "args", "id", "type": "tool_call"}
    dicts -- confirmed directly against LangChain's own docs. This is the
    exact shape _tool_node already consumes, so _tool_node needed NO
    changes at all -- only _agent_node's internals and the provider type
    changed.
  - This removes an entire conversion step that used to live here: no more
    manual json.loads() of tool_call.function.arguments, no more building
    OpenAI-dict messages via _to_openai_dict() (now dead code, removed).
    LangChain message objects are passed straight to ChatOllama.

[KNOWN RISK -- read before assuming this "just works"] Multiple dated
GitHub issues (langchain-ai/langchain #28781, #30271, #26335) report real
tool-calling reliability problems with ChatOllama specifically -- ranging
from "calls a tool on every input regardless of relevance" to "doesn't
pick up tool calls at all" depending on model/version/time period. I
can't verify which (if any) apply to your installed langchain-ollama
version or to paeka-qwen's specific chat template. Test this against
POST /api/agent/react immediately after deploying this change -- if tool
calls aren't firing, check `langchain_ollama` version first (pin to a
recent release if old), then check whether paeka-qwen's Modelfile
TEMPLATE actually declares tool-calling support.

[FIX] from __future__ import annotations removed. This project runs on
Python 3.14, where annotations are lazily evaluated to real objects by
default (PEP 649) -- the future-import opts back into the OLDER PEP 563
behaviour (eager stringification: every annotation becomes a plain str at
runtime instead). LangGraph's add_node() does runtime introspection on
node signatures expecting a real RunnableConfig type object; with the
future-import active it was very likely receiving the literal string
"RunnableConfig | None" instead, producing the confusing
"should be typed as RunnableConfig | None, not RunnableConfig | None"
warning (comparing a string against a type that look identical when
printed, but aren't the same object). Python 3.14 supports `X | Y` union
syntax natively without this import, so nothing else in this file needed
to change to remove it.

Phase 1 observability (unchanged from before): every agent_node call and
every tool_node round wrapped in a logfire span via the _span() helper,
guardrails (ToolCallGuard) unchanged.
"""

import asyncio
import logging
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, END

from backend.agent.guardrails import ToolCallGuard
from backend.mcp.client import call_tool, get_tool_schemas

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS: int = 24_000
KEEP_RECENT: int       = 6
MAX_ROUNDS: int         = 10

_SYSTEM_PROMPT = """\
You are PAEKA, a local AI engineering and knowledge assistant running on the user's machine.

You have access to tools for searching the local knowledge base, executing code, \
searching the web, and managing files. Use them proactively when they would help \
answer the question accurately.

Guidelines:
- Call tools when you need information you don't already have.
- You can call multiple tools in parallel in a single turn.
- After receiving tool results, synthesise them into a clear response.
- For code: always test in the sandbox before presenting it to the user.
- Cite document sources (filename, page) when drawing from the knowledge base.
- If a tool fails, try an alternative approach or explain the limitation clearly.
- If a tool tells you it has been disabled or that you are repeating a call,
  stop using that tool and change your approach -- do not retry it.
"""


# ---------------------------------------------------------------------------
# Logfire helper: a real span when logfire is installed, a harmless no-op
# otherwise.
# ---------------------------------------------------------------------------

class _NoOpSpan:
    def set_attribute(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *_args) -> bool:
        return False


def _span(name: str, **attrs: Any):
    try:
        import logfire
        return logfire.span(name, **attrs)
    except ImportError:
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# ReActGraph
# ---------------------------------------------------------------------------

class ReActGraph:
    """Minimal two-node ReAct graph: agent_node <-> tool_node."""

    def __init__(
        self,
        llm: ChatOllama,
        mcp_url: str = "http://localhost:8000/mcp/",
        system_prompt: str = _SYSTEM_PROMPT,
        max_rounds: int = MAX_ROUNDS,
    ) -> None:
        self._llm           = llm
        self._mcp_url       = mcp_url
        self._system_prompt = system_prompt
        self._max_rounds    = max_rounds
        self._graph         = self._build()

    async def run(
        self,
        user_message: str,
        conversation_memory=None,
    ) -> str:
        tool_schemas = await get_tool_schemas(mcp_url=self._mcp_url)

        system_content = self._system_prompt
        if conversation_memory is not None:
            try:
                past = await conversation_memory.get_context(
                    query=user_message, include_retrieved=True, retrieved_limit=3
                )
                for msg in past:
                    if msg.get("role") == "system":
                        system_content = self._system_prompt + "\n\n" + msg["content"]
                        break
            except Exception as exc:
                logger.warning("ConversationMemory.get_context() failed: %s", exc)

        initial_messages: list[BaseMessage] = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ]

        # One guard instance per run, threaded through config so it persists
        # across agent<->tool rounds within this single conversation turn.
        guard = ToolCallGuard()

        config = {
            "configurable": {
                "tool_schemas": tool_schemas,
                "mcp_url":      self._mcp_url,
                "max_rounds":   self._max_rounds,
                "llm":          self._llm,
                "guard":        guard,
            }
        }

        with _span("react_graph.run", user_message_chars=len(user_message)):
            result = await self._graph.ainvoke({"messages": initial_messages}, config=config)

        final = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                final = msg.content
                break

        if conversation_memory is not None and final:
            try:
                await conversation_memory.add_turn("user",      user_message)
                await conversation_memory.add_turn("assistant", final)
            except Exception as exc:
                logger.warning("ConversationMemory.add_turn() failed: %s", exc)

        return final

    def _build(self) -> Any:
        g = StateGraph(MessagesState)
        g.add_node("agent", _agent_node)
        g.add_node("tools", _tool_node)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile()


# ---------------------------------------------------------------------------
# Node: agent
# ---------------------------------------------------------------------------

async def _agent_node(state: MessagesState, config: RunnableConfig | None = None) -> dict:
    cfg          = (config or {}).get("configurable", {})
    tool_schemas = cfg.get("tool_schemas", [])
    messages     = _trim_messages(state["messages"])

    round_num = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)

    with _span("react_graph.agent_node", round=round_num, message_count=len(messages)) as span:
        try:
            llm: ChatOllama = _get_llm_from_config(cfg)
            # bind_tools() accepts our OpenAI-format tool dicts directly --
            # confirmed against LangChain's bind_tools() implementation,
            # which passes already-OpenAI-shaped dicts through unchanged.
            bound = llm.bind_tools(tool_schemas) if tool_schemas else llm
            ai_msg: AIMessage = await bound.ainvoke(messages)
        except Exception as exc:
            logger.error("agent_node LLM call failed: %s", exc)
            span.set_attribute("error", str(exc))
            return {"messages": [AIMessage(content=f"I encountered an error: {exc}")]}

        # done_reason is Ollama's native field name (confirmed via
        # LangChain's own ChatOllama docs example response_metadata),
        # analogous to OpenAI's finish_reason but not identical wording.
        done_reason = (ai_msg.response_metadata or {}).get("done_reason", "unknown")
        tool_calls  = ai_msg.tool_calls or []

        span.set_attribute("done_reason", done_reason)
        span.set_attribute("tool_calls_returned", len(tool_calls))

        logger.info("agent_node: round=%d done_reason=%s tool_calls=%d",
                    round_num, done_reason, len(tool_calls))

        return {"messages": [ai_msg]}


# ---------------------------------------------------------------------------
# Node: tools
# ---------------------------------------------------------------------------
# Unchanged from the LiteLLM version. AIMessage.tool_calls from ChatOllama
# is already in the same {"name", "args", "id", "type": "tool_call"} shape
# this code was written against, so no changes were needed here at all.

async def _tool_node(state: MessagesState, config: RunnableConfig | None = None) -> dict:
    cfg     = (config or {}).get("configurable", {})
    mcp_url = cfg.get("mcp_url", "http://localhost:8000/mcp/")
    guard: ToolCallGuard | None = cfg.get("guard")

    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai = msg
            break

    if last_ai is None:
        logger.warning("tool_node called but no AIMessage with tool_calls found")
        return {"messages": []}

    async def _call_one(tc: dict) -> ToolMessage:
        name    = tc["name"]
        args    = tc["args"]
        call_id = tc["id"]

        if guard is not None:
            decision = guard.check(name, args)
            if not decision.allowed:
                logger.warning("Tool '%s' blocked by guardrail: %s", name, decision.reason)
                return ToolMessage(content=f"[BLOCKED] {decision.reason}",
                                   tool_call_id=call_id, name=name)
            if decision.result is not None:
                return ToolMessage(content=decision.result, tool_call_id=call_id, name=name)

        logger.info("Calling tool: %s(%s)", name, list(args.keys()))

        with _span("react_graph.tool_call", tool_name=name):
            result = await call_tool(name, args, mcp_url=mcp_url)

        success = not result.startswith("[MCP ERROR]")
        if guard is not None:
            guard.record_result(name, args, success=success, result=result)

        return ToolMessage(content=result, tool_call_id=call_id, name=name)

    with _span("react_graph.tool_node", tool_call_count=len(last_ai.tool_calls)):
        tool_messages: list[ToolMessage] = list(
            await asyncio.gather(*[_call_one(tc) for tc in last_ai.tool_calls])
        )

    for tm in tool_messages:
        status = "OK" if not (tm.content.startswith("[MCP ERROR]")
                              or tm.content.startswith("[BLOCKED]")) else "FAILED/BLOCKED"
        logger.info("Tool %s: %s (%d chars)", tm.name, status, len(tm.content))

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _should_continue(state: MessagesState, config: RunnableConfig | None = None) -> Literal["tools", "__end__"]:
    cfg        = (config or {}).get("configurable", {})
    max_rounds = cfg.get("max_rounds", MAX_ROUNDS)
    messages   = state["messages"]

    rounds = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    if rounds >= max_rounds:
        logger.warning("ReAct loop hit max_rounds=%d -- forcing final answer.", max_rounds)
        return END

    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    total = sum(len(getattr(m, "content", "") or "") for m in messages)
    if total <= MAX_CONTEXT_CHARS:
        return messages
    if len(messages) <= KEEP_RECENT + 1:
        return messages

    system  = messages[0]
    recent  = messages[-KEEP_RECENT:]
    trimmed = messages[1 : len(messages) - KEEP_RECENT]
    summary = SystemMessage(
        content=f"[{len(trimmed)} earlier messages trimmed to stay within context window]"
    )
    return [system, summary, *recent]


def _get_llm_from_config(cfg: dict) -> ChatOllama:
    llm = cfg.get("llm")
    if llm is not None:
        return llm
    # Fallback default, in case config["configurable"]["llm"] wasn't set --
    # should not normally happen since ReActGraph.run() always sets it.
    return ChatOllama(model="paeka-qwen", base_url="http://localhost:11434")
