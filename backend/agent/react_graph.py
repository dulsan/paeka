"""
backend/agent/react_graph.py
=============================
ReAct tool-calling loop using native LiteLLM function calling.

Why this replaces SelfHealingToolGraph
---------------------------------------
The original Selector → Executor → Evaluator → Reflector → Synthesiser
pattern has the LLM produce a JSON list of tool calls as plain text, which
is then parsed, validated, and dispatched. This pattern fails ~10-15% of the
time due to JSON formatting errors, wrong field names, or the LLM choosing
to explain rather than call tools. Each failure burns a Reflector LLM call
before the correct tool call is retried.

The approach used by Claude, GPT-4, and Gemini (and supported by llama-server
via --jinja + Qwen3.5's native tool-use format) is simpler:

    [Agent] <-- LiteLLM acompletion(tools=schemas) --> [llama-server]
       |
       | If response contains tool_calls:
       v
    [Tool Executor] -- calls MCP tools concurrently --> [results as ToolMessages]
       |
       v (back to Agent with tool results in message history)
    [Agent] ... (loop continues until finish_reason == "stop")

The LLM generates tool calls as structured tokens constrained by the
function-calling grammar — JSON is never "parsed" from free text. The model
sees tool results in its own message history and self-corrects naturally
without a separate Reflector node.

This matches how Claude Code, Cursor, and similar coding assistants operate.

LangGraph wiring
----------------
Two nodes:

    agent_node  → conditional edge →  tool_node  → back to agent_node
                     (finish_reason == "stop" → END)

The state is a list of OpenAI-format messages (MessagesState). LangGraph's
built-in ToolNode is NOT used because tools are served via MCP (not as
LangChain @tool objects). We implement a thin async ToolNode that calls the
MCP client instead.

Parallel tool execution
-----------------------
When the LLM requests N tools in one round, all N are dispatched concurrently
via asyncio.gather(). This reduces N×latency to ~1×latency per round.

Conversation memory integration
---------------------------------
Context from ConversationMemory (semantically retrieved past turns) is
injected into the system message at the start of each run() call.
New turns are archived after the loop exits.

Token budget management
-----------------------
LangGraph runs the loop until finish_reason == "stop". For long agentic
tasks this can exhaust the 8192-token context window. A lightweight message
trimmer runs before each agent_node call: if the message list exceeds
MAX_CONTEXT_CHARS, it:
  1. Keeps the system message (always first)
  2. Keeps the last KEEP_RECENT messages intact
  3. Replaces intermediate messages with "[N messages trimmed]"
This avoids the context overflow without losing the most recent tool results.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, MessagesState, END

from backend.llm.litellm_provider import LiteLLMProvider
from backend.mcp.client import call_tool, get_tool_schemas

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context window budget
# ---------------------------------------------------------------------------
# Qwen3.5-9B is loaded with --ctx-size 8192. We leave 2048 tokens for the
# model's own generation. At ~4 chars/token, 6144 tokens ≈ 24576 chars.
MAX_CONTEXT_CHARS: int = 24_000
KEEP_RECENT: int       = 6    # always preserve last N messages verbatim

# Maximum tool-call rounds before forcing a final answer
MAX_ROUNDS: int = 10


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
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
"""


# ---------------------------------------------------------------------------
# ReActGraph
# ---------------------------------------------------------------------------

class ReActGraph:
    """
    Minimal two-node ReAct graph: agent_node ↔ tool_node.

    Parameters
    ----------
    llm:
        LiteLLMProvider instance. Must be configured with
        LITELLM_MODEL=openai/Qwen3.5-9B-Q4_K_M.
    mcp_url:
        URL of the PAEKA MCP server (mounted at /mcp in FastAPI).
    system_prompt:
        Override the default PAEKA system prompt.
    max_rounds:
        Maximum agent→tool→agent cycles before forcing a final answer.
    """

    def __init__(
        self,
        llm: LiteLLMProvider,
        mcp_url: str = "http://localhost:8000/mcp",
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
        conversation_memory=None,   # ConversationMemory | None
    ) -> str:
        """
        Run the ReAct loop for a single user turn.

        Parameters
        ----------
        user_message:
            The user's natural language input.
        conversation_memory:
            Optional ConversationMemory for context retrieval and archival.

        Returns
        -------
        str
            The assistant's final response text.
        """
        # Fetch available tool schemas from MCP (cached after first call)
        tool_schemas = await get_tool_schemas(mcp_url=self._mcp_url)

        # Build context from conversation memory
        system_content = self._system_prompt
        if conversation_memory is not None:
            try:
                past = await conversation_memory.get_context(
                    query=user_message, include_retrieved=True, retrieved_limit=3
                )
                # Extract the system-level retrieved context injected by get_context()
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

        config = {
            "configurable": {
                "tool_schemas": tool_schemas,
                "mcp_url":      self._mcp_url,
                "max_rounds":   self._max_rounds,
            }
        }

        result = await self._graph.ainvoke(
            {"messages": initial_messages}, config=config
        )

        # Extract the final assistant message
        final = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                final = msg.content
                break

        # Archive to conversation memory
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
        g.add_conditional_edges("agent", _should_continue,
                                {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile()


# ---------------------------------------------------------------------------
# Node: agent
# ---------------------------------------------------------------------------

async def _agent_node(state: MessagesState, config: dict) -> dict:
    """
    Call the LLM with the current message history and available tools.

    If the response contains tool_calls, they flow to _tool_node.
    If finish_reason is "stop" (no tool calls), the graph ends.
    """
    from backend.shared.config import get_settings
    settings = get_settings()

    cfg          = config.get("configurable", {})
    tool_schemas = cfg.get("tool_schemas", [])
    messages     = _trim_messages(state["messages"])

    # Build call kwargs
    lm_messages = [_to_openai_dict(m) for m in messages]

    try:
        llm: LiteLLMProvider = _get_llm_from_config(cfg)
        response = await llm._acompletion_raw(
            messages=lm_messages,
            tools=tool_schemas if tool_schemas else None,
        )
    except Exception as exc:
        logger.error("agent_node LLM call failed: %s", exc)
        return {"messages": [AIMessage(content=f"I encountered an error: {exc}")]}

    choice  = response.choices[0]
    message = choice.message

    # Build a LangChain AIMessage from the LiteLLM response
    tool_calls_raw = message.tool_calls or []
    lc_tool_calls  = [
        {
            "id":   tc.id,
            "name": tc.function.name,
            "args": json.loads(tc.function.arguments or "{}"),
            "type": "tool_call",
        }
        for tc in tool_calls_raw
    ]

    ai_msg = AIMessage(
        content=message.content or "",
        tool_calls=lc_tool_calls,
    )

    logger.info(
        "agent_node: finish_reason=%s tool_calls=%d",
        choice.finish_reason, len(lc_tool_calls),
    )
    return {"messages": [ai_msg]}


# ---------------------------------------------------------------------------
# Node: tools
# ---------------------------------------------------------------------------

async def _tool_node(state: MessagesState, config: dict) -> dict:
    """
    Execute all tool calls from the last AIMessage concurrently.

    All tools dispatched in a single agent round run in parallel via
    asyncio.gather(). Results come back as ToolMessage objects appended
    to the message history so the LLM sees them in the next agent call.
    """
    cfg      = config.get("configurable", {})
    mcp_url  = cfg.get("mcp_url", "http://localhost:8000/mcp")
    messages = state["messages"]

    # Find the most recent AIMessage with tool calls
    last_ai: AIMessage | None = None
    for msg in reversed(messages):
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
        logger.info("Calling tool: %s(%s)", name, list(args.keys()))
        result = await call_tool(name, args, mcp_url=mcp_url)
        return ToolMessage(content=result, tool_call_id=call_id, name=name)

    # [PARALLEL] All tools in this round run concurrently
    tool_messages: list[ToolMessage] = list(
        await asyncio.gather(*[_call_one(tc) for tc in last_ai.tool_calls])
    )

    for tm in tool_messages:
        status = "OK" if not tm.content.startswith("[MCP ERROR]") else "FAILED"
        logger.info("Tool %s: %s (%d chars)", tm.name, status, len(tm.content))

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _should_continue(
    state: MessagesState,
    config: dict,
) -> Literal["tools", "__end__"]:
    """
    Continue to tool_node if the LLM made tool calls.
    End if it produced a plain text response (finish_reason == "stop").
    Also enforce a maximum round limit to prevent infinite loops.
    """
    cfg        = config.get("configurable", {})
    max_rounds = cfg.get("max_rounds", MAX_ROUNDS)
    messages   = state["messages"]

    # Count how many AIMessages with tool_calls have occurred
    rounds = sum(
        1 for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
    )
    if rounds >= max_rounds:
        logger.warning(
            "ReAct loop hit max_rounds=%d — forcing final answer.", max_rounds
        )
        return END

    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Trim the message list to stay within MAX_CONTEXT_CHARS.

    Always keeps:
      - messages[0]  (system prompt)
      - messages[-KEEP_RECENT:]  (most recent N messages)

    Replaces everything in between with a single SystemMessage summary.
    """
    total = sum(len(getattr(m, "content", "") or "") for m in messages)
    if total <= MAX_CONTEXT_CHARS:
        return messages

    if len(messages) <= KEEP_RECENT + 1:
        return messages  # nothing to trim

    system   = messages[0]
    recent   = messages[-KEEP_RECENT:]
    trimmed  = messages[1 : len(messages) - KEEP_RECENT]
    n        = len(trimmed)

    summary = SystemMessage(
        content=f"[{n} earlier messages trimmed to stay within context window]"
    )
    return [system, summary, *recent]


def _to_openai_dict(msg: BaseMessage) -> dict:
    """Convert a LangChain message to OpenAI-format dict for LiteLLM."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, AIMessage):
        d: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id":       tc["id"],
                    "type":     "function",
                    "function": {
                        "name":      tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in msg.tool_calls
            ]
        return d
    if isinstance(msg, ToolMessage):
        return {
            "role":         "tool",
            "content":      msg.content,
            "tool_call_id": msg.tool_call_id,
        }
    return {"role": "user", "content": str(msg.content)}


def _get_llm_from_config(cfg: dict) -> LiteLLMProvider:
    """Retrieve LiteLLM provider from config or build one from env."""
    llm = cfg.get("llm")
    if llm is not None:
        return llm
    return LiteLLMProvider()


# ---------------------------------------------------------------------------
# Patch LiteLLMProvider with a raw acompletion helper
# ---------------------------------------------------------------------------
# The react_graph needs direct access to the raw LiteLLM response object
# (to inspect finish_reason and tool_calls), not just the string that
# LiteLLMProvider.complete() returns. This module-level patch adds
# _acompletion_raw() to LiteLLMProvider if it doesn't already exist.
# This is cleaner than subclassing or modifying litellm_provider.py.

import litellm as _litellm

async def _acompletion_raw(self, messages: list[dict], tools: list | None = None, **kwargs):
    """Raw LiteLLM acompletion returning the full response object."""
    call_kw: dict = {
        "model":       self._model,
        "api_base":    self._api_base,
        "api_key":     self._api_key,
        "messages":    messages,
        "max_tokens":  self._max_tokens,
        "temperature": self._temperature,
        "timeout":     self._timeout,
        **kwargs,
    }
    if tools:
        call_kw["tools"]       = tools
        call_kw["tool_choice"] = "auto"
    return await _litellm.acompletion(**call_kw)

if not hasattr(LiteLLMProvider, "_acompletion_raw"):
    LiteLLMProvider._acompletion_raw = _acompletion_raw  # type: ignore[attr-defined]
