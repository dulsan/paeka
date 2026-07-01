"""
backend/agent/deep_orchestrator.py
====================================
deepagents-based orchestrator for PAEKA.

Architecture
------------
The orchestrator is the top-level agent created via ``create_deep_agent()``.
It has direct access to three externally-visible tools (web_search,
graph_search, qdrant_ingest) which are gated by HITL approval before
execution. Read-only local retrieval is delegated to a RAG sub-agent that
wraps the existing ``AgenticRAGPipeline`` (Planner → Retriever → Critic →
Synthesiser in backend/agent/graph.py) -- those tool calls happen inside
the sub-agent and are not surfaced to the orchestrator's HITL middleware,
which is intentional: local retrieval doesn't need human approval.

                ┌─────────────────────────────────────────┐
                │          DeepOrchestrator (HITL)         │
                │                                          │
                │  tools (gated):                          │
                │    web_search ───── interrupt_on ─► HOLD │
                │    graph_search ─── interrupt_on ─► HOLD │
                │    qdrant_ingest ── interrupt_on ─► HOLD │
                │                                          │
                │  subagents (ungated):                    │
                │    rag_researcher ──► AgenticRAGPipeline │
                │      (qdrant_search, falkor traverse      │
                │       all happen inside the sub-agent)   │
                └─────────────────────────────────────────┘

HITL resume protocol (API consumers)
-------------------------------------
1. POST /api/agent/deep  → may return {"interrupted": true,
                                       "thread_id": "...",
                                       "pending": [...]}
2. POST /api/agent/deep/resume  → body: {"thread_id": "...",
                                         "decisions": [{"type": "approve"}]}
   Resume types: "approve" | "reject" | "edit" (edit also needs "args" key).

Transition note
---------------
The existing ReActGraph (react_graph.py) and tool_graph.py are intentionally
kept as fallback paths during this transition -- they are the /api/agent/react
and /api/agent/tool routes respectively. The new orchestrator is exposed
at /api/agent/deep. Once the deep path has been verified in production,
react_graph.py and tool_graph.py will be removed (Step 6 plan, step 2).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from backend.shared.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic input/output schemas
# ---------------------------------------------------------------------------

class OrchestratorInput(BaseModel):
    """Input to DeepOrchestrator.run()."""

    query: str = Field(..., min_length=1, max_length=16_000)
    conversation_id: str = Field(default="")
    thread_id: str = Field(
        default="",
        description=(
            "LangGraph checkpoint thread ID.  Leave blank to start a new "
            "thread; supply the ID returned by a previous interrupted run "
            "when resuming."
        ),
    )


class OrchestratorOutput(BaseModel):
    """Output from DeepOrchestrator.run()."""

    answer: str
    thread_id: str
    interrupted: bool = False
    pending: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-agent: RAG researcher (wraps existing AgenticRAGPipeline)
# ---------------------------------------------------------------------------


def _build_rag_state():
    """Construct the TypedDict state class for the RAG sub-agent's LangGraph."""
    # Annotated and add_messages must be resolvable in the class body's
    # global scope so LangGraph's get_type_hints() can evaluate them at
    # runtime. Importing inside the function and then using them as
    # annotations works only if the same names exist in the *enclosing
    # module* globals -- so we inject them into the module namespace.
    import sys as _sys
    import typing as _typing
    from typing_extensions import TypedDict
    _mod = _sys.modules[__name__]
    if not hasattr(_mod, "Annotated"):
        _mod.Annotated = _typing.Annotated          # type: ignore[attr-defined]
    if not hasattr(_mod, "_add_messages_marker"):
        _mod._add_messages_marker = add_messages    # type: ignore[attr-defined]

    class _RAGState(TypedDict):
        messages: "Annotated[list, _add_messages_marker]"

    # Patch __annotations__ to use the already-resolved runtime values
    # (bypasses the forward-ref string evaluation entirely).
    import typing
    _RAGState.__annotations__["messages"] = typing.Annotated[list, add_messages]
    return _RAGState


def build_rag_subagent(pipeline) -> dict:
    """
    Wrap an ``AgenticRAGPipeline`` instance as a ``CompiledSubAgent`` dict
    for ``create_deep_agent(subagents=[...])``.

    The sub-agent receives a ``HumanMessage`` (the delegated task text from
    the orchestrator) and returns an ``AIMessage`` with the RAG answer.
    Internal retrieval (qdrant_search, graph_search) is entirely opaque to
    the orchestrator -- it runs inside this sub-agent's LangGraph node and
    is NOT surfaced to the HITL middleware.

    Parameters
    ----------
    pipeline:
        An ``AgenticRAGPipeline`` instance (from backend/agent/graph.py).
    """
    State = _build_rag_state()

    async def _rag_node(state: State) -> dict:
        # Extract the last human message as the query for the RAG pipeline.
        last_human = next(
            (m for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {"messages": [AIMessage(content="No query provided to RAG sub-agent.")]}

        try:
            result = await pipeline.run(query=last_human.content)
            answer = result.get("answer", "No answer returned from RAG pipeline.")
        except Exception as exc:
            logger.error("RAG sub-agent pipeline error: %s", exc)
            answer = f"Error running RAG pipeline: {exc}"

        return {"messages": [AIMessage(content=answer)]}

    builder = StateGraph(State)
    builder.add_node("rag", _rag_node)
    builder.set_entry_point("rag")
    builder.add_edge("rag", END)
    runnable = builder.compile()

    return {
        "name": "rag_researcher",
        "description": (
            "Answers questions using the local knowledge base (vector + graph RAG). "
            "Runs a multi-hop Planner → Retriever → Critic → Synthesiser pipeline "
            "over ingested documents. Use this sub-agent for any question about "
            "documents, papers, or technical content that has been ingested into PAEKA. "
            "Do NOT use for current events or real-time information -- use web_search instead."
        ),
        "runnable": runnable,
    }


# ---------------------------------------------------------------------------
# Orchestrator tools (thin MCP wrappers, gated by HITL)
# ---------------------------------------------------------------------------

def _make_mcp_tools(mcp_url: str, falkor=None) -> list:
    """
    Build the three HITL-gated tool callables the orchestrator is directly
    equipped with.  Each one is a thin async wrapper over mcp.client.call_tool
    with Pydantic-validated args, so the tool schema surfaced to the LLM and
    the enforcement boundary around call_tool are consistent (TOOL_ARG_SCHEMAS
    from tools/schemas.py owns both).

    These are plain async functions -- deepagents accepts them directly via
    its ``tools`` parameter, and langchain wraps them internally.
    """
    from backend.mcp.client import call_tool
    from backend.tools.schemas import tools_in_groups

    web_allowed = tools_in_groups(["web"])
    retrieval_allowed = tools_in_groups(["retrieval"])

    async def web_search(query: str, num_results: int = 3) -> str:
        """
        Search the web (DuckDuckGo) for current information not available
        in the local knowledge base. Use for recent events, latest versions,
        or anything that post-dates the ingested documents.

        Args:
            query: Search query string.
            num_results: Number of results to return (1-10, default 3).
        """
        return await call_tool(
            "web_search",
            {"query": query, "num_results": num_results},
            mcp_url=mcp_url,
            allowed_tools=web_allowed,
        )

    async def graph_search(entity: str, max_hops: int = 2) -> str:
        """
        Traverse the knowledge graph from a named entity to find related
        entities up to max_hops away. Use for "how is X related to Y" or
        "what connects to X" questions -- finds multi-hop relationships
        that a single text search would miss.

        Args:
            entity: Entity label to start from (e.g. 'Transformer', 'BERT').
            max_hops: Traversal depth (1-4, default 2).
        """
        return await call_tool(
            "graph_search",
            {"entity": entity, "max_hops": max_hops},
            mcp_url=mcp_url,
            allowed_tools=retrieval_allowed,
        )

    async def qdrant_ingest(content: str, metadata: str = "") -> str:
        """
        Add new content to the local knowledge base (vector store). Use
        when the user explicitly asks to save, remember, or store something
        for future retrieval. Requires human approval before executing.

        Args:
            content: Text content to ingest.
            metadata: Optional JSON metadata string (e.g. source URL or title).
        """
        return await call_tool(
            "qdrant_ingest",
            {"content": content, "metadata": metadata},
            mcp_url=mcp_url,
            allowed_tools=retrieval_allowed,
        )

    return [web_search, graph_search, qdrant_ingest]


# ---------------------------------------------------------------------------
# DeepOrchestrator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are PAEKA, a precise, research-grade AI assistant. You have access to:

1. **rag_researcher sub-agent** — Use this for any question about ingested \
documents, papers, code, or technical content. It runs multi-hop retrieval \
over the local knowledge base.

2. **web_search** (requires approval) — Use for current events, recent \
releases, or anything post-dating the knowledge base. Human approval is \
required before this executes.

3. **graph_search** (requires approval) — Use to explore entity relationships \
in the knowledge graph ("how is X connected to Y"). Human approval is required \
before this executes.

4. **qdrant_ingest** (requires approval) — Use ONLY when the user explicitly \
asks to save or remember something. Human approval is required before this executes.

Guidelines:
- Prefer rag_researcher for knowledge-base questions before trying web_search.
- Be explicit about uncertainty. Cite sources when you can.
- If a tool call is rejected by the human, explain why you were going to use \
it and offer alternatives.
"""

_INTERRUPT_ON: dict[str, bool] = {
    "web_search": True,
    "graph_search": True,
    "qdrant_ingest": True,
}


class DeepOrchestrator:
    """
    Top-level orchestrator wrapping ``create_deep_agent()``.

    Create via ``DeepOrchestrator.create(app_state, settings)`` in the
    FastAPI lifespan (app.py step 11) -- not by direct construction, since
    it depends on lifespan-managed services (RAG pipeline, MCP URL, Falkor).

    Thread-safety: LangGraph's MemorySaver is per-instance, so each
    ``DeepOrchestrator`` instance owns its own checkpoint store. Create once
    at app startup and share across requests via ``app.state.deep_orchestrator``.
    """

    def __init__(
        self,
        agent,                 # CompiledStateGraph from create_deep_agent()
        checkpointer,          # MemorySaver
        settings,              # Settings
    ) -> None:
        self._agent = agent
        self._checkpointer = checkpointer
        self._settings = settings

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        app_state,
        settings,
        mcp_url: str = "http://localhost:8000/mcp",
    ) -> "DeepOrchestrator":
        """
        Build and return a configured ``DeepOrchestrator``.

        Called once in app.py lifespan step 11 after all other services
        are initialised.  Degrades gracefully when optional services
        (RAG pipeline, Falkor) aren't available -- the orchestrator still
        works with just direct tools.
        """
        from deepagents import create_deep_agent

        chat_ollama = getattr(app_state, "chat_ollama", None)
        if chat_ollama is None:
            raise RuntimeError(
                "DeepOrchestrator.create() requires app.state.chat_ollama "
                "(ChatOllama instance built in app.py step 3)."
            )

        # Tools -- thin MCP wrappers, gated by HITL
        falkor = getattr(app_state, "kg_falkor", None)
        tools = _make_mcp_tools(mcp_url=mcp_url, falkor=falkor)

        # Sub-agents
        subagents = []
        rag_pipeline = getattr(app_state, "agent_pipeline", None)
        if rag_pipeline is not None:
            subagents.append(build_rag_subagent(rag_pipeline))
            logger.info("DeepOrchestrator: RAG researcher sub-agent attached.")
        else:
            logger.warning(
                "DeepOrchestrator: RAG pipeline not available -- "
                "rag_researcher sub-agent not registered. Orchestrator "
                "will rely on direct tools only."
            )

        checkpointer = MemorySaver()

        agent = create_deep_agent(
            model=chat_ollama,
            tools=tools,
            system_prompt=_SYSTEM_PROMPT,
            subagents=subagents if subagents else None,
            interrupt_on=_INTERRUPT_ON,
            checkpointer=checkpointer,
        )

        logger.info(
            "DeepOrchestrator ready: tools=%s subagents=%s",
            [t.__name__ for t in tools],
            [s["name"] for s in subagents],
        )
        return cls(agent=agent, checkpointer=checkpointer, settings=settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, inp: OrchestratorInput) -> OrchestratorOutput:
        """
        Execute one turn of the orchestrator.

        Returns immediately if the orchestrator is interrupted waiting for
        HITL approval -- ``output.interrupted`` will be True and
        ``output.pending`` will contain the tool calls requiring review.
        The caller should then call ``resume()`` with the thread_id and
        human decisions.
        """
        from backend.shared.logging import bind_context
        if inp.conversation_id:
            bind_context(conversation_id=inp.conversation_id)

        thread_id = inp.thread_id or _new_thread_id()
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=inp.query)]},
                config=config,
            )
        except Exception as exc:
            logger.error("DeepOrchestrator.run failed: %s", exc)
            return OrchestratorOutput(
                answer=f"Orchestrator error: {exc}",
                thread_id=thread_id,
            )

        return _parse_result(result, thread_id)

    async def resume(
        self,
        thread_id: str,
        decisions: list[dict],
    ) -> OrchestratorOutput:
        """
        Resume an interrupted orchestrator run.

        Parameters
        ----------
        thread_id:
            Thread ID from the interrupted ``OrchestratorOutput``.
        decisions:
            List of decision dicts.  Each must have a ``"type"`` key with
            value ``"approve"`` | ``"reject"`` | ``"edit"``.  Edit decisions
            also need an ``"args"`` key with the corrected tool arguments.

        Example
        -------
        ::

            await orchestrator.resume(
                thread_id="abc123",
                decisions=[{"type": "approve"}],
            )
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = await self._agent.ainvoke(
                Command(resume={"decisions": decisions}),
                config=config,
            )
        except Exception as exc:
            logger.error("DeepOrchestrator.resume failed (thread=%s): %s", thread_id, exc)
            return OrchestratorOutput(
                answer=f"Resume error: {exc}",
                thread_id=thread_id,
            )

        return _parse_result(result, thread_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_thread_id() -> str:
    import uuid
    return uuid.uuid4().hex[:16]


def _parse_result(result: dict, thread_id: str) -> OrchestratorOutput:
    """
    Convert a raw LangGraph result dict into an ``OrchestratorOutput``,
    handling both the normal-completion and interrupted cases.
    """
    if "__interrupt__" in result:
        pending = []
        for interrupt in result["__interrupt__"]:
            val = interrupt.value if hasattr(interrupt, "value") else interrupt
            if isinstance(val, dict):
                for req in val.get("action_requests", []):
                    pending.append({
                        "tool": req.get("name", "unknown"),
                        "args": req.get("args", {}),
                        "description": req.get("description", ""),
                        "allowed_decisions": next(
                            (
                                cfg.get("allowed_decisions", ["approve", "reject"])
                                for cfg in val.get("review_configs", [])
                                if cfg.get("action_name") == req.get("name")
                            ),
                            ["approve", "reject"],
                        ),
                    })
        return OrchestratorOutput(
            answer="",
            thread_id=thread_id,
            interrupted=True,
            pending=pending,
        )

    messages = result.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    return OrchestratorOutput(
        answer=last_ai.content if last_ai else "No answer generated.",
        thread_id=thread_id,
    )
