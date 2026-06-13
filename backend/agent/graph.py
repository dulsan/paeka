"""
backend/agent/graph.py
=======================
Agentic RAG pipeline — Planner → Retriever → Critic → Synthesiser.
Now includes optional SearXNG web client as a fourth retrieval tool.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes.planner     import planner_node
from backend.agent.nodes.retriever   import retriever_node
from backend.agent.nodes.critic      import critic_node
from backend.agent.nodes.synthesiser import synthesiser_node
from backend.llm.client import LLMClient
from backend.shared.config import get_settings

logger = logging.getLogger(__name__)


class AgenticRAGPipeline:
    def __init__(
        self,
        llm: LLMClient,
        retrieval_engine: Any | None = None,
        graph_retriever: Any | None = None,
        web_client: Any | None = None,
        max_hops: int = 2,
    ) -> None:
        self._llm              = llm
        self._retrieval_engine = retrieval_engine
        self._graph_retriever  = graph_retriever
        self._web_client       = web_client
        self._max_hops         = max_hops
        self._graph            = self._build_graph()

    async def run(
        self,
        query: str,
        conversation_id: str = "",
        system_prompt: str = "",
    ) -> dict[str, Any]:
        if not system_prompt:
            system_prompt = get_settings().llm.system_prompt

        initial: AgentState = {
            "user_query":           query,
            "conversation_id":      conversation_id,
            "system_prompt":        system_prompt,
            "sub_queries":          [],
            "research_plan":        "",
            "retrieved_passages":   [],
            "hop_count":            0,
            "max_hops":             self._max_hops,
            "critique":             "",
            "needs_more_retrieval": False,
            "approved_passages":    [],
            "final_answer":         "",
            "citations":            [],
            "graph_context":        "",
            "error":                None,
            "metadata":             {},
        }

        try:
            final: AgentState = await self._graph.ainvoke(initial)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline error: %s", exc)
            return {"answer": f"Pipeline error: {exc}",
                    "citations": [], "plan": "", "hops": 0, "graph_context": ""}

        return {
            "answer":        final.get("final_answer", ""),
            "citations":     final.get("citations", []),
            "plan":          final.get("research_plan", ""),
            "hops":          final.get("hop_count", 0),
            "graph_context": final.get("graph_context", ""),
        }

    def _build_graph(self) -> StateGraph:
        llm, engine, graph_ret, web = (
            self._llm, self._retrieval_engine,
            self._graph_retriever, self._web_client,
        )

        async def _planner(s: AgentState) -> AgentState:
            return await planner_node(s, llm)

        async def _retriever(s: AgentState) -> AgentState:
            return await retriever_node(s, engine, graph_ret, web)

        async def _critic(s: AgentState) -> AgentState:
            return await critic_node(s, llm)

        async def _synthesiser(s: AgentState) -> AgentState:
            return await synthesiser_node(s, llm)

        def _route(s: AgentState) -> str:
            if s.get("needs_more_retrieval") and s.get("hop_count", 0) < s.get("max_hops", 2):
                return "retriever"
            return "synthesiser"

        b = StateGraph(AgentState)
        b.add_node("planner",     _planner)
        b.add_node("retriever",   _retriever)
        b.add_node("critic",      _critic)
        b.add_node("synthesiser", _synthesiser)
        b.set_entry_point("planner")
        b.add_edge("planner",   "retriever")
        b.add_edge("retriever", "critic")
        b.add_conditional_edges("critic", _route,
                                {"retriever": "retriever", "synthesiser": "synthesiser"})
        b.add_edge("synthesiser", END)
        return b.compile()
