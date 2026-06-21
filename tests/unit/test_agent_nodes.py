"""
tests/unit/test_agent_nodes.py
================================
Unit tests for individual agentic RAG nodes.
No GPU, no Weaviate, no SGLang required.
All LLM calls are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agent.state import AgentState, SubQuery


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "user_query":          "What is the transformer architecture?",
        "conversation_id":     "test-conv",
        "system_prompt":       "You are PAEKA.",
        "sub_queries":         [],
        "research_plan":       "",
        "retrieved_passages":  [],
        "hop_count":           0,
        "max_hops":            2,
        "critique":            "",
        "needs_more_retrieval": False,
        "approved_passages":   [],
        "final_answer":        "",
        "citations":           [],
        "graph_context":       "",
        "error":               None,
        "metadata":            {},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_planner_valid_json():
    from backend.agent.nodes.planner import planner_node

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='''{
        "plan": "Search for transformer papers then check KG.",
        "sub_queries": [
            {"query": "transformer attention mechanism", "tool": "vector", "priority": 1},
            {"query": "Vaswani 2017", "tool": "keyword", "priority": 2}
        ]
    }''')

    state = _base_state()
    result = await planner_node(state, llm)

    assert len(result["sub_queries"]) >= 2
    assert result["research_plan"] != ""
    assert all(sq["tool"] in ("vector", "graph", "keyword") for sq in result["sub_queries"])


@pytest.mark.anyio
async def test_planner_fallback_on_bad_json():
    """Planner must not crash on malformed LLM output."""
    from backend.agent.nodes.planner import planner_node

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Sorry, I cannot do that.")

    state = _base_state()
    result = await planner_node(state, llm)

    # Should fall back to a single sub-query with the original question
    assert len(result["sub_queries"]) >= 1
    assert result["sub_queries"][0]["query"] == state["user_query"]


@pytest.mark.anyio
async def test_planner_llm_error_fallback():
    """Planner must degrade gracefully on LLM connection error."""
    from backend.agent.nodes.planner import planner_node

    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=Exception("SGLang unavailable"))

    state = _base_state()
    result = await planner_node(state, llm)

    assert len(result["sub_queries"]) >= 1


# ---------------------------------------------------------------------------
# Retriever node
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_retriever_no_engine():
    """Retriever must not crash when no retrieval engine is available."""
    from backend.agent.nodes.retriever import retriever_node

    state = _base_state(sub_queries=[
        SubQuery(query="transformer", tool="vector", priority=1)
    ])
    # [FIX] retriever_node's actual parameter is named `engine`, not
    # `retrieval_engine` -- confirmed against backend/agent/nodes/retriever.py's
    # real signature. This test was written against an older signature and
    # never updated after the parameter was renamed.
    result = await retriever_node(state, engine=None, graph_retriever=None)

    # hop_count must increment
    assert result["hop_count"] == 1
    # No passages but no crash
    assert result["retrieved_passages"] == []


@pytest.mark.anyio
async def test_retriever_deduplicates_passages():
    """Passages with identical content should not be added twice."""
    from backend.agent.nodes.retriever import retriever_node
    from backend.agent.state import RetrievalResult

    existing_passage: RetrievalResult = {
        "content": "The transformer uses self-attention.",
        "source": "paper.pdf",
        "heading": "Introduction",
        "page": 1,
        "score": 0.9,
        "element_type": "text",
        "trust_tier": "local",  # [FIX] required field on RetrievalResult, was missing
    }

    mock_engine = MagicMock()
    # [FIX] RetrievalEngine.retrieve() is genuinely async (confirmed against
    # backend/retrieval/engine.py -- it's a deliberate async def, with its
    # own historical [FIX-ASYNC] comment in retriever.py noting it used to
    # block the event loop before being made async). A plain MagicMock()
    # here returns the list directly instead of something awaitable, so
    # `await engine.retrieve(...)` in retriever_node fails with
    # "object list can't be used in 'await' expression". AsyncMock is the
    # correct mock type for an async method.
    mock_engine.retrieve = AsyncMock(return_value=[
        MagicMock(
            content=existing_passage["content"],
            score=0.9,
            metadata={"filename": "paper.pdf", "heading": "Intro", "page": 1, "element_type": "text"},
        )
    ])

    state = _base_state(
        sub_queries=[SubQuery(query="transformer", tool="vector", priority=1)],
        retrieved_passages=[existing_passage],
        hop_count=0,
    )
    result = await retriever_node(state, engine=mock_engine, graph_retriever=None)

    # Duplicate passage must be filtered out
    assert len(result["retrieved_passages"]) == 1


# ---------------------------------------------------------------------------
# Critic node
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_critic_sufficient_passages():
    from backend.agent.nodes.critic import critic_node
    from backend.agent.state import RetrievalResult

    passages: list[RetrievalResult] = [
        {
            "content": "The transformer architecture uses multi-head attention.",
            "source": "paper.pdf", "heading": "Abstract",
            "page": 1, "score": 0.95, "element_type": "text", "trust_tier": "local",
        }
    ]

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='''{
        "sufficient": true,
        "missing_topics": [],
        "contradictions": [],
        "keep_passage_indices": [0],
        "reasoning": "The passage directly answers the query."
    }''')

    state = _base_state(retrieved_passages=passages, hop_count=0)
    result = await critic_node(state, llm)

    assert result["needs_more_retrieval"] is False
    assert len(result["approved_passages"]) == 1


@pytest.mark.anyio
async def test_critic_requests_more_retrieval():
    from backend.agent.nodes.critic import critic_node
    from backend.agent.state import RetrievalResult

    passages: list[RetrievalResult] = [
        {
            "content": "The transformer uses attention layers.",
            "source": "paper.pdf", "heading": "Intro",
            "page": 1, "score": 0.7, "element_type": "text", "trust_tier": "local",
        }
    ]

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='''{
        "sufficient": false,
        "missing_topics": ["positional encoding", "feed-forward layers"],
        "contradictions": [],
        "keep_passage_indices": [0],
        "reasoning": "Missing details on positional encoding."
    }''')

    state = _base_state(retrieved_passages=passages, hop_count=0, max_hops=2)
    result = await critic_node(state, llm)

    assert result["needs_more_retrieval"] is True
    # Missing topics become new sub-queries
    new_queries = [sq["query"] for sq in result["sub_queries"]]
    assert any("positional encoding" in q for q in new_queries)


@pytest.mark.anyio
async def test_critic_forces_synthesis_at_max_hops():
    """Critic must never loop beyond max_hops regardless of LLM output."""
    from backend.agent.nodes.critic import critic_node

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='''{
        "sufficient": false,
        "missing_topics": ["something"],
        "contradictions": [],
        "keep_passage_indices": [],
        "reasoning": "Still missing info."
    }''')

    state = _base_state(hop_count=2, max_hops=2)
    result = await critic_node(state, llm)

    assert result["needs_more_retrieval"] is False


# ---------------------------------------------------------------------------
# Synthesiser node
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_synthesiser_produces_answer():
    from backend.agent.nodes.synthesiser import synthesiser_node
    from backend.agent.state import RetrievalResult

    passages: list[RetrievalResult] = [
        {
            "content": "Transformers use multi-head self-attention.",
            "source": "paper.pdf", "heading": "Method",
            "page": 3, "score": 0.95, "element_type": "text", "trust_tier": "local",
        }
    ]

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="The Transformer architecture uses multi-head attention.")

    state = _base_state(approved_passages=passages)
    result = await synthesiser_node(state, llm)

    assert "Transformer" in result["final_answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["filename"] == "paper.pdf"


@pytest.mark.anyio
async def test_synthesiser_empty_passages():
    """Synthesiser must work with no retrieved passages (parametric answer)."""
    from backend.agent.nodes.synthesiser import synthesiser_node

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="I'll answer from my training knowledge.")

    state = _base_state(approved_passages=[])
    result = await synthesiser_node(state, llm)

    assert result["final_answer"] != ""
    assert result["citations"] == []
