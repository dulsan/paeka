"""
backend/agent/state.py
=======================
LangGraph state shared across all agent nodes.

Covers three graph types:
  1. RAG pipeline:    Planner → Retriever → Critic → Synthesiser
  2. Tool calling:    ToolSelector → ToolExecutor → ToolEvaluator → Reflector
  3. Iteration loop:  Generator → Evaluator → Reflector → (loop or finish)
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# RAG state
# ---------------------------------------------------------------------------

class RetrievalResult(TypedDict):
    content: str
    source: str
    heading: str
    page: int
    score: float
    element_type: str
    trust_tier: str    # "local" | "graph" | "web"


class SubQuery(TypedDict):
    query: str
    tool: str        # "vector" | "graph" | "keyword" | "web"
    priority: int


class AgentState(TypedDict):
    # Input
    user_query: str
    conversation_id: str
    system_prompt: str

    # Planner
    sub_queries: list[SubQuery]
    research_plan: str

    # Retrieval
    retrieved_passages: list[RetrievalResult]
    hop_count: int
    max_hops: int

    # Critic
    critique: str
    needs_more_retrieval: bool
    approved_passages: list[RetrievalResult]

    # Output
    final_answer: str
    citations: list[dict]
    graph_context: str

    # Metadata
    error: str | None
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool calling state
# ---------------------------------------------------------------------------

class ToolCall(TypedDict):
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


class ToolResult(TypedDict):
    call_id: str
    tool_name: str
    output: str
    success: bool
    error: str


class ToolCallingState(TypedDict):
    # Input
    user_request: str
    system_prompt: str
    available_tools: list[dict]    # tool schemas

    # Execution
    tool_calls: list[ToolCall]     # planned calls
    tool_results: list[ToolResult] # executed results
    iteration: int
    max_iterations: int

    # Self-healing
    failed_calls: list[ToolResult] # calls that errored
    reflections: list[str]         # LLM reasoning about failures
    retry_count: int
    max_retries: int

    # Output
    final_response: str
    succeeded: bool
    error: str | None


# ---------------------------------------------------------------------------
# Autonomous iteration state
# ---------------------------------------------------------------------------

class IterationState(TypedDict):
    # Input
    task: str
    system_prompt: str
    context: str

    # Iteration
    current_output: str
    iteration: int
    max_iterations: int

    # Evaluation
    score: float              # 0.0–1.0 quality score
    score_threshold: float    # exit loop when score >= threshold
    evaluation: str           # evaluator's assessment
    critique: str             # reflector's specific critique

    # History (for reflection)
    output_history: list[str]
    critique_history: list[str]

    # Output
    final_output: str
    converged: bool
    error: str | None
