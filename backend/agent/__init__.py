"""
backend/agent
=============
Deliberately empty of re-exports.

[FIX] This used to eagerly import AgenticRAGPipeline (backend.agent.graph)
at package-init time, which pulls in the entire retrieval/knowledge-graph
stack (langgraph -> retriever_node -> knowledge.retriever ->
retrieval.reranker -> torch/FlagEmbedding) just because something did
`from backend.agent.sandbox import get_sandbox`. Importing *any* submodule
of this package runs this file first, so the stdlib-only sandbox.py was
unable to be imported (or unit-tested) without installing the full ML
stack. Confirmed nothing in the codebase actually used the package-level
shortcut (`from backend.agent import X`) -- every real caller already
imports the specific submodule directly (backend.agent.sandbox,
backend.agent.graph, backend.agent.tool_graph, etc.), so there is no
re-export to preserve. Import what you need from its own submodule.
"""
