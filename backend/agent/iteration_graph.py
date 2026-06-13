"""
backend/agent/iteration_graph.py
=================================
Autonomous Iteration pipeline built with LangGraph.

Inspired by Unsloth Studio's iterative improvement pattern.

Graph topology:
                ┌───────────────┐
                │   GENERATOR   │  Produces initial output for the task
                └───────┬───────┘
                        │
                ┌───────▼───────┐
                │   EVALUATOR   │  Scores the output (0.0 – 1.0)
                └───────┬───────┘
                        │
         score < thresh ┼─ yes ──► REFLECTOR ──► GENERATOR (improve)
                        │ no (or max_iterations)
                ┌───────▼───────┐
                │    FINISH     │  Final output returned
                └───────────────┘

Use cases:
  - Code generation with quality checks
  - Report writing with factual scoring
  - Answer refinement with completeness scoring
  - Any task where the first attempt is rarely the best attempt

The loop exits when:
  1. Evaluator score >= score_threshold  (converged)
  2. iteration >= max_iterations         (hard cap)
  3. Evaluator signals "already_good"    (no improvement possible)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import StateGraph, END

from backend.agent.state import IterationState
from backend.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """\
{system_prompt}

Task: {task}

{context_block}

{critique_block}

Produce the best possible output for this task.
"""

_EVALUATOR_PROMPT = """\
Evaluate the following output for the given task.
Score it from 0.0 (completely wrong) to 1.0 (perfect).

Task: {task}

Output to evaluate:
{output}

Criteria:
- Accuracy: is the content factually correct?
- Completeness: does it fully address the task?
- Clarity: is it well-structured and readable?
- Specificity: does it avoid vague generalities?

Respond ONLY with a JSON object:
{{
  "score": 0.0–1.0,
  "evaluation": "one paragraph assessment",
  "already_good": true/false
}}
"""

_REFLECTOR_PROMPT = """\
You are a critic helping improve an output iteratively.

Task: {task}

Current output:
{output}

Evaluator assessment:
{evaluation}

Iteration {iteration} of {max_iterations}.

Identify the MOST IMPORTANT specific improvements needed.
Be concrete — say exactly what is wrong and what should replace it.

Respond with a critique (plain text, 2-4 sentences max):
"""


class AutonomousIterationGraph:
    """
    LangGraph autonomous iteration pipeline.

    Parameters
    ----------
    llm:
        LLMProvider instance.
    score_threshold:
        Exit loop when evaluator score >= this value (default 0.85).
    max_iterations:
        Hard cap on iterations (default 4).
    """

    def __init__(
        self,
        llm: LLMProvider,
        score_threshold: float = 0.85,
        max_iterations: int = 4,
    ) -> None:
        self._llm             = llm
        self._score_threshold = score_threshold
        self._max_iterations  = max_iterations
        self._graph           = self._build()

    async def run(
        self,
        task: str,
        system_prompt: str = "",
        context: str = "",
    ) -> dict[str, Any]:
        """
        Run the autonomous iteration loop.

        Returns
        -------
        dict with:
          final_output:    str
          iterations:      int
          final_score:     float
          converged:       bool
          critique_history: list[str]
        """
        initial: IterationState = {
            "task":            task,
            "system_prompt":   system_prompt,
            "context":         context,
            "current_output":  "",
            "iteration":       0,
            "max_iterations":  self._max_iterations,
            "score":           0.0,
            "score_threshold": self._score_threshold,
            "evaluation":      "",
            "critique":        "",
            "output_history":  [],
            "critique_history": [],
            "final_output":    "",
            "converged":       False,
            "error":           None,
        }

        try:
            final: IterationState = await self._graph.ainvoke(initial)
        except Exception as exc:  # noqa: BLE001
            logger.error("Iteration graph error: %s", exc)
            return {
                "final_output":    f"Iteration error: {exc}",
                "iterations":      0,
                "final_score":     0.0,
                "converged":       False,
                "critique_history": [],
            }

        return {
            "final_output":    final.get("final_output", ""),
            "iterations":      final.get("iteration", 0),
            "final_score":     final.get("score", 0.0),
            "converged":       final.get("converged", False),
            "critique_history": final.get("critique_history", []),
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self) -> StateGraph:
        llm = self._llm

        async def _gen(s: IterationState) -> IterationState:
            return await _generator(s, llm)

        async def _eval(s: IterationState) -> IterationState:
            return await _evaluator(s, llm)

        async def _reflect(s: IterationState) -> IterationState:
            return await _reflector(s, llm)

        async def _finish(s: IterationState) -> IterationState:
            return {**s, "final_output": s["current_output"]}

        def _route(s: IterationState) -> str:
            converged     = s.get("score", 0.0) >= s.get("score_threshold", 0.85)
            maxed_out     = s.get("iteration", 0) >= s.get("max_iterations", 4)
            already_good  = s.get("converged", False)
            if converged or maxed_out or already_good:
                return "finish"
            return "reflector"

        b = StateGraph(IterationState)
        b.add_node("generator", _gen)
        b.add_node("evaluator", _eval)
        b.add_node("reflector", _reflect)
        b.add_node("finish",    _finish)

        b.set_entry_point("generator")
        b.add_edge("generator", "evaluator")
        b.add_conditional_edges("evaluator", _route,
                                {"reflector": "reflector", "finish": "finish"})
        b.add_edge("reflector", "generator")
        b.add_edge("finish", END)

        return b.compile()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


async def _generator(s: IterationState, llm: LLMProvider) -> IterationState:
    """Generate (or re-generate with critique) the task output."""
    context_block  = f"Context:\n{s['context']}" if s.get("context") else ""
    critique_block = ""
    if s.get("critique"):
        critique_block = (
            f"Critique from previous attempt (iteration {s['iteration']}):\n"
            f"{s['critique']}\n\n"
            f"Address ALL points in the critique in this improved version."
        )

    prompt = _GENERATOR_PROMPT.format(
        system_prompt=s.get("system_prompt", ""),
        task=s["task"],
        context_block=context_block,
        critique_block=critique_block,
    ).strip()

    try:
        output = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.6 if s.get("iteration", 0) == 0 else 0.4,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Generator LLM error: %s", exc)
        output = s.get("current_output", "")

    history = list(s.get("output_history", []))
    history.append(output)

    logger.info(
        "Generator: iteration=%d output_len=%d",
        s.get("iteration", 0) + 1, len(output),
    )
    return {**s, "current_output": output, "output_history": history}


async def _evaluator(s: IterationState, llm: LLMProvider) -> IterationState:
    """Score the current output and decide if another iteration is needed."""
    prompt = _EVALUATOR_PROMPT.format(
        task=s["task"],
        output=s["current_output"][:3000],  # truncate to keep within context
    )

    try:
        raw  = await llm.complete([{"role": "user", "content": prompt}],
                                   max_tokens=300, temperature=0.1)
        data = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evaluator LLM error: %s", exc)
        data = {"score": 0.5, "evaluation": str(exc), "already_good": False}

    score       = float(data.get("score", 0.5))
    score       = max(0.0, min(1.0, score))
    evaluation  = str(data.get("evaluation", ""))
    already_good = bool(data.get("already_good", False))
    converged   = score >= s.get("score_threshold", 0.85) or already_good

    logger.info(
        "Evaluator: score=%.2f converged=%s iteration=%d",
        score, converged, s.get("iteration", 0),
    )
    return {
        **s,
        "score":      score,
        "evaluation": evaluation,
        "converged":  converged,
        "iteration":  s.get("iteration", 0) + 1,
    }


async def _reflector(s: IterationState, llm: LLMProvider) -> IterationState:
    """Produce a specific critique to guide the next generation attempt."""
    prompt = _REFLECTOR_PROMPT.format(
        task=s["task"],
        output=s["current_output"][:2000],
        evaluation=s.get("evaluation", ""),
        iteration=s.get("iteration", 1),
        max_iterations=s.get("max_iterations", 4),
    )

    try:
        critique = await llm.complete([{"role": "user", "content": prompt}],
                                       max_tokens=300, temperature=0.3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reflector LLM error: %s", exc)
        critique = s.get("evaluation", "No specific critique available.")

    history = list(s.get("critique_history", []))
    history.append(f"[Iteration {s.get('iteration', 1)}] {critique}")

    logger.info("Reflector: critique_len=%d", len(critique))
    return {**s, "critique": critique, "critique_history": history}


def _parse_json(raw: str) -> dict:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
