"""
backend/agent/iteration_graph.py
==================================
Autonomous iteration graph: Generate → Evaluate → Reflect → (loop or finish).

Fixes applied:
  [FIX-PARSE]  _parse_json() used str.lstrip("```json") which strips
               individual characters {, `, j, s, o, n from the left edge.
               A JSON object starting with { had its opening brace stripped,
               producing invalid JSON every time. Now uses str.removeprefix().

  [FIX-EVAL]   _evaluator() prompt only showed the CURRENT output. It had
               no visibility into whether this iteration improved over the
               previous one. The evaluator could legitimately score iteration
               N lower than N-1 if the output actually regressed, but the
               router only used the raw threshold with no delta awareness.
               Now passes last two outputs and corresponding critiques so the
               evaluator can reason about progress.

  [FIX-SILENT] _parse_json() previously returned {} on parse failure with
               only a debug log. Now logs a WARNING with the raw text so
               failures are visible in production logs without being noisy.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any

from langgraph.graph import StateGraph, END

from backend.agent.state import IterationState
from backend.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """\
{system_prompt}
You are an expert assistant working on the following task:

{task}

{context}

Previous attempt (if any):
{previous_output}

Critique of previous attempt (if any):
{critique}

Produce an improved response that directly addresses the critique.
Be specific, accurate, and complete.
"""

_EVALUATOR_PROMPT = """\
You are evaluating the quality of an AI-generated response.

Task:
{task}

Current response (iteration {iteration}):
{current_output}

Previous response (iteration {prev_iteration}):
{previous_output}

Critique that led to the current response:
{critique}

Score the CURRENT response on a scale of 0.0 to 1.0 where:
  1.0 = perfect, complete, and accurate
  0.8 = very good, minor gaps only
  0.6 = acceptable but notable issues remain
  0.4 = significant problems
  0.2 = largely incorrect or incomplete
  0.0 = completely wrong or off-topic

Also assess: is the current response better than the previous? Has the critique been addressed?

Respond ONLY with a JSON object:
{{
  "score": 0.85,
  "improved_over_previous": true,
  "already_good": false,
  "reasoning": "brief explanation"
}}
"""

_REFLECTOR_PROMPT = """\
You are a critic improving an AI-generated response.

Task:
{task}

Current response:
{current_output}

Current quality score: {score:.2f} (threshold: {threshold:.2f})

Evaluator reasoning: {reasoning}

Provide a specific, actionable critique. Do NOT rewrite the response — only
identify the exact problems and what must be changed. Be concise.
"""


class AutonomousIterationGraph:
    def __init__(
        self,
        llm: LLMProvider,
        max_iterations: int = 5,
        score_threshold: float = 0.85,
    ) -> None:
        self._llm       = llm
        self._max_iter  = max_iterations
        self._threshold = score_threshold
        self._graph     = self._build()

    async def run(self, task: str, system_prompt: str = "", context: str = "") -> dict[str, Any]:
        # [FIX] run() previously had no system_prompt parameter at all, but
        # backend/api/routes/agent.py's /agent/iterate route calls
        # graph.run(task=..., system_prompt=..., context=...) -- that was
        # raising TypeError on every single call to that endpoint, before
        # ever reaching any of the other bugs below. The IterationState
        # schema already had a "system_prompt" field; run() just never
        # accepted or threaded it through.
        initial: IterationState = {
            "task":            task,
            "system_prompt":   system_prompt,
            "context":         context,
            "current_output":  "",
            "iteration":       0,
            "max_iterations":  self._max_iter,
            "score":           0.0,
            "score_threshold": self._threshold,
            "evaluation":      "",
            "critique":        "",
            "output_history":  [],
            "critique_history": [],
            "final_output":    "",
            "converged":       False,
            "error":           None,
        }
        final = await self._graph.ainvoke(initial)
        # [FIX] Previously returned the raw graph state dict directly. The
        # API route destructures result["iterations"] (plural) twice, but
        # the state only ever had "iteration" (singular, the in-progress
        # counter) -- a guaranteed KeyError on every call. Mapping
        # explicitly here also makes this the single place that defines
        # the public contract of run()'s return value, separate from the
        # internal state schema's field names.
        return {
            "final_output":      final.get("final_output", "") or final.get("current_output", ""),
            "iterations":        final.get("iteration", 0),
            "final_score":       final.get("score", 0.0),
            "converged":         final.get("converged", False),
            "critique_history":  final.get("critique_history", []),
        }

    def _build(self) -> Any:
        llm       = self._llm
        threshold = self._threshold

        g = StateGraph(IterationState)  # type: ignore[arg-type]
        g.add_node("generator",  partial(_generator,  llm=llm))
        g.add_node("evaluator",  partial(_evaluator,  llm=llm))
        g.add_node("reflector",  partial(_reflector,  llm=llm))

        g.set_entry_point("generator")
        g.add_edge("generator", "evaluator")

        def _route(s: IterationState) -> str:
            if s.get("error"):
                return END
            score       = s.get("score", 0.0)
            iteration   = s.get("iteration", 0)
            max_iter    = s.get("max_iterations", self._max_iter)
            already_good = s.get("already_good", False)

            if score >= threshold or already_good:
                return END
            if iteration >= max_iter:
                logger.info("Iteration graph: max_iterations=%d reached, score=%.2f", max_iter, score)
                return END
            return "reflector"

        g.add_conditional_edges("evaluator", _route,
                                {"reflector": "reflector", END: END})
        g.add_edge("reflector", "generator")
        return g.compile()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def _generator(s: IterationState, llm: LLMProvider) -> IterationState:
    output_history  = list(s.get("output_history",  []))
    critique_history = list(s.get("critique_history", []))

    previous_output = output_history[-1] if output_history else ""
    critique        = critique_history[-1] if critique_history else ""

    temp = 0.6 if s.get("iteration", 0) == 0 else 0.4

    prompt = _GENERATOR_PROMPT.format(
        system_prompt=s.get("system_prompt", "") or "",
        task=s["task"],
        context=s.get("context", ""),
        previous_output=previous_output or "None — this is the first attempt.",
        critique=critique or "None.",
    )
    try:
        output = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=temp,
        )
    except Exception as exc:
        logger.error("Generator error: %s", exc)
        return {**s, "error": str(exc)}

    new_history = output_history + [output]
    return {
        **s,
        "current_output": output,
        "output_history": new_history,
        "iteration":      s.get("iteration", 0) + 1,
        "error":          None,
    }


async def _evaluator(s: IterationState, llm: LLMProvider) -> IterationState:
    output_history = s.get("output_history", [])
    iteration      = s.get("iteration", 0)

    # [FIX-EVAL] Pass last two outputs so the evaluator can assess progress.
    current_output  = s.get("current_output", "")
    previous_output = output_history[-2] if len(output_history) >= 2 else "N/A (first attempt)"
    critique        = (s.get("critique_history") or [""])[-1]

    prompt = _EVALUATOR_PROMPT.format(
        task=s["task"],
        current_output=current_output,
        previous_output=previous_output,
        critique=critique,
        iteration=iteration,
        prev_iteration=max(0, iteration - 1),
    )
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
    except Exception as exc:
        logger.error("Evaluator error: %s", exc)
        return {**s, "score": 0.5, "evaluation": str(exc), "already_good": False}

    data = _parse_json(raw)
    # [FIX] Score was used as-is with no validation -- an LLM reporting
    # 1.5 or -0.3 (outside the documented 0.0-1.0 scale) would silently
    # propagate that out-of-range value into score_threshold comparisons
    # and the API response. Clamp to the documented range.
    raw_score    = float(data.get("score", 0.5))
    score        = max(0.0, min(1.0, raw_score))
    already_good = bool(data.get("already_good", False))
    reasoning    = str(data.get("reasoning", ""))
    threshold    = s.get("score_threshold", 0.85)

    # [FIX] converged was never set anywhere in this function's return --
    # it stayed at its initial False for the entire run regardless of
    # actual outcome, since _evaluator only ever did {**s, ...} without
    # touching this key. /api/agent/iterate's response always reported
    # converged=False as a result, even on a genuine, clean convergence.
    converged = score >= threshold or already_good

    logger.info(
        "Evaluator: iteration=%d score=%.2f improved=%s already_good=%s converged=%s",
        iteration, score, data.get("improved_over_previous"), already_good, converged,
    )

    final_output = current_output if converged else s.get("final_output", "")

    return {
        **s,
        "score":        score,
        "evaluation":   reasoning,
        "already_good": already_good,
        "converged":    converged,
        "final_output": final_output or current_output,
    }


async def _reflector(s: IterationState, llm: LLMProvider) -> IterationState:
    prompt = _REFLECTOR_PROMPT.format(
        task=s["task"],
        current_output=s.get("current_output", ""),
        score=s.get("score", 0.0),
        threshold=s.get("score_threshold", 0.85),
        reasoning=s.get("evaluation", ""),
    )
    try:
        critique = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
    except Exception as exc:
        logger.error("Reflector error: %s", exc)
        critique = f"Reflector failed: {exc}"

    history = list(s.get("critique_history", []))
    # [FIX] Critique entries were stored with no indication of which
    # iteration they came from -- made the history hard to read back when
    # debugging a run that took several rounds to converge (or didn't).
    iteration = s.get("iteration", 0)
    history.append(f"[Iteration {iteration}] {critique}")
    return {**s, "critique": critique, "critique_history": history}


# ---------------------------------------------------------------------------
# JSON parse helper
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """
    Extract the first JSON object from an LLM response string.

    [FIX-PARSE] The previous implementation used:
        raw.lstrip("```json").lstrip("```")
    str.lstrip(chars) strips CHARACTERS from the set, not a prefix substring.
    "```json".lstrip("```json") strips any leading {,`,j,s,o,n chars.
    A JSON object starting with { had its opening brace removed every time.

    Correct approach: find the first { ... } span using str.find/rfind,
    then fall back to removeprefix() if no braces appear (plain JSON).
    """
    text = raw.strip()

    # Fast path: try the whole string first (no fences)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences with removeprefix / removesuffix (Python 3.9+)
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract the first complete {...} block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # [FIX-SILENT] Was a silent debug log. Now a visible WARNING.
    logger.warning(
        "Evaluator: could not parse JSON from response (returning defaults). "
        "Raw (first 200 chars): %s",
        raw[:200],
    )
    return {}
