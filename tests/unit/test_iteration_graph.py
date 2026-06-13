"""
tests/unit/test_iteration_graph.py
====================================
Unit tests for the autonomous iteration graph.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agent.state import IterationState


def _base_state(**overrides) -> IterationState:
    s: IterationState = {
        "task":             "Write a haiku about Python.",
        "system_prompt":    "",
        "context":          "",
        "current_output":   "",
        "iteration":        0,
        "max_iterations":   4,
        "score":            0.0,
        "score_threshold":  0.85,
        "evaluation":       "",
        "critique":         "",
        "output_history":   [],
        "critique_history": [],
        "final_output":     "",
        "converged":        False,
        "error":            None,
    }
    s.update(overrides)
    return s


@pytest.mark.anyio
async def test_generator_produces_output():
    from backend.agent.iteration_graph import _generator

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Indented lines flow\nSpaces, not tabs, define scope\nPython breathes clean")

    state = _base_state()
    result = await _generator(state, llm)

    assert len(result["current_output"]) > 0
    assert len(result["output_history"]) == 1


@pytest.mark.anyio
async def test_generator_incorporates_critique():
    """Second-pass generator should include critique in its prompt."""
    from backend.agent.iteration_graph import _generator

    prompts_seen = []

    async def capture_complete(messages, **kwargs):
        prompts_seen.append(messages[0]["content"])
        return "Improved haiku here"

    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=capture_complete)

    state = _base_state(
        critique="The haiku lacks a seasonal reference (kigo).",
        iteration=1,
    )
    await _generator(state, llm)

    assert any("critique" in p.lower() or "kigo" in p for p in prompts_seen)


@pytest.mark.anyio
async def test_evaluator_extracts_score():
    from backend.agent.iteration_graph import _evaluator

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='{"score": 0.92, "evaluation": "Excellent haiku.", "already_good": true}')

    state = _base_state(current_output="Indented lines flow...")
    result = await _evaluator(state, llm)

    assert result["score"] == pytest.approx(0.92)
    assert result["converged"] is True
    assert result["iteration"] == 1


@pytest.mark.anyio
async def test_evaluator_clamps_score():
    """Score must be clamped to [0.0, 1.0] regardless of LLM output."""
    from backend.agent.iteration_graph import _evaluator

    llm = MagicMock()
    llm.complete = AsyncMock(return_value='{"score": 1.5, "evaluation": "Perfect.", "already_good": false}')

    state = _base_state(current_output="some output")
    result = await _evaluator(state, llm)

    assert 0.0 <= result["score"] <= 1.0


@pytest.mark.anyio
async def test_reflector_produces_critique():
    from backend.agent.iteration_graph import _reflector

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Add a seasonal reference and tighten the syllable count.")

    state = _base_state(current_output="Some haiku", evaluation="Missing kigo.", iteration=1)
    result = await _reflector(state, llm)

    assert len(result["critique"]) > 0
    assert len(result["critique_history"]) == 1
    assert "[Iteration 1]" in result["critique_history"][0]


@pytest.mark.anyio
async def test_full_pipeline_converges():
    from backend.agent.iteration_graph import AutonomousIterationGraph

    llm = MagicMock()
    responses = [
        "Draft haiku attempt 1",                               # generator
        '{"score": 0.9, "evaluation": "Good.", "already_good": true}',  # evaluator → converge
        "Final synthesised response",                           # finish
    ]
    llm.complete = AsyncMock(side_effect=responses)

    graph = AutonomousIterationGraph(llm=llm, score_threshold=0.85, max_iterations=4)
    result = await graph.run("Write a haiku about Python.")

    assert result["converged"] is True
    assert result["iterations"] == 1
    assert result["final_score"] == pytest.approx(0.9)


@pytest.mark.anyio
async def test_full_pipeline_respects_max_iterations():
    from backend.agent.iteration_graph import AutonomousIterationGraph

    # Always score below threshold — should stop at max_iterations
    low_score_eval = '{"score": 0.3, "evaluation": "Needs improvement.", "already_good": false}'
    gen_response   = "Some output"
    critique       = "Make it better."

    responses = [
        gen_response, low_score_eval, critique,  # iteration 1
        gen_response, low_score_eval, critique,  # iteration 2
        gen_response, low_score_eval,            # iteration 3 (max)
    ]
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=responses)

    graph = AutonomousIterationGraph(llm=llm, score_threshold=0.85, max_iterations=3)
    result = await graph.run("Impossible task")

    assert result["converged"] is False
    assert result["iterations"] <= 3
