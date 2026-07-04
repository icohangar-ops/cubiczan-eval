"""Shared test fixtures: offline stub scorers and helpers.

No live API keys are required anywhere in the test suite — the judge and the
self-improvement engine both take an injected completion callable, and here we
inject deterministic stubs (the analogue of strata's ``MockLLM``).
"""

from __future__ import annotations

import json
from typing import Callable

import pytest

from cubiczan_eval import DimensionScore, EvalResult, Rubric, default_rubric


def make_stub_scorer(scores: dict[str, int]) -> Callable[[str, str], str]:
    """Return a completion fn that always emits ``scores`` as judge JSON.

    Args:
        scores: dimension key -> integer score.
    """

    payload = {
        "scores": [
            {"dimension": dim, "score": s, "justification": f"stub for {dim}"}
            for dim, s in scores.items()
        ]
    }
    blob = "```json\n" + json.dumps(payload) + "\n```"

    def complete(system: str, user: str) -> str:  # noqa: ARG001
        return blob

    return complete


@pytest.fixture
def rubric() -> Rubric:
    return default_rubric()


@pytest.fixture
def high_scorer() -> Callable[[str, str], str]:
    return make_stub_scorer(
        {"correctness": 9, "completeness": 9, "grounding": 9, "clarity": 9}
    )


@pytest.fixture
def low_grounding_scorer() -> Callable[[str, str], str]:
    """All dimensions healthy except grounding, which is well below threshold."""
    return make_stub_scorer(
        {"correctness": 8, "completeness": 8, "grounding": 3, "clarity": 8}
    )


def eval_result(agent: str, scores: dict[str, int]) -> EvalResult:
    """Build an EvalResult directly (no judge call needed)."""
    return EvalResult(
        agent_name=agent,
        output_text="…",
        scores=[
            DimensionScore(dimension=d, score=s, justification=f"j-{d}")
            for d, s in scores.items()
        ],
    )
