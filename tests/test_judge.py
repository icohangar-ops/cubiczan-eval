"""Judge scoring tests — offline, using stub completion callables."""

from __future__ import annotations

import pytest

from cubiczan_eval import InMemoryEvalStore, LLMJudge
from tests.conftest import make_stub_scorer


@pytest.mark.asyncio
async def test_scoring_returns_per_dimension_scores(rubric, high_scorer):
    judge = LLMJudge(rubric, high_scorer)
    result = await judge.evaluate("some output", "some context", agent_name="agentA")

    got = {s.dimension: s.score for s in result.scores}
    assert got == {"correctness": 9, "completeness": 9, "grounding": 9, "clarity": 9}
    assert all(s.justification for s in result.scores)


@pytest.mark.asyncio
async def test_overall_is_weighted_average(rubric):
    # correctness .30, completeness .25, grounding .25, clarity .20
    scorer = make_stub_scorer(
        {"correctness": 10, "completeness": 6, "grounding": 8, "clarity": 4}
    )
    judge = LLMJudge(rubric, scorer)
    result = await judge.evaluate("o", "c")
    expected = 10 * 0.30 + 6 * 0.25 + 8 * 0.25 + 4 * 0.20
    assert result.overall_score == pytest.approx(round(expected, 2))


@pytest.mark.asyncio
async def test_scores_are_clamped_into_range(rubric):
    scorer = make_stub_scorer(
        {"correctness": 99, "completeness": -5, "grounding": 7, "clarity": 7}
    )
    judge = LLMJudge(rubric, scorer)
    result = await judge.evaluate("o", "c")
    by_dim = {s.dimension: s.score for s in result.scores}
    assert by_dim["correctness"] == 10  # clamped to max
    assert by_dim["completeness"] == 1  # clamped to min


@pytest.mark.asyncio
async def test_async_completion_fn_is_awaited(rubric):
    async def acomplete(system: str, user: str) -> str:
        return make_stub_scorer(
            {"correctness": 7, "completeness": 7, "grounding": 7, "clarity": 7}
        )(system, user)

    judge = LLMJudge(rubric, acomplete)
    result = await judge.evaluate("o", "c")
    assert result.overall_score == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_evaluate_persists_to_store(rubric, high_scorer):
    store = InMemoryEvalStore()
    judge = LLMJudge(rubric, high_scorer, store=store)
    await judge.evaluate("o", "c", agent_name="agentA")
    recent = store.recent_results()
    assert len(recent) == 1
    assert recent[0]["agent_name"] == "agentA"


def test_unparseable_response_yields_no_scores(rubric):
    judge = LLMJudge(rubric, lambda s, u: "not json at all")
    result = judge.evaluate_sync("o", "c")
    assert result.scores == []
    assert result.overall_score == 0.0
