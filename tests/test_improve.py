"""Self-improvement engine tests — the grade -> revise step, offline."""

from __future__ import annotations

import pytest

from cubiczan_eval import (
    DEFAULT_AMENDMENT_TEMPLATES,
    InMemoryEvalStore,
    SelfImprovementEngine,
    revise_prompt,
)
from tests.conftest import eval_result


def _seed(store: InMemoryEvalStore, agent: str, scores_list: list[dict[str, int]]):
    for scores in scores_list:
        store.add_result(eval_result(agent, scores))


@pytest.mark.asyncio
async def test_below_threshold_triggers_amendment_from_template():
    store = InMemoryEvalStore()
    _seed(
        store,
        "agentA",
        [
            {"correctness": 8, "grounding": 3},
            {"correctness": 8, "grounding": 2},
            {"correctness": 8, "grounding": 4},
        ],
    )
    engine = SelfImprovementEngine(
        store, templates=DEFAULT_AMENDMENT_TEMPLATES, threshold=6.0, min_evals=3
    )
    report = await engine.run_cycle()

    assert "agentA" in report.amendments
    amendment = report.amendments["agentA"]
    assert amendment.dimension == "grounding"
    assert amendment.source == "template"
    assert amendment.text == DEFAULT_AMENDMENT_TEMPLATES["grounding"]


@pytest.mark.asyncio
async def test_healthy_scores_produce_no_amendments():
    store = InMemoryEvalStore()
    _seed(store, "agentA", [{"grounding": 9}] * 5)
    engine = SelfImprovementEngine(store, templates=DEFAULT_AMENDMENT_TEMPLATES)
    report = await engine.run_cycle()
    assert report.amendments == {}
    assert report.flags == []


@pytest.mark.asyncio
async def test_model_drafts_amendment_when_no_template():
    store = InMemoryEvalStore()
    _seed(store, "agentA", [{"grounding": 2}] * 3)

    calls: list[tuple[str, str]] = []

    def fake_llm(system: str, user: str) -> str:
        calls.append((system, user))
        return "Drafted addendum: cite your sources."

    engine = SelfImprovementEngine(
        store, templates={}, complete=fake_llm, threshold=6.0, min_evals=3
    )
    report = await engine.run_cycle()
    assert report.amendments["agentA"].source == "model"
    assert report.amendments["agentA"].text == "Drafted addendum: cite your sources."
    assert len(calls) == 1  # model was actually consulted


@pytest.mark.asyncio
async def test_fallback_when_no_template_and_no_model():
    store = InMemoryEvalStore()
    _seed(store, "agentA", [{"grounding": 2}] * 3)
    engine = SelfImprovementEngine(store, templates={}, threshold=6.0, min_evals=3)
    report = await engine.run_cycle()
    assert report.amendments["agentA"].source == "fallback"
    assert "grounding" in report.amendments["agentA"].text


@pytest.mark.asyncio
async def test_per_agent_cap_is_respected():
    store = InMemoryEvalStore()
    # two weak dimensions for the same agent
    _seed(store, "agentA", [{"grounding": 2, "clarity": 2}] * 3)
    engine = SelfImprovementEngine(
        store,
        templates=DEFAULT_AMENDMENT_TEMPLATES,
        threshold=6.0,
        min_evals=3,
        max_amendments_per_agent=1,
    )
    report = await engine.run_cycle()
    assert len(report.amendments) == 1  # capped
    assert "agentA" in report.skipped


@pytest.mark.asyncio
async def test_report_is_persisted():
    store = InMemoryEvalStore()
    _seed(store, "agentA", [{"grounding": 2}] * 3)
    engine = SelfImprovementEngine(store, templates=DEFAULT_AMENDMENT_TEMPLATES)
    await engine.run_cycle()
    reports = store.recent_reports()
    assert len(reports) == 1
    assert "agentA" in reports[0]["amendments"]


def test_apply_amendment_appends():
    out = SelfImprovementEngine.apply_amendment("BASE", "ADDENDUM")
    assert out == "BASE\n\nADDENDUM"


@pytest.mark.asyncio
async def test_revise_prompt_rewrites_via_model():
    def fake_llm(system: str, user: str) -> str:
        assert "grounding" in user  # feedback made it into the prompt
        return "REVISED PROMPT"

    result = eval_result("agentA", {"grounding": 2, "clarity": 8})
    revised = await revise_prompt("ORIGINAL PROMPT", result, fake_llm)
    assert revised == "REVISED PROMPT"


@pytest.mark.asyncio
async def test_revise_prompt_falls_back_to_original_on_empty():
    revised = await revise_prompt("ORIGINAL", eval_result("a", {"clarity": 2}), lambda s, u: "")
    assert revised == "ORIGINAL"
