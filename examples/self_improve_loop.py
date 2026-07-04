"""End-to-end usage example — runs fully offline (no API keys).

Demonstrates the whole harness:

1. Build a rubric (the default 4-dim one).
2. Score several outputs with an injected stub judge.
3. Track rolling averages and flag a degrading dimension.
4. Generate a prompt amendment and apply it to the agent's system prompt.

Run:  python examples/self_improve_loop.py
"""

from __future__ import annotations

import asyncio
import json

from cubiczan_eval import (
    DEFAULT_AMENDMENT_TEMPLATES,
    InMemoryEvalStore,
    LLMJudge,
    SelfImprovementEngine,
    default_rubric,
)

SYSTEM_PROMPTS = {
    "report_writer": "You are a financial analyst. Write concise, useful memos."
}


def make_stub_judge(scores: dict[str, int]):
    """A fake judge completion fn. In production, call your LLM here instead."""

    def complete(system: str, user: str) -> str:  # noqa: ARG001
        payload = {
            "scores": [
                {"dimension": d, "score": s, "justification": f"stub: {d}={s}"}
                for d, s in scores.items()
            ]
        }
        return "```json\n" + json.dumps(payload) + "\n```"

    return complete


async def main() -> None:
    store = InMemoryEvalStore()
    rubric = default_rubric()

    # Simulate 4 evaluations where 'grounding' is consistently weak.
    for grounding in (3, 2, 4, 3):
        judge = LLMJudge(
            rubric,
            make_stub_judge(
                {"correctness": 8, "completeness": 8, "grounding": grounding, "clarity": 8}
            ),
            store=store,
        )
        await judge.evaluate(
            output="Extend DPO to 45 days.",
            context="Optimise working capital for a manufacturer.",
            agent_name="report_writer",
        )

    print("Stored evaluations:", len(store.recent_results()))

    # Close the loop: flag weak dimensions and generate amendments.
    engine = SelfImprovementEngine(
        store,
        templates=DEFAULT_AMENDMENT_TEMPLATES,
        threshold=6.0,
        min_evals=3,
    )
    report = await engine.run_cycle()

    print("\nFlags:")
    for f in report.flags:
        print(f"  {f.agent_name}/{f.dimension}: mean={f.rolling_mean} < {f.threshold}")

    print("\nAmendments + application:")
    for agent, amendment in report.amendments.items():
        SYSTEM_PROMPTS[agent] = SelfImprovementEngine.apply_amendment(
            SYSTEM_PROMPTS[agent], amendment
        )
        print(f"  {agent}: fixing '{amendment.dimension}' (source={amendment.source})")

    print("\nAmended system prompt for report_writer:\n")
    print(SYSTEM_PROMPTS["report_writer"])


if __name__ == "__main__":
    asyncio.run(main())
