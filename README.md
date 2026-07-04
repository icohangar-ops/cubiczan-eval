# cubiczan-eval

A small, reusable **LLM-as-a-Judge + self-improvement** harness for Python
agent systems. Score every output against a rubric, watch rolling averages per
dimension, and when a dimension slips below threshold, automatically produce a
prompt amendment (the "grade → revise" step).

It has **no hard LLM dependency**: the judge and the self-improvement engine
both take an injected `complete(system, user) -> str` callable (sync or async),
so the whole loop runs offline in tests with a stub scorer — no API keys.

```
Evaluate ──► Track rolling means ──► Flag dims below threshold ──► Amend prompts ──► (next run)
 LLMJudge      RollingScoreTracker          Flag                  SelfImprovementEngine
```

## Install

```bash
pip install -e ".[dev]"          # dev = pytest, pytest-asyncio, mypy
pip install -e ".[tracing]"      # optional: opentelemetry + httpx
```

Requires Python ≥ 3.10.

## Quick start

```python
import asyncio
from cubiczan_eval import (
    LLMJudge, default_rubric, InMemoryEvalStore,
    SelfImprovementEngine, DEFAULT_AMENDMENT_TEMPLATES,
)

# 1. A pluggable completion fn. In production, wrap your LLM SDK here.
#    In tests, return canned judge JSON — no keys needed.
def complete(system: str, user: str) -> str:
    ...  # call your model; return its text

store = InMemoryEvalStore()
rubric = default_rubric()   # correctness, completeness, grounding, clarity (1–10)
judge = LLMJudge(rubric, complete, store=store)

async def main():
    # 2. Score outputs as your agents produce them.
    await judge.evaluate(output="…agent output…", context="…the task…",
                         agent_name="report_writer")

    # 3. Periodically close the loop: flag weak dims, generate amendments.
    engine = SelfImprovementEngine(
        store,
        templates=DEFAULT_AMENDMENT_TEMPLATES,  # deterministic, offline
        complete=complete,                      # used only when no template fits
        threshold=6.0, min_evals=3, window=20,
    )
    report = await engine.run_cycle()
    for agent, amendment in report.amendments.items():
        new_prompt = SelfImprovementEngine.apply_amendment(SYSTEM_PROMPTS[agent], amendment)
        SYSTEM_PROMPTS[agent] = new_prompt   # apply on the next run

asyncio.run(main())
```

## Public API

### Rubric layer — `cubiczan_eval.rubric`
- **`Rubric`** — weighted set of `RubricDimension`s; `min_score`/`max_score`
  (default 1–10). `render_system_prompt(role=None)` produces the judge prompt.
- **`RubricDimension`** — `key`, `name`, `weight`, `description`, `bands`.
- **`ScoreBand`** — an anchored score range (`min_score`, `max_score`, `anchor`).
- **`DimensionScore`** / **`EvalResult`** — per-dimension score and full result
  (with weighted `overall_score`, ids, timestamp; `to_dict()`).
- **`default_rubric()`** — the generic 4-dim rubric: **correctness (.30),
  completeness (.25), grounding/provenance (.25), clarity (.20)**.

### Judge layer — `cubiczan_eval.judge`
- **`LLMJudge(rubric, complete, *, store=None, system_prompt=None, judge_role=None)`**
  - `await evaluate(output, context, *, agent_name="", item_id=None, store=True)`
  - `await evaluate_batch(items, context)`
  - `evaluate_sync(...)` — wrapper (not inside a running loop)
- **`CompletionFn`** — `Callable[[str, str], str | Awaitable[str]]`. The single
  injection point for any model provider.

### Tracking layer — `cubiczan_eval.tracker`
- **`RollingScoreTracker(*, threshold=6.0, min_evals=3, window=20)`**
  - `compute(results, *, newest_first=True) -> list[DimensionStats]`
  - `flags(results, *, newest_first=True, max_issues=3) -> list[Flag]`
- **`DimensionStats`** — `agent_name`, `dimension`, `rolling_mean`, `sample_count`.
- **`Flag`** — a dimension below threshold, with `top_issues` (dedup'd
  justifications) as evidence.

### Self-improvement layer — `cubiczan_eval.improve`
- **`SelfImprovementEngine(store, *, templates=None, complete=None, tracker=None,
  threshold=6.0, min_evals=3, window=20, max_amendments_per_agent=1,
  history_limit=100)`**
  - `await run_cycle(*, persist=True) -> ImprovementReport`
  - `apply_amendment(system_prompt, amendment) -> str` (static; appends)
  - Amendment source order: **template → model (`complete`) → built-in fallback**.
- **`Amendment`** / **`ImprovementReport`** — dataclasses; `to_dict()` on report.
- **`revise_prompt(system_prompt, grades, complete, *, context="")`** — the
  *rewrite* variant of the revise step (strata-style): given grades, ask the
  model to rewrite the whole prompt (vs. appending an amendment).
- **`DEFAULT_AMENDMENT_TEMPLATES`** — templates keyed by default-rubric dimension.

### Persistence — `cubiczan_eval.store`
- **`EvalStore`** — runtime `Protocol`: `add_result`, `recent_results`,
  `add_report`, `recent_reports`.
- **`InMemoryEvalStore`** (tests), **`JSONEvalStore`** (atomic file),
  **`SQLiteEvalStore`** (stdlib). Bring your own by implementing `EvalStore`
  (e.g. wrap Postgres/CockroachDB).

### Tracing — `cubiczan_eval.tracing` (optional, no-op by default)
- **`get_tracer(name="cubiczan_eval", *, provider=None) -> Tracer`** — returns
  an `OTelTracer` if `opentelemetry` is importable, else a `NoOpTracer`. Never
  raises, needs no collector.
- **`Tracer`** protocol: `with tracer.span(name, **attrs) as span: ...`.

## Running the tests offline

No API keys required — tests inject stub scorers (the analogue of strata's
`MockLLM`).

```bash
pip install -e ".[dev]"
pytest -q
```

Coverage highlights:
- `test_judge.py` — scoring returns per-dimension scores; weighted overall;
  clamping; async completion; store persistence.
- `test_tracker.py` — rolling mean per dimension; **below-threshold flagging**;
  `min_evals` gate; window limiting; JSON-string score columns.
- `test_improve.py` — **below threshold triggers an amendment** (template,
  model, and fallback paths); per-agent cap; report persistence; `revise_prompt`.
- `test_store_and_tracing.py` — JSON/SQLite roundtrips; no-op tracer safety.

## How the two donor implementations map onto this harness

Both donors implement the same pattern; this package is their common core.

### working-capital-optimizer (`wco/agent/src/wco/`)
| Donor piece | Harness equivalent |
|---|---|
| `eval/evaluator.py` `RecommendationEvaluator` + `_JUDGE_SYSTEM_PROMPT` | `LLMJudge` + `Rubric.render_system_prompt()` |
| 4 dims relevance/actionability/financial_impact/risk_awareness (1–10, weighted) | `default_rubric()` dims correctness/completeness/grounding/clarity (configurable) |
| Direct Gemini client | injected `CompletionFn` |
| `self_improvement.py` `SCORE_THRESHOLD=6.0`, `MIN_EVALS_FOR_ANALYSIS=3` | `RollingScoreTracker(threshold=6.0, min_evals=3)` |
| `_identify_weaknesses()` (group by agent+dim, mean, `< 6.0`) | `RollingScoreTracker.flags()` → `Flag` |
| `SelfImprovementEngine` + `_DIMENSION_AMENDMENT_TEMPLATES` + Gemini fallback | `SelfImprovementEngine` + `DEFAULT_AMENDMENT_TEMPLATES` + `complete` fallback |
| `MAX_AMENDMENTS_PER_AGENT=1`, `apply_amendment()` | `max_amendments_per_agent=1`, `SelfImprovementEngine.apply_amendment()` |
| CockroachDB reads/writes of `evaluations` + `self_improvement_reports` | `EvalStore` protocol (`SQLiteEvalStore`/`JSONEvalStore`, or your own DB adapter) |
| `tracing/phoenix_setup.py` OpenInference → Phoenix Cloud | `cubiczan_eval.tracing` (`OTelTracer`/`NoOpTracer`); adopters keep their Phoenix exporter and pass a provider to `get_tracer` |

### strata (`src/strata/deliverable/`)
| Donor piece | Harness equivalent |
|---|---|
| `factory.py` `DeliverableFactory.run()` persona→draft→grade→revise loop (`max_iterations`, stop on `passed`) | drive with `LLMJudge.evaluate()` + `revise_prompt()`; `passed` = your threshold on `EvalResult.overall_score` |
| `grader.py` `LLMClient` protocol `complete(system, user) -> str` | `CompletionFn` (identical signature) |
| `grader.py` `MockLLM` (deterministic, offline) | the stub scorers in `tests/conftest.py` |
| `AnthropicLLM` / `OpenAICompatibleLLM` | concrete `CompletionFn`s you supply |
| `GRADER_SYSTEM` prompt + hierarchical rubric, `RubricScoreReport.compute()` | `Rubric.render_system_prompt()` + `LLMJudge.compute_overall()` (flat weighted; hierarchy flattens to weighted dims) |
| `author.py` `build_author_prompt()` injecting weakest-5 feedback into next draft | `revise_prompt()` (rewrite) or `apply_amendment()` (append) |
| Postgres `RubricScoreRow` per iteration | `EvalStore` |

**Semantic note.** strata revises *implicitly* by feeding prior grades back
into the next author call; wco revises *explicitly* by appending a targeted
amendment when a rolling average degrades. The harness supports both:
`revise_prompt()` for the rewrite style, `SelfImprovementEngine` +
`apply_amendment()` for the append/threshold style.

## How adopters wire it in

Docs only — these repos are **not** modified.

- **meshcfo** — score each agent's memo with `default_rubric()`; persist to a
  `SQLiteEvalStore` (or a Postgres adapter); run `SelfImprovementEngine.run_cycle()`
  nightly and `apply_amendment()` the results onto each agent's system prompt.
- **stratifi-core** — reuse the strata mapping directly: wrap its existing
  `LLMClient` as a `CompletionFn` and replace the bespoke grader with `LLMJudge`;
  keep the factory loop but use `revise_prompt()` for the revise step.
- **cfo-command-center** — a customer-facing rubric (e.g. correctness, grounding,
  compliance, clarity) via `Rubric`/`RubricDimension`; feed the judge from the
  command bus; surface `RollingScoreTracker.flags()` on a quality dashboard.
- **convergence** — per-sub-agent `agent_name` scoping means one store tracks the
  whole mesh; `run_cycle()` returns per-agent amendments the orchestrator applies
  before the next convergence round.

Common recipe:
1. Define (or reuse) a `Rubric`.
2. Wrap your model as `complete(system, user) -> str`.
3. `judge = LLMJudge(rubric, complete, store=store)` and `evaluate(...)` each output.
4. Periodically `SelfImprovementEngine(store, templates=..., complete=complete).run_cycle()`.
5. `apply_amendment()` (append) or `revise_prompt()` (rewrite) onto the agent prompt.

## Caveats

- **Amendments are not auto-applied.** `run_cycle()` produces amendments and a
  report; wiring them back onto live prompts is the adopter's call (mirrors the
  donors' deliberately staged design).
- **Flat weighting.** The overall score is a flat weighted mean of dimensions.
  strata's 3-level (group→characteristic→attribute) hierarchy must be flattened
  to weighted dimensions.
- **Judge JSON parsing is tolerant but not infallible.** Unparseable responses
  yield an empty score list (overall 0.0) and a warning; consider ret/reprompt
  in production.
- **`SQLiteEvalStore` opens a connection per op** and does not support
  `:memory:`; use `InMemoryEvalStore` for that.
- **Tracing** ships as a no-op. Full Phoenix/OTel export requires the `tracing`
  extra plus a provider you configure (see the wco `phoenix_setup.py` donor).
```
