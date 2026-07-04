"""cubiczan-eval — a reusable LLM-as-a-Judge + self-improvement harness.

Extracted from two donor implementations of the same pattern:

- **working-capital-optimizer** — OpenInference/OTel tracing to Arize Phoenix,
  a 4-dimension rubric that scores every recommendation, rolling averages that
  trigger automatic prompt amendments when a dimension drops below threshold.
- **strata** — a persona -> draft -> grade -> revise loop with a hierarchical
  rubric and a pluggable ``LLMClient`` (MockLLM for offline use).

Public surface (import from the package root):

Rubric layer
    :class:`Rubric`, :class:`RubricDimension`, :class:`ScoreBand`,
    :class:`DimensionScore`, :class:`EvalResult`,
    :func:`default_rubric`.

Judge layer
    :class:`LLMJudge`, :data:`CompletionFn`.

Tracking layer
    :class:`RollingScoreTracker`, :class:`DimensionStats`, :class:`Flag`.

Self-improvement layer
    :class:`SelfImprovementEngine`, :class:`Amendment`,
    :class:`ImprovementReport`, :func:`revise_prompt`.

Persistence
    :class:`EvalStore`, :class:`InMemoryEvalStore`, :class:`JSONEvalStore`,
    :class:`SQLiteEvalStore`.

Tracing (optional, no-op unless configured)
    :class:`Tracer`, :class:`NoOpTracer`, :class:`OTelTracer`,
    :func:`get_tracer`.

Everything works offline: inject a fake ``complete`` callable (or use a
template-only :class:`SelfImprovementEngine`) and no API keys are required.
"""

from __future__ import annotations

from cubiczan_eval.improve import (
    MAX_AMENDMENTS_PER_AGENT,
    Amendment,
    ImprovementReport,
    SelfImprovementEngine,
    revise_prompt,
)
from cubiczan_eval.judge import CompletionFn, LLMJudge
from cubiczan_eval.rubric import (
    DimensionScore,
    EvalResult,
    Rubric,
    RubricDimension,
    ScoreBand,
)
from cubiczan_eval.store import (
    EvalStore,
    InMemoryEvalStore,
    JSONEvalStore,
    SQLiteEvalStore,
)
from cubiczan_eval.tracker import (
    DEFAULT_MIN_EVALS,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    DimensionStats,
    Flag,
    RollingScoreTracker,
)
from cubiczan_eval.tracing import (
    NoOpTracer,
    OTelTracer,
    Tracer,
    get_tracer,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # rubric
    "Rubric",
    "RubricDimension",
    "ScoreBand",
    "DimensionScore",
    "EvalResult",
    "default_rubric",
    # judge
    "LLMJudge",
    "CompletionFn",
    # tracker
    "RollingScoreTracker",
    "DimensionStats",
    "Flag",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MIN_EVALS",
    "DEFAULT_WINDOW",
    # improve
    "SelfImprovementEngine",
    "Amendment",
    "ImprovementReport",
    "revise_prompt",
    "MAX_AMENDMENTS_PER_AGENT",
    # store
    "EvalStore",
    "InMemoryEvalStore",
    "JSONEvalStore",
    "SQLiteEvalStore",
    # tracing
    "Tracer",
    "NoOpTracer",
    "OTelTracer",
    "get_tracer",
    # amendment templates
    "DEFAULT_AMENDMENT_TEMPLATES",
]


def default_rubric() -> Rubric:
    """Return the default 4-dimension rubric.

    Dimensions: ``correctness``, ``completeness``, ``grounding`` (provenance),
    and ``clarity`` — scored 1–10.  This is the generic analogue of
    working-capital-optimizer's relevance/actionability/financial_impact/
    risk_awareness rubric.
    """
    return Rubric(
        name="default_quality_rubric",
        description="Generic quality rubric for LLM-generated outputs.",
        min_score=1,
        max_score=10,
        dimensions=[
            RubricDimension(
                key="correctness",
                name="Correctness",
                weight=0.30,
                description="Is the output factually and logically correct?",
                bands=[
                    ScoreBand(min_score=9, max_score=10, anchor="Fully correct; no errors."),
                    ScoreBand(min_score=7, max_score=8, anchor="Correct with minor slips."),
                    ScoreBand(min_score=4, max_score=6, anchor="Partly correct; notable errors."),
                    ScoreBand(min_score=1, max_score=3, anchor="Largely incorrect."),
                ],
            ),
            RubricDimension(
                key="completeness",
                name="Completeness",
                weight=0.25,
                description="Does it address every part of the task?",
                bands=[
                    ScoreBand(min_score=9, max_score=10, anchor="Covers everything asked."),
                    ScoreBand(min_score=7, max_score=8, anchor="Minor omissions."),
                    ScoreBand(min_score=4, max_score=6, anchor="Significant gaps."),
                    ScoreBand(min_score=1, max_score=3, anchor="Mostly incomplete."),
                ],
            ),
            RubricDimension(
                key="grounding",
                name="Grounding / Provenance",
                weight=0.25,
                description="Are claims grounded in the provided context with traceable sources?",
                bands=[
                    ScoreBand(min_score=9, max_score=10, anchor="Every claim tied to a specific source."),
                    ScoreBand(min_score=7, max_score=8, anchor="Mostly grounded; a few unsupported claims."),
                    ScoreBand(min_score=4, max_score=6, anchor="Some grounding; several unsupported claims."),
                    ScoreBand(min_score=1, max_score=3, anchor="Ungrounded / hallucinated."),
                ],
            ),
            RubricDimension(
                key="clarity",
                name="Clarity",
                weight=0.20,
                description="Is the output clear, well-structured, and readable?",
                bands=[
                    ScoreBand(min_score=9, max_score=10, anchor="Crisp and well-organised."),
                    ScoreBand(min_score=7, max_score=8, anchor="Clear with minor friction."),
                    ScoreBand(min_score=4, max_score=6, anchor="Somewhat confusing."),
                    ScoreBand(min_score=1, max_score=3, anchor="Hard to follow."),
                ],
            ),
        ],
    )


#: Ready-made amendment templates keyed by default-rubric dimension.  Mirrors
#: working-capital-optimizer's ``_DIMENSION_AMENDMENT_TEMPLATES`` but for the
#: generic dimensions.  Pass to :class:`SelfImprovementEngine(templates=...)`.
DEFAULT_AMENDMENT_TEMPLATES: dict[str, str] = {
    "correctness": (
        "CRITICAL ADDENDUM — Correctness improvement:\n"
        "Recent outputs have scored below the correctness threshold. Double-check "
        "every factual and numerical claim before presenting it. State your "
        "reasoning for non-obvious conclusions, and flag anything you are unsure of "
        "rather than asserting it confidently."
    ),
    "completeness": (
        "CRITICAL ADDENDUM — Completeness improvement:\n"
        "Recent outputs have scored below the completeness threshold. Before "
        "finishing, re-read the task and enumerate every requirement; address each "
        "one explicitly. Do not drop sub-questions or edge cases."
    ),
    "grounding": (
        "CRITICAL ADDENDUM — Grounding / provenance improvement:\n"
        "Recent outputs have scored below the grounding threshold. Ground every "
        "claim in a specific item from the provided context and cite it inline. If "
        "you cannot find supporting evidence, state the gap explicitly instead of "
        "guessing."
    ),
    "clarity": (
        "CRITICAL ADDENDUM — Clarity improvement:\n"
        "Recent outputs have scored below the clarity threshold. Lead with a "
        "one-line synthesis, use short paragraphs and structure, and prefer plain "
        "language over jargon."
    ),
}
