"""Self-improvement engine: turn flagged weaknesses into prompt amendments.

The :class:`SelfImprovementEngine` closes the loop:

1. Read recent evaluations from an :class:`~cubiczan_eval.store.EvalStore`.
2. Use a :class:`~cubiczan_eval.tracker.RollingScoreTracker` to flag any
   ``(agent, dimension)`` whose rolling mean has dropped below threshold.
3. For each flag, produce a **prompt amendment** — a block of text to append
   to the offending agent's system prompt (the "grade -> revise" step).
4. Cap amendments per agent per cycle and persist a report.

Amendment text comes from one of three sources, in order of preference:

1. A caller-supplied **template** for the dimension (verbatim, deterministic —
   no model call, so this path works offline and in tests).
2. A pluggable **completion function** (same
   :data:`~cubiczan_eval.judge.CompletionFn` signature as the judge) that
   drafts a bespoke amendment from the judge's justifications.
3. A safe built-in fallback string.

Generalised from working-capital-optimizer's
``wco.eval.self_improvement.SelfImprovementEngine`` — the CockroachDB read is
replaced by the ``EvalStore`` protocol, the four hard-coded WCO templates are
replaced by a caller-supplied dict, and the direct Gemini call is replaced by
the injected ``CompletionFn``.  It also captures strata's persona -> draft ->
grade -> revise loop: :func:`revise_prompt` is the "revise" step applied to a
system prompt given its grades.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from cubiczan_eval.judge import CompletionFn
from cubiczan_eval.prism import trace_prism_llm
from cubiczan_eval.tracker import (
    DEFAULT_MIN_EVALS,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    Flag,
    RollingScoreTracker,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_AMENDMENTS_PER_AGENT",
    "Amendment",
    "ImprovementReport",
    "SelfImprovementEngine",
    "revise_prompt",
]

#: Default cap on amendments generated per agent per cycle (feedback stability).
MAX_AMENDMENTS_PER_AGENT = 1


@dataclass
class Amendment:
    """A generated prompt amendment for one weak dimension.

    Attributes:
        agent_name: The agent whose prompt this amends.
        dimension: The weak dimension that triggered it.
        text: The amendment text to append to the system prompt.
        rolling_mean: The offending rolling mean at generation time.
        source: Where the text came from — ``template``, ``model``, or
            ``fallback``.
        amendment_id: Short unique id.
    """

    agent_name: str
    dimension: str
    text: str
    rolling_mean: float
    source: str = "template"
    amendment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class ImprovementReport:
    """Summary of one self-improvement cycle.

    Attributes:
        report_id: Unique id.
        timestamp: ISO-8601 UTC timestamp.
        flags: Every flag the tracker raised this cycle.
        amendments: Amendments generated (agent -> Amendment).
        skipped: Agents skipped because the per-agent cap was reached.
    """

    report_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    flags: list[Flag] = field(default_factory=list)
    amendments: dict[str, Amendment] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "flags": [asdict(f) for f in self.flags],
            "amendments": {k: asdict(v) for k, v in self.amendments.items()},
            "skipped": list(self.skipped),
        }


class SelfImprovementEngine:
    """Flag weak dimensions and generate prompt amendments to fix them.

    Parameters:
        store: An :class:`~cubiczan_eval.store.EvalStore` to read evaluations
            from and (optionally) write reports to.
        templates: Optional mapping of ``dimension -> amendment text``.  When a
            flagged dimension has a template, it is used verbatim (no model
            call).  This is the offline-friendly path.
        complete: Optional :data:`~cubiczan_eval.judge.CompletionFn` used to
            draft an amendment when no template exists.  If omitted, a safe
            built-in fallback string is used instead.
        tracker: Optional pre-configured
            :class:`~cubiczan_eval.tracker.RollingScoreTracker`.  If omitted,
            one is built from ``threshold`` / ``min_evals`` / ``window``.
        threshold, min_evals, window: Passed to a new tracker when ``tracker``
            is not supplied.
        max_amendments_per_agent: Per-agent-per-cycle cap.
        history_limit: How many recent evaluations to read from the store.
    """

    def __init__(
        self,
        store: Any,
        *,
        templates: dict[str, str] | None = None,
        complete: CompletionFn | None = None,
        tracker: RollingScoreTracker | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_evals: int = DEFAULT_MIN_EVALS,
        window: int = DEFAULT_WINDOW,
        max_amendments_per_agent: int = MAX_AMENDMENTS_PER_AGENT,
        history_limit: int = 100,
    ) -> None:
        self._store = store
        self._templates = dict(templates or {})
        self._complete = complete
        self._tracker = tracker or RollingScoreTracker(
            threshold=threshold, min_evals=min_evals, window=window
        )
        self._max_per_agent = max_amendments_per_agent
        self._history_limit = history_limit

    @property
    def tracker(self) -> RollingScoreTracker:
        """The rolling-score tracker in use."""
        return self._tracker

    # ── Public API ───────────────────────────────────────────────────────

    async def run_cycle(self, *, persist: bool = True) -> ImprovementReport:
        """Run one improvement cycle over the store's recent evaluations.

        Reads evaluations, flags weak dimensions, generates one amendment per
        flag (respecting the per-agent cap), and optionally persists the report.

        Returns:
            An :class:`ImprovementReport`.
        """
        started = perf_counter()
        report = ImprovementReport()

        try:
            results = self._store.recent_results(self._history_limit)
        except Exception as exc:
            logger.warning("Cannot read evaluations for self-improvement: %s", exc)
            return report

        flags = self._tracker.flags(results)
        report.flags = flags

        per_agent: dict[str, int] = {}
        for flag in flags:
            agent = flag.agent_name
            if per_agent.get(agent, 0) >= self._max_per_agent:
                if agent not in report.skipped:
                    report.skipped.append(agent)
                logger.info("Skipping amendment for %s — cap reached", agent)
                continue

            amendment = await self._make_amendment(flag)
            report.amendments[agent] = amendment
            per_agent[agent] = per_agent.get(agent, 0) + 1
            logger.info(
                "Amendment for %s/%s (mean=%.2f, source=%s)",
                agent, flag.dimension, flag.rolling_mean, amendment.source,
            )

        if persist and hasattr(self._store, "add_report"):
            try:
                self._store.add_report(report.to_dict())
            except Exception as exc:
                logger.warning("Failed to persist improvement report: %s", exc)

        trace_prism_llm(
            agent_id="cubiczan-eval",
            agent_name="cubiczan-eval",
            model="improvement-cycle",
            input_messages=[
                {"role": "system", "content": "Review recent evaluations and generate prompt amendments."},
                {"role": "user", "content": str(len(report.flags))},
            ],
            output=str(report.to_dict()),
            latency_ms=int((perf_counter() - started) * 1000),
            metadata={"flags": len(report.flags), "amendments": len(report.amendments)},
            trace_id=report.report_id,
        )

        return report

    @staticmethod
    def apply_amendment(system_prompt: str, amendment: str | Amendment) -> str:
        """Append an amendment to a system prompt.

        Accepts either raw amendment text or an :class:`Amendment`.
        """
        text = amendment.text if isinstance(amendment, Amendment) else amendment
        return f"{system_prompt}\n\n{text}"

    # ── Internals ────────────────────────────────────────────────────────

    async def _make_amendment(self, flag: Flag) -> Amendment:
        """Produce an amendment for a flag: template -> model -> fallback."""
        template = self._templates.get(flag.dimension)
        if template:
            return Amendment(
                agent_name=flag.agent_name,
                dimension=flag.dimension,
                text=template,
                rolling_mean=flag.rolling_mean,
                source="template",
            )

        if self._complete is not None:
            text = await self._draft_via_model(flag)
            if text:
                return Amendment(
                    agent_name=flag.agent_name,
                    dimension=flag.dimension,
                    text=text,
                    rolling_mean=flag.rolling_mean,
                    source="model",
                )

        fallback = (
            f"CRITICAL ADDENDUM — {flag.dimension} improvement:\n"
            f"Recent outputs scored {flag.rolling_mean:.1f} on '{flag.dimension}', "
            f"below the {flag.threshold:.1f} threshold. Materially raise the quality "
            f"of this dimension in every response."
        )
        return Amendment(
            agent_name=flag.agent_name,
            dimension=flag.dimension,
            text=fallback,
            rolling_mean=flag.rolling_mean,
            source="fallback",
        )

    async def _draft_via_model(self, flag: Flag) -> str:
        """Ask the injected completion fn to draft an amendment."""
        issues = "\n".join(f"- {i}" for i in flag.top_issues) or "- (none recorded)"
        system = (
            "You improve AI agents by writing concise system-prompt addenda. "
            "Respond ONLY with the addendum text, nothing else."
        )
        user = (
            f'The agent "{flag.agent_name}" has been scoring poorly on the '
            f'"{flag.dimension}" dimension: rolling average {flag.rolling_mean:.1f} '
            f"(threshold {flag.threshold:.1f}) over {flag.sample_count} evaluations.\n\n"
            f"Recent issues from judge justifications:\n{issues}\n\n"
            "Write a concise addendum (3-5 sentences) to append to the agent's "
            "system prompt that would fix this weakness. Be specific and actionable. "
            "Do NOT repeat the original prompt."
        )
        try:
            out = self._complete(system, user)  # type: ignore[misc]
            if inspect.isawaitable(out):
                out = await out
            return (out or "").strip()
        except Exception as exc:
            logger.warning("Model amendment drafting failed: %s", exc)
            return ""


async def revise_prompt(
    system_prompt: str,
    grades: "list[Any] | Any",
    complete: CompletionFn,
    *,
    context: str = "",
) -> str:
    """The "revise" step: rewrite a system prompt given its grades.

    This mirrors strata's ``draft -> grade -> revise`` loop as a single call:
    given a prompt, the judge's per-dimension feedback, and a completion fn, it
    asks the model for an improved prompt.  Unlike
    :meth:`SelfImprovementEngine.apply_amendment` (which *appends*), this
    *rewrites* the whole prompt.

    Args:
        system_prompt: The prompt to revise.
        grades: An :class:`~cubiczan_eval.rubric.EvalResult`, a list of
            :class:`~cubiczan_eval.rubric.DimensionScore`, or any object that
            stringifies to useful grade feedback.
        complete: Pluggable completion fn ``(system, user) -> str``.
        context: Optional extra context about the task.

    Returns:
        The revised prompt (falls back to the original on empty/failed model
        output).
    """
    feedback = _format_grades(grades)
    system = (
        "You are a prompt engineer. Given a system prompt and evaluation "
        "feedback, rewrite the prompt to fix the weaknesses. Respond ONLY with "
        "the full revised system prompt."
    )
    user = (
        f"## Current system prompt\n{system_prompt}\n\n"
        f"## Evaluation feedback\n{feedback}\n"
        + (f"\n## Task context\n{context}\n" if context else "")
        + "\nRewrite the system prompt to address the feedback."
    )
    out = complete(system, user)
    if inspect.isawaitable(out):
        out = await out
    revised = (out or "").strip()
    return revised or system_prompt


def _format_grades(grades: Any) -> str:
    """Best-effort render of grade feedback for a revise prompt."""
    scores = getattr(grades, "scores", grades)
    lines: list[str] = []
    if isinstance(scores, list):
        for s in scores:
            dim = getattr(s, "dimension", None)
            score = getattr(s, "score", None)
            just = getattr(s, "justification", "")
            if dim is not None:
                lines.append(f"- {dim}: {score} — {just}".rstrip(" —"))
    if not lines:
        return str(grades)
    return "\n".join(lines)
