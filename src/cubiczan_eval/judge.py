"""LLM-as-a-Judge scoring against a :class:`~cubiczan_eval.rubric.Rubric`.

The judge has **no LLM provider dependency**: the caller injects a completion
callable ``complete(system_prompt, user_prompt) -> str`` (sync or async), and
the judge builds the prompt, parses the strict-JSON scores, and computes the
weighted overall score.

Ported from working-capital-optimizer's ``wco.eval.evaluator`` and
generalised: *recommendation* -> *output*, Gemini client -> injected callable.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from typing import Any, Awaitable, Callable, Union

from cubiczan_eval.rubric import DimensionScore, EvalResult, Rubric

logger = logging.getLogger(__name__)

__all__ = ["CompletionFn", "LLMJudge"]

#: A pluggable LLM completion function: ``(system_prompt, user_prompt) -> text``.
#: May be a plain function or an async coroutine function.
CompletionFn = Callable[[str, str], Union[str, Awaitable[str]]]

# Truncation limits (ported from wco.eval.evaluator)
_CONTEXT_PROMPT_LIMIT = 3000
_OUTPUT_PROMPT_LIMIT = 2000
_CONTEXT_SUMMARY_LIMIT = 1000


class LLMJudge:
    """Scores outputs against a rubric via an injected LLM callable.

    Parameters:
        rubric: The scoring rubric.
        complete: Completion callable ``(system_prompt, user_prompt) -> str``;
            sync or async.
        store: Optional persistence backend implementing
            :class:`~cubiczan_eval.store.EvalStore` (``add_result``).
        system_prompt: Optional full override of the judge system prompt.
            Defaults to ``rubric.render_system_prompt()``.
        judge_role: Optional role paragraph passed to
            :meth:`Rubric.render_system_prompt` (ignored when
            ``system_prompt`` is given).
    """

    def __init__(
        self,
        rubric: Rubric,
        complete: CompletionFn,
        *,
        store: Any | None = None,
        system_prompt: str | None = None,
        judge_role: str | None = None,
    ) -> None:
        self._rubric = rubric
        self._complete = complete
        self._store = store
        self._system_prompt = system_prompt or rubric.render_system_prompt(judge_role)

    @property
    def rubric(self) -> Rubric:
        """The rubric this judge scores against."""
        return self._rubric

    @property
    def system_prompt(self) -> str:
        """The judge system prompt in use."""
        return self._system_prompt

    # ── Public API ───────────────────────────────────────────────────────

    async def evaluate(
        self,
        output: str,
        context: str,
        *,
        agent_name: str = "",
        item_id: str | None = None,
        store: bool = True,
    ) -> EvalResult:
        """Evaluate a single output.

        Args:
            output: The output text to evaluate.
            context: The context / problem description the output was
                generated for.
            agent_name: Name of the producing agent/prompt (for tracking).
            item_id: Optional ID linking to the source record.
            store: Whether to persist the result to the configured store.

        Returns:
            An :class:`EvalResult` with dimension scores and overall score.
        """
        prompt = self.build_user_prompt(output, context)
        raw = await self._call(self._system_prompt, prompt)

        scores = self.parse_scores(raw)
        overall = self.compute_overall(scores)

        result = EvalResult(
            item_id=item_id,
            agent_name=agent_name,
            output_text=output,
            context_summary=context[:_CONTEXT_SUMMARY_LIMIT],
            scores=scores,
            overall_score=round(overall, 2),
        )

        logger.info(
            "Evaluation complete: overall=%.1f/%d (%s)",
            overall,
            self._rubric.max_score,
            ", ".join(f"{s.dimension}={s.score}" for s in scores),
        )

        if store and self._store is not None:
            try:
                self._store.add_result(result)
            except Exception as exc:
                logger.warning("Failed to store evaluation result: %s", exc)

        return result

    async def evaluate_batch(
        self,
        items: list[dict[str, str]],
        context: str,
    ) -> list[EvalResult]:
        """Evaluate multiple outputs in sequence.

        Args:
            items: List of dicts with ``text`` and optionally ``agent_name``
                and ``id`` keys.
            context: Shared context / problem description.

        Returns:
            List of :class:`EvalResult` in the same order.
        """
        results: list[EvalResult] = []
        for item in items:
            result = await self.evaluate(
                output=item["text"],
                context=context,
                agent_name=item.get("agent_name", ""),
                item_id=item.get("id"),
                store=False,  # Store after batch
            )
            results.append(result)

        if self._store is not None:
            try:
                for r in results:
                    self._store.add_result(r)
            except Exception as exc:
                logger.warning("Failed to batch-store evaluation results: %s", exc)

        return results

    def evaluate_sync(
        self,
        output: str,
        context: str,
        *,
        agent_name: str = "",
        item_id: str | None = None,
        store: bool = True,
    ) -> EvalResult:
        """Synchronous wrapper around :meth:`evaluate`.

        Must not be called from inside a running event loop.
        """
        return asyncio.run(
            self.evaluate(
                output,
                context,
                agent_name=agent_name,
                item_id=item_id,
                store=store,
            )
        )

    # ── Internals ────────────────────────────────────────────────────────

    async def _call(self, system: str, user: str) -> str:
        """Invoke the injected completion callable (sync or async)."""
        result = self._complete(system, user)
        if inspect.isawaitable(result):
            result = await result
        return result or ""

    @staticmethod
    def build_user_prompt(output: str, context: str) -> str:
        """Construct the evaluation prompt."""
        return f"""## Context / Problem Description
{context[:_CONTEXT_PROMPT_LIMIT]}

## Output to Evaluate
{output[:_OUTPUT_PROMPT_LIMIT]}

Evaluate this output on all dimensions of the rubric. Be rigorous and fair."""

    def parse_scores(self, raw: str) -> list[DimensionScore]:
        """Parse the JSON scores from the judge response.

        Tolerates markdown code fences and leading/trailing prose; clamps
        scores into the rubric range.  Returns ``[]`` on parse failure.
        """
        try:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            json_str = match.group(1).strip() if match else raw
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                json_str = json_str[start : end + 1]
            data = json.loads(json_str)
            return [
                DimensionScore(
                    dimension=s["dimension"],
                    score=self._rubric.clamp(int(s["score"])),
                    justification=s.get("justification", ""),
                )
                for s in data.get("scores", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse judge scores: %s — raw: %s", exc, raw[:200])
            return []

    def compute_overall(self, scores: list[DimensionScore]) -> float:
        """Compute the weighted overall score."""
        if not scores:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for s in scores:
            w = self._rubric.weight_for(s.dimension)
            weighted_sum += s.score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight else 0.0
