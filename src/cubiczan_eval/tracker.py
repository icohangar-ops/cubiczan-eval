"""Rolling-average score tracking and threshold flagging.

The :class:`RollingScoreTracker` groups historical evaluation results by
``(agent_name, dimension)`` and computes a rolling mean over a fixed window
of the most recent evaluations.  When a dimension's rolling mean drops below
a configurable threshold — and enough samples exist to be meaningful — it is
reported as a :class:`Flag`.

This module has **no I/O and no LLM dependency**: it consumes plain result
dicts (as returned by :meth:`~cubiczan_eval.store.EvalStore.recent_results`
or :meth:`~cubiczan_eval.rubric.EvalResult.to_dict`) so it can be unit-tested
without a store, a database, or API keys.

Ported and generalised from working-capital-optimizer's
``wco.eval.self_improvement._identify_weaknesses`` (which hard-coded a
``SCORE_THRESHOLD = 6.0`` / ``MIN_EVALS_FOR_ANALYSIS = 3`` policy over a
CockroachDB read).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_MIN_EVALS",
    "DEFAULT_WINDOW",
    "DimensionStats",
    "Flag",
    "RollingScoreTracker",
]

#: Default rolling-mean threshold below which a dimension is flagged.
DEFAULT_THRESHOLD = 6.0
#: Minimum number of samples before a dimension can be flagged.
DEFAULT_MIN_EVALS = 3
#: Number of most-recent evaluations per (agent, dimension) to average over.
DEFAULT_WINDOW = 20


@dataclass
class DimensionStats:
    """Rolling statistics for one ``(agent, dimension)`` pair.

    Attributes:
        agent_name: The producing agent/prompt name.
        dimension: The rubric dimension key.
        rolling_mean: Mean score over the rolling window.
        sample_count: Number of samples included in ``rolling_mean``.
        justifications: Judge justifications for the windowed samples
            (newest first), useful as evidence when generating amendments.
    """

    agent_name: str
    dimension: str
    rolling_mean: float
    sample_count: int
    justifications: list[str] = field(default_factory=list)


@dataclass
class Flag:
    """A dimension whose rolling mean is below the configured threshold.

    Attributes:
        agent_name: The producing agent/prompt name.
        dimension: The under-performing rubric dimension key.
        rolling_mean: The offending rolling mean.
        threshold: The threshold it fell below.
        sample_count: Samples backing the rolling mean.
        top_issues: A few representative judge justifications (deduplicated).
    """

    agent_name: str
    dimension: str
    rolling_mean: float
    threshold: float
    sample_count: int
    top_issues: list[str] = field(default_factory=list)


class RollingScoreTracker:
    """Compute rolling per-dimension means and flag threshold breaches.

    Parameters:
        threshold: Rolling mean at/above which a dimension is healthy;
            below it the dimension is flagged.
        min_evals: Minimum samples before a dimension is eligible to flag.
        window: Number of most-recent samples per ``(agent, dimension)`` to
            average over.  Results are assumed newest-first (the order
            returned by the store); pass ``newest_first=False`` to
            :meth:`compute` / :meth:`flags` otherwise.
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_evals: int = DEFAULT_MIN_EVALS,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if min_evals <= 0:
            raise ValueError("min_evals must be positive")
        self.threshold = threshold
        self.min_evals = min_evals
        self.window = window

    # ── Public API ───────────────────────────────────────────────────────

    def compute(
        self,
        results: Iterable[dict[str, Any]],
        *,
        newest_first: bool = True,
    ) -> list[DimensionStats]:
        """Compute rolling stats for every ``(agent, dimension)`` pair.

        Args:
            results: Evaluation result dicts, each with ``agent_name`` and a
                ``scores`` list of ``{"dimension", "score", "justification"}``.
            newest_first: Whether ``results`` is ordered newest-first (the
                default store ordering).  The rolling window keeps the newest
                ``window`` samples per pair.

        Returns:
            One :class:`DimensionStats` per pair, sorted by rolling mean
            ascending (weakest first).
        """
        ordered = list(results)
        if not newest_first:
            ordered = list(reversed(ordered))

        # agent -> dimension -> (scores, justifications), newest first
        grouped: dict[str, dict[str, tuple[list[float], list[str]]]] = {}
        for ev in ordered:
            agent = ev.get("agent_name", "") or "unknown"
            for s in _iter_scores(ev):
                dim = s.get("dimension", "")
                if not dim:
                    continue
                try:
                    score = float(s.get("score", 0))
                except (TypeError, ValueError):
                    continue
                bucket = grouped.setdefault(agent, {}).setdefault(dim, ([], []))
                if len(bucket[0]) < self.window:
                    bucket[0].append(score)
                    bucket[1].append(str(s.get("justification", "")))

        stats: list[DimensionStats] = []
        for agent, dims in grouped.items():
            for dim, (scores, justifications) in dims.items():
                if not scores:
                    continue
                mean = sum(scores) / len(scores)
                stats.append(
                    DimensionStats(
                        agent_name=agent,
                        dimension=dim,
                        rolling_mean=round(mean, 4),
                        sample_count=len(scores),
                        justifications=justifications,
                    )
                )

        stats.sort(key=lambda st: st.rolling_mean)
        return stats

    def flags(
        self,
        results: Iterable[dict[str, Any]],
        *,
        newest_first: bool = True,
        max_issues: int = 3,
    ) -> list[Flag]:
        """Return dimensions whose rolling mean is below the threshold.

        A pair is flagged only when ``sample_count >= min_evals`` and
        ``rolling_mean < threshold``.  Flags are sorted weakest-first.
        """
        out: list[Flag] = []
        for st in self.compute(results, newest_first=newest_first):
            if st.sample_count >= self.min_evals and st.rolling_mean < self.threshold:
                out.append(
                    Flag(
                        agent_name=st.agent_name,
                        dimension=st.dimension,
                        rolling_mean=st.rolling_mean,
                        threshold=self.threshold,
                        sample_count=st.sample_count,
                        top_issues=_dedupe(st.justifications, max_issues),
                    )
                )
        return out


# ── Helpers ──────────────────────────────────────────────────────────────


def _iter_scores(ev: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the scores list from a result dict, tolerating JSON strings."""
    raw = ev.get("scores", [])
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    return list(raw) if isinstance(raw, list) else []


def _dedupe(items: list[str], limit: int) -> list[str]:
    """Preserve-order deduplicate non-empty strings, capped at ``limit``."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= limit:
            break
    return out
