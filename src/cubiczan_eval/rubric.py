"""Generic rubric schema for LLM-as-a-Judge evaluation.

A :class:`Rubric` is a weighted set of :class:`RubricDimension` objects, each
with optional :class:`ScoreBand` anchors describing what each score range
looks like in practice.  The rubric renders itself into a judge system prompt
(:meth:`Rubric.render_system_prompt`) that instructs the judge LLM to return
strict JSON scores, which :class:`~cubiczan_eval.judge.LLMJudge` parses into
:class:`DimensionScore` / :class:`EvalResult` objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "ScoreBand",
    "RubricDimension",
    "Rubric",
    "DimensionScore",
    "EvalResult",
]


class ScoreBand(BaseModel):
    """One anchored band on a dimension's scoring scale.

    Attributes:
        min_score: Lowest score covered by this band (inclusive).
        max_score: Highest score covered by this band (inclusive).
        anchor: What outputs in this band look like in practice.
    """

    min_score: int
    max_score: int
    anchor: str = Field(min_length=4)

    @model_validator(mode="after")
    def _ordered(self) -> "ScoreBand":
        if self.min_score > self.max_score:
            raise ValueError("min_score must be <= max_score")
        return self

    def label(self) -> str:
        """Render the band range, e.g. ``9-10`` or ``7``."""
        if self.min_score == self.max_score:
            return str(self.min_score)
        return f"{self.min_score}–{self.max_score}"


class RubricDimension(BaseModel):
    """A weighted evaluation dimension.

    Attributes:
        key: Machine name (snake_case) used in judge JSON output.
        name: Human-readable name.
        weight: Relative weight in the overall score (any positive number;
            weights are normalised when the overall score is computed).
        description: Optional one-line description of the dimension.
        bands: Optional anchored score bands, highest first by convention.
    """

    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    weight: float = Field(gt=0)
    description: str = ""
    bands: list[ScoreBand] = Field(default_factory=list)


class Rubric(BaseModel):
    """A complete scoring rubric.

    Attributes:
        name: Rubric identifier.
        description: What this rubric evaluates.
        min_score: Minimum score per dimension (default 1).
        max_score: Maximum score per dimension (default 10).
        dimensions: The weighted dimensions.
    """

    name: str
    description: str = ""
    min_score: int = 1
    max_score: int = 10
    dimensions: list[RubricDimension] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_keys(self) -> "Rubric":
        keys = [d.key for d in self.dimensions]
        if len(set(keys)) != len(keys):
            raise ValueError("dimension keys must be unique")
        if self.min_score >= self.max_score:
            raise ValueError("min_score must be < max_score")
        return self

    # ── Helpers ──────────────────────────────────────────────────────────

    def dimension_keys(self) -> list[str]:
        """Return dimension keys in rubric order."""
        return [d.key for d in self.dimensions]

    def weight_for(self, key: str) -> float:
        """Weight for a dimension key.

        Unknown keys fall back to ``1 / len(dimensions)`` so a judge that
        invents a dimension does not zero out the overall score.
        """
        for d in self.dimensions:
            if d.key == key:
                return d.weight
        return 1.0 / len(self.dimensions)

    def clamp(self, score: int) -> int:
        """Clamp a raw score into the rubric's valid range."""
        return max(self.min_score, min(self.max_score, score))

    # ── Prompt rendering ─────────────────────────────────────────────────

    def render_system_prompt(self, role: str | None = None) -> str:
        """Render the judge system prompt for this rubric.

        Args:
            role: Opening paragraph establishing the judge persona and the
                task under evaluation.  Defaults to a generic impartial-judge
                framing built from the rubric description.

        Returns:
            A full system prompt instructing the judge to score every
            dimension and respond with strict JSON.
        """
        lo, hi = self.min_score, self.max_score
        role_text = role or (
            "You are an expert evaluator acting as an impartial judge. "
            f"Your job is to score AI-generated outputs. {self.description}".strip()
        )

        lines: list[str] = [
            role_text,
            "",
            "For each dimension, provide:",
            f"1. A score from {lo} to {hi} (integer only)",
            "2. A brief justification (1–2 sentences)",
            "",
            "Scoring rubric:",
            "",
        ]
        for dim in self.dimensions:
            lines.append(f"**{dim.name} ({lo}–{hi})**")
            if dim.description:
                lines.append(dim.description)
            for band in dim.bands:
                lines.append(f"- {band.label()}: {band.anchor}")
            lines.append("")

        score_entries = ",\n".join(
            "    {\n"
            f'      "dimension": "{dim.key}",\n'
            '      "score": <int>,\n'
            '      "justification": "<explanation>"\n'
            "    }"
            for dim in self.dimensions
        )
        lines.append(
            "Respond ONLY with valid JSON in this format:\n"
            "```json\n"
            "{\n"
            '  "scores": [\n'
            f"{score_entries}\n"
            "  ]\n"
            "}\n"
            "```"
        )
        return "\n".join(lines)


# ── Result models ──────────────────────────────────────────────────────────


class DimensionScore(BaseModel):
    """A single evaluation dimension score.

    Attributes:
        dimension: Key of the dimension (e.g. ``relevance``).
        score: Integer score within the rubric range.
        justification: Judge's explanation for the score.
    """

    dimension: str
    score: int
    justification: str = ""


class EvalResult(BaseModel):
    """Complete evaluation result for one output.

    Attributes:
        id: Unique evaluation identifier.
        item_id: ID of the evaluated item (if available).
        agent_name: Name of the agent/prompt that produced the output.
        output_text: The full evaluated output text.
        context_summary: Summary of the context provided.
        scores: List of dimension scores.
        overall_score: Weighted average of dimension scores.
        created_at: ISO-8601 timestamp.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    item_id: str | None = None
    agent_name: str = ""
    output_text: str = ""
    context_summary: str = ""
    scores: list[DimensionScore] = Field(default_factory=list)
    overall_score: float = 0.0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return self.model_dump()
