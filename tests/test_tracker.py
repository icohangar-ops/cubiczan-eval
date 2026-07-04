"""Rolling-average tracker tests."""

from __future__ import annotations

from cubiczan_eval import RollingScoreTracker
from tests.conftest import eval_result


def _results(agent: str, per_eval_scores: list[dict[str, int]]):
    # newest first, matching store ordering
    return [eval_result(agent, s).to_dict() for s in reversed(per_eval_scores)]


def test_rolling_mean_computed_per_dimension():
    results = _results(
        "agentA",
        [
            {"correctness": 8, "grounding": 4},
            {"correctness": 6, "grounding": 2},
            {"correctness": 7, "grounding": 3},
        ],
    )
    tracker = RollingScoreTracker(threshold=6.0, min_evals=3)
    stats = {(s.agent_name, s.dimension): s for s in tracker.compute(results)}
    assert stats[("agentA", "correctness")].rolling_mean == 7.0
    assert stats[("agentA", "grounding")].rolling_mean == 3.0


def test_below_threshold_dimension_is_flagged():
    results = _results(
        "agentA",
        [
            {"correctness": 8, "grounding": 4},
            {"correctness": 8, "grounding": 2},
            {"correctness": 8, "grounding": 3},
        ],
    )
    tracker = RollingScoreTracker(threshold=6.0, min_evals=3)
    flags = tracker.flags(results)
    assert [f.dimension for f in flags] == ["grounding"]
    assert flags[0].rolling_mean == 3.0
    assert flags[0].sample_count == 3


def test_min_evals_gates_flagging():
    results = _results("agentA", [{"grounding": 1}, {"grounding": 2}])
    tracker = RollingScoreTracker(threshold=6.0, min_evals=3)
    assert tracker.flags(results) == []


def test_window_limits_samples():
    # 5 evals, window of 2 -> only the 2 newest count
    results = _results(
        "agentA",
        [
            {"clarity": 2},  # oldest
            {"clarity": 2},
            {"clarity": 2},
            {"clarity": 9},
            {"clarity": 9},  # newest
        ],
    )
    tracker = RollingScoreTracker(threshold=6.0, min_evals=2, window=2)
    stats = tracker.compute(results)
    assert stats[0].rolling_mean == 9.0  # only the two newest 9s
    assert stats[0].sample_count == 2
    assert tracker.flags(results) == []  # 9.0 not below threshold


def test_scores_stored_as_json_string_are_parsed():
    import json

    d = eval_result("agentA", {"grounding": 2}).to_dict()
    d["scores"] = json.dumps(d["scores"])  # simulate a SQL JSON column
    tracker = RollingScoreTracker(threshold=6.0, min_evals=1)
    flags = tracker.flags([d])
    assert len(flags) == 1 and flags[0].dimension == "grounding"
