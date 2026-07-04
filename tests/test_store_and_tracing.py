"""Store backends and the optional (no-op) tracing hook."""

from __future__ import annotations

from cubiczan_eval import (
    EvalStore,
    JSONEvalStore,
    NoOpTracer,
    SQLiteEvalStore,
    get_tracer,
)
from cubiczan_eval.tracing import Tracer
from tests.conftest import eval_result


def test_json_store_roundtrip(tmp_path):
    store = JSONEvalStore(tmp_path / "evals.json")
    store.add_result(eval_result("agentA", {"clarity": 5}))
    store.add_report({"report_id": "r1", "timestamp": "t", "amendments": {}})
    assert isinstance(store, EvalStore)
    recent = store.recent_results()
    assert len(recent) == 1 and recent[0]["agent_name"] == "agentA"
    assert store.recent_reports()[0]["report_id"] == "r1"


def test_sqlite_store_roundtrip(tmp_path):
    store = SQLiteEvalStore(tmp_path / "evals.db")
    store.add_result(eval_result("agentA", {"grounding": 4}))
    store.add_report({"report_id": "r1", "timestamp": "2020", "x": 1})
    assert isinstance(store, EvalStore)
    recent = store.recent_results()
    assert len(recent) == 1
    assert recent[0]["scores"][0]["dimension"] == "grounding"
    assert store.recent_reports()[0]["report_id"] == "r1"


def test_noop_tracer_is_safe():
    tracer = NoOpTracer()
    with tracer.span("eval", agent="a") as span:
        span.set_attribute("k", "v")  # must not raise


def test_get_tracer_returns_a_tracer_without_otel():
    tracer = get_tracer("test")
    assert isinstance(tracer, Tracer)
    with tracer.span("op", foo=1) as span:
        span.set_attribute("bar", 2)  # never raises regardless of backend
