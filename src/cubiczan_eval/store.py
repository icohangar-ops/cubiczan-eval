"""Persistence backends for evaluation results and improvement reports.

Three built-in backends:

- :class:`InMemoryEvalStore` — ephemeral, for tests and short sessions.
- :class:`JSONEvalStore` — a single JSON file, atomic-rename writes.
- :class:`SQLiteEvalStore` — SQLite database (stdlib ``sqlite3``).

Applications with their own database (e.g. CockroachDB/Postgres) can
implement the :class:`EvalStore` protocol instead.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from cubiczan_eval.rubric import EvalResult

__all__ = [
    "EvalStore",
    "InMemoryEvalStore",
    "JSONEvalStore",
    "SQLiteEvalStore",
]


@runtime_checkable
class EvalStore(Protocol):
    """Storage protocol used by :class:`LLMJudge` and :class:`SelfImprovementEngine`."""

    def add_result(self, result: EvalResult) -> None:
        """Persist one evaluation result."""
        ...

    def recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent evaluation results, newest first."""
        ...

    def add_report(self, report: dict[str, Any]) -> None:
        """Persist one self-improvement report (as a plain dict)."""
        ...

    def recent_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent improvement reports, newest first."""
        ...


class InMemoryEvalStore:
    """Ephemeral store — useful for tests and single-process sessions."""

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []
        self._reports: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_result(self, result: EvalResult) -> None:
        with self._lock:
            self._results.append(result.to_dict())

    def recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._results[-limit:]))

    def add_report(self, report: dict[str, Any]) -> None:
        with self._lock:
            self._reports.append(dict(report))

    def recent_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._reports[-limit:]))


class JSONEvalStore:
    """Single-file JSON store with atomic-rename writes.

    File layout::

        {"evaluations": [...], "reports": [...]}

    Parameters:
        path: Path of the JSON file (created on first write).
        max_evaluations: Cap on retained evaluations (oldest dropped).
    """

    def __init__(self, path: str | os.PathLike[str], *, max_evaluations: int = 1000) -> None:
        self._path = Path(path)
        self._max = max_evaluations
        self._lock = threading.Lock()

    # ── Internals ────────────────────────────────────────────────────────

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("store file is not a JSON object")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            data = {}
        data.setdefault("evaluations", [])
        data.setdefault("reports", [])
        return data

    def _save(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=self._path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── EvalStore API ────────────────────────────────────────────────────

    def add_result(self, result: EvalResult) -> None:
        with self._lock:
            data = self._load()
            data["evaluations"].append(result.to_dict())
            data["evaluations"] = data["evaluations"][-self._max :]
            self._save(data)

    def recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load()
        return list(reversed(data["evaluations"][-limit:]))

    def add_report(self, report: dict[str, Any]) -> None:
        with self._lock:
            data = self._load()
            data["reports"].append(dict(report))
            self._save(data)

    def recent_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load()
        return list(reversed(data["reports"][-limit:]))


class SQLiteEvalStore:
    """SQLite-backed store using only the standard library.

    Parameters:
        path: Database file path (``:memory:`` is *not* supported because a
            new connection is opened per operation; use
            :class:`InMemoryEvalStore` instead).
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    item_id TEXT,
                    agent_name TEXT NOT NULL DEFAULT '',
                    output_text TEXT NOT NULL DEFAULT '',
                    context_summary TEXT NOT NULL DEFAULT '',
                    scores TEXT NOT NULL DEFAULT '[]',
                    overall_score REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS improvement_reports (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    report TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

    # ── EvalStore API ────────────────────────────────────────────────────

    def add_result(self, result: EvalResult) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluations (
                    id, item_id, agent_name, output_text, context_summary,
                    scores, overall_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    overall_score = excluded.overall_score,
                    scores = excluded.scores
                """,
                (
                    result.id,
                    result.item_id,
                    result.agent_name,
                    result.output_text,
                    result.context_summary,
                    json.dumps([s.model_dump() for s in result.scores]),
                    result.overall_score,
                    result.created_at,
                ),
            )

    def recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, item_id, agent_name, output_text, context_summary,
                       scores, overall_score, created_at
                FROM evaluations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            try:
                entry["scores"] = json.loads(entry["scores"])
            except (json.JSONDecodeError, TypeError):
                entry["scores"] = []
            results.append(entry)
        return results

    def add_report(self, report: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO improvement_reports (id, timestamp, report)
                VALUES (?, ?, ?)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    str(report.get("report_id", "")),
                    str(report.get("timestamp", "")),
                    json.dumps(report, default=str),
                ),
            )

    def recent_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT report FROM improvement_reports ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        reports = []
        for row in rows:
            try:
                reports.append(json.loads(row["report"]))
            except (json.JSONDecodeError, TypeError):
                continue
        return reports
