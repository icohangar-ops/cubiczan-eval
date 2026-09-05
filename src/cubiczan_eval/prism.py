"""Best-effort PRISMtrace helper for eval runs."""

from __future__ import annotations

import json
import os
import uuid
from urllib import error, request

DEFAULT_PRISM_HOST = "https://api.prism.blockconvey.com"


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _resolve_host() -> str:
    return _env("PRISMTRACE_HOST") or _env("PRISMTRACE_ENDPOINT") or DEFAULT_PRISM_HOST


def trace_prism_llm(
    *,
    agent_id: str,
    agent_name: str,
    model: str,
    input_messages: list[dict[str, str]],
    output: str,
    latency_ms: int,
    metadata: dict[str, object] | None = None,
    trace_id: str | None = None,
) -> None:
    api_key = _env("PRISMTRACE_API_KEY")
    project_id = _env("PRISMTRACE_PROJECT_ID")
    if not api_key or not project_id:
        return

    payload = {
        "project_id": project_id,
        "model": model,
        "input_messages": input_messages,
        "output_message": output,
        "latency_ms": latency_ms,
        "token_count_input": 0,
        "token_count_output": 0,
        "trace_id": trace_id or uuid.uuid4().hex,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "metadata": metadata or {},
    }
    req = request.Request(
        f"{_resolve_host()}/api/traces",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-PRISMtrace-Key": api_key,
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            resp.read()
    except (error.URLError, TimeoutError, OSError):
        return
