"""Optional tracing hook — a no-op unless explicitly configured.

The harness should not force an OpenTelemetry / OpenInference / Arize Phoenix
dependency on every adopter.  This module provides a tiny :class:`Tracer`
protocol and two implementations:

- :class:`NoOpTracer` — the default; records nothing, costs nothing.
- :class:`OTelTracer` — wraps an OpenTelemetry tracer if the SDK is installed
  and a provider is available; otherwise it degrades to a no-op.

:func:`get_tracer` returns a usable tracer given a name, preferring OTel when
available.  Everything is safe to call with no keys, no collector, and no
``opentelemetry`` package installed.

Generalised from working-capital-optimizer's ``wco.tracing.phoenix_setup``
(``setup_phoenix`` / ``store_trace``), which wired OpenInference instrumentors
to a Phoenix Cloud OTLP exporter.  Adopters that want full Phoenix export can
call :meth:`OTelTracer.from_provider` with a provider they configure exactly as
that donor does.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = ["Span", "Tracer", "NoOpTracer", "OTelTracer", "get_tracer"]


@runtime_checkable
class Span(Protocol):
    """Minimal span surface used by the harness."""

    def set_attribute(self, key: str, value: Any) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracer surface: a context manager yielding a :class:`Span`."""

    def span(self, name: str, **attributes: Any) -> Any:
        """Return a context manager that yields a :class:`Span`."""
        ...


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        return None


class NoOpTracer:
    """A tracer that records nothing.  The default when tracing is unconfigured."""

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()


class OTelTracer:
    """Adapter over an OpenTelemetry tracer.

    Prefer :func:`get_tracer`, which builds this only when the SDK is present.

    Parameters:
        otel_tracer: An ``opentelemetry.trace.Tracer`` instance.
    """

    def __init__(self, otel_tracer: Any) -> None:
        self._tracer = otel_tracer

    @classmethod
    def from_provider(cls, name: str, provider: Any | None = None) -> "OTelTracer":
        """Build from an OTel ``TracerProvider`` (or the global one)."""
        from opentelemetry import trace  # type: ignore[import-not-found]

        tracer = trace.get_tracer(name, tracer_provider=provider)
        return cls(tracer)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        with self._tracer.start_as_current_span(name) as sp:
            for key, value in attributes.items():
                try:
                    sp.set_attribute(key, value)
                except Exception:  # pragma: no cover - defensive
                    pass
            yield sp


def get_tracer(name: str = "cubiczan_eval", *, provider: Any | None = None) -> Tracer:
    """Return the best available tracer for ``name``.

    Returns an :class:`OTelTracer` when ``opentelemetry`` is importable,
    otherwise a :class:`NoOpTracer`.  Never raises.

    Args:
        name: Instrumentation scope name.
        provider: Optional OTel ``TracerProvider``; defaults to the global one.
    """
    try:
        return OTelTracer.from_provider(name, provider)
    except Exception:
        logger.debug("OpenTelemetry unavailable — using NoOpTracer")
        return NoOpTracer()
