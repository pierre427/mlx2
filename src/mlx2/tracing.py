"""Optional host-only OTLP request tracing.

OpenTelemetry is imported only when an endpoint is configured.  The SDK's
BatchSpanProcessor owns export I/O, so request threads never perform a
synchronous network export.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Mapping
from typing import Any


class _CountingExporter:
    def __init__(self, owner: OptionalRequestTracer, delegate: Any) -> None:
        self._owner = owner
        self._delegate = delegate

    def export(self, spans):
        try:
            result = self._delegate.export(spans)
        except Exception:
            self._owner._bump("export_errors")
            raise
        if getattr(result, "name", "") == "SUCCESS":
            self._owner._bump("exported", len(spans))
        else:
            self._owner._bump("export_errors")
        return result

    def shutdown(self, *args, **kwargs):
        return self._delegate.shutdown(*args, **kwargs)

    def force_flush(self, *args, **kwargs):
        flush = getattr(self._delegate, "force_flush", None)
        return True if flush is None else flush(*args, **kwargs)


class RequestTrace:
    def __init__(self, owner: OptionalRequestTracer, span: Any = None) -> None:
        self._owner = owner
        self._span = span
        self._finished = False

    def finish(self, status: int) -> None:
        if self._finished:
            return
        self._finished = True
        if self._span is None:
            return
        try:
            self._span.set_attribute("http.response.status_code", int(status))
            self._span.end()
        except Exception:
            self._owner._bump("failed")
        else:
            self._owner._bump("completed")


class OptionalRequestTracer:
    """Lazy W3C-context server spans with asynchronous OTLP export."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._tracer = None
        self._provider = None
        self._propagate = None
        self._span_kind = None
        if endpoint is None:
            return
        try:
            from opentelemetry import propagate
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.trace import SpanKind

            provider = TracerProvider(
                resource=Resource.create({"service.name": "mlx2"})
            )
            provider.add_span_processor(
                BatchSpanProcessor(
                    _CountingExporter(
                        self, OTLPSpanExporter(endpoint=endpoint)
                    )
                )
            )
            self._provider = provider
            self._tracer = provider.get_tracer("mlx2.server")
            self._propagate = propagate
            self._span_kind = SpanKind.SERVER
        except Exception as error:
            self._bump("setup_errors")
            raise RuntimeError(
                "OTLP tracing requires the mlx2[observability] dependencies"
            ) from error

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    def _bump(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[key] += int(amount)

    def start(
        self, name: str, headers: Mapping[str, str], attributes: Mapping[str, Any]
    ) -> RequestTrace:
        if self._tracer is None:
            return RequestTrace(self)
        try:
            context = self._propagate.extract(headers)
            span = self._tracer.start_span(
                name,
                context=context,
                kind=self._span_kind,
                attributes=dict(attributes),
            )
        except Exception:
            self._bump("failed")
            return RequestTrace(self)
        self._bump("started")
        return RequestTrace(self, span)

    def prometheus_snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                key: int(self._counts.get(key, 0))
                for key in (
                    "started", "completed", "failed", "exported",
                    "setup_errors", "export_errors",
                )
            }

    def close(self) -> None:
        """Flush and stop the asynchronous exporter when tracing is enabled."""

        if self._provider is not None:
            self._provider.shutdown()
