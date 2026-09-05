"""Tracer implementation with async/sync context managers and @traceable decorator."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
import functools
import inspect
from typing import Any, Callable, Generator, Optional

from evalforge.tracing.context import (
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from evalforge.tracing.exporter import InMemorySpanExporter, SpanExporter
from evalforge.tracing.span import Span, Trace
from evalforge.tracing.types import SpanStatus, SpanType


class Tracer:
    """Core tracing manager responsible for span lifecycle and active context tracking."""

    def __init__(self, exporters: Optional[list[SpanExporter]] = None):
        self.exporters: list[SpanExporter] = exporters or [InMemorySpanExporter()]

    def add_exporter(self, exporter: SpanExporter) -> None:
        self.exporters.append(exporter)

    def start_trace(
        self,
        name: str = "root",
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Trace:
        trace = Trace(name=name, session_id=session_id, metadata=metadata or {})
        return trace

    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        parent: Optional[Span] = None,
        attributes: Optional[dict[str, Any]] = None,
        trace: Optional[Trace] = None,
    ) -> Span:
        active_trace = trace or get_current_trace()
        if active_trace is None:
            active_trace = self.start_trace(name=f"trace_{name}")
            set_current_trace(active_trace)

        parent_span = parent if parent is not None else get_current_span()
        parent_id = parent_span.span_id if parent_span else None

        span = Span(
            name=name,
            trace_id=active_trace.trace_id,
            parent_span_id=parent_id,
            span_type=span_type,
            attributes=attributes or {},
        )
        active_trace.add_span(span)
        return span

    @contextmanager
    def trace_span(
        self,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Synchronous context manager for span tracing."""
        active_trace = get_current_trace()
        is_root = active_trace is None
        if is_root:
            active_trace = self.start_trace(name=name)
            token_trace = set_current_trace(active_trace)

        parent_span = get_current_span()
        span = self.start_span(
            name=name,
            span_type=span_type,
            parent=parent_span,
            attributes=attributes,
            trace=active_trace,
        )
        token_span = set_current_span(span)

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.finish(SpanStatus.OK)
            else:
                span.finish()
        except Exception as exc:
            span.record_exception(exc)
            span.finish(SpanStatus.ERROR)
            raise
        finally:
            set_current_span(parent_span)
            if is_root:
                active_trace.end_time = datetime.now(timezone.utc)
                for exp in self.exporters:
                    exp.export(active_trace)
                set_current_trace(None)

    @asynccontextmanager
    async def trace_span_async(
        self,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """Asynchronous context manager for span tracing."""
        active_trace = get_current_trace()
        is_root = active_trace is None
        if is_root:
            active_trace = self.start_trace(name=name)
            token_trace = set_current_trace(active_trace)

        parent_span = get_current_span()
        span = self.start_span(
            name=name,
            span_type=span_type,
            parent=parent_span,
            attributes=attributes,
            trace=active_trace,
        )
        token_span = set_current_span(span)

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.finish(SpanStatus.OK)
            else:
                span.finish()
        except Exception as exc:
            span.record_exception(exc)
            span.finish(SpanStatus.ERROR)
            raise
        finally:
            set_current_span(parent_span)
            if is_root:
                active_trace.end_time = datetime.now(timezone.utc)
                for exp in self.exporters:
                    exp.export(active_trace)
                set_current_trace(None)


_GLOBAL_TRACER: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Retrieve or initialize the global Tracer instance."""
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = Tracer()
    return _GLOBAL_TRACER


def init_tracer(exporters: Optional[list[SpanExporter]] = None) -> Tracer:
    """Configure global Tracer with specified exporters."""
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = Tracer(exporters=exporters)
    return _GLOBAL_TRACER


def traceable(
    name: Optional[str] = None,
    span_type: SpanType = SpanType.CUSTOM,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable:
    """Decorator to trace functions and coroutines with automatic span management."""

    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                async with tracer.trace_span_async(span_name, span_type=span_type, attributes=attributes) as span:
                    span.set_attribute("function.args_count", len(args))
                    span.set_attribute("function.kwargs_keys", list(kwargs.keys()))
                    res = await fn(*args, **kwargs)
                    return res
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.trace_span(span_name, span_type=span_type, attributes=attributes) as span:
                    span.set_attribute("function.args_count", len(args))
                    span.set_attribute("function.kwargs_keys", list(kwargs.keys()))
                    res = fn(*args, **kwargs)
                    return res
            return sync_wrapper

    return decorator
