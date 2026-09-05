"""Contextvars-based thread-safe and coroutine-safe trace propagation."""

from __future__ import annotations

import contextvars
from typing import Optional

from evalforge.tracing.span import Span, Trace

# Current active trace
_CURRENT_TRACE: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar(
    "current_trace", default=None
)

# Current active span
_CURRENT_SPAN: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "current_span", default=None
)


def get_current_trace() -> Optional[Trace]:
    """Retrieve the currently active trace in this context, if any."""
    return _CURRENT_TRACE.get()


def set_current_trace(trace: Optional[Trace]) -> contextvars.Token:
    """Set the active trace in this context."""
    return _CURRENT_TRACE.set(trace)


def get_current_span() -> Optional[Span]:
    """Retrieve the currently active span in this context, if any."""
    return _CURRENT_SPAN.get()


def set_current_span(span: Optional[Span]) -> contextvars.Token:
    """Set the active span in this context."""
    return _CURRENT_SPAN.set(span)


def get_baggage(key: str) -> Optional[str]:
    """Retrieve baggage item from the active trace."""
    trace = get_current_trace()
    if trace:
        return trace.baggage.get(key)
    return None


def set_baggage(key: str, value: str) -> None:
    """Set baggage item on the active trace, propagating through context."""
    trace = get_current_trace()
    if trace:
        trace.baggage[key] = value
