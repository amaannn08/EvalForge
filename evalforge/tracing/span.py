"""Span and Trace domain abstractions with hierarchical tree rendering."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from evalforge.tracing.types import SpanStatus, SpanType


def generate_trace_id() -> str:
    """Generate 32-hex-character W3C OpenTelemetry standard trace ID."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate 16-hex-character W3C OpenTelemetry standard span ID."""
    return uuid.uuid4().hex[:16]


@dataclass
class SpanEvent:
    """An annotated point in time within a span (e.g. prompt tokenized, stream chunk)."""
    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """A single unit of work in a trace tree (e.g., LLM inference, guard check, tool call)."""
    name: str
    trace_id: str
    span_id: str = field(default_factory=generate_span_id)
    parent_span_id: Optional[str] = None
    span_type: SpanType = SpanType.CUSTOM
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error_message: Optional[str] = None
    _start_perf: float = field(default_factory=time.perf_counter)

    def set_attribute(self, key: str, value: Any) -> Span:
        self.attributes[key] = value
        return self

    def set_attributes(self, mapping: dict[str, Any]) -> Span:
        self.attributes.update(mapping)
        return self

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> Span:
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))
        return self

    def set_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: Optional[str] = None,
        cost_usd: Optional[float] = None,
    ) -> Span:
        total = prompt_tokens + completion_tokens
        self.set_attributes({
            "llm.usage.prompt_tokens": prompt_tokens,
            "llm.usage.completion_tokens": completion_tokens,
            "llm.usage.total_tokens": total,
        })
        if model_name:
            self.set_attribute("llm.model", model_name)
        if cost_usd is not None:
            self.set_attribute("llm.usage.cost_usd", cost_usd)
        return self

    def record_exception(self, exc: BaseException, escaped: bool = True) -> Span:
        self.status = SpanStatus.ERROR
        self.error_message = f"{type(exc).__name__}: {str(exc)}"
        self.add_event(
            "exception",
            {
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
                "exception.escaped": escaped,
            },
        )
        return self

    def finish(self, status: Optional[SpanStatus] = None) -> Span:
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)
            self.duration_ms = max(0.0, (time.perf_counter() - self._start_perf) * 1000.0)
            if status is not None:
                self.status = status
            elif self.status == SpanStatus.UNSET:
                self.status = SpanStatus.OK
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type.value if isinstance(self.span_type, SpanType) else self.span_type,
            "status": self.status.value if isinstance(self.status, SpanStatus) else self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "error_message": self.error_message,
        }


@dataclass
class Trace:
    """A distributed trace composed of spans forming a directed acyclic execution graph."""
    trace_id: str = field(default_factory=generate_trace_id)
    name: str = "root"
    spans: list[Span] = field(default_factory=list)
    baggage: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    @property
    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        root = self.get_root_span()
        if root and root.duration_ms > 0:
            return root.duration_ms
        return sum(s.duration_ms for s in self.spans if s.parent_span_id is None)

    @property
    def total_tokens(self) -> int:
        return sum(int(s.attributes.get("llm.usage.total_tokens", 0)) for s in self.spans)

    @property
    def estimated_cost_usd(self) -> float:
        return sum(float(s.attributes.get("llm.usage.cost_usd", 0.0)) for s in self.spans)

    @property
    def status(self) -> SpanStatus:
        if any(s.status == SpanStatus.ERROR for s in self.spans):
            return SpanStatus.ERROR
        return SpanStatus.OK

    def get_root_span(self) -> Optional[Span]:
        for s in self.spans:
            if s.parent_span_id is None:
                return s
        return self.spans[0] if self.spans else None

    def build_span_tree(self) -> list[dict[str, Any]]:
        """Construct hierarchical nested span tree structure."""
        span_map: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []

        for s in self.spans:
            node = s.to_dict()
            node["children"] = []
            span_map[s.span_id] = node

        for s in self.spans:
            node = span_map[s.span_id]
            if s.parent_span_id and s.parent_span_id in span_map:
                span_map[s.parent_span_id]["children"].append(node)
            else:
                roots.append(node)

        return roots

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "status": self.status.value,
            "total_duration_ms": round(self.total_duration_ms, 3),
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "baggage": self.baggage,
            "span_count": len(self.spans),
            "tree": self.build_span_tree(),
        }
