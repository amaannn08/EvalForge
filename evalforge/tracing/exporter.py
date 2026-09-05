"""Span and trace exporters for persistence, memory storage, and console rendering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from rich.console import Console
from rich.tree import Tree

from evalforge.db.repository import TraceRepository
from evalforge.db.session import SessionLocal
from evalforge.tracing.span import Trace


class SpanExporter(ABC):
    """Abstract base class for trace and span exporters."""

    @abstractmethod
    def export(self, trace: Trace) -> None:
        """Export finished trace and its spans."""
        pass


class InMemorySpanExporter(SpanExporter):
    """In-memory trace collector, ideal for test suites and interactive evaluation."""

    def __init__(self):
        self.traces: list[Trace] = []

    def export(self, trace: Trace) -> None:
        self.traces.append(trace)

    def get_by_id(self, trace_id: str) -> Optional[Trace]:
        for t in self.traces:
            if t.trace_id == trace_id:
                return t
        return None

    def clear(self) -> None:
        self.traces.clear()

    @property
    def latest(self) -> Optional[Trace]:
        return self.traces[-1] if self.traces else None


class DatabaseSpanExporter(SpanExporter):
    """Exports traces and spans to SQLite via SQLAlchemy repository."""

    def export(self, trace: Trace) -> None:
        with SessionLocal() as db:
            repo = TraceRepository(db)
            spans_data = [
                {
                    "span_id": s.span_id,
                    "parent_span_id": s.parent_span_id,
                    "name": s.name,
                    "span_type": s.span_type.value if hasattr(s.span_type, "value") else str(s.span_type),
                    "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration_ms": s.duration_ms,
                    "attributes": s.attributes,
                    "events": [e.to_dict() for e in s.events],
                    "error_message": s.error_message,
                }
                for s in trace.spans
            ]
            repo.save_trace(
                trace_id=trace.trace_id,
                name=trace.name,
                status=trace.status.value,
                total_duration_ms=trace.total_duration_ms,
                total_tokens=trace.total_tokens,
                estimated_cost_usd=trace.estimated_cost_usd,
                metadata=trace.metadata,
                spans=spans_data,
                start_time=trace.start_time,
                end_time=trace.end_time,
            )


class ConsoleSpanExporter(SpanExporter):
    """Renders formatted trace tree to stdout using Rich."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def export(self, trace: Trace) -> None:
        tree = Tree(
            f"[bold cyan]Trace: {trace.name}[/] [dim]({trace.trace_id[:8]}...)[/] "
            f"[{'green' if trace.status.value == 'OK' else 'red'}]{trace.status.value}[/] "
            f"[yellow]{trace.total_duration_ms:.2f}ms[/]"
        )

        def _add_nodes(parent_tree, nodes):
            for node in nodes:
                color = "green" if node["status"] == "OK" else "red"
                label = (
                    f"[{color}]{node['name']}[/] [dim]({node['span_type']})[/] "
                    f"[yellow]{node['duration_ms']:.2f}ms[/]"
                )
                if node.get("error_message"):
                    label += f" - [red]{node['error_message']}[/]"
                sub_tree = parent_tree.add(label)
                if node.get("children"):
                    _add_nodes(sub_tree, node["children"])

        span_tree = trace.build_span_tree()
        _add_nodes(tree, span_tree)
        self.console.print(tree)
