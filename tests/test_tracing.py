"""Unit tests for OpenTelemetry-compatible distributed tracing."""

import asyncio
import pytest
from evalforge.tracing import (
    ConsoleSpanExporter,
    InMemorySpanExporter,
    Span,
    SpanStatus,
    SpanType,
    Tracer,
    get_baggage,
    get_current_span,
    get_current_trace,
    get_tracer,
    set_baggage,
    traceable,
)


def test_span_lifecycle():
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporters=[exporter])

    with tracer.trace_span("root_operation", span_type=SpanType.CHAIN) as root:
        root.set_attribute("env", "test")
        root.set_token_usage(prompt_tokens=100, completion_tokens=50, model_name="gpt-4o", cost_usd=0.002)
        assert root.status == SpanStatus.UNSET

    assert len(exporter.traces) == 1
    trace = exporter.traces[0]
    assert trace.status == SpanStatus.OK
    assert trace.total_tokens == 150
    assert trace.estimated_cost_usd == 0.002
    assert trace.total_duration_ms >= 0.0


def test_nested_span_tree():
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporters=[exporter])

    with tracer.trace_span("parent_span", span_type=SpanType.CHAIN) as parent:
        with tracer.trace_span("child_1", span_type=SpanType.RETRIEVAL) as c1:
            c1.set_attribute("docs_count", 5)
        with tracer.trace_span("child_2", span_type=SpanType.LLM) as c2:
            c2.set_token_usage(20, 30)

    trace = exporter.latest
    assert trace is not None
    assert len(trace.spans) == 3

    tree = trace.build_span_tree()
    assert len(tree) == 1
    root_node = tree[0]
    assert root_node["name"] == "parent_span"
    assert len(root_node["children"]) == 2
    assert {c["name"] for c in root_node["children"]} == {"child_1", "child_2"}


def test_baggage_propagation():
    tracer = Tracer()
    with tracer.trace_span("request_entry") as span:
        set_baggage("user_id", "user_12345")
        set_baggage("org_id", "org_987")

        with tracer.trace_span("sub_action"):
            assert get_baggage("user_id") == "user_12345"
            assert get_baggage("org_id") == "org_987"


def test_exception_recording():
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporters=[exporter])

    with pytest.raises(ValueError):
        with tracer.trace_span("failing_span"):
            raise ValueError("Something went wrong inside the span")

    trace = exporter.latest
    assert trace is not None
    assert trace.status == SpanStatus.ERROR
    failing_span = trace.spans[0]
    assert failing_span.status == SpanStatus.ERROR
    assert "ValueError" in (failing_span.error_message or "")


@pytest.mark.asyncio
async def test_traceable_decorator_async_and_sync():
    exporter = InMemorySpanExporter()
    tracer = get_tracer()
    tracer.exporters = [exporter]

    @traceable("sync_calc", span_type=SpanType.TOOL)
    def compute(a, b):
        return a + b

    @traceable("async_fetch", span_type=SpanType.RETRIEVAL)
    async def fetch_data(x):
        await asyncio.sleep(0.01)
        return x * 2

    res1 = compute(3, 4)
    assert res1 == 7

    res2 = await fetch_data(10)
    assert res2 == 20

    assert len(exporter.traces) >= 2
