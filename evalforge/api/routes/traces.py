"""Distributed tracing API routes."""

import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from evalforge.db.repository import TraceRepository
from evalforge.db.session import get_db

router = APIRouter(prefix="/api/v1/traces", tags=["Tracing"])


@router.get("")
def list_traces(limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)):
    repo = TraceRepository(db)
    traces = repo.list_traces(limit=limit)
    return [
        {
            "trace_id": t.id,
            "name": t.name,
            "status": t.status,
            "total_duration_ms": t.total_duration_ms,
            "total_tokens": t.total_tokens,
            "estimated_cost_usd": t.estimated_cost_usd,
            "start_time": t.start_time.isoformat() if t.start_time else None,
            "end_time": t.end_time.isoformat() if t.end_time else None,
        }
        for t in traces
    ]


@router.get("/{trace_id}")
def get_trace(trace_id: str, db: Session = Depends(get_db)):
    repo = TraceRepository(db)
    trace = repo.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found.")

    spans_data = [
        {
            "span_id": s.id,
            "parent_span_id": s.parent_span_id,
            "name": s.name,
            "span_type": s.span_type,
            "status": s.status,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "duration_ms": s.duration_ms,
            "attributes": json.loads(s.attributes_json),
            "events": json.loads(s.events_json),
            "error_message": s.error_message,
        }
        for s in trace.spans
    ]

    # Build tree
    span_map = {s["span_id"]: dict(s, children=[]) for s in spans_data}
    roots = []
    for s in spans_data:
        node = span_map[s["span_id"]]
        if s["parent_span_id"] and s["parent_span_id"] in span_map:
            span_map[s["parent_span_id"]]["children"].append(node)
        else:
            roots.append(node)

    return {
        "trace_id": trace.id,
        "name": trace.name,
        "status": trace.status,
        "total_duration_ms": trace.total_duration_ms,
        "total_tokens": trace.total_tokens,
        "estimated_cost_usd": trace.estimated_cost_usd,
        "metadata": json.loads(trace.metadata_json),
        "spans_count": len(trace.spans),
        "tree": roots,
    }
