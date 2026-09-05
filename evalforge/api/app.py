"""FastAPI application factory and middleware configuration for EvalForge."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from evalforge.api.routes.health import router as health_router
from evalforge.api.routes.guardrails import router as guardrails_router
from evalforge.api.routes.traces import router as traces_router
from evalforge.api.routes.evaluations import router as evaluations_router
from evalforge.api.routes.datasets import router as datasets_router
from evalforge.config import settings
from evalforge.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing DB schema migration."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Create and configure production FastAPI application."""
    app = FastAPI(
        title="EvalForge API",
        description=(
            "Production LLM Evaluation & Guardrails Platform.\n\n"
            "Features:\n"
            "- OpenTelemetry-style Span Tree Tracing\n"
            "- Composite Scoring Engine (Deterministic, Lexical, LLM-as-a-Judge)\n"
            "- Statistical Regression Detection with paired z-scores and bootstrap CI\n"
            "- Guardrail Middleware (PII redaction, Prompt Injection, Toxicity, JSON Schema, Cost SLA)\n"
            "- Dataset Snapshot Versioning with SHA-256 content hashing"
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Observability and timing middleware
    @app.middleware("http")
    async def add_process_time_and_trace_headers(request: Request, call_next):
        start_time = time.perf_counter()
        trace_id = request.headers.get("x-trace-id", uuid.uuid4().hex)
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000.0
        response.headers["x-trace-id"] = trace_id
        response.headers["x-latency-ms"] = f"{process_time:.2f}"
        return response

    # Include routers
    app.include_router(health_router)
    app.include_router(guardrails_router)
    app.include_router(traces_router)
    app.include_router(evaluations_router)
    app.include_router(datasets_router)

    return app


app = create_app()
