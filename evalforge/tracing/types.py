"""Types and enumerations for EvalForge OpenTelemetry-compatible tracing."""

from __future__ import annotations

from enum import Enum


class SpanType(str, Enum):
    LLM = "llm"
    CHAIN = "chain"
    GUARDRAIL = "guardrail"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    EVALUATION = "evaluation"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"
