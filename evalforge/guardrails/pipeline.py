"""Guardrail pipeline and chain-of-responsibility middleware execution engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Optional

from evalforge.guardrails.base import (
    BaseGuardrail,
    GuardrailAction,
    GuardrailResult,
    GuardrailViolation,
)
from evalforge.tracing.tracer import get_tracer
from evalforge.tracing.types import SpanType


class ExecutionMode(str, Enum):
    FAIL_FAST = "FAIL_FAST"
    COLLECT_ALL = "COLLECT_ALL"
    PARALLEL = "PARALLEL"


@dataclass
class PipelineResult:
    """Aggregate result from executing a guardrail chain."""
    passed: bool
    final_action: GuardrailAction
    original_text: str
    sanitized_text: str
    overall_score: float
    total_latency_ms: float
    results: list[GuardrailResult] = field(default_factory=list)
    violations: list[GuardrailViolation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "final_action": self.final_action.value if hasattr(self.final_action, "value") else str(self.final_action),
            "original_text": self.original_text,
            "sanitized_text": self.sanitized_text,
            "overall_score": round(self.overall_score, 4),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "violation_count": len(self.violations),
            "results": [r.to_dict() for r in self.results],
            "violations": [v.to_dict() for v in self.violations],
            "metadata": self.metadata,
        }


class GuardrailPipeline:
    """Configurable middleware pipeline executing a sequence of guardrails."""

    def __init__(
        self,
        guardrails: Optional[list[BaseGuardrail]] = None,
        mode: ExecutionMode = ExecutionMode.COLLECT_ALL,
        enable_tracing: bool = True,
    ):
        self.guardrails: list[BaseGuardrail] = guardrails or []
        self.mode = mode
        self.enable_tracing = enable_tracing

    def add(self, guardrail: BaseGuardrail) -> GuardrailPipeline:
        self.guardrails.append(guardrail)
        return self

    def run(self, text: str, context: Optional[dict[str, Any]] = None) -> PipelineResult:
        """Synchronously execute the guardrail chain."""
        start_t = time.perf_counter()
        tracer = get_tracer() if self.enable_tracing else None
        current_text = text
        results: list[GuardrailResult] = []
        all_violations: list[GuardrailViolation] = []
        overall_passed = True
        highest_action = GuardrailAction.PASS

        action_priority = {
            GuardrailAction.PASS: 0,
            GuardrailAction.WARN: 1,
            GuardrailAction.REDACT: 2,
            GuardrailAction.BLOCK: 3,
        }

        # Tracing context
        span_ctx = (
            tracer.trace_span("guardrail.pipeline", span_type=SpanType.GUARDRAIL)
            if tracer
            else None
        )

        with span_ctx if span_ctx else _dummy_context() as root_span:
            if root_span:
                root_span.set_attribute("pipeline.mode", self.mode.value)
                root_span.set_attribute("pipeline.guards_count", len(self.guardrails))

            for guard in self.guardrails:
                sub_span_ctx = (
                    tracer.trace_span(f"guardrail.{guard.name}", span_type=SpanType.GUARDRAIL)
                    if tracer
                    else None
                )
                with sub_span_ctx if sub_span_ctx else _dummy_context() as sub_span:
                    res = guard.check(current_text, context)
                    results.append(res)
                    all_violations.extend(res.violations)

                    if sub_span:
                        sub_span.set_attribute("guard.name", guard.name)
                        sub_span.set_attribute("guard.passed", res.passed)
                        sub_span.set_attribute("guard.action", res.action.value)
                        sub_span.set_attribute("guard.score", res.score)

                    # Update text if sanitized/redacted
                    if res.sanitized_text != current_text:
                        current_text = res.sanitized_text

                    if action_priority[res.action] > action_priority[highest_action]:
                        highest_action = res.action

                    if not res.passed:
                        overall_passed = False
                        if self.mode == ExecutionMode.FAIL_FAST:
                            break

        total_latency = (time.perf_counter() - start_t) * 1000.0
        scores = [r.score for r in results]
        overall_score = sum(scores) / len(scores) if scores else 1.0

        return PipelineResult(
            passed=overall_passed,
            final_action=highest_action,
            original_text=text,
            sanitized_text=current_text,
            overall_score=overall_score,
            total_latency_ms=total_latency,
            results=results,
            violations=all_violations,
            metadata={"guardrails_executed": [r.guardrail_name for r in results]},
        )

    async def run_async(self, text: str, context: Optional[dict[str, Any]] = None) -> PipelineResult:
        """Asynchronously execute the guardrail chain with parallel support."""
        if self.mode == ExecutionMode.PARALLEL:
            start_t = time.perf_counter()
            tracer = get_tracer() if self.enable_tracing else None

            async def _check_one(g: BaseGuardrail) -> GuardrailResult:
                if tracer:
                    async with tracer.trace_span_async(f"guardrail.{g.name}", span_type=SpanType.GUARDRAIL) as s:
                        res = await g.check_async(text, context)
                        s.set_attribute("guard.passed", res.passed)
                        return res
                return await g.check_async(text, context)

            results: list[GuardrailResult] = await asyncio.gather(
                *[_check_one(g) for g in self.guardrails]
            )

            current_text = text
            all_violations: list[GuardrailViolation] = []
            overall_passed = True
            highest_action = GuardrailAction.PASS
            action_priority = {
                GuardrailAction.PASS: 0,
                GuardrailAction.WARN: 1,
                GuardrailAction.REDACT: 2,
                GuardrailAction.BLOCK: 3,
            }

            for res in results:
                all_violations.extend(res.violations)
                if res.sanitized_text != text and current_text == text:
                    current_text = res.sanitized_text
                if action_priority[res.action] > action_priority[highest_action]:
                    highest_action = res.action
                if not res.passed:
                    overall_passed = False

            total_latency = (time.perf_counter() - start_t) * 1000.0
            scores = [r.score for r in results]
            overall_score = sum(scores) / len(scores) if scores else 1.0

            return PipelineResult(
                passed=overall_passed,
                final_action=highest_action,
                original_text=text,
                sanitized_text=current_text,
                overall_score=overall_score,
                total_latency_ms=total_latency,
                results=results,
                violations=all_violations,
                metadata={"parallel": True},
            )
        else:
            # Fallback to sync run in thread
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.run, text, context)


class _dummy_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
