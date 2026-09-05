"""Composite scoring engine combining multiple weighted evaluators."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Optional

from evalforge.evaluators.base import BaseEvaluator, EvaluationMetric
from evalforge.tracing.tracer import get_tracer
from evalforge.tracing.types import SpanType


@dataclass
class CompositeEvaluationResult:
    """Aggregate result from a composite evaluation execution."""
    overall_score: float  # [0.0, 1.0]
    passed: bool
    metrics: dict[str, EvaluationMetric] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    input_text: str = ""
    actual_output: str = ""
    expected_output: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "total_latency_ms": round(self.total_latency_ms, 3),
            "input_text": self.input_text,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
        }


class CompositeScorer:
    """Combines deterministic, lexical, and LLM-judge evaluators into a unified composite metric."""

    def __init__(
        self,
        evaluators: list[tuple[BaseEvaluator, float]],
        pass_threshold: float = 0.70,
        enable_tracing: bool = True,
    ):
        if not evaluators:
            raise ValueError("CompositeScorer requires at least one evaluator.")

        self.evaluators = evaluators
        self.pass_threshold = pass_threshold
        self.enable_tracing = enable_tracing

        # Normalize weights so sum equals 1.0
        total_weight = sum(w for _, w in self.evaluators)
        if total_weight <= 0:
            raise ValueError("Sum of evaluator weights must be strictly positive.")
        self.normalized_weights = [w / total_weight for _, w in self.evaluators]

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> CompositeEvaluationResult:
        start_t = time.perf_counter()
        tracer = get_tracer() if self.enable_tracing else None
        metrics_dict: dict[str, EvaluationMetric] = {}
        weighted_score_sum = 0.0

        span_ctx = (
            tracer.trace_span("evaluator.composite", span_type=SpanType.EVALUATION)
            if tracer
            else None
        )

        with span_ctx if span_ctx else _dummy_context() as root_span:
            if root_span:
                root_span.set_attribute("eval.pass_threshold", self.pass_threshold)
                root_span.set_attribute("eval.num_evaluators", len(self.evaluators))

            for (evaluator, raw_weight), norm_weight in zip(self.evaluators, self.normalized_weights):
                sub_span_ctx = (
                    tracer.trace_span(f"evaluator.{evaluator.name}", span_type=SpanType.EVALUATION)
                    if tracer
                    else None
                )
                with sub_span_ctx if sub_span_ctx else _dummy_context() as sub_span:
                    metric = evaluator.evaluate(
                        actual=actual,
                        expected=expected,
                        input_prompt=input_prompt,
                        context=context,
                    )
                    metric.weight = norm_weight
                    metrics_dict[metric.name] = metric
                    weighted_score_sum += norm_weight * metric.score

                    if sub_span:
                        sub_span.set_attribute("metric.name", metric.name)
                        sub_span.set_attribute("metric.score", metric.score)
                        sub_span.set_attribute("metric.passed", metric.passed)
                        sub_span.set_attribute("metric.weight", norm_weight)

        total_latency = (time.perf_counter() - start_t) * 1000.0
        overall_score = min(1.0, max(0.0, weighted_score_sum))
        passed = overall_score >= self.pass_threshold

        return CompositeEvaluationResult(
            overall_score=overall_score,
            passed=passed,
            metrics=metrics_dict,
            total_latency_ms=total_latency,
            input_text=input_prompt or "",
            actual_output=actual,
            expected_output=expected,
            metadata={
                "pass_threshold": self.pass_threshold,
                "weights": {e.name: round(w, 4) for (e, _), w in zip(self.evaluators, self.normalized_weights)},
            },
        )

    async def evaluate_async(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> CompositeEvaluationResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.evaluate, actual, expected, input_prompt, context)


class _dummy_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
