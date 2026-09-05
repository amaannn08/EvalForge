"""Base abstractions for LLM response evaluators and metrics."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvaluationMetric:
    """Evaluation result for a single metric."""
    name: str
    score: float  # Normalized between 0.0 and 1.0
    passed: bool
    weight: float = 1.0
    reason: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "weight": self.weight,
            "reason": self.reason,
            "details": self.details,
            "latency_ms": round(self.latency_ms, 3),
        }


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation metrics."""

    name: str = "base_evaluator"
    description: str = "Base evaluator"

    @abstractmethod
    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        """Evaluate candidate actual output against criteria or expected ground truth."""
        pass

    async def evaluate_async(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        """Async execution fallback."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.evaluate, actual, expected, input_prompt, context)
