"""Base classes and types for Guardrail middleware."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Optional


class GuardrailAction(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    REDACT = "REDACT"


class GuardrailSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class GuardrailViolation:
    """Detailed record of a specific guardrail check failure."""
    rule_name: str
    severity: GuardrailSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value if hasattr(self.severity, "value") else str(self.severity),
            "message": self.message,
            "details": self.details,
        }


@dataclass
class GuardrailResult:
    """Outcome of a guardrail inspection."""
    guardrail_name: str
    passed: bool
    action: GuardrailAction
    score: float = 1.0  # 1.0 = pristine/safe, 0.0 = completely unsafe
    violations: list[GuardrailViolation] = field(default_factory=list)
    original_text: str = ""
    sanitized_text: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "guardrail_name": self.guardrail_name,
            "passed": self.passed,
            "action": self.action.value if hasattr(self.action, "value") else str(self.action),
            "score": round(self.score, 4),
            "violations": [v.to_dict() for v in self.violations],
            "original_text": self.original_text,
            "sanitized_text": self.sanitized_text,
            "latency_ms": round(self.latency_ms, 3),
            "metadata": self.metadata,
        }


class BaseGuardrail(ABC):
    """Abstract base class for all input and output guardrails."""

    name: str = "base_guardrail"
    description: str = "Base guardrail validator"

    @abstractmethod
    def check(self, text: str, context: Optional[dict[str, Any]] = None) -> GuardrailResult:
        """Synchronously check text for guardrail compliance."""
        pass

    async def check_async(self, text: str, context: Optional[dict[str, Any]] = None) -> GuardrailResult:
        """Asynchronously check text (defaults to running check in default threadpool)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check, text, context)
