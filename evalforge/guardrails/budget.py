"""Cost, token consumption, and latency SLA budget guardrail."""

from __future__ import annotations

import time
from typing import Any, Optional

from evalforge.guardrails.base import (
    BaseGuardrail,
    GuardrailAction,
    GuardrailResult,
    GuardrailSeverity,
    GuardrailViolation,
)


class CostBudgetGuardrail(BaseGuardrail):
    """Enforces latency SLAs, token consumption ceilings, and cost budgets."""

    name: str = "cost_budget_sla"
    description: str = "Checks response latency, token count, and estimated inference cost against SLAs."

    def __init__(
        self,
        max_latency_ms: float = 3000.0,
        max_prompt_tokens: int = 4096,
        max_completion_tokens: int = 2048,
        max_total_tokens: int = 6000,
        max_cost_usd: float = 0.05,
        action: GuardrailAction = GuardrailAction.WARN,
    ):
        self.max_latency_ms = max_latency_ms
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.max_total_tokens = max_total_tokens
        self.max_cost_usd = max_cost_usd
        self.action = action

    def check(self, text: str, context: Optional[dict[str, Any]] = None) -> GuardrailResult:
        start_t = time.perf_counter()
        ctx = context or {}
        violations: list[GuardrailViolation] = []

        # 1. Check Latency
        latency_val = ctx.get("latency_ms")
        if latency_val is not None and float(latency_val) > self.max_latency_ms:
            violations.append(
                GuardrailViolation(
                    rule_name="budget.latency_sla_exceeded",
                    severity=GuardrailSeverity.MEDIUM,
                    message=f"Response latency {latency_val:.1f}ms exceeded SLA limit of {self.max_latency_ms:.1f}ms",
                    details={"latency_ms": latency_val, "limit_ms": self.max_latency_ms},
                )
            )

        # 2. Check Token Usage
        prompt_tokens = ctx.get("prompt_tokens")
        completion_tokens = ctx.get("completion_tokens")
        if completion_tokens is None and text:
            # Estimate roughly ~4 chars per token if not provided in context
            completion_tokens = max(1, len(text) // 4)

        if prompt_tokens is not None and prompt_tokens > self.max_prompt_tokens:
            violations.append(
                GuardrailViolation(
                    rule_name="budget.prompt_tokens_exceeded",
                    severity=GuardrailSeverity.HIGH,
                    message=f"Prompt token count {prompt_tokens} exceeded limit of {self.max_prompt_tokens}",
                    details={"prompt_tokens": prompt_tokens, "limit": self.max_prompt_tokens},
                )
            )

        if completion_tokens is not None and completion_tokens > self.max_completion_tokens:
            violations.append(
                GuardrailViolation(
                    rule_name="budget.completion_tokens_exceeded",
                    severity=GuardrailSeverity.HIGH,
                    message=f"Completion token count {completion_tokens} exceeded limit of {self.max_completion_tokens}",
                    details={"completion_tokens": completion_tokens, "limit": self.max_completion_tokens},
                )
            )

        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if total_tokens > self.max_total_tokens:
            violations.append(
                GuardrailViolation(
                    rule_name="budget.total_tokens_exceeded",
                    severity=GuardrailSeverity.HIGH,
                    message=f"Total token count {total_tokens} exceeded limit of {self.max_total_tokens}",
                    details={"total_tokens": total_tokens, "limit": self.max_total_tokens},
                )
            )

        # 3. Check Cost
        cost_usd = ctx.get("cost_usd")
        if cost_usd is not None and float(cost_usd) > self.max_cost_usd:
            violations.append(
                GuardrailViolation(
                    rule_name="budget.cost_exceeded",
                    severity=GuardrailSeverity.CRITICAL,
                    message=f"Inference cost ${cost_usd:.4f} exceeded budget of ${self.max_cost_usd:.4f}",
                    details={"cost_usd": cost_usd, "limit": self.max_cost_usd},
                )
            )

        check_latency = (time.perf_counter() - start_t) * 1000.0
        passed = len(violations) == 0

        score = 1.0 if passed else max(0.0, 1.0 - 0.25 * len(violations))

        return GuardrailResult(
            guardrail_name=self.name,
            passed=passed if self.action == GuardrailAction.BLOCK else True,
            action=GuardrailAction.PASS if passed else self.action,
            score=score,
            violations=violations,
            original_text=text,
            sanitized_text=text,
            latency_ms=check_latency,
            metadata={
                "measured_latency_ms": latency_val,
                "measured_tokens": total_tokens,
                "measured_cost_usd": cost_usd,
            },
        )
