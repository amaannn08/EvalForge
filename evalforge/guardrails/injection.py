"""Prompt injection, jailbreak detection, and delimiter escape guardrail."""

from __future__ import annotations

import base64
import re
import time
from typing import Any, Optional

from evalforge.guardrails.base import (
    BaseGuardrail,
    GuardrailAction,
    GuardrailResult,
    GuardrailSeverity,
    GuardrailViolation,
)


class PromptInjectionGuardrail(BaseGuardrail):
    """Detects adversarial prompt injections, DAN/jailbreak patterns, and system prompt leaks."""

    name: str = "prompt_injection"
    description: str = "Analyzes text for adversarial prompt injection, jailbreak keywords, and system overrides."

    # Direct override patterns
    INJECTION_PATTERNS = [
        (
            re.compile(
                r"(ignore|disregard|forget|override|negate)\s+(all\s+)?(previous|prior|above|system)\s+(instructions|directives|prompts|rules)",
                re.IGNORECASE,
            ),
            GuardrailSeverity.CRITICAL,
            0.6,
            "Direct instruction override attempt",
        ),
        (
            re.compile(
                r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(in\s+)?(developer\s+mode|dan|jailbreak|unfiltered|god\s+mode|evil|unrestricted)",
                re.IGNORECASE,
            ),
            GuardrailSeverity.CRITICAL,
            0.5,
            "Persona hijack / Jailbreak mode attempt",
        ),
        (
            re.compile(
                r"(repeat|print|output|display|show|reveal)\s+(the\s+)?(entire|full|exact|verbatim)?\s*(system\s+prompt|initial\s+prompt|developer\s+instructions|hidden\s+rules)",
                re.IGNORECASE,
            ),
            GuardrailSeverity.HIGH,
            0.4,
            "System prompt extraction attempt",
        ),
        (
            re.compile(
                r"(\[INST\]|\[\/INST\]|<\s*\/?[sS][yY][sS]\s*>|<<SYS>>|<\/s>|<\|im_start\|>|<\|im_end\|>)",
                re.IGNORECASE,
            ),
            GuardrailSeverity.HIGH,
            0.4,
            "Chat template delimiter injection attempt",
        ),
        (
            re.compile(
                r"(do\s+not\s+follow|bypass|disable|evade)\s+(safety|content|security|ethical)\s+(guidelines|filters|guards|policies)",
                re.IGNORECASE,
            ),
            GuardrailSeverity.CRITICAL,
            0.5,
            "Safety filter bypass attempt",
        ),
        (
            re.compile(
                r"(\bSystem\s*:|\bAssistant\s*:|\bHuman\s*:|\bUser\s*:)\s*\n",
                re.IGNORECASE,
            ),
            GuardrailSeverity.MEDIUM,
            0.25,
            "Role spoofing delimiter pattern detected",
        ),
    ]

    def __init__(
        self,
        risk_threshold: float = 0.45,
        action: GuardrailAction = GuardrailAction.BLOCK,
        check_base64: bool = True,
    ):
        self.risk_threshold = risk_threshold
        self.action = action
        self.check_base64 = check_base64

    def _inspect_base64_payloads(self, text: str) -> list[tuple[str, str]]:
        """Inspect embedded base64 blocks to prevent obfuscated injection attacks."""
        suspicious: list[tuple[str, str]] = []
        b64_matches = re.findall(r"\b[A-Za-z0-9+/]{20,}={0,2}\b", text)
        for cand in b64_matches:
            try:
                decoded = base64.b64decode(cand, validate=True).decode("utf-8", errors="ignore")
                if len(decoded) > 10 and any(
                    term in decoded.lower()
                    for term in ["ignore", "system prompt", "jailbreak", "instructions", "bypass"]
                ):
                    suspicious.append((cand, decoded))
            except Exception:
                continue
        return suspicious

    def check(self, text: str, context: Optional[dict[str, Any]] = None) -> GuardrailResult:
        start_t = time.perf_counter()
        violations: list[GuardrailViolation] = []
        accumulated_risk = 0.0

        # 1. Regex pattern matching
        for pattern, severity, weight, desc in self.INJECTION_PATTERNS:
            if pattern.search(text):
                accumulated_risk += weight
                violations.append(
                    GuardrailViolation(
                        rule_name="injection.pattern_match",
                        severity=severity,
                        message=desc,
                        details={"weight": weight},
                    )
                )

        # 2. Obfuscated Base64 checks
        if self.check_base64:
            b64_payloads = self._inspect_base64_payloads(text)
            for raw_b64, decoded in b64_payloads:
                accumulated_risk += 0.5
                violations.append(
                    GuardrailViolation(
                        rule_name="injection.obfuscated_base64",
                        severity=GuardrailSeverity.CRITICAL,
                        message=f"Obfuscated base64 injection payload detected: {decoded[:30]}...",
                        details={"decoded_sample": decoded[:50]},
                    )
                )

        accumulated_risk = min(1.0, accumulated_risk)
        safety_score = max(0.0, 1.0 - accumulated_risk)
        is_attack = accumulated_risk >= self.risk_threshold

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if not is_attack:
            return GuardrailResult(
                guardrail_name=self.name,
                passed=True,
                action=GuardrailAction.PASS if not violations else GuardrailAction.WARN,
                score=safety_score,
                violations=violations,
                original_text=text,
                sanitized_text=text,
                latency_ms=latency_ms,
                metadata={"risk_score": round(accumulated_risk, 3)},
            )

        return GuardrailResult(
            guardrail_name=self.name,
            passed=False,
            action=self.action,
            score=safety_score,
            violations=violations,
            original_text=text,
            sanitized_text=text,
            latency_ms=latency_ms,
            metadata={"risk_score": round(accumulated_risk, 3), "blocked": True},
        )
