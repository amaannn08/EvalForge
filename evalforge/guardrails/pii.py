"""PII (Personally Identifiable Information) detection and redaction engine."""

from __future__ import annotations

import hashlib
import math
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


def luhn_checksum(card_number_str: str) -> bool:
    """Validate credit card number using standard ISO/IEC 7812 Luhn algorithm."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def calculate_shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy to distinguish random keys/tokens from natural text."""
    if not data:
        return 0.0
    entropy = 0.0
    length = len(data)
    frequencies: dict[str, int] = {}
    for char in data:
        frequencies[char] = frequencies.get(char, 0) + 1
    for count in frequencies.values():
        p_x = count / length
        entropy += -p_x * math.log2(p_x)
    return entropy


class PIIGuardrail(BaseGuardrail):
    """Detects and redacts PII and confidential secrets from LLM inputs/outputs."""

    name: str = "pii_redaction"
    description: str = "Scans for emails, phone numbers, SSNs, credit cards, IP addresses, and API keys."

    # Regex patterns
    PATTERNS = {
        "email": (
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
            GuardrailSeverity.MEDIUM,
            "Email address detected",
        ),
        "ssn": (
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            GuardrailSeverity.CRITICAL,
            "US Social Security Number detected",
        ),
        "phone": (
            re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            GuardrailSeverity.MEDIUM,
            "Phone number detected",
        ),
        "ipv4": (
            re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            GuardrailSeverity.LOW,
            "IPv4 address detected",
        ),
        "aws_key": (
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            GuardrailSeverity.CRITICAL,
            "AWS Access Key ID detected",
        ),
        "jwt": (
            re.compile(r"\beyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b"),
            GuardrailSeverity.HIGH,
            "JSON Web Token (JWT) detected",
        ),
    }

    # Credit card candidates pattern
    CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

    def __init__(
        self,
        action: GuardrailAction = GuardrailAction.REDACT,
        redaction_mode: str = "MASK",  # MASK, HASH, PARTIAL
        entropy_threshold: float = 4.2,
        check_entropy_secrets: bool = True,
    ):
        self.default_action = action
        self.redaction_mode = redaction_mode
        self.entropy_threshold = entropy_threshold
        self.check_entropy_secrets = check_entropy_secrets

    def _redact_value(self, category: str, original: str) -> str:
        if self.redaction_mode == "HASH":
            h = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
            return f"[REDACTED_{category.upper()}:{h}]"
        elif self.redaction_mode == "PARTIAL":
            clean = "".join(c for c in original if c.isalnum())
            if len(clean) > 4:
                return f"***{clean[-4:]}"
            return "[REDACTED]"
        else:
            return f"[REDACTED_{category.upper()}]"

    def check(self, text: str, context: Optional[dict[str, Any]] = None) -> GuardrailResult:
        start_t = time.perf_counter()
        violations: list[GuardrailViolation] = []
        sanitized = text

        # 1. Regex checks
        for cat, (regex, severity, desc) in self.PATTERNS.items():
            matches = list(regex.finditer(sanitized))
            if matches:
                for match in matches:
                    matched_str = match.group(0)
                    # Filter out obvious false positives for IPv4
                    if cat == "ipv4":
                        parts = [int(p) for p in matched_str.split(".")]
                        if any(p > 255 for p in parts) or parts[0] == 0:
                            continue

                    violations.append(
                        GuardrailViolation(
                            rule_name=f"pii.{cat}",
                            severity=severity,
                            message=desc,
                            details={"category": cat, "sample": matched_str[:4] + "***"},
                        )
                    )

                # Redact
                sanitized = regex.sub(lambda m: self._redact_value(cat, m.group(0)), sanitized)

        # 2. Credit card validation with Luhn algorithm
        card_matches = list(self.CARD_PATTERN.finditer(sanitized))
        for match in card_matches:
            raw = match.group(0)
            digits = "".join(c for c in raw if c.isdigit())
            if luhn_checksum(digits):
                violations.append(
                    GuardrailViolation(
                        rule_name="pii.credit_card",
                        severity=GuardrailSeverity.CRITICAL,
                        message="Valid Credit Card number detected via Luhn algorithm",
                        details={"category": "credit_card", "last4": digits[-4:]},
                    )
                )
                sanitized = sanitized.replace(raw, self._redact_value("credit_card", raw))

        # 3. Shannon entropy secret detection
        if self.check_entropy_secrets:
            words = re.findall(r"\b[A-Za-z0-9+/=_-]{24,}\b", sanitized)
            for word in words:
                if not word.startswith("[REDACTED"):
                    entropy = calculate_shannon_entropy(word)
                    if entropy >= self.entropy_threshold:
                        violations.append(
                            GuardrailViolation(
                                rule_name="pii.high_entropy_secret",
                                severity=GuardrailSeverity.HIGH,
                                message=f"High entropy secret token detected (entropy={entropy:.2f})",
                                details={"entropy": round(entropy, 2), "length": len(word)},
                            )
                        )
                        sanitized = sanitized.replace(word, self._redact_value("secret", word))

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if not violations:
            return GuardrailResult(
                guardrail_name=self.name,
                passed=True,
                action=GuardrailAction.PASS,
                score=1.0,
                violations=[],
                original_text=text,
                sanitized_text=text,
                latency_ms=latency_ms,
            )

        # If violations found
        severity_weights = {
            GuardrailSeverity.LOW: 0.1,
            GuardrailSeverity.MEDIUM: 0.25,
            GuardrailSeverity.HIGH: 0.5,
            GuardrailSeverity.CRITICAL: 1.0,
        }
        total_penalty = sum(severity_weights.get(v.severity, 0.2) for v in violations)
        score = max(0.0, 1.0 - total_penalty)

        passed = self.default_action != GuardrailAction.BLOCK
        final_action = self.default_action if self.default_action == GuardrailAction.BLOCK else GuardrailAction.REDACT

        return GuardrailResult(
            guardrail_name=self.name,
            passed=passed,
            action=final_action,
            score=score,
            violations=violations,
            original_text=text,
            sanitized_text=sanitized,
            latency_ms=latency_ms,
            metadata={"violation_count": len(violations)},
        )
