"""Toxicity, harassment, hate-speech, and self-harm guardrail."""

from __future__ import annotations

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


class ToxicityGuardrail(BaseGuardrail):
    """Detects hate speech, violence threats, harassment, self-harm, and profanity."""

    name: str = "toxicity_filter"
    description: str = "Scans for toxic language, hate speech, self-harm encouragement, and severe profanity."

    # Normalized leetspeak translation table
    LEET_TRANS = str.maketrans({
        "@": "a", "4": "a", "8": "b", "3": "e", "1": "i", "!": "i",
        "0": "o", "5": "s", "$": "s", "7": "t", "+": "t",
    })

    # Categorized toxicity dictionaries
    CATEGORIES = {
        "self_harm": {
            "severity": GuardrailSeverity.CRITICAL,
            "patterns": [
                r"\b(kill|hang|shoot|slit)\s+(your\s*self|ur\s*self)\b",
                r"\bhow\s+to\s+(commit\s+suicide|end\s+my\s+life)\b",
                r"\bencouraging\s+(suicide|self[- ]harm)\b",
            ],
            "penalty": 1.0,
        },
        "violence_threat": {
            "severity": GuardrailSeverity.CRITICAL,
            "patterns": [
                r"\bi('ll| will)\s+(murder|kill|bomb|execute|decapitate|slaughter)\s+(you|them|all)\b",
                r"\bhow\s+to\s+build\s+a\s+(bomb|explosive|weapon\s+of\s+mass\s+destruction)\b",
                r"\bthreat\s+of\s+(violence|mass\s+shooting)\b",
            ],
            "penalty": 0.9,
        },
        "hate_speech": {
            "severity": GuardrailSeverity.HIGH,
            "patterns": [
                r"\b(inferior|subhuman|vermin|parasite|plague)\s+(race|ethnicity|religion|people|refugees)\b",
                r"\bexterminate\s+(all\s+)?(jews|muslims|christians|hindus|blacks|whites|immigrants)\b",
            ],
            "penalty": 0.7,
        },
        "harassment": {
            "severity": GuardrailSeverity.MEDIUM,
            "patterns": [
                r"\byou('re| are)\s+(a\s+piece\s+of\s+trash|worthless|retarded|idiotic\s+moron)\b",
                r"\bdoxx(ing)?\s+(address|phone|identity)\b",
            ],
            "penalty": 0.4,
        },
        "profanity": {
            "severity": GuardrailSeverity.LOW,
            "patterns": [
                r"\b(fuck|shit|bitch|asshole|bastard|cunt|dickhead)\b",
            ],
            "penalty": 0.15,
        },
    }

    def __init__(
        self,
        threshold: float = 0.3,
        action: GuardrailAction = GuardrailAction.BLOCK,
        block_profanity: bool = False,
    ):
        self.threshold = threshold
        self.action = action
        self.block_profanity = block_profanity

    def _normalize_text(self, text: str) -> str:
        """Normalize text by translating leetspeak and collapsing repeated punctuation."""
        lower = text.lower()
        translated = lower.translate(self.LEET_TRANS)
        # Collapse multiple spaces or repeating characters
        collapsed = re.sub(r"(.)\1{3,}", r"\1\1", translated)
        return collapsed

    def check(self, text: str, context: Optional[dict[str, Any]] = None) -> GuardrailResult:
        start_t = time.perf_counter()
        normalized = self._normalize_text(text)
        violations: list[GuardrailViolation] = []
        total_penalty = 0.0

        for cat_name, config in self.CATEGORIES.items():
            if cat_name == "profanity" and not self.block_profanity:
                continue

            for pat_str in config["patterns"]:
                match = re.search(pat_str, normalized, re.IGNORECASE)
                if match:
                    penalty = config["penalty"]
                    total_penalty += penalty
                    violations.append(
                        GuardrailViolation(
                            rule_name=f"toxicity.{cat_name}",
                            severity=config["severity"],
                            message=f"Toxicity violation detected in category: {cat_name}",
                            details={"category": cat_name, "match": match.group(0)},
                        )
                    )
                    break

        score = max(0.0, 1.0 - min(1.0, total_penalty))
        passed = total_penalty < self.threshold
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if passed:
            return GuardrailResult(
                guardrail_name=self.name,
                passed=True,
                action=GuardrailAction.PASS if not violations else GuardrailAction.WARN,
                score=score,
                violations=violations,
                original_text=text,
                sanitized_text=text,
                latency_ms=latency_ms,
                metadata={"toxicity_penalty": round(total_penalty, 3)},
            )

        return GuardrailResult(
            guardrail_name=self.name,
            passed=False,
            action=self.action,
            score=score,
            violations=violations,
            original_text=text,
            sanitized_text=text,
            latency_ms=latency_ms,
            metadata={"toxicity_penalty": round(total_penalty, 3), "blocked": True},
        )
