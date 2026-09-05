"""LLM-as-a-Judge evaluation with rubrics and deterministic mock evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
import time
from typing import Any, Optional

from evalforge.evaluators.base import BaseEvaluator, EvaluationMetric


@dataclass
class RubricCriterion:
    """A qualitative evaluation criterion with 1-5 scale descriptors."""
    name: str
    description: str
    weight: float = 1.0
    rubric_levels: dict[int, str] = field(default_factory=lambda: {
        1: "Completely fails to meet criteria.",
        2: "Poorly meets criteria with significant deficiencies.",
        3: "Acceptably meets criteria with minor flaws.",
        4: "Good adherence to criteria with strong quality.",
        5: "Flawless adherence to criteria.",
    })


# Built-in industry standard rubrics
RELEVANCE_RUBRIC = RubricCriterion(
    name="relevance",
    description="Evaluates whether the response directly and comprehensively answers the user prompt.",
    rubric_levels={
        1: "Completely irrelevant or unresponsive to the query.",
        2: "Mentions query topics but goes off on unrelated tangents.",
        3: "Partially answers the query but leaves out key information.",
        4: "Directly and accurately answers the query with minor omissions.",
        5: "Exceptionally relevant, comprehensive, and concise answer.",
    },
)

FAITHFULNESS_RUBRIC = RubricCriterion(
    name="faithfulness",
    description="Evaluates whether the response is strictly supported by provided context without hallucinations.",
    rubric_levels={
        1: "Entirely hallucinated or contradicts the reference context.",
        2: "Contains major unsupported claims or fabrications.",
        3: "Mostly grounded but makes unverified extrapolations.",
        4: "Strongly grounded in the context with negligible ambiguity.",
        5: "100% faithful to the context with zero unsupported claims.",
    },
)

COHERENCE_RUBRIC = RubricCriterion(
    name="coherence",
    description="Evaluates logical clarity, fluency, syntax, and organization.",
    rubric_levels={
        1: "Incoherent, fragmented, or unreadable.",
        2: "Disorganized with frequent grammatical and reasoning errors.",
        3: "Generally readable but lacks smooth transitions or structure.",
        4: "Well-structured, clear flow of thought, and fluent language.",
        5: "Exemplary structure, compelling reasoning, and flawless clarity.",
    },
)


class BaseJudgeProvider(ABC):
    """Abstract provider for LLM judge generation."""

    @abstractmethod
    def judge(
        self,
        prompt: str,
        criterion: RubricCriterion,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[int, str]:
        """Returns (score_1_to_5, reasoning)."""
        pass


class MockJudgeProvider(BaseJudgeProvider):
    """Deterministic heuristic mock judge for reproducible tests and offline CI evaluation."""

    def judge(
        self,
        prompt: str,
        criterion: RubricCriterion,
        actual: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[int, str]:
        act_lower = actual.lower()
        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        act_words = set(re.findall(r"\w+", act_lower))

        if len(act_words) == 0:
            return 1, "Output is empty."

        overlap = len(prompt_words & act_words) / max(1, len(prompt_words))

        # Check expected match if available
        if expected:
            exp_words = set(re.findall(r"\w+", expected.lower()))
            exp_overlap = len(exp_words & act_words) / max(1, len(exp_words))
            score_factor = 0.5 * overlap + 0.5 * exp_overlap
        else:
            score_factor = overlap

        # Scale factor 0.0-1.0 to 1-5 integer scale
        if score_factor >= 0.75:
            score = 5
            reason = f"Excellent adherence to {criterion.name}. Strong conceptual and vocabulary alignment."
        elif score_factor >= 0.5:
            score = 4
            reason = f"Good adherence to {criterion.name}. Well addressed with solid contextual grounding."
        elif score_factor >= 0.3:
            score = 3
            reason = f"Moderate adherence to {criterion.name}. Basic requirements covered with some gaps."
        elif score_factor >= 0.15:
            score = 2
            reason = f"Weak adherence to {criterion.name}. Significant divergence from expected topics."
        else:
            score = 1
            reason = f"Fails {criterion.name}. Negligible alignment with user intent."

        return score, reason


class JudgeEvaluator(BaseEvaluator):
    """LLM-as-a-Judge evaluator using structured rubrics and chain-of-thought reasoning."""

    def __init__(
        self,
        criterion: RubricCriterion,
        provider: Optional[BaseJudgeProvider] = None,
        pass_threshold: float = 0.6,  # 3/5 is 0.6
        weight: float = 1.0,
    ):
        self.criterion = criterion
        self.name = f"judge_{criterion.name}"
        self.description = criterion.description
        self.provider = provider or MockJudgeProvider()
        self.pass_threshold = pass_threshold
        self.weight = weight

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        start_t = time.perf_counter()
        prompt_str = input_prompt or ""

        score_1_to_5, reason = self.provider.judge(
            prompt=prompt_str,
            criterion=self.criterion,
            actual=actual,
            expected=expected,
            context=context,
        )

        normalized_score = (score_1_to_5 - 1) / 4.0  # Maps [1, 5] -> [0.0, 1.0]
        passed = normalized_score >= self.pass_threshold
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return EvaluationMetric(
            name=self.name,
            score=normalized_score,
            passed=passed,
            weight=self.weight,
            reason=reason,
            details={
                "raw_rubric_score": score_1_to_5,
                "rubric_criterion": self.criterion.name,
                "rubric_description": self.criterion.description,
            },
            latency_ms=latency_ms,
        )
