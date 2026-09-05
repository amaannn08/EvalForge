"""Evaluation engine and metrics for EvalForge."""

from evalforge.evaluators.base import BaseEvaluator, EvaluationMetric
from evalforge.evaluators.lexical import (
    ExactMatchEvaluator,
    RegexMatchEvaluator,
    TokenF1Evaluator,
    LevenshteinSimilarityEvaluator,
    RougeEvaluator,
    BleuEvaluator,
)
from evalforge.evaluators.judge import (
    BaseJudgeProvider,
    MockJudgeProvider,
    RubricCriterion,
    JudgeEvaluator,
    RELEVANCE_RUBRIC,
    FAITHFULNESS_RUBRIC,
    COHERENCE_RUBRIC,
)
from evalforge.evaluators.composite import (
    CompositeScorer,
    CompositeEvaluationResult,
)

__all__ = [
    "BaseEvaluator",
    "EvaluationMetric",
    "ExactMatchEvaluator",
    "RegexMatchEvaluator",
    "TokenF1Evaluator",
    "LevenshteinSimilarityEvaluator",
    "RougeEvaluator",
    "BleuEvaluator",
    "BaseJudgeProvider",
    "MockJudgeProvider",
    "RubricCriterion",
    "JudgeEvaluator",
    "RELEVANCE_RUBRIC",
    "FAITHFULNESS_RUBRIC",
    "COHERENCE_RUBRIC",
    "CompositeScorer",
    "CompositeEvaluationResult",
]
