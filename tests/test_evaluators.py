"""Unit tests for lexical, deterministic, LLM-judge, and composite evaluators."""

from evalforge.evaluators import (
    BleuEvaluator,
    CompositeScorer,
    ExactMatchEvaluator,
    JudgeEvaluator,
    LevenshteinSimilarityEvaluator,
    RELEVANCE_RUBRIC,
    RegexMatchEvaluator,
    RougeEvaluator,
    TokenF1Evaluator,
)


def test_exact_match():
    em_case_ins = ExactMatchEvaluator(case_sensitive=False)
    assert em_case_ins.evaluate("Paris", "paris").passed is True

    em_case_sens = ExactMatchEvaluator(case_sensitive=True)
    assert em_case_sens.evaluate("Paris", "paris").passed is False


def test_regex_match():
    reg = RegexMatchEvaluator(r"\b\d{4}-\d{2}-\d{2}\b")
    assert reg.evaluate("Date of event: 2026-09-06.").passed is True
    assert reg.evaluate("Date not specified.").passed is False


def test_token_f1():
    f1_eval = TokenF1Evaluator(pass_threshold=0.6)
    m = f1_eval.evaluate(
        actual="The quick brown fox jumps over the lazy dog",
        expected="The brown fox jumps over a lazy dog",
    )
    assert m.score > 0.7
    assert m.passed is True


def test_levenshtein_similarity():
    lev = LevenshteinSimilarityEvaluator(pass_threshold=0.8)
    m = lev.evaluate("EvalForge Platform", "EvalForge Platform")
    assert m.score == 1.0
    assert m.passed is True

    m_diff = lev.evaluate("EvalForge", "Forge")
    assert 0.4 < m_diff.score < 0.7


def test_rouge():
    rouge = RougeEvaluator("rougeL", pass_threshold=0.5)
    m = rouge.evaluate(
        actual="Deep learning uses deep artificial neural networks for representation learning.",
        expected="Deep learning is based on artificial neural networks for representation learning.",
    )
    assert m.score > 0.7
    assert m.passed is True
    assert "rougeL_f1" in m.details


def test_bleu():
    bleu = BleuEvaluator(max_order=3, pass_threshold=0.4)
    m = bleu.evaluate(
        actual="The cat is sitting quietly on the mat.",
        expected="The cat is sitting quietly on the mat.",
    )
    assert m.score > 0.8
    assert m.passed is True


def test_mock_judge():
    judge = JudgeEvaluator(criterion=RELEVANCE_RUBRIC, pass_threshold=0.5)
    m = judge.evaluate(
        actual="Photosynthesis converts solar light energy into chemical energy inside plant chloroplasts.",
        expected="Photosynthesis produces glucose from sunlight, water, and CO2.",
        input_prompt="Explain photosynthesis.",
    )
    assert m.score >= 0.5
    assert m.passed is True
    assert "relevance" in m.reason.lower()


def test_composite_scorer():
    scorer = CompositeScorer([
        (ExactMatchEvaluator(), 0.2),
        (TokenF1Evaluator(), 0.3),
        (RougeEvaluator("rougeL"), 0.3),
        (JudgeEvaluator(RELEVANCE_RUBRIC), 0.2),
    ], pass_threshold=0.6)

    res = scorer.evaluate(
        actual="Binary search algorithm operates in O(log n) time on sorted data.",
        expected="Binary search runs in logarithmic O(log n) time on a sorted array.",
        input_prompt="What is the complexity of binary search?",
    )
    assert 0.0 <= res.overall_score <= 1.0
    assert len(res.metrics) == 4
    assert res.total_latency_ms >= 0.0
