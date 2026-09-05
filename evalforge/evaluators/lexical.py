"""Deterministic and lexical evaluation metrics (ExactMatch, Token-F1, Levenshtein, ROUGE, BLEU)."""

from __future__ import annotations

from collections import Counter
import math
import re
import string
import time
from typing import Any, Optional

from evalforge.evaluators.base import BaseEvaluator, EvaluationMetric


def _tokenize(text: str) -> list[str]:
    """Tokenize and normalize text into word tokens."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in cleaned.split() if t]


def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract consecutive n-grams from a token sequence."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _longest_common_subsequence_length(tokens1: list[str], tokens2: list[str]) -> int:
    """Compute LCS length using dynamic programming."""
    m, n = len(tokens1), len(tokens2)
    if m == 0 or n == 0:
        return 0
    # Two-row DP memory optimization
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)

    return curr[n]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance via DP."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = list(v1)

    return v0[len(s2)]


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluates whether actual output matches expected string exactly."""

    name: str = "exact_match"
    description: str = "Strict or normalized exact string equality."

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        ignore_punctuation: bool = False,
    ):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.ignore_punctuation = ignore_punctuation

    def _normalize(self, s: str) -> str:
        if self.strip_whitespace:
            s = s.strip()
        if not self.case_sensitive:
            s = s.lower()
        if self.ignore_punctuation:
            s = s.translate(str.maketrans("", "", string.punctuation))
            s = " ".join(s.split())
        return s

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        start_t = time.perf_counter()
        if expected is None:
            return EvaluationMetric(
                name=self.name,
                score=0.0,
                passed=False,
                reason="No expected ground truth provided for exact match",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        norm_act = self._normalize(actual)
        norm_exp = self._normalize(expected)
        matched = norm_act == norm_exp
        score = 1.0 if matched else 0.0

        return EvaluationMetric(
            name=self.name,
            score=score,
            passed=matched,
            reason="Outputs match exactly" if matched else "Mismatch between actual and expected",
            details={"actual_normalized": norm_act, "expected_normalized": norm_exp},
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )


class RegexMatchEvaluator(BaseEvaluator):
    """Evaluates whether actual output matches a specified regex pattern."""

    name: str = "regex_match"
    description: str = "Regular expression pattern match."

    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        self.pattern_str = pattern
        self.pattern = re.compile(pattern, flags)

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        start_t = time.perf_counter()
        match = self.pattern.search(actual)
        passed = match is not None
        score = 1.0 if passed else 0.0

        return EvaluationMetric(
            name=self.name,
            score=score,
            passed=passed,
            reason=f"Regex pattern '{self.pattern_str}' {'matched' if passed else 'failed'}",
            details={"matched_text": match.group(0) if match else None},
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )


class TokenF1Evaluator(BaseEvaluator):
    """Calculates token-level precision, recall, and harmonic F1 score."""

    name: str = "token_f1"
    description: str = "Token-level unigram precision, recall, and F1 overlap."

    def __init__(self, pass_threshold: float = 0.6):
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        start_t = time.perf_counter()
        if not expected:
            return EvaluationMetric(
                name=self.name,
                score=0.0,
                passed=False,
                reason="Expected ground truth missing",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        act_tokens = _tokenize(actual)
        exp_tokens = _tokenize(expected)

        if not act_tokens and not exp_tokens:
            return EvaluationMetric(name=self.name, score=1.0, passed=True, latency_ms=0.0)
        if not act_tokens or not exp_tokens:
            return EvaluationMetric(name=self.name, score=0.0, passed=False, latency_ms=0.0)

        act_counter = Counter(act_tokens)
        exp_counter = Counter(exp_tokens)
        common_tokens = sum((act_counter & exp_counter).values())

        precision = common_tokens / len(act_tokens)
        recall = common_tokens / len(exp_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        passed = f1 >= self.pass_threshold
        return EvaluationMetric(
            name=self.name,
            score=f1,
            passed=passed,
            reason=f"F1: {f1:.3f} (P: {precision:.3f}, R: {recall:.3f})",
            details={"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )


class LevenshteinSimilarityEvaluator(BaseEvaluator):
    """Normalized edit distance similarity metric [0.0, 1.0]."""

    name: str = "levenshtein_similarity"
    description: str = "Character-level normalized Levenshtein similarity."

    def __init__(self, pass_threshold: float = 0.75):
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        start_t = time.perf_counter()
        if not expected:
            return EvaluationMetric(
                name=self.name,
                score=0.0,
                passed=False,
                reason="Expected target string required",
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        max_len = max(len(actual), len(expected))
        if max_len == 0:
            sim = 1.0
        else:
            dist = _levenshtein_distance(actual, expected)
            sim = max(0.0, 1.0 - (dist / max_len))

        passed = sim >= self.pass_threshold
        return EvaluationMetric(
            name=self.name,
            score=sim,
            passed=passed,
            reason=f"Normalized edit distance similarity: {sim:.3f}",
            details={"similarity": round(sim, 4)},
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )


class RougeEvaluator(BaseEvaluator):
    """Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores using Longest Common Subsequence."""

    name: str = "rouge"
    description: str = "ROUGE-1, ROUGE-2, and ROUGE-L overlap metrics."

    def __init__(self, rouge_type: str = "rougeL", pass_threshold: float = 0.5):
        self.rouge_type = rouge_type  # rouge1, rouge2, rougeL
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        start_t = time.perf_counter()
        if not expected:
            return EvaluationMetric(name=self.name, score=0.0, passed=False, reason="Target expected text missing")

        act_tokens = _tokenize(actual)
        exp_tokens = _tokenize(expected)

        if not act_tokens or not exp_tokens:
            return EvaluationMetric(name=self.name, score=0.0, passed=False, latency_ms=0.0)

        # ROUGE-1
        r1_common = sum((Counter(act_tokens) & Counter(exp_tokens)).values())
        r1_prec = r1_common / len(act_tokens)
        r1_rec = r1_common / len(exp_tokens)
        r1_f1 = (2 * r1_prec * r1_rec) / (r1_prec + r1_rec) if (r1_prec + r1_rec) > 0 else 0.0

        # ROUGE-2
        act_bg = Counter(_get_ngrams(act_tokens, 2))
        exp_bg = Counter(_get_ngrams(exp_tokens, 2))
        r2_common = sum((act_bg & exp_bg).values())
        r2_prec = r2_common / max(1, len(act_tokens) - 1)
        r2_rec = r2_common / max(1, len(exp_tokens) - 1)
        r2_f1 = (2 * r2_prec * r2_rec) / (r2_prec + r2_rec) if (r2_prec + r2_rec) > 0 else 0.0

        # ROUGE-L
        lcs_len = _longest_common_subsequence_length(act_tokens, exp_tokens)
        rl_prec = lcs_len / len(act_tokens)
        rl_rec = lcs_len / len(exp_tokens)
        rl_f1 = (2 * rl_prec * rl_rec) / (rl_prec + rl_rec) if (rl_prec + rl_rec) > 0 else 0.0

        scores = {"rouge1": r1_f1, "rouge2": r2_f1, "rougeL": rl_f1}
        target_score = scores.get(self.rouge_type, rl_f1)
        passed = target_score >= self.pass_threshold

        return EvaluationMetric(
            name=f"rouge_{self.rouge_type}",
            score=target_score,
            passed=passed,
            reason=f"{self.rouge_type.upper()} F1: {target_score:.3f}",
            details={
                "rouge1_f1": round(r1_f1, 4),
                "rouge2_f1": round(r2_f1, 4),
                "rougeL_f1": round(rl_f1, 4),
            },
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )


class BleuEvaluator(BaseEvaluator):
    """Calculates smoothed sentence BLEU score up to 4-grams with Brevity Penalty."""

    name: str = "bleu"
    description: str = "BLEU score with n-gram precision and brevity penalty."

    def __init__(self, max_order: int = 4, pass_threshold: float = 0.4):
        self.max_order = max_order
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        actual: str,
        expected: Optional[str] = None,
        input_prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> EvaluationMetric:
        start_t = time.perf_counter()
        if not expected:
            return EvaluationMetric(name=self.name, score=0.0, passed=False, reason="Target expected text missing")

        act_tokens = _tokenize(actual)
        exp_tokens = _tokenize(expected)

        if not act_tokens or not exp_tokens:
            return EvaluationMetric(name=self.name, score=0.0, passed=False, latency_ms=0.0)

        precisions: list[float] = []
        for n in range(1, self.max_order + 1):
            act_ng = Counter(_get_ngrams(act_tokens, n))
            exp_ng = Counter(_get_ngrams(exp_tokens, n))
            overlap = sum((act_ng & exp_ng).values())
            total = max(1, len(act_tokens) - n + 1)
            # Add-1 smoothing for higher order ngrams if needed
            p = (overlap + 0.1) / (total + 0.1) if total > 0 else 0.0
            precisions.append(p)

        # Brevity penalty
        ref_len = len(exp_tokens)
        hyp_len = len(act_tokens)
        bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len)) if hyp_len > 0 else 0.0

        # Geometric mean
        log_prec_sum = sum((1.0 / self.max_order) * math.log(p) for p in precisions if p > 0)
        bleu = bp * math.exp(log_prec_sum)
        bleu = min(1.0, max(0.0, bleu))
        passed = bleu >= self.pass_threshold

        return EvaluationMetric(
            name=self.name,
            score=bleu,
            passed=passed,
            reason=f"BLEU-{self.max_order}: {bleu:.3f} (BP={bp:.3f})",
            details={"brevity_penalty": round(bp, 4), "bleu": round(bleu, 4)},
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )
