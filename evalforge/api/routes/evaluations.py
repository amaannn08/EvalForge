"""Evaluation execution and statistical comparison routes."""

import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from evalforge.api.schemas import CompareRunsRequest, RunEvaluationRequest
from evalforge.datasets.manager import DatasetManager
from evalforge.db.repository import EvaluationRepository
from evalforge.db.session import get_db
from evalforge.evaluators import (
    BleuEvaluator,
    CompositeScorer,
    ExactMatchEvaluator,
    JudgeEvaluator,
    LevenshteinSimilarityEvaluator,
    RELEVANCE_RUBRIC,
    RougeEvaluator,
    TokenF1Evaluator,
)
from evalforge.stats import StatisticalRegressionDetector

router = APIRouter(prefix="/api/v1/evaluations", tags=["Evaluations"])


def _build_composite_scorer(eval_names: list[str], pass_threshold: float) -> CompositeScorer:
    evaluators = []
    for name in eval_names:
        n = name.lower().strip()
        if n == "exact_match":
            evaluators.append((ExactMatchEvaluator(), 1.0))
        elif n in ("token_f1", "f1"):
            evaluators.append((TokenF1Evaluator(), 1.0))
        elif n in ("rouge", "rougel", "rouge_rougel"):
            evaluators.append((RougeEvaluator("rougeL"), 1.0))
        elif n in ("levenshtein", "edit_distance"):
            evaluators.append((LevenshteinSimilarityEvaluator(), 1.0))
        elif n in ("bleu", "bleu4"):
            evaluators.append((BleuEvaluator(), 1.0))
        elif n in ("judge", "judge_relevance", "relevance"):
            evaluators.append((JudgeEvaluator(RELEVANCE_RUBRIC), 1.5))

    if not evaluators:
        evaluators = [(TokenF1Evaluator(), 1.0), (RougeEvaluator("rougeL"), 1.0)]

    return CompositeScorer(evaluators=evaluators, pass_threshold=pass_threshold)


@router.post("/run")
def run_evaluation(req: RunEvaluationRequest, db: Session = Depends(get_db)):
    mgr = DatasetManager(db=db)
    try:
        cases = mgr.get_test_cases(req.dataset_name, req.version_tag)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    scorer = _build_composite_scorer(req.evaluators, req.pass_threshold)
    eval_repo = EvaluationRepository(db)

    run = eval_repo.create_run(
        run_name=req.run_name,
        candidate_model=req.candidate_model,
        baseline_run_id=req.baseline_run_id,
        config={
            "evaluators": req.evaluators,
            "pass_threshold": req.pass_threshold,
            "dataset_name": req.dataset_name,
            "version_tag": req.version_tag,
        },
    )

    results_data = []
    total_score = 0.0
    passed_count = 0
    total_latency = 0.0

    mock_map = req.mock_responses or {}

    for c in cases:
        # If mock response is provided for this prompt, use it; otherwise fallback to expected or prompt
        actual = mock_map.get(c.input_prompt, c.expected_output or f"Response to: {c.input_prompt}")
        res = scorer.evaluate(actual=actual, expected=c.expected_output, input_prompt=c.input_prompt)

        total_score += res.overall_score
        if res.passed:
            passed_count += 1
        total_latency += res.total_latency_ms

        results_data.append({
            "test_case_id": c.id,
            "input_text": c.input_prompt,
            "actual_output": actual,
            "expected_output": c.expected_output,
            "score": res.overall_score,
            "passed": res.passed,
            "metrics": {k: v.to_dict() for k, v in res.metrics.items()},
            "latency_ms": res.total_latency_ms,
            "token_usage": {"prompt_tokens": len(c.input_prompt.split()), "completion_tokens": len(actual.split())},
        })

    n = len(cases)
    pass_rate = passed_count / n if n > 0 else 0.0
    avg_score = total_score / n if n > 0 else 0.0

    updated_run = eval_repo.record_results(
        run_id=run.id,
        results=results_data,
        summary_metrics={"total_cases": n, "passed_cases": passed_count, "avg_score": avg_score},
        pass_rate=pass_rate,
        avg_score=avg_score,
        total_latency_ms=total_latency,
    )

    return {
        "run_id": updated_run.id,
        "run_name": updated_run.run_name,
        "candidate_model": updated_run.candidate_model,
        "status": updated_run.status,
        "total_cases": n,
        "pass_rate": round(pass_rate, 4),
        "avg_score": round(avg_score, 4),
        "total_latency_ms": round(total_latency, 2),
    }


@router.get("")
def list_runs(limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)):
    repo = EvaluationRepository(db)
    runs = repo.list_runs(limit=limit)
    return [
        {
            "run_id": r.id,
            "run_name": r.run_name,
            "candidate_model": r.candidate_model,
            "status": r.status,
            "pass_rate": r.pass_rate,
            "avg_score": r.avg_score,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in runs
    ]


@router.get("/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    repo = EvaluationRepository(db)
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    return {
        "run_id": run.id,
        "run_name": run.run_name,
        "candidate_model": run.candidate_model,
        "status": run.status,
        "pass_rate": run.pass_rate,
        "avg_score": run.avg_score,
        "total_latency_ms": run.total_latency_ms,
        "summary_metrics": json.loads(run.summary_metrics_json),
        "results": [
            {
                "test_case_id": res.test_case_id,
                "input_text": res.input_text,
                "actual_output": res.actual_output,
                "expected_output": res.expected_output,
                "score": res.score,
                "passed": res.passed,
                "metrics": json.loads(res.metrics_json),
                "latency_ms": res.latency_ms,
            }
            for res in run.results
        ],
    }


@router.post("/compare")
def compare_runs(req: CompareRunsRequest, db: Session = Depends(get_db)):
    repo = EvaluationRepository(db)
    base_run = repo.get_run(req.baseline_run_id)
    cand_run = repo.get_run(req.candidate_run_id)

    if not base_run or not cand_run:
        raise HTTPException(status_code=404, detail="One or both evaluation runs not found.")

    b_scores = [r.score for r in base_run.results]
    c_scores = [r.score for r in cand_run.results]
    b_lats = [r.latency_ms for r in base_run.results]
    c_lats = [r.latency_ms for r in cand_run.results]
    b_pass = [r.passed for r in base_run.results]
    c_pass = [r.passed for r in cand_run.results]

    detector = StatisticalRegressionDetector(
        alpha=req.alpha,
        regression_threshold_delta=req.regression_threshold,
        max_latency_regression_pct=req.max_latency_regression_pct,
    )

    report = detector.compare(
        baseline_scores=b_scores,
        candidate_scores=c_scores,
        baseline_latencies=b_lats,
        candidate_latencies=c_lats,
        baseline_passed=b_pass,
        candidate_passed=c_pass,
        baseline_run_id=base_run.id,
        candidate_run_id=cand_run.id,
    )

    return {
        "report": report.to_dict(),
        "markdown": report.to_markdown(),
    }
