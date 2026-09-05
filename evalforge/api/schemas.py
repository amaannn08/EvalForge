"""Pydantic schemas for FastAPI REST request and response validation."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    service: str = "EvalForge"


class GuardrailCheckRequest(BaseModel):
    text: str = Field(..., description="The input or output text to evaluate.")
    guardrails: list[str] = Field(
        default=["pii", "injection", "toxicity"],
        description="List of guardrails: pii, injection, toxicity, budget, schema",
    )
    mode: str = Field(default="COLLECT_ALL", description="FAIL_FAST, COLLECT_ALL, or PARALLEL")
    context: dict[str, Any] = Field(default_factory=dict)


class GuardrailCheckResponse(BaseModel):
    passed: bool
    final_action: str
    original_text: str
    sanitized_text: str
    overall_score: float
    total_latency_ms: float
    violation_count: int
    violations: list[dict[str, Any]]
    results: list[dict[str, Any]]


class RunEvaluationRequest(BaseModel):
    run_name: str
    candidate_model: str
    dataset_name: str
    version_tag: str
    baseline_run_id: Optional[str] = None
    pass_threshold: float = 0.70
    mock_responses: Optional[dict[str, str]] = None  # Mapping prompt -> response for offline testing
    evaluators: list[str] = Field(default=["token_f1", "rougeL", "judge_relevance"])


class CompareRunsRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    alpha: float = 0.05
    regression_threshold: float = 0.02
    max_latency_regression_pct: float = 25.0


class DatasetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class TestCaseInput(BaseModel):
    input_prompt: str
    system_prompt: Optional[str] = None
    expected_output: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class DatasetVersionCreateRequest(BaseModel):
    version_tag: str
    test_cases: list[TestCaseInput]
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
