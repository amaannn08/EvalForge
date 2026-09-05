"""Integration tests for FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient
from evalforge.api.app import app
from evalforge.db.session import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert "x-latency-ms" in res.headers


def test_guardrails_check_endpoint():
    payload = {
        "text": "Send credentials to admin@mycorp.com immediately.",
        "guardrails": ["pii"],
        "mode": "COLLECT_ALL",
    }
    res = client.post("/api/v1/guardrails/check", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["passed"] is True
    assert "[REDACTED_EMAIL]" in data["sanitized_text"]
    assert data["violation_count"] >= 1


def test_dataset_and_evaluation_lifecycle_api():
    # 1. Create dataset
    d_res = client.post("/api/v1/datasets", json={"name": "api_test_dataset", "description": "API test dataset"})
    assert d_res.status_code in (200, 400)

    # 2. Register version
    v_res = client.post(
        "/api/v1/datasets/api_test_dataset/versions",
        json={
            "version_tag": "v1.0.0",
            "test_cases": [
                {"input_prompt": "What is Python?", "expected_output": "A high-level programming language."},
                {"input_prompt": "What is SQL?", "expected_output": "Structured Query Language."},
            ],
        },
    )
    assert v_res.status_code == 200
    assert v_res.json()["num_cases"] == 2

    # 3. Run evaluation
    run_res = client.post(
        "/api/v1/evaluations/run",
        json={
            "run_name": "api_run_1",
            "candidate_model": "test-gpt",
            "dataset_name": "api_test_dataset",
            "version_tag": "v1.0.0",
            "evaluators": ["token_f1", "rougeL"],
        },
    )
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["total_cases"] == 2
    assert run_data["status"] == "COMPLETED"

    # 4. Get run details
    get_res = client.get(f"/api/v1/evaluations/{run_data['run_id']}")
    assert get_res.status_code == 200
    assert len(get_res.json()["results"]) == 2
