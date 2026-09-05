"""Unit tests for dataset versioning, parsing, and hashing."""

import json
from evalforge.datasets import (
    DatasetManager,
    TestCase,
    compute_dataset_hash,
)
from evalforge.db.session import init_db


def test_content_hashing_determinism():
    c1 = [
        TestCase(input_prompt="Q1", expected_output="A1", tags=["b", "a"]),
        TestCase(input_prompt="Q2", expected_output="A2"),
    ]
    c2 = [
        TestCase(input_prompt="Q1", expected_output="A1", tags=["a", "b"]),
        TestCase(input_prompt="Q2", expected_output="A2"),
    ]
    h1 = compute_dataset_hash(c1)
    h2 = compute_dataset_hash(c2)
    assert h1 == h2
    assert len(h1) == 64


def test_dataset_manager_registration_and_fetch(tmp_path):
    init_db()
    mgr = DatasetManager()

    sample_json = tmp_path / "sample.json"
    data = [
        {"input_prompt": "What is AI?", "expected_output": "Artificial Intelligence"},
        {"input_prompt": "What is ML?", "expected_output": "Machine Learning"},
    ]
    sample_json.write_text(json.dumps(data), encoding="utf-8")

    cases = mgr.parse_test_cases(sample_json)
    assert len(cases) == 2

    info = mgr.register_dataset_version("ai-fundamentals", "v1.0.0", cases)
    assert info.version_tag == "v1.0.0"
    assert info.num_cases == 2

    fetched = mgr.get_test_cases("ai-fundamentals", "v1.0.0")
    assert len(fetched) == 2
    assert fetched[0].input_prompt == "What is AI?"
