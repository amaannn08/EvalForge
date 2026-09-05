"""Integration tests for Typer CLI commands."""

import json
from typer.testing import CliRunner
from evalforge.cli.main import app
from evalforge.db.session import init_db

runner = CliRunner()


def setup_module():
    init_db()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Production LLM Evaluation & Guardrails CLI platform" in result.output


def test_cli_guard_check():
    result = runner.invoke(app, ["guard", "check", "Please email alex@work.com for details."])
    assert result.exit_code == 0
    assert "[REDACTED_EMAIL]" in result.output


def test_cli_run_evaluation(tmp_path):
    dataset_file = tmp_path / "bench.json"
    dataset_file.write_text(
        json.dumps([
            {"input_prompt": "State the capital of Germany.", "expected_output": "The capital of Germany is Berlin."},
            {"input_prompt": "State the capital of Italy.", "expected_output": "The capital of Italy is Rome."},
        ]),
        encoding="utf-8",
    )

    out_file = tmp_path / "out.json"
    result = runner.invoke(
        app,
        ["run", "--dataset", str(dataset_file), "--evaluators", "token_f1,rougeL", "--output", str(out_file)],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data["results"]) == 2


def test_cli_gate_command(tmp_path):
    # Candidate and baseline reports
    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"

    base_results = [{"score": 0.90, "latency_ms": 100.0, "passed": True} for _ in range(10)]
    cand_results = [{"score": 0.91, "latency_ms": 105.0, "passed": True} for _ in range(10)]

    base_file.write_text(json.dumps({"results": base_results}), encoding="utf-8")
    cand_file.write_text(json.dumps({"results": cand_results}), encoding="utf-8")

    result = runner.invoke(
        app,
        ["gate", "--candidate", str(cand_file), "--baseline", str(base_file), "--alpha", "0.05", "--threshold", "0.02"],
    )
    assert result.exit_code == 0
    assert "CI GATE PASSED" in result.output
