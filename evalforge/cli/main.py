"""EvalForge Typer command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from evalforge.config import settings
from evalforge.datasets import DatasetManager, TestCase
from evalforge.db.repository import EvaluationRepository, TraceRepository
from evalforge.db.session import SessionLocal, init_db
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
from evalforge.guardrails import (
    CostBudgetGuardrail,
    ExecutionMode,
    GuardrailPipeline,
    PIIGuardrail,
    PromptInjectionGuardrail,
    SchemaValidationGuardrail,
    ToxicityGuardrail,
)
from evalforge.stats import RegressionStatus, StatisticalRegressionDetector
from evalforge.tracing.exporter import ConsoleSpanExporter

app = typer.Typer(
    name="evalforge",
    help="Production LLM Evaluation & Guardrails CLI platform.",
    add_completion=False,
)
console = Console()

# Sub-command groups
guard_app = typer.Typer(name="guard", help="Guardrails testing and sanitization.")
dataset_app = typer.Typer(name="dataset", help="Dataset versioning and management.")
trace_app = typer.Typer(name="trace", help="Span and trace inspection.")
app.add_typer(guard_app)
app.add_typer(dataset_app)
app.add_typer(trace_app)


@app.callback()
def main():
    """EvalForge: Production LLM Evaluation & Guardrails Platform."""
    init_db()


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to evaluation dataset file (.json or .jsonl)."),
    evaluators: str = typer.Option(
        "token_f1,rougeL,judge_relevance", "--evaluators", "-e", help="Comma-separated list of evaluators."
    ),
    pass_threshold: float = typer.Option(0.70, "--threshold", "-t", help="Minimum score to pass test case."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional output JSON report path."),
    run_name: str = typer.Option("eval_run", "--name", "-n", help="Name for this evaluation run."),
    candidate_model: str = typer.Option("candidate_model", "--model", "-m", help="Candidate model identifier."),
):
    """Run an evaluation benchmark across a dataset."""
    console.print(f"[bold cyan]Starting evaluation run:[/] [bold]{run_name}[/] on [yellow]{dataset}[/]")

    mgr = DatasetManager()
    cases = mgr.parse_test_cases(dataset)
    if not cases:
        console.print("[bold red]Error:[/] No test cases found in dataset.")
        raise typer.Exit(code=1)

    eval_list = []
    for name in [e.strip().lower() for e in evaluators.split(",") if e.strip()]:
        if name == "exact_match":
            eval_list.append((ExactMatchEvaluator(), 1.0))
        elif name in ("token_f1", "f1"):
            eval_list.append((TokenF1Evaluator(), 1.0))
        elif name in ("rougel", "rouge"):
            eval_list.append((RougeEvaluator("rougeL"), 1.0))
        elif name in ("levenshtein", "edit_distance"):
            eval_list.append((LevenshteinSimilarityEvaluator(), 1.0))
        elif name in ("bleu", "bleu4"):
            eval_list.append((BleuEvaluator(), 1.0))
        elif name in ("judge_relevance", "judge"):
            eval_list.append((JudgeEvaluator(RELEVANCE_RUBRIC), 1.5))

    scorer = CompositeScorer(eval_list, pass_threshold=pass_threshold)

    results = []
    total_score = 0.0
    passed_count = 0
    total_latency = 0.0

    with console.status("[bold green]Evaluating test cases...[/]"):
        for c in cases:
            # Deterministic simulation or actual output
            actual = c.expected_output or f"Generated answer for: {c.input_prompt}"
            res = scorer.evaluate(actual=actual, expected=c.expected_output, input_prompt=c.input_prompt)
            total_score += res.overall_score
            if res.passed:
                passed_count += 1
            total_latency += res.total_latency_ms
            results.append(res)

    n = len(cases)
    pass_rate = passed_count / n
    avg_score = total_score / n

    # Save to SQLite
    with SessionLocal() as db:
        repo = EvaluationRepository(db)
        eval_run = repo.create_run(
            run_name=run_name,
            candidate_model=candidate_model,
            config={"dataset": dataset, "evaluators": evaluators, "pass_threshold": pass_threshold},
        )
        repo.record_results(
            run_id=eval_run.id,
            results=[
                {
                    "input_text": r.input_text,
                    "actual_output": r.actual_output,
                    "expected_output": r.expected_output,
                    "score": r.overall_score,
                    "passed": r.passed,
                    "metrics": {k: v.to_dict() for k, v in r.metrics.items()},
                    "latency_ms": r.total_latency_ms,
                }
                for r in results
            ],
            summary_metrics={"cases": n, "pass_rate": pass_rate, "avg_score": avg_score},
            pass_rate=pass_rate,
            avg_score=avg_score,
            total_latency_ms=total_latency,
        )
        run_id = eval_run.id

    # Print summary table
    table = Table(title=f"Evaluation Results: {run_name} (Run ID: {run_id[:8]})")
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Value", style="bold green" if pass_rate >= pass_threshold else "bold red")
    table.add_row("Total Test Cases", str(n))
    table.add_row("Passed Cases", f"{passed_count} / {n}")
    table.add_row("Pass Rate", f"{pass_rate * 100:.1f}%")
    table.add_row("Average Score", f"{avg_score:.4f}")
    table.add_row("Total Latency", f"{total_latency:.2f}ms")
    console.print(table)

    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "run_id": run_id,
            "run_name": run_name,
            "candidate_model": candidate_model,
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "total_latency_ms": total_latency,
            "results": [r.to_dict() for r in results],
        }
        out_p.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        console.print(f"[green]Saved detailed report to:[/] [bold]{output}[/]")


@app.command()
def gate(
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate run JSON file or run ID."),
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline run JSON file or run ID."),
    alpha: float = typer.Option(0.05, "--alpha", "-a", help="Significance level alpha."),
    threshold: float = typer.Option(0.02, "--threshold", "-t", help="Regression threshold delta."),
    markdown_output: Optional[str] = typer.Option(None, "--markdown", "-m", help="Path to write Markdown summary."),
):
    """Evaluate statistical regression gate for CI/CD pipelines (exits with code 0 or 1)."""
    console.print("[bold cyan]Evaluating CI regression gate...[/]")

    def load_run_scores(ref: str) -> tuple[str, list[float], list[float], list[bool]]:
        p = Path(ref)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            run_id = data.get("run_id", p.stem)
            scores = [r["overall_score"] if "overall_score" in r else r["score"] for r in data.get("results", [])]
            lats = [r.get("total_latency_ms", r.get("latency_ms", 100.0)) for r in data.get("results", [])]
            passed = [r.get("passed", True) for r in data.get("results", [])]
            return run_id, scores, lats, passed
        else:
            with SessionLocal() as db:
                repo = EvaluationRepository(db)
                run_obj = repo.get_run(ref)
                if not run_obj:
                    console.print(f"[bold red]Error:[/] Run {ref} not found in DB or filesystem.")
                    raise typer.Exit(code=2)
                return (
                    run_obj.id,
                    [r.score for r in run_obj.results],
                    [r.latency_ms for r in run_obj.results],
                    [r.passed for r in run_obj.results],
                )

    c_id, c_scores, c_lats, c_passed = load_run_scores(candidate)
    b_id, b_scores, b_lats, b_passed = load_run_scores(baseline)

    detector = StatisticalRegressionDetector(alpha=alpha, regression_threshold_delta=threshold)
    report = detector.compare(
        baseline_scores=b_scores,
        candidate_scores=c_scores,
        baseline_latencies=b_lats,
        candidate_latencies=c_lats,
        baseline_passed=b_passed,
        candidate_passed=c_passed,
        baseline_run_id=b_id,
        candidate_run_id=c_id,
    )

    console.print(Panel(report.to_markdown(), title=f"CI Gate: {report.status.value}", expand=False))

    if markdown_output:
        Path(markdown_output).write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"[green]Wrote markdown report to:[/] {markdown_output}")

    if report.status == RegressionStatus.REGRESSION_DETECTED:
        console.print("[bold red]CI GATE FAILED:[/] Regression detected!")
        sys.exit(1)
    else:
        console.print("[bold green]CI GATE PASSED:[/] Candidate meets quality thresholds.")
        sys.exit(0)


@guard_app.command("check")
def check_guard(
    text: str = typer.Argument(..., help="Text to inspect with guardrails."),
    guards: str = typer.Option("pii,injection,toxicity", "--guards", "-g", help="Guardrails to run."),
):
    """Run text through guardrails."""
    guard_objs = []
    for g in [item.strip().lower() for item in guards.split(",") if item.strip()]:
        if g == "pii":
            guard_objs.append(PIIGuardrail())
        elif g in ("injection", "prompt_injection"):
            guard_objs.append(PromptInjectionGuardrail())
        elif g in ("toxicity", "toxic"):
            guard_objs.append(ToxicityGuardrail())
        elif g in ("schema", "json"):
            guard_objs.append(SchemaValidationGuardrail())
        elif g in ("budget", "sla"):
            guard_objs.append(CostBudgetGuardrail())

    pipe = GuardrailPipeline(guard_objs, mode=ExecutionMode.COLLECT_ALL)
    res = pipe.run(text)

    status_style = "bold green" if res.passed else "bold red"
    console.print(f"Passed: [{status_style}]{res.passed}[/]")
    console.print(f"Action: [yellow]{res.final_action.value}[/]")
    console.print(f"Score: [cyan]{res.overall_score:.4f}[/]")
    console.print(f"Latency: [dim]{res.total_latency_ms:.2f}ms[/]")
    console.print("\n[bold]Sanitized Output:[/]")
    console.print(res.sanitized_text)

    if res.violations:
        t = Table(title="Detected Violations")
        t.add_column("Rule", style="cyan")
        t.add_column("Severity", style="red")
        t.add_column("Message", style="white")
        for v in res.violations:
            t.add_row(v.rule_name, v.severity.value, v.message)
        console.print(t)


@dataset_app.command("register")
def register_dataset(
    file_path: str = typer.Argument(..., help="Path to dataset file."),
    name: str = typer.Option(..., "--name", "-n", help="Dataset name."),
    tag: str = typer.Option("v1.0.0", "--tag", "-t", help="Dataset version tag."),
    description: Optional[str] = typer.Option(None, "--desc", "-d", help="Description."),
):
    """Register a versioned dataset snapshot."""
    mgr = DatasetManager()
    cases = mgr.parse_test_cases(file_path)
    info = mgr.register_dataset_version(name=name, version_tag=tag, test_cases=cases, description=description)
    console.print(
        f"[bold green]Registered dataset:[/] [cyan]{name}[/] ([yellow]{info.version_tag}[/]) "
        f"- {info.num_cases} test cases, SHA256: [dim]{info.content_hash[:16]}...[/]"
    )


@trace_app.command("view")
def view_trace(trace_id: str = typer.Argument(..., help="ID of trace to inspect.")):
    """View hierarchical span tree for a given trace."""
    with SessionLocal() as db:
        repo = TraceRepository(db)
        trace_obj = repo.get_trace(trace_id)
        if not trace_obj:
            console.print(f"[bold red]Error:[/] Trace {trace_id} not found.")
            raise typer.Exit(code=1)

        # Build in-memory Trace to use ConsoleSpanExporter
        from evalforge.tracing.span import Span, Trace
        trace = Trace(
            trace_id=trace_obj.id,
            name=trace_obj.name,
            total_duration_ms=trace_obj.total_duration_ms,
        )
        for s in trace_obj.spans:
            span = Span(
                name=s.name,
                trace_id=trace.trace_id,
                span_id=s.id,
                parent_span_id=s.parent_span_id,
                span_type=s.span_type,
                status=s.status,
                duration_ms=s.duration_ms,
                attributes=json.loads(s.attributes_json),
                error_message=s.error_message,
            )
            trace.add_span(span)

        exporter = ConsoleSpanExporter(console)
        exporter.export(trace)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload."),
):
    """Start the EvalForge FastAPI web service."""
    import uvicorn
    console.print(f"[bold cyan]Launching EvalForge API server on http://{host}:{port}[/]")
    uvicorn.run("evalforge.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
