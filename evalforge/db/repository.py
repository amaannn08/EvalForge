"""Database repository access layer for EvalForge entities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy import desc, select
from sqlalchemy.orm import Session, joinedload

from evalforge.db.models import (
    DatasetModel,
    DatasetVersionModel,
    EvaluationResultModel,
    EvaluationRunModel,
    GuardrailLogModel,
    ProjectModel,
    SpanModel,
    TestCaseModel,
    TraceModel,
)


class ProjectRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create(self, name: str, description: Optional[str] = None) -> ProjectModel:
        project = self.db.execute(select(ProjectModel).where(ProjectModel.name == name)).scalar_one_or_none()
        if not project:
            project = ProjectModel(name=name, description=description)
            self.db.add(project)
            self.db.commit()
            self.db.refresh(project)
        return project

    def get(self, project_id: str) -> Optional[ProjectModel]:
        return self.db.execute(select(ProjectModel).where(ProjectModel.id == project_id)).scalar_one_or_none()

    def list(self) -> list[ProjectModel]:
        return list(self.db.execute(select(ProjectModel).order_by(desc(ProjectModel.created_at))).scalars().all())


class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_dataset(self, name: str, description: Optional[str] = None, project_id: Optional[str] = None) -> DatasetModel:
        dataset = DatasetModel(name=name, description=description, project_id=project_id)
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def get_dataset(self, dataset_id: str) -> Optional[DatasetModel]:
        return self.db.execute(select(DatasetModel).where(DatasetModel.id == dataset_id)).scalar_one_or_none()

    def get_dataset_by_name(self, name: str) -> Optional[DatasetModel]:
        return self.db.execute(select(DatasetModel).where(DatasetModel.name == name)).scalar_one_or_none()

    def list_datasets(self) -> list[DatasetModel]:
        return list(self.db.execute(select(DatasetModel).order_by(desc(DatasetModel.created_at))).scalars().all())

    def create_version(
        self,
        dataset_id: str,
        version_tag: str,
        content_hash: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DatasetVersionModel:
        version = DatasetVersionModel(
            dataset_id=dataset_id,
            version_tag=version_tag,
            content_hash=content_hash,
            metadata_json=json.dumps(metadata or {}),
        )
        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)
        return version

    def get_version(self, version_id: str) -> Optional[DatasetVersionModel]:
        return self.db.execute(
            select(DatasetVersionModel)
            .options(joinedload(DatasetVersionModel.test_cases))
            .where(DatasetVersionModel.id == version_id)
        ).unique().scalar_one_or_none()

    def get_version_by_tag(self, dataset_id: str, version_tag: str) -> Optional[DatasetVersionModel]:
        return self.db.execute(
            select(DatasetVersionModel)
            .options(joinedload(DatasetVersionModel.test_cases))
            .where(DatasetVersionModel.dataset_id == dataset_id, DatasetVersionModel.version_tag == version_tag)
        ).unique().scalar_one_or_none()

    def add_test_cases(self, version_id: str, cases: list[dict[str, Any]]) -> list[TestCaseModel]:
        models = []
        for c in cases:
            m = TestCaseModel(
                dataset_version_id=version_id,
                input_prompt=c["input_prompt"],
                system_prompt=c.get("system_prompt"),
                expected_output=c.get("expected_output"),
                context_json=json.dumps(c.get("context", {})),
                metadata_json=json.dumps(c.get("metadata", {})),
                tags_json=json.dumps(c.get("tags", [])),
            )
            models.append(m)
            self.db.add(m)

        version = self.get_version(version_id)
        if version:
            version.num_cases = len(models)
            self.db.add(version)

        self.db.commit()
        for m in models:
            self.db.refresh(m)
        return models


class EvaluationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_run(
        self,
        run_name: str,
        candidate_model: str,
        project_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        baseline_run_id: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> EvaluationRunModel:
        run = EvaluationRunModel(
            run_name=run_name,
            candidate_model=candidate_model,
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            baseline_run_id=baseline_run_id,
            config_json=json.dumps(config or {}),
            status="RUNNING",
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def record_results(
        self,
        run_id: str,
        results: list[dict[str, Any]],
        summary_metrics: dict[str, Any],
        pass_rate: float,
        avg_score: float,
        total_latency_ms: float,
        status: str = "COMPLETED",
    ) -> EvaluationRunModel:
        for r in results:
            result_model = EvaluationResultModel(
                run_id=run_id,
                test_case_id=r.get("test_case_id"),
                input_text=r["input_text"],
                actual_output=r["actual_output"],
                expected_output=r.get("expected_output"),
                score=r.get("score", 0.0),
                passed=r.get("passed", False),
                metrics_json=json.dumps(r.get("metrics", {})),
                latency_ms=r.get("latency_ms", 0.0),
                token_usage_json=json.dumps(r.get("token_usage", {})),
                error_message=r.get("error_message"),
            )
            self.db.add(result_model)

        run = self.get_run(run_id)
        if run:
            run.status = status
            run.summary_metrics_json = json.dumps(summary_metrics)
            run.pass_rate = pass_rate
            run.avg_score = avg_score
            run.total_latency_ms = total_latency_ms
            run.completed_at = datetime.now(timezone.utc)
            self.db.add(run)

        self.db.commit()
        if run:
            self.db.refresh(run)
        return run

    def get_run(self, run_id: str) -> Optional[EvaluationRunModel]:
        return self.db.execute(
            select(EvaluationRunModel)
            .options(joinedload(EvaluationRunModel.results))
            .where(EvaluationRunModel.id == run_id)
        ).unique().scalar_one_or_none()

    def list_runs(self, limit: int = 50) -> list[EvaluationRunModel]:
        return list(
            self.db.execute(
                select(EvaluationRunModel).order_by(desc(EvaluationRunModel.started_at)).limit(limit)
            ).scalars().all()
        )


class TraceRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_trace(
        self,
        trace_id: str,
        name: str,
        status: str,
        total_duration_ms: float,
        total_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
        spans: Optional[list[dict[str, Any]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> TraceModel:
        trace = TraceModel(
            id=trace_id,
            name=name,
            status=status,
            total_duration_ms=total_duration_ms,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            metadata_json=json.dumps(metadata or {}),
            start_time=start_time or datetime.now(timezone.utc),
            end_time=end_time,
        )
        self.db.add(trace)

        if spans:
            for s in spans:
                span_model = SpanModel(
                    id=s["span_id"],
                    trace_id=trace_id,
                    parent_span_id=s.get("parent_span_id"),
                    name=s["name"],
                    span_type=s.get("span_type", "custom"),
                    status=s.get("status", "OK"),
                    start_time=s.get("start_time", datetime.now(timezone.utc)),
                    end_time=s.get("end_time"),
                    duration_ms=s.get("duration_ms", 0.0),
                    attributes_json=json.dumps(s.get("attributes", {})),
                    events_json=json.dumps(s.get("events", [])),
                    error_message=s.get("error_message"),
                )
                self.db.add(span_model)

        self.db.commit()
        self.db.refresh(trace)
        return trace

    def get_trace(self, trace_id: str) -> Optional[TraceModel]:
        return self.db.execute(
            select(TraceModel).options(joinedload(TraceModel.spans)).where(TraceModel.id == trace_id)
        ).unique().scalar_one_or_none()

    def list_traces(self, limit: int = 50) -> list[TraceModel]:
        return list(
            self.db.execute(select(TraceModel).order_by(desc(TraceModel.start_time)).limit(limit))
            .scalars()
            .all()
        )
