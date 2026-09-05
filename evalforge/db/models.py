"""SQLAlchemy 2.0 ORM models for EvalForge."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid
from typing import Any, Optional
from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evalforge.db.session import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return uuid.uuid4().hex


class ProjectModel(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    datasets: Mapped[list[DatasetModel]] = relationship(
        "DatasetModel", back_populates="project", cascade="all, delete-orphan"
    )
    evaluation_runs: Mapped[list[EvaluationRunModel]] = relationship(
        "EvaluationRunModel", back_populates="project", cascade="all, delete-orphan"
    )
    traces: Mapped[list[TraceModel]] = relationship(
        "TraceModel", back_populates="project", cascade="all, delete-orphan"
    )


class DatasetModel(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    project: Mapped[Optional[ProjectModel]] = relationship("ProjectModel", back_populates="datasets")
    versions: Mapped[list[DatasetVersionModel]] = relationship(
        "DatasetVersionModel", back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetVersionModel(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_tag: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    num_cases: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    dataset: Mapped[DatasetModel] = relationship("DatasetModel", back_populates="versions")
    test_cases: Mapped[list[TestCaseModel]] = relationship(
        "TestCaseModel", back_populates="dataset_version", cascade="all, delete-orphan"
    )
    evaluation_runs: Mapped[list[EvaluationRunModel]] = relationship(
        "EvaluationRunModel", back_populates="dataset_version"
    )

    __table_args__ = (
        Index("idx_dataset_version_tag", "dataset_id", "version_tag", unique=True),
    )


class TestCaseModel(Base):
    __tablename__ = "test_cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    input_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_json: Mapped[str] = mapped_column(Text, default="{}")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    dataset_version: Mapped[DatasetVersionModel] = relationship(
        "DatasetVersionModel", back_populates="test_cases"
    )


class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    dataset_version_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    run_name: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_model: Mapped[str] = mapped_column(String(128), nullable=False)
    baseline_run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_score: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(default=utc_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    project: Mapped[Optional[ProjectModel]] = relationship(
        "ProjectModel", back_populates="evaluation_runs"
    )
    dataset_version: Mapped[Optional[DatasetVersionModel]] = relationship(
        "DatasetVersionModel", back_populates="evaluation_runs"
    )
    results: Mapped[list[EvaluationResultModel]] = relationship(
        "EvaluationResultModel", back_populates="evaluation_run", cascade="all, delete-orphan"
    )


class EvaluationResultModel(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_case_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("test_cases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    actual_output: Mapped[str] = mapped_column(Text, nullable=False)
    expected_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    token_usage_json: Mapped[str] = mapped_column(Text, default="{}")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    evaluation_run: Mapped[EvaluationRunModel] = relationship(
        "EvaluationRunModel", back_populates="results"
    )


class TraceModel(Base):
    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="OK")
    total_duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    start_time: Mapped[datetime] = mapped_column(default=utc_now)
    end_time: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    project: Mapped[Optional[ProjectModel]] = relationship("ProjectModel", back_populates="traces")
    spans: Mapped[list[SpanModel]] = relationship(
        "SpanModel", back_populates="trace", cascade="all, delete-orphan"
    )
    guardrail_logs: Mapped[list[GuardrailLogModel]] = relationship(
        "GuardrailLogModel", back_populates="trace", cascade="all, delete-orphan"
    )


class SpanModel(Base):
    __tablename__ = "spans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("traces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_span_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    span_type: Mapped[str] = mapped_column(String(32), default="custom")
    status: Mapped[str] = mapped_column(String(32), default="OK")
    start_time: Mapped[datetime] = mapped_column(default=utc_now)
    end_time: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")
    events_json: Mapped[str] = mapped_column(Text, default="[]")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    trace: Mapped[TraceModel] = relationship("TraceModel", back_populates="spans")


class GuardrailLogModel(Base):
    __tablename__ = "guardrail_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    trace_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("traces.id", ondelete="CASCADE"), nullable=True, index=True
    )
    span_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    guardrail_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action_taken: Mapped[str] = mapped_column(String(32), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=True)
    score: Mapped[float] = mapped_column(Float, default=1.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    output_text: Mapped[str] = mapped_column(Text, nullable=False)
    violations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    trace: Mapped[Optional[TraceModel]] = relationship("TraceModel", back_populates="guardrail_logs")
