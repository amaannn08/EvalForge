"""Database package for EvalForge."""

from evalforge.db.session import Base, SessionLocal, engine, get_db, init_db
from evalforge.db.models import (
    ProjectModel,
    DatasetModel,
    DatasetVersionModel,
    TestCaseModel,
    EvaluationRunModel,
    EvaluationResultModel,
    TraceModel,
    SpanModel,
    GuardrailLogModel,
)
from evalforge.db.repository import (
    ProjectRepository,
    DatasetRepository,
    EvaluationRepository,
    TraceRepository,
)

__all__ = [
    "Base",
    "SessionLocal",
    "engine",
    "get_db",
    "init_db",
    "ProjectModel",
    "DatasetModel",
    "DatasetVersionModel",
    "TestCaseModel",
    "EvaluationRunModel",
    "EvaluationResultModel",
    "TraceModel",
    "SpanModel",
    "GuardrailLogModel",
    "ProjectRepository",
    "DatasetRepository",
    "EvaluationRepository",
    "TraceRepository",
]
