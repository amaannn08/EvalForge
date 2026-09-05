"""EvalForge configuration settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Global configuration settings for EvalForge."""

    env: Literal["development", "testing", "production"] = Field(
        default_factory=lambda: os.getenv("EVALFORGE_ENV", "development")  # type: ignore
    )
    database_url: str = Field(
        default_factory=lambda: os.getenv("EVALFORGE_DATABASE_URL", "sqlite:///./evalforge.db")
    )
    app_title: str = "EvalForge API"
    app_version: str = "0.1.0"
    debug: bool = Field(
        default_factory=lambda: os.getenv("EVALFORGE_DEBUG", "false").lower() in ("true", "1", "yes")
    )

    # Tracing configuration
    trace_sample_rate: float = 1.0
    export_batch_size: int = 100
    export_interval_ms: int = 1000

    # Guardrails SLA defaults
    default_latency_sla_ms: float = 2500.0
    default_max_prompt_tokens: int = 4096
    default_max_completion_tokens: int = 2048

    # Statistical evaluation thresholds
    default_significance_alpha: float = 0.05
    default_regression_threshold: float = 0.02
    min_paired_samples: int = 10

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"

    model_config = {"arbitrary_types_allowed": True}


settings = Settings()
