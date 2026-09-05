"""Dataset management and versioning package for EvalForge."""

from evalforge.datasets.schema import (
    TestCase,
    DatasetVersionInfo,
    DatasetInfo,
    compute_dataset_hash,
)
from evalforge.datasets.manager import DatasetManager

__all__ = [
    "TestCase",
    "DatasetVersionInfo",
    "DatasetInfo",
    "compute_dataset_hash",
    "DatasetManager",
]
