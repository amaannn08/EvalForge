"""Domain models for Datasets, Versions, and Test Cases with content hashing."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Optional
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single evaluation benchmark test case."""
    __test__ = False  # Prevent pytest from treating this as a test class

    id: Optional[str] = None
    input_prompt: str
    system_prompt: Optional[str] = None
    expected_output: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic canonical representation for content hash computation."""
        return {
            "input_prompt": self.input_prompt.strip(),
            "system_prompt": (self.system_prompt or "").strip(),
            "expected_output": (self.expected_output or "").strip(),
            "context": self.context,
            "metadata": self.metadata,
            "tags": sorted(self.tags),
        }


def compute_dataset_hash(cases: list[TestCase]) -> str:
    """Computes a deterministic SHA-256 content hash across test cases."""
    canonical_list = [c.canonical_dict() for c in cases]
    serialized = json.dumps(canonical_list, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DatasetVersionInfo(BaseModel):
    """Metadata for a specific immutable dataset snapshot."""
    version_id: Optional[str] = None
    version_tag: str
    content_hash: str
    num_cases: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetInfo(BaseModel):
    """High-level dataset container representation."""
    dataset_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    versions: list[DatasetVersionInfo] = Field(default_factory=list)
