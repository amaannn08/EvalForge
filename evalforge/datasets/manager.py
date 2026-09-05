"""Dataset loader, exporter, and version management engine."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Optional
from sqlalchemy.orm import Session

from evalforge.datasets.schema import DatasetInfo, DatasetVersionInfo, TestCase, compute_dataset_hash
from evalforge.db.repository import DatasetRepository
from evalforge.db.session import SessionLocal


class DatasetManager:
    """Manages versioned datasets with cryptographic content hashing and SQLite persistence."""

    def __init__(self, db: Optional[Session] = None):
        self._custom_db = db

    def _get_db(self) -> Session:
        return self._custom_db if self._custom_db is not None else SessionLocal()

    def parse_test_cases(self, file_path: str | Path) -> list[TestCase]:
        """Load test cases from JSON, JSONL, or CSV file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = p.suffix.lower()
        content = p.read_text(encoding="utf-8")

        if ext == ".jsonl":
            cases = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    data = json.loads(line)
                    cases.append(TestCase(**data))
            return cases

        elif ext == ".json":
            raw_data = json.loads(content)
            if isinstance(raw_data, list):
                return [TestCase(**item) for item in raw_data]
            elif isinstance(raw_data, dict) and "test_cases" in raw_data:
                return [TestCase(**item) for item in raw_data["test_cases"]]
            raise ValueError(f"Invalid JSON format for test cases in {file_path}")

        elif ext == ".csv":
            cases = []
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                prompt = row.get("input_prompt") or row.get("prompt") or row.get("input") or ""
                expected = row.get("expected_output") or row.get("target") or row.get("ground_truth")
                system = row.get("system_prompt")
                tags = [t.strip() for t in row.get("tags", "").split(",") if t.strip()]
                cases.append(
                    TestCase(
                        input_prompt=prompt,
                        system_prompt=system,
                        expected_output=expected,
                        tags=tags,
                    )
                )
            return cases

        else:
            raise ValueError(f"Unsupported file extension: {ext}. Supported formats: .json, .jsonl, .csv")

    def register_dataset_version(
        self,
        name: str,
        version_tag: str,
        test_cases: list[TestCase],
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        project_id: Optional[str] = None,
    ) -> DatasetVersionInfo:
        """Register a dataset version with deterministic SHA-256 hash in SQLite."""
        content_hash = compute_dataset_hash(test_cases)
        db = self._get_db()
        should_close = self._custom_db is None

        try:
            repo = DatasetRepository(db)
            dataset = repo.get_dataset_by_name(name)
            if not dataset:
                dataset = repo.create_dataset(name=name, description=description, project_id=project_id)

            existing_version = repo.get_version_by_tag(dataset.id, version_tag)
            if existing_version:
                return DatasetVersionInfo(
                    version_id=existing_version.id,
                    version_tag=existing_version.version_tag,
                    content_hash=existing_version.content_hash,
                    num_cases=existing_version.num_cases,
                )

            cases_data = [
                {
                    "input_prompt": c.input_prompt,
                    "system_prompt": c.system_prompt,
                    "expected_output": c.expected_output,
                    "context": c.context,
                    "metadata": c.metadata,
                    "tags": c.tags,
                }
                for c in test_cases
            ]

            version_model = repo.create_version(
                dataset_id=dataset.id,
                version_tag=version_tag,
                content_hash=content_hash,
                metadata=metadata,
            )
            repo.add_test_cases(version_model.id, cases_data)

            return DatasetVersionInfo(
                version_id=version_model.id,
                version_tag=version_tag,
                content_hash=content_hash,
                num_cases=len(test_cases),
                metadata=metadata or {},
            )
        finally:
            if should_close:
                db.close()

    def get_test_cases(self, dataset_name: str, version_tag: str) -> list[TestCase]:
        """Fetch test cases for a specific dataset version."""
        db = self._get_db()
        should_close = self._custom_db is None

        try:
            repo = DatasetRepository(db)
            dataset = repo.get_dataset_by_name(dataset_name)
            if not dataset:
                raise ValueError(f"Dataset '{dataset_name}' not found.")
            version = repo.get_version_by_tag(dataset.id, version_tag)
            if not version:
                raise ValueError(f"Version '{version_tag}' not found for dataset '{dataset_name}'.")

            return [
                TestCase(
                    id=c.id,
                    input_prompt=c.input_prompt,
                    system_prompt=c.system_prompt,
                    expected_output=c.expected_output,
                    context=json.loads(c.context_json),
                    metadata=json.loads(c.metadata_json),
                    tags=json.loads(c.tags_json),
                )
                for c in version.test_cases
            ]
        finally:
            if should_close:
                db.close()

    def export_dataset(self, dataset_name: str, version_tag: str, output_path: str | Path) -> Path:
        """Export dataset version test cases to a JSON file."""
        cases = self.get_test_cases(dataset_name, version_tag)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [c.model_dump() for c in cases]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p
