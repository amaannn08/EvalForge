"""Dataset management API routes."""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from evalforge.api.schemas import DatasetCreateRequest, DatasetVersionCreateRequest
from evalforge.datasets import DatasetManager, TestCase
from evalforge.db.repository import DatasetRepository
from evalforge.db.session import get_db

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    repo = DatasetRepository(db)
    datasets = repo.list_datasets()
    return [
        {
            "id": d.id,
            "name": d.name,
            "description": d.description,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "version_count": len(d.versions),
            "versions": [{"version_tag": v.version_tag, "num_cases": v.num_cases} for v in d.versions],
        }
        for d in datasets
    ]


@router.post("")
def create_dataset(req: DatasetCreateRequest, db: Session = Depends(get_db)):
    repo = DatasetRepository(db)
    existing = repo.get_dataset_by_name(req.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Dataset with name '{req.name}' already exists.")
    dataset = repo.create_dataset(name=req.name, description=req.description)
    return {"id": dataset.id, "name": dataset.name, "description": dataset.description}


@router.post("/{name}/versions")
def create_dataset_version(name: str, req: DatasetVersionCreateRequest, db: Session = Depends(get_db)):
    mgr = DatasetManager(db=db)
    cases = [
        TestCase(
            input_prompt=c.input_prompt,
            system_prompt=c.system_prompt,
            expected_output=c.expected_output,
            context=c.context,
            tags=c.tags,
        )
        for c in req.test_cases
    ]
    info = mgr.register_dataset_version(
        name=name,
        version_tag=req.version_tag,
        test_cases=cases,
        description=req.description,
        metadata=req.metadata,
    )
    return {
        "version_id": info.version_id,
        "version_tag": info.version_tag,
        "content_hash": info.content_hash,
        "num_cases": info.num_cases,
    }


@router.get("/{name}/versions/{version_tag}")
def get_dataset_version(name: str, version_tag: str, db: Session = Depends(get_db)):
    mgr = DatasetManager(db=db)
    try:
        cases = mgr.get_test_cases(name, version_tag)
        return {
            "dataset_name": name,
            "version_tag": version_tag,
            "num_cases": len(cases),
            "test_cases": [c.model_dump() for c in cases],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
