"""Health check API route."""

from fastapi import APIRouter
from evalforge.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse()
