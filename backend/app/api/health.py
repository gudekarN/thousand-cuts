"""Health check endpoint: GET /api/health"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Return server liveness status."""
    return HealthResponse(status="ok")
