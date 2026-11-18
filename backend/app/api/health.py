"""Health check endpoint: GET /api/health"""
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.jobs.worker import worker_status

router = APIRouter()


class WorkerHealth(BaseModel):
    status: str
    heartbeat_utc: Optional[str] = None
    current_run_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    worker: Optional[WorkerHealth] = None


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Return server liveness and custom-worker status."""
    return HealthResponse(status="ok", worker=WorkerHealth(**worker_status()))
