"""FastAPI application entry point for Death by a Thousand Cuts.

Serves the REST API at /api with CORS for the React frontend (http://localhost:5173).
The custom-experiment worker is started as a separate spawned process during
lifespan unless START_WORKER=false is set.
"""
import logging
import multiprocessing
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.config_routes import router as config_router
from app.api.results_routes import router as results_router
from app.api.experiments_routes import router as experiments_router
from app.jobs.worker import worker_main

logger = logging.getLogger(__name__)

_WORKER_PROCESS: Optional[multiprocessing.Process] = None


def _start_worker_process() -> Optional[multiprocessing.Process]:
    """Start the dedicated worker with the spawn method (Windows safe)."""
    if os.environ.get("START_WORKER", "true").strip().lower() == "false":
        logger.info("START_WORKER=false — custom worker must be started manually")
        return None
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=worker_main, name="custom-worker", daemon=True)
    process.start()
    logger.info("Started custom experiment worker process (pid=%s)", process.pid)
    return process


def _stop_worker_process(process: Optional[multiprocessing.Process]) -> None:
    if process is None or not process.is_alive():
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _WORKER_PROCESS
    _WORKER_PROCESS = _start_worker_process()
    try:
        yield
    finally:
        _stop_worker_process(_WORKER_PROCESS)
        _WORKER_PROCESS = None


app = FastAPI(
    title="Death by a Thousand Cuts — API",
    description="Backend API for the ML robustness experiment platform.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register global error handlers (produces {"error": {...}} shape)
register_exception_handlers(app)

# Mount routers under /api
app.include_router(health_router, prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(results_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
