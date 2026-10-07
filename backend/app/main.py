"""FastAPI application entry point for Death by a Thousand Cuts.

Serves the REST API at /api with CORS for the React frontend (http://localhost:5173).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.config_routes import router as config_router
from app.api.results_routes import router as results_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Placeholder: worker startup will be added in Task 6.8
    yield
    # Placeholder: worker shutdown will be added in Task 6.8


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
