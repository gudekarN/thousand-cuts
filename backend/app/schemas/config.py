"""Pydantic schemas for the /api/config endpoint."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class LevelRow(BaseModel):
    level: int
    label: float
    gaussian: float
    outliers: float
    missing: float


class NoiseTypeInfo(BaseModel):
    id: str
    order: int  # canonical noise application order (1-based)


class ModelInfo(BaseModel):
    id: str
    label: str


class ConfigResponse(BaseModel):
    datasets: List[str]
    models: List[ModelInfo]
    noise_types: List[NoiseTypeInfo]
    active_table: str
    levels: List[Dict[str, Any]]  # rows of the active level table
    levels_version: str
    frozen: bool
    config_hash: Optional[str]
    available_official_stages: List[str]
