"""Pydantic schemas for custom experiment jobs.

`seeds` is a COUNT (1-10), not a list of seed values. Actual seeds are
`0 .. seeds-1`, per PRD 4.3 and rules.md section 4.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import DATASET_IDS, MODEL_IDS
from app.engine.noise import canonical_combo


class ExperimentRequest(BaseModel):
    """Payload for POST /api/experiments and POST /api/experiments/estimate."""

    dataset: str
    models: List[str]
    noises: List[str]
    mode: Literal["single_level", "sweep"]
    level: Optional[int] = None
    seeds: int = Field(..., ge=1, le=10)

    @field_validator("dataset")
    @classmethod
    def _dataset_known(cls, value: str) -> str:
        if value not in DATASET_IDS:
            raise ValueError(f"Unknown dataset '{value}'. Must be one of {list(DATASET_IDS)}")
        return value

    @field_validator("models")
    @classmethod
    def _models_valid(cls, value: List[str]) -> List[str]:
        if not value or len(value) > 4:
            raise ValueError("models must contain between 1 and 4 entries")
        if len(set(value)) != len(value):
            raise ValueError("models must not contain duplicates")
        for model in value:
            if model not in MODEL_IDS:
                raise ValueError(f"Unknown model '{model}'. Must be one of {list(MODEL_IDS)}")
        return value

    @field_validator("noises")
    @classmethod
    def _noises_valid(cls, value: List[str]) -> List[str]:
        if not value or len(value) > 4:
            raise ValueError("noises must contain between 1 and 4 entries")
        if len(set(value)) != len(value):
            raise ValueError("noises must not contain duplicates")
        for noise in value:
            try:
                canonical_combo(noise)
            except ValueError as exc:
                raise ValueError(f"Invalid noise selection '{noise}': {exc}") from exc
        return value

    @model_validator(mode="after")
    def _level_matches_mode(self) -> "ExperimentRequest":
        if self.mode == "single_level":
            if self.level is None or not (1 <= self.level <= 5):
                raise ValueError("level must be 1-5 when mode is 'single_level'")
        return self

    def to_planner_request(self) -> Dict[str, Any]:
        """Return the dict consumed by the shared plan builder."""
        return {
            "dataset": self.dataset,
            "models": list(self.models),
            "noises": list(self.noises),
            "mode": self.mode,
            "level": self.level,
            "seeds": self.seeds,
        }


class JobProgress(BaseModel):
    done: int = 0
    total: int = 0
    percent: float = 0.0
    current: Optional[Dict[str, Any]] = None
    eta_seconds: Optional[float] = None


class JobResponse(BaseModel):
    run_id: str
    run_type: Literal["custom"] = "custom"
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    request: Dict[str, Any]
    planned_fits: int
    progress: JobProgress
    created_utc: str
    started_utc: Optional[str] = None
    finished_utc: Optional[str] = None
    heartbeat_utc: Optional[str] = None
    cancel_requested: bool = False
    error: Optional[str] = None


class EstimateResponse(BaseModel):
    planned_fits: int
    estimated_seconds: float
    mean_fit_time_s: float
    estimate_source: str
    combos: List[str]
    seeds: List[int]
