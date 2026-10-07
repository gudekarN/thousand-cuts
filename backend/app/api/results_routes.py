"""Results read-layer API endpoints.

Serves summary, breaking_points, robustness and manifest data
from the official/full results directory as JSON.
No ML logic — pure CSV/JSON reads.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.storage.paths import official_dir

router = APIRouter()


def _csv(stage: str, name: str) -> Path:
    return official_dir(stage) / name


def _read_csv(stage: str, filename: str) -> pd.DataFrame:
    path = _csv(stage, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found for stage '{stage}'")
    df = pd.read_csv(path)
    # Replace NaN with None so json serialises cleanly
    df = df.where(pd.notna(df), other=None)
    return df


@router.get("/results/summary", tags=["results"])
async def get_summary(stage: str = "full", dataset: Optional[str] = None,
                       model: Optional[str] = None, combo: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return summary.csv rows with optional filters."""
    df = _read_csv(stage, "summary.csv")
    if dataset:
        df = df[df["dataset"] == dataset]
    if model:
        df = df[df["model"] == model]
    if combo:
        df = df[df["combo"] == combo]
    return df.to_dict(orient="records")


@router.get("/results/breaking-points", tags=["results"])
async def get_breaking_points(stage: str = "full", dataset: Optional[str] = None,
                               model: Optional[str] = None, combo: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return breaking_points.csv rows with optional filters."""
    df = _read_csv(stage, "breaking_points.csv")
    if dataset:
        df = df[df["dataset"] == dataset]
    if model:
        df = df[df["model"] == model]
    if combo:
        df = df[df["combo"] == combo]
    return df.to_dict(orient="records")


@router.get("/results/robustness", tags=["results"])
async def get_robustness(stage: str = "full", dataset: Optional[str] = None,
                          scope: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return robustness.csv rows (BPI / RS rankings) with optional filters."""
    df = _read_csv(stage, "robustness.csv")
    if dataset:
        df = df[df["dataset"] == dataset]
    if scope:
        df = df[df["scope"] == scope]
    return df.to_dict(orient="records")


@router.get("/results/baselines", tags=["results"])
async def get_baselines(stage: str = "full", dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return baselines.csv rows."""
    df = _read_csv(stage, "baselines.csv")
    if dataset:
        df = df[df["dataset"] == dataset]
    return df.to_dict(orient="records")


@router.get("/results/manifest", tags=["results"])
async def get_manifest(stage: str = "full") -> Dict[str, Any]:
    """Return the manifest.json for a stage."""
    import json
    path = _csv(stage, "manifest.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"manifest.json not found for stage '{stage}'")
    return json.loads(path.read_text())


@router.get("/results/synergy", tags=["results"])
async def get_synergy(stage: str = "full", dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return synergy.csv rows."""
    df = _read_csv(stage, "synergy.csv")
    if dataset:
        df = df[df["dataset"] == dataset]
    return df.to_dict(orient="records")
