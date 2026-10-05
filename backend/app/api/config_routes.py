"""Config endpoint: GET /api/config

Returns datasets, models, noise types, active level table, frozen state,
config hash, and available official stages found on disk.
No ML logic here — pure metadata from config.py and levels.json.
"""
from typing import Any, Dict, List
from pathlib import Path

from fastapi import APIRouter

from app.core.config import (
    DATASET_IDS,
    MODEL_IDS,
    NOISE_ORDER,
    NOISE_IDS,
    STAGES,
)
from app.core.levels import load_levels, DEFAULT_LEVELS_PATH
from app.schemas.config import ConfigResponse, ModelInfo, NoiseTypeInfo
from app.storage.paths import official_dir

router = APIRouter()

# Human-readable labels for model IDs
_MODEL_LABELS: Dict[str, str] = {
    "logreg": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def _available_official_stages() -> List[str]:
    """Return stages that have a raw_results.csv on disk."""
    found = []
    for stage in STAGES:
        try:
            csv = official_dir(stage) / "raw_results.csv"
            if csv.exists():
                found.append(stage)
        except Exception:
            pass
    return found


@router.get("/config", response_model=ConfigResponse, tags=["config"])
async def get_config() -> ConfigResponse:
    """Return the current experiment configuration and metadata."""
    lvl = load_levels(str(DEFAULT_LEVELS_PATH))

    active_table_name: str = lvl.get("active_table", "A")
    tables: Dict[str, Any] = lvl.get("tables", {})
    active_table: Dict[str, Any] = tables.get(active_table_name, {})

    # Build level rows as a list of dicts (level index + per-param values)
    # Table structure: {param_name: [val_L0, val_L1, ..., val_L5]}
    level_indices: list = lvl.get("level_indices", [0, 1, 2, 3, 4, 5])
    level_rows: List[Dict[str, Any]] = []
    for li in level_indices:
        row: Dict[str, Any] = {"level": li}
        for param_name, values in active_table.items():
            row[param_name] = values[li] if li < len(values) else 0.0
        level_rows.append(row)

    models = [
        ModelInfo(id=m, label=_MODEL_LABELS.get(m, m))
        for m in MODEL_IDS
    ]

    noise_types = [
        NoiseTypeInfo(id=n, order=NOISE_IDS[n])
        for n in NOISE_ORDER
    ]

    return ConfigResponse(
        datasets=list(DATASET_IDS),
        models=models,
        noise_types=noise_types,
        active_table=active_table_name,
        levels=level_rows,
        levels_version=lvl.get("levels_version", ""),
        frozen=bool(lvl.get("frozen", False)),
        config_hash=lvl.get("config_hash"),
        available_official_stages=_available_official_stages(),
    )
