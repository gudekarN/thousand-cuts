"""Path resolution utilities for official and custom experiment results.

Guarantees clean separation of official stages and custom run directories,
supporting RESULTS_ROOT override via environment variable.
"""

import os
from pathlib import Path
from typing import Union

from app.core.config import STAGES

RESULTS_ROOT_ENV_VAR = "RESULTS_ROOT"


def get_results_root() -> Path:
    """Return the base directory for all experiment results.

    Defaults to <project_root>/results, but can be overridden
    by the RESULTS_ROOT environment variable (e.g. for testing).
    """
    env_root = os.environ.get(RESULTS_ROOT_ENV_VAR)
    if env_root:
        return Path(env_root).resolve()

    # backend/app/storage/paths.py -> parents[3] is project root
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / "results").resolve()


def official_dir(stage: str) -> Path:
    """Return the directory for an official experiment stage.

    Args:
        stage: Stage name, must be one of STAGES ('mvp', 'stage2', 'full').

    Returns:
        Path to results/official/<stage>

    Raises:
        ValueError: If stage is unknown.
    """
    if stage not in STAGES:
        raise ValueError(
            f"Unknown official stage '{stage}'. Must be one of: {', '.join(STAGES)}"
        )
    return get_results_root() / "official" / stage


def custom_dir(run_id: str) -> Path:
    """Return the directory for a custom experiment run.

    Args:
        run_id: Unique identifier for the custom run.

    Returns:
        Path to results/custom/<run_id>

    Raises:
        ValueError: If run_id is empty or contains path traversal characters ('/', '\\', '..').
    """
    if not run_id or not isinstance(run_id, str):
        raise ValueError("run_id must be a non-empty string.")

    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError(
            f"Invalid run_id '{run_id}': cannot contain path separators ('/', '\\') or '..'."
        )

    return get_results_root() / "custom" / run_id
