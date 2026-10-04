"""Manifest and provenance utilities for tracking experiment executions."""

import datetime
import platform
import subprocess
import importlib.metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core.config import (
    METHODOLOGY_VERSION, TEST_SIZE, OUTLIER_SIGMA, BREAK_RATIO, 
    MODEL_CONFIGS, NOISE_ORDER, DATASET_IDS
)
from app.core.levels import load_levels
from app.core.hashing import compute_config_hash
from app.engine.data import load_dataset, split_hashes
from app.storage.json_store import write_json_atomic, read_json

def _get_git_commit() -> Optional[str]:
    """Retrieve the current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return None

def _get_package_versions() -> Dict[str, str]:
    """Retrieve versions of installed packages."""
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            versions[name] = dist.version
    return versions

def build_manifest(
    run_id: str, 
    run_type: str, 
    stage: str, 
    plan_summary: Dict[str, Any], 
    requested_config: Any
) -> Dict[str, Any]:
    """Build the initial manifest dictionary for a run.
    
    Args:
        run_id: Unique identifier for the run.
        run_type: 'official' or 'custom'.
        stage: The stage name ('mvp', 'stage2', 'full', or custom identifier).
        plan_summary: Summary containing 'planned_fits' and 'seeds'.
        requested_config: The configuration requested for this run (e.g. CLI args or API payload).
        
    Returns:
        A dictionary representing the manifest.
    """
    levels_data = load_levels()
    
    # Extract planned fits and seeds
    planned_fits = plan_summary.get("planned_fits", 0)
    seeds = plan_summary.get("seeds", [])
    if isinstance(seeds, int):
        seeds = list(range(seeds))
        
    # Datasets metadata
    datasets_meta = {}
    all_split_hashes = {}
    for d in DATASET_IDS:
        _, _, meta = load_dataset(d)
        datasets_meta[d] = meta
        
        # Split hashes
        all_split_hashes[d] = {}
        for s in seeds:
            tr_h, te_h = split_hashes(d, s)
            all_split_hashes[d][str(s)] = {"train": tr_h, "test": te_h}

    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    manifest = {
        "run_id": run_id,
        "run_type": run_type,
        "stage": stage,
        "created_utc": now_utc,
        "finished_utc": None,
        "status": "started",
        "git_commit": _get_git_commit(),
        "requested_config": requested_config,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _get_package_versions(),
        "config_hash": compute_config_hash(),
        "methodology_snapshot": {
            "version": METHODOLOGY_VERSION,
            "test_size": TEST_SIZE,
            "outlier_sigma": OUTLIER_SIGMA,
            "break_ratio": BREAK_RATIO,
            "noise_order": list(NOISE_ORDER)
        },
        "levels_snapshot": {
            "active_table": levels_data.get("active_table"),
            "values": levels_data.get("tables", {}).get(levels_data.get("active_table")),
            "version": levels_data.get("version", "1.0.0"),
            "frozen": levels_data.get("frozen", False),
            "frozen_at": levels_data.get("frozen_at")
        },
        "datasets_meta": datasets_meta,
        "models": MODEL_CONFIGS,
        "seeds": seeds,
        "planned_fits": planned_fits,
        "completed_fits": 0,
        "split_hashes": all_split_hashes
    }
    
    return manifest

def finalize_manifest(path: Union[str, Path], status: str, completed_fits: int) -> None:
    """Update an existing manifest on disk with final status.
    
    Args:
        path: Path to the manifest.json file.
        status: The final status (e.g., 'completed', 'error', 'cancelled').
        completed_fits: Number of successfully completed fits.
    """
    manifest = read_json(path)
    manifest["status"] = status
    manifest["completed_fits"] = completed_fits
    manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json_atomic(path, manifest)
