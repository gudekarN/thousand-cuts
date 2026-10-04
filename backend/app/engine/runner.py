"""Experiment execution and planning.

Builds deterministic execution plans for official stages and custom runs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import (
    DATASET_BREAST_CANCER,
    DATASET_IDS,
    MODEL_IDS,
    STAGES,
    STAGE_MVP,
    STAGE_STAGE2,
    STAGE_FULL,
    COMBO_CLEAN,
    STAGE_SEEDS,
)
import time
import datetime
import json
import numpy as np

from app.engine.data import split
from app.engine.noise import apply_noise, list_all_combos, canonical_combo
from app.engine.pipelines import build_pipeline
from app.engine.metrics import evaluate


@dataclass(frozen=True)
class FitSpec:
    """Specification for a single model training and evaluation fit."""
    dataset: str
    model: str
    combo: str
    level: int
    seed: int


def build_plan(
    stage: Optional[str] = None, 
    custom_request: Optional[Dict[str, Any]] = None
) -> List[FitSpec]:
    """Build the execution plan (list of FitSpecs) for an experiment.
    
    Args:
        stage: Name of the official stage ('mvp', 'stage2', 'full').
        custom_request: Dictionary of custom run configuration.
        
    Returns:
        List of FitSpec objects ordered by dataset, seed, combo, level, model.
    """
    if stage is not None and custom_request is not None:
        raise ValueError("Cannot provide both stage and custom_request")
        
    if stage is None and custom_request is None:
        raise ValueError("Must provide either stage or custom_request")

    if stage:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
            
        if stage == STAGE_MVP:
            datasets = [DATASET_BREAST_CANCER]
            combos = ["label", "gaussian", "outliers", "missing", "label+gaussian"]
            seeds = list(STAGE_SEEDS[STAGE_MVP])
        elif stage == STAGE_STAGE2:
            datasets = list(DATASET_IDS)
            combos = list_all_combos()
            seeds = list(STAGE_SEEDS[STAGE_STAGE2])
        else:  # STAGE_FULL
            datasets = list(DATASET_IDS)
            combos = list_all_combos()
            seeds = list(STAGE_SEEDS[STAGE_FULL])
            
        models = list(MODEL_IDS)
        levels_to_run = [1, 2, 3, 4, 5]
        
    else:
        # Custom request
        dataset = custom_request.get("dataset")
        models = custom_request.get("models", [])
        noises = custom_request.get("noises", [])
        mode = custom_request.get("mode")
        level = custom_request.get("level")
        seed_count = custom_request.get("seed_count")
        
        if dataset not in DATASET_IDS:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        if not models or not isinstance(models, list):
            raise ValueError("models must be a non-empty list")
        for m in models:
            if m not in MODEL_IDS:
                raise ValueError(f"Unknown model: {m}")
                
        if not noises or not isinstance(noises, list):
            raise ValueError("noises must be a non-empty list")
            
        combos_set = set()
        for n in noises:
            try:
                can_n = canonical_combo(n)
                combos_set.add(can_n)
                # Automatically add single-noise components
                for single_component in can_n.split("+"):
                    combos_set.add(single_component)
            except ValueError as e:
                raise ValueError(f"Invalid noise combo: {e}")
                
        # Preserve canonical ordering from list_all_combos
        all_canonical_combos = list_all_combos()
        combos = [c for c in all_canonical_combos if c in combos_set]
        
        if mode not in ["single_level", "sweep"]:
            raise ValueError(f"Unknown mode: {mode}")
            
        if mode == "single_level":
            if level not in [1, 2, 3, 4, 5]:
                raise ValueError(f"level must be 1-5 for single_level, got {level}")
            levels_to_run = [level]
        else:
            levels_to_run = [1, 2, 3, 4, 5]
            
        if not isinstance(seed_count, int) or not (1 <= seed_count <= 10):
            raise ValueError(f"seed_count must be an integer between 1 and 10, got {seed_count}")
            
        datasets = [dataset]
        seeds = list(range(seed_count))

    fits_final = []
    seen = set()
    
    # Clean runs have n_noises=0 and use combo="clean"
    all_combos = [COMBO_CLEAN] + combos
    
    # Ordering: dataset -> seed -> combo -> level -> model
    for d in datasets:
        for s in seeds:
            for c in all_combos:
                current_levels = [0] if c == COMBO_CLEAN else levels_to_run
                for l in current_levels:
                    for m in models:
                        spec = FitSpec(dataset=d, model=m, combo=c, level=l, seed=s)
                        if spec not in seen:
                            fits_final.append(spec)
                            seen.add(spec)
                            
    return fits_final


_SPLIT_CACHE = {}


def _get_split_cached(dataset: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve train/test split from cache, computing it if necessary.
    Returns copies to ensure immutability of cached arrays.
    """
    key = (dataset, seed)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = split(dataset, seed)
    X_tr, X_te, y_tr, y_te = _SPLIT_CACHE[key]
    return X_tr.copy(), X_te.copy(), y_tr.copy(), y_te.copy()


def run_fit(spec: FitSpec, run_meta: dict) -> dict:
    """Execute exactly one fit specified by FitSpec.
    
    Args:
        spec: FitSpec object defining the run parameters.
        run_meta: Dictionary with run metadata (run_id, config_hash, etc.).
        
    Returns:
        A dictionary row mapping to RAW_COLUMNS in CSV storage.
    """
    row = {
        "run_id": run_meta.get("run_id"),
        "run_type": run_meta.get("run_type"),
        "stage": run_meta.get("stage"),
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": spec.seed,
        "combo": spec.combo,
        "n_noises": len(spec.combo.split("+")) if spec.combo != COMBO_CLEAN else 0,
        "level": spec.level,
        "macro_f1": float("nan"),
        "accuracy": float("nan"),
        "fit_time_s": float("nan"),
        "n_train": 0,
        "n_test": 0,
        "noise_stats": "{}",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_hash": run_meta.get("config_hash"),
        "methodology_version": run_meta.get("methodology_version"),
        "levels_version": run_meta.get("levels_version"),
        "levels_frozen": run_meta.get("levels_frozen"),
        "status": "success",
        "error_msg": "",
    }
    
    try:
        X_train, X_test, y_train, y_test = _get_split_cached(spec.dataset, spec.seed)
        row["n_train"] = X_train.shape[0]
        row["n_test"] = X_test.shape[0]
        
        if spec.level == 0 or spec.combo == COMBO_CLEAN:
            X_train_fit = X_train
            y_train_fit = y_train
            noise_stats = {}
        else:
            X_train_fit, y_train_fit, noise_stats = apply_noise(
                X_train, y_train, spec.combo, spec.level, spec.seed
            )
            
        row["noise_stats"] = noise_stats  # Will be serialized to JSON string in csv_store
        
        pipeline = build_pipeline(spec.model, spec.seed)
        
        start_time = time.time()
        pipeline.fit(X_train_fit, y_train_fit)
        end_time = time.time()
        
        row["fit_time_s"] = end_time - start_time
        
        y_pred = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        
        row["macro_f1"] = metrics["macro_f1"]
        row["accuracy"] = metrics["accuracy"]
        
    except Exception as e:
        row["status"] = "error"
        row["error_msg"] = str(e)
        row["macro_f1"] = float("nan")
        row["accuracy"] = float("nan")
        row["fit_time_s"] = float("nan")
        
    row["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return row
