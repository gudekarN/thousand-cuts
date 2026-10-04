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
from app.engine.noise import list_all_combos, canonical_combo


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
