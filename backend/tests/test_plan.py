"""Tests for experiment plan builder."""

import pytest

from app.core.config import DATASET_BREAST_CANCER, DATASET_DIGITS
from app.engine.runner import build_plan

def test_build_plan_mvp_count():
    """MVP stage must produce exactly 312 fits with no duplicates."""
    plan = build_plan(stage="mvp")
    assert len(plan) == 312
    assert len(set(plan)) == 312

def test_build_plan_mvp_constraints():
    """MVP stage must only include breast_cancer and specific 5 combos."""
    plan = build_plan(stage="mvp")
    datasets = {spec.dataset for spec in plan}
    assert datasets == {DATASET_BREAST_CANCER}
    
    combos = {spec.combo for spec in plan}
    expected_combos = {"clean", "label", "gaussian", "outliers", "missing", "label+gaussian"}
    assert combos == expected_combos

def test_build_plan_stage2_count():
    """Stage 2 must produce exactly 1,824 fits with no duplicates."""
    plan = build_plan(stage="stage2")
    assert len(plan) == 1824
    assert len(set(plan)) == 1824

def test_build_plan_full_count():
    """Full stage must produce exactly 6,080 fits with no duplicates."""
    plan = build_plan(stage="full")
    assert len(plan) == 6080
    assert len(set(plan)) == 6080

def test_build_plan_custom_compound_includes_components():
    """Custom run with compound noise must automatically include its single components."""
    req = {
        "dataset": DATASET_BREAST_CANCER,
        "models": ["logreg"],
        "noises": ["label+gaussian"],
        "mode": "sweep",
        "seed_count": 1
    }
    plan = build_plan(custom_request=req)
    
    combos = {spec.combo for spec in plan}
    # Should include "clean" plus "label", "gaussian", and "label+gaussian"
    assert "clean" in combos
    assert "label" in combos
    assert "gaussian" in combos
    assert "label+gaussian" in combos

def test_build_plan_custom_single_level():
    """Custom single_level mode generates only clean and the chosen level."""
    req = {
        "dataset": DATASET_DIGITS,
        "models": ["logreg"],
        "noises": ["label"],
        "mode": "single_level",
        "level": 3,
        "seed_count": 1
    }
    plan = build_plan(custom_request=req)
    
    levels = {spec.level for spec in plan}
    assert levels == {0, 3}

def test_build_plan_validation_errors():
    """Various invalid inputs raise ValueError."""
    # Both stage and custom
    with pytest.raises(ValueError):
        build_plan(stage="mvp", custom_request={})
        
    # Neither stage nor custom
    with pytest.raises(ValueError):
        build_plan()
        
    # Unknown stage
    with pytest.raises(ValueError):
        build_plan(stage="unknown_stage")
        
    # Unknown dataset
    req = {"dataset": "unknown", "models": ["logreg"], "noises": ["label"], "mode": "sweep", "seed_count": 1}
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_plan(custom_request=req)
        
    # Unknown model
    req = {"dataset": DATASET_BREAST_CANCER, "models": ["unknown"], "noises": ["label"], "mode": "sweep", "seed_count": 1}
    with pytest.raises(ValueError, match="Unknown model"):
        build_plan(custom_request=req)
        
    # Empty models
    req = {"dataset": DATASET_BREAST_CANCER, "models": [], "noises": ["label"], "mode": "sweep", "seed_count": 1}
    with pytest.raises(ValueError, match="models must be a non-empty list"):
        build_plan(custom_request=req)
        
    # Unknown noise
    req = {"dataset": DATASET_BREAST_CANCER, "models": ["logreg"], "noises": ["unknown_noise"], "mode": "sweep", "seed_count": 1}
    with pytest.raises(ValueError, match="Invalid noise combo"):
        build_plan(custom_request=req)
        
    # Empty noises
    req = {"dataset": DATASET_BREAST_CANCER, "models": ["logreg"], "noises": [], "mode": "sweep", "seed_count": 1}
    with pytest.raises(ValueError, match="noises must be a non-empty list"):
        build_plan(custom_request=req)
        
    # Invalid mode
    req = {"dataset": DATASET_BREAST_CANCER, "models": ["logreg"], "noises": ["label"], "mode": "invalid", "seed_count": 1}
    with pytest.raises(ValueError, match="Unknown mode"):
        build_plan(custom_request=req)
        
    # Invalid level for single_level
    req = {"dataset": DATASET_BREAST_CANCER, "models": ["logreg"], "noises": ["label"], "mode": "single_level", "level": 6, "seed_count": 1}
    with pytest.raises(ValueError, match="level must be 1-5"):
        build_plan(custom_request=req)
        
    # Invalid seed count
    req = {"dataset": DATASET_BREAST_CANCER, "models": ["logreg"], "noises": ["label"], "mode": "sweep", "seed_count": 11}
    with pytest.raises(ValueError, match="seed_count must be an integer between 1 and 10"):
        build_plan(custom_request=req)
        
    # Missing level for single_level (None)
    req = {"dataset": DATASET_BREAST_CANCER, "models": ["logreg"], "noises": ["label"], "mode": "single_level", "seed_count": 1}
    with pytest.raises(ValueError, match="level must be 1-5"):
        build_plan(custom_request=req)
