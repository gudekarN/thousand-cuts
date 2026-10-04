"""Tests for single fit execution."""

import json
import pytest
import numpy as np

from app.core.config import DATASET_BREAST_CANCER, COMBO_CLEAN
from app.engine.runner import run_fit, FitSpec, _SPLIT_CACHE, _get_split_cached

@pytest.fixture(autouse=True)
def clear_split_cache():
    """Clear the split cache before each test to ensure isolation."""
    _SPLIT_CACHE.clear()
    yield
    _SPLIT_CACHE.clear()

def test_run_fit_clean_breast_cancer_logreg_f1():
    """Clean breast_cancer logreg should have Macro F1 above 0.9."""
    spec = FitSpec(
        dataset=DATASET_BREAST_CANCER,
        model="logreg",
        combo=COMBO_CLEAN,
        level=0,
        seed=42
    )
    run_meta = {"run_id": "test_run"}
    
    row = run_fit(spec, run_meta)
    
    assert row["status"] == "success"
    assert row["macro_f1"] > 0.9
    assert row["accuracy"] > 0.9
    assert row["n_train"] > 0
    assert row["n_test"] > 0
    assert row["noise_stats"] == {}

def test_run_fit_noisy_row_differs_from_clean():
    """A noisy row (level > 0) should yield different metrics than clean."""
    spec_clean = FitSpec(
        dataset=DATASET_BREAST_CANCER,
        model="logreg",
        combo=COMBO_CLEAN,
        level=0,
        seed=42
    )
    spec_noisy = FitSpec(
        dataset=DATASET_BREAST_CANCER,
        model="logreg",
        combo="label",
        level=5,
        seed=42
    )
    run_meta = {"run_id": "test_run"}
    
    row_clean = run_fit(spec_clean, run_meta)
    row_noisy = run_fit(spec_noisy, run_meta)
    
    assert row_noisy["status"] == "success"
    assert row_clean["macro_f1"] != row_noisy["macro_f1"]
    assert "label_flipped" in row_noisy["noise_stats"]

def test_run_fit_test_set_unmodified():
    """X_test and y_test must remain byte-identical after fit and predict."""
    spec = FitSpec(
        dataset=DATASET_BREAST_CANCER,
        model="logreg",
        combo=COMBO_CLEAN,
        level=0,
        seed=42
    )
    run_meta = {}
    
    # Pre-cache the split so we can hold a copy
    X_tr_init, X_te_init, y_tr_init, y_te_init = _get_split_cached(spec.dataset, spec.seed)
    
    X_te_bytes = X_te_init.tobytes()
    y_te_bytes = y_te_init.tobytes()
    
    run_fit(spec, run_meta)
    
    # Check if the cached array was modified
    X_tr_after, X_te_after, y_tr_after, y_te_after = _SPLIT_CACHE[(spec.dataset, spec.seed)]
    
    assert X_te_after.tobytes() == X_te_bytes
    assert y_te_after.tobytes() == y_te_bytes

def test_run_fit_determinism():
    """Same spec run twice gives identical metrics."""
    spec = FitSpec(
        dataset=DATASET_BREAST_CANCER,
        model="logreg",
        combo="gaussian",
        level=3,
        seed=42
    )
    run_meta = {}
    
    row1 = run_fit(spec, run_meta)
    
    # Clear cache to force full recompute
    _SPLIT_CACHE.clear()
    
    row2 = run_fit(spec, run_meta)
    
    assert row1["macro_f1"] == row2["macro_f1"]
    assert row1["noise_stats"] == row2["noise_stats"]

def test_run_fit_forced_failure(monkeypatch):
    """Exceptions are caught and returned as error row."""
    spec = FitSpec(
        dataset=DATASET_BREAST_CANCER,
        model="logreg",
        combo=COMBO_CLEAN,
        level=0,
        seed=42
    )
    run_meta = {"run_id": "test_error"}
    
    def mock_build_pipeline(*args, **kwargs):
        raise ValueError("Forced pipeline failure")
        
    import app.engine.runner
    monkeypatch.setattr(app.engine.runner, "build_pipeline", mock_build_pipeline)
    
    row = run_fit(spec, run_meta)
    
    assert row["status"] == "error"
    assert "Forced pipeline failure" in row["error_msg"]
    assert np.isnan(row["macro_f1"])
    assert np.isnan(row["fit_time_s"])
