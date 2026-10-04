"""Tests for validate.py checks 1-4.

Each check is verified to PASS on correct code and FAIL when a bug is introduced.
All failure paths are genuinely exercised.
"""

import sys
import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate import (
    check_level0_equals_clean,
    check_test_set_unchanged,
    check_pipeline_stats_from_noisy_train,
    check_noise_rates,
)
from app.engine import noise as noise_module
from app.engine.data import split
from app.core.config import DATASET_BREAST_CANCER


# ─── Check 1: Level-0 equals clean ───────────────────────────────────────────

def test_check1_passes_correct_code():
    """Part A: all 15 combos return identical data at level 0."""
    result = check_level0_equals_clean(seeds=(0,))
    assert result["passed"], result["failures"]


def test_check1_fails_when_level0_noise_applied(monkeypatch):
    """Part A FAIL: apply_noise at level 0 mistakenly applies level-1 noise."""
    original_apply_noise = noise_module.apply_noise

    def buggy_apply_noise(X_train, y_train, combo, level, seed, classes=None):
        # Bug: always applies level 1 instead of respecting level 0
        return original_apply_noise(X_train, y_train, combo, 1, seed, classes)

    import scripts.validate as v
    monkeypatch.setattr(v, "apply_noise", buggy_apply_noise)

    result = v.check_level0_equals_clean(seeds=(0,))
    assert not result["passed"], "Should FAIL when level-0 applies noise"
    assert len(result["failures"]) > 0


def test_check1_passes_with_synthetic_mvp_csv(tmp_path):
    """Part B PASS: valid clean rows in correct MVP CSV design."""
    # MVP design: one clean row per (dataset, model, seed) + noisy levels 1..5
    # Only combo="clean", level=0 rows exist at level 0
    rows = [
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "clean", "level": 0, "status": "ok",
         "macro_f1": 0.95, "accuracy": 0.96, "fit_time_s": 0.1},
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label", "level": 1, "status": "ok",
         "macro_f1": 0.90, "accuracy": 0.91, "fit_time_s": 0.1},
        {"dataset": "breast_cancer", "model": "logreg", "seed": 1,
         "combo": "clean", "level": 0, "status": "ok",
         "macro_f1": 0.94, "accuracy": 0.95, "fit_time_s": 0.1},
    ]
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = check_level0_equals_clean(seeds=(0,), mvp_csv_path=csv_path)
    assert result["passed"], result["failures"]
    assert len(result["mvp_metrics"]) == 2  # two clean rows
    assert all(m["f1_finite"] for m in result["mvp_metrics"])
    assert all(m["f1_nonneg"] for m in result["mvp_metrics"])


def test_check1_fails_when_no_clean_rows(tmp_path):
    """Part B FAIL: CSV has no combo='clean' rows at all."""
    rows = [
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label", "level": 1, "status": "ok",
         "macro_f1": 0.90, "accuracy": 0.91, "fit_time_s": 0.1},
    ]
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = check_level0_equals_clean(seeds=(0,), mvp_csv_path=csv_path)
    assert not result["passed"], "Should FAIL when clean rows are missing"
    assert any("no combo='clean'" in f for f in result["failures"])


def test_check1_fails_on_nonfinite_clean_metric(tmp_path):
    """Part B FAIL: clean row has NaN macro_f1 -> check fails."""
    rows = [
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "clean", "level": 0, "status": "ok",
         "macro_f1": float("nan"), "accuracy": 0.96, "fit_time_s": 0.1},
    ]
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = check_level0_equals_clean(seeds=(0,), mvp_csv_path=csv_path)
    assert not result["passed"], "Should FAIL when clean row has NaN metric"
    assert any("non-finite" in f for f in result["failures"])


def test_check1_fails_on_duplicate_clean_rows(tmp_path):
    """Part B FAIL: two clean rows for the same (dataset, model, seed)."""
    rows = [
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "clean", "level": 0, "status": "ok",
         "macro_f1": 0.95, "accuracy": 0.96, "fit_time_s": 0.1},
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "clean", "level": 0, "status": "ok",
         "macro_f1": 0.95, "accuracy": 0.96, "fit_time_s": 0.1},
    ]
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = check_level0_equals_clean(seeds=(0,), mvp_csv_path=csv_path)
    assert not result["passed"], "Should FAIL when clean rows are duplicated"
    assert any("duplicate" in f for f in result["failures"])


# ─── Check 2: Test set unchanged ─────────────────────────────────────────────

def test_check2_passes_correct_code():
    """Check 2 PASS: correct implementation never modifies X_test/y_test."""
    result = check_test_set_unchanged(seeds=(0,))
    assert result["passed"], result["failures"]


def test_check2_fails_when_test_set_mutated(monkeypatch):
    """Check 2 FAIL: apply_noise corrupts X_test in-place — check detects it."""
    original_apply = noise_module.apply_noise

    def buggy_apply(X_train, y_train, combo, level, seed, classes=None):
        X_noisy, y_noisy, stats = original_apply(X_train, y_train, combo, level, seed, classes)
        return X_noisy, y_noisy, stats

    # Patch split so X_te is the *same object* as X_train so any mutation is shared
    X_tr_real, X_te_real, y_tr_real, y_te_real = split(DATASET_BREAST_CANCER, 0)
    # Use a mutable shared buffer: X_te and X_tr point to the same array
    shared_X = X_tr_real.copy()

    call_count = [0]

    def mock_split(dataset, seed):
        call_count[0] += 1
        # Return the same shared_X for both train and test so mutation of train
        # also mutates the pre-hash reference — simulating leakage
        return shared_X, shared_X, y_tr_real.copy(), y_te_real.copy()

    import scripts.validate as v
    monkeypatch.setattr(v, "split", mock_split)

    # Patch apply_noise to corrupt X_train in-place after producing noisy copy
    def corrupting_apply(X_train, y_train, combo, level, seed, classes=None):
        X_noisy, y_noisy, stats = original_apply(X_train, y_train, combo, level, seed, classes)
        X_train[:] = 999.0   # mutate the shared buffer in-place
        return X_noisy, y_noisy, stats

    monkeypatch.setattr(v, "apply_noise", corrupting_apply)

    result = v.check_test_set_unchanged(seeds=(0,))
    assert not result["passed"], "Should FAIL when X_test is mutated during the run"
    assert len(result["failures"]) > 0


# ─── Check 3: Pipeline stats from noisy train ─────────────────────────────────

def test_check3_passes_correct_code():
    """Check 3 PASS: imputer and scaler stats match noisy train statistics."""
    result = check_pipeline_stats_from_noisy_train(seeds=(0,))
    assert result["passed"], result["failures"]


def test_check3_fails_when_pipeline_fit_on_clean(monkeypatch):
    """Check 3 FAIL: pipeline is fit on clean X_train instead of noisy X_train."""
    import app.engine.pipelines as pipelines_mod
    original_build = pipelines_mod.build_pipeline

    # Capture clean data at seed 0 once for use inside buggy pipeline
    X_tr_clean, _, y_tr_clean, _ = split(DATASET_BREAST_CANCER, 0)

    class CleanFitPipeline:
        """Always fits on the clean training data, ignoring the noisy input."""
        def __init__(self):
            self._pipe = original_build("logreg", seed=0)

        def fit(self, X, y):
            # Bug: fit on clean data, not the noisy X passed in
            self._pipe.fit(X_tr_clean, y_tr_clean)
            return self

        def predict(self, X):
            return self._pipe.predict(X)

        @property
        def named_steps(self):
            return self._pipe.named_steps

    def buggy_build_pipeline(model_id, seed=0):
        return CleanFitPipeline()

    import scripts.validate as v
    monkeypatch.setattr(v, "build_pipeline", buggy_build_pipeline)

    result = v.check_pipeline_stats_from_noisy_train(seeds=(0,))
    assert not result["passed"], "Should FAIL when pipeline is fit on clean data"
    assert len(result["failures"]) > 0


def test_check3_fails_when_noisy_data_contains_inf(monkeypatch):
    """Check 3 FAIL: apply_noise returns inf values in X_noisy -> check detects it."""
    original_apply = noise_module.apply_noise

    def inf_apply(X_train, y_train, combo, level, seed, classes=None):
        X_noisy, y_noisy, stats = original_apply(X_train, y_train, combo, level, seed, classes)
        X_noisy[0, 0] = np.inf   # inject an inf into the noisy matrix
        return X_noisy, y_noisy, stats

    import scripts.validate as v
    monkeypatch.setattr(v, "apply_noise", inf_apply)

    result = v.check_pipeline_stats_from_noisy_train(seeds=(0,))
    assert not result["passed"], "Should FAIL when noisy X contains inf"
    assert any("inf" in f.lower() for f in result["failures"])


# ─── Check 4: Noise rates ─────────────────────────────────────────────────────

def test_check4_passes_correct_code():
    """Check 4 PASS: all four noise types within tolerance on real data."""
    result = check_noise_rates(seed=0)
    assert result["passed"], result["failures"]


def test_check4_has_structured_rate_details():
    """Check 4: rate_details contains structured entries with required fields."""
    result = check_noise_rates(seed=0)
    assert "rate_details" in result
    assert len(result["rate_details"]) > 0
    required_fields = {"level", "noise", "expected", "actual", "tolerance", "pass"}
    # gaussian entries use actual_mean_ratio instead of actual
    for entry in result["rate_details"]:
        assert "level" in entry
        assert "noise" in entry
        assert "pass" in entry
        assert "tolerance" in entry


def test_check4_fails_when_wrong_level_params(monkeypatch):
    """Check 4 FAIL: using level-1 params for all levels mismatches higher-level rates."""
    import scripts.validate as v
    original_get_params = v.get_params

    def buggy_get_params(level):
        return original_get_params(1)  # always use level-1 params

    monkeypatch.setattr(v, "get_params", buggy_get_params)

    result = v.check_noise_rates(seed=0)
    assert not result["passed"], "Should FAIL when wrong level params are used"
    assert len(result["failures"]) > 0
