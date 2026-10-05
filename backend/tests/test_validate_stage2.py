"""Tests for Stage 2 validation checks (Task 4.2).

Covers S1-S7 with pass and fail cases on synthetic data where possible.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate import (
    _check_s2_digits_10_classes,
    _check_s3_digits_zero_std_cols_unchanged,
    _check_s4_s5_completeness,
    _check_s6_compound_order,
    _check_s7_synergy_components,
    _ALL_NOISY_COMBOS,
    _COMPOUND_COMBOS,
    _STAGE2_DATASETS,
    _STAGE2_MODELS,
    _STAGE2_SEEDS,
    _STAGE2_NOISY_LEVELS,
    _STAGE2_EXPECTED_OK,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic CSV generation
# ---------------------------------------------------------------------------

def _make_full_stage2_csv(tmp_path: Path) -> Path:
    """Write a minimal but complete Stage 2 raw_results.csv (1824 ok rows)."""
    rows = []
    # clean rows: 2 datasets x 4 models x 3 seeds = 24
    for dataset in _STAGE2_DATASETS:
        for model in _STAGE2_MODELS:
            for seed in _STAGE2_SEEDS:
                rows.append({
                    "dataset": dataset, "model": model, "seed": seed,
                    "combo": "clean", "level": 0,
                    "macro_f1": 0.90, "accuracy": 0.90,
                    "status": "ok", "error_msg": "",
                })
    # noisy rows: 2 x 4 x 3 x 15 x 5 = 1800
    for dataset in _STAGE2_DATASETS:
        for model in _STAGE2_MODELS:
            for seed in _STAGE2_SEEDS:
                for combo in _ALL_NOISY_COMBOS:
                    for level in _STAGE2_NOISY_LEVELS:
                        rows.append({
                            "dataset": dataset, "model": model, "seed": seed,
                            "combo": combo, "level": level,
                            "macro_f1": max(0.5, 0.90 - 0.05 * level),
                            "accuracy": max(0.5, 0.90 - 0.04 * level),
                            "status": "ok", "error_msg": "",
                        })
    assert len(rows) == _STAGE2_EXPECTED_OK
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# S2: Digits 10-class check
# ---------------------------------------------------------------------------

def test_s2_digits_10_classes_pass():
    """Digits dataset actually has 10 classes in y_test for all seeds."""
    result = _check_s2_digits_10_classes()
    assert result["passed"], result["failures"]


def test_s2_digits_10_classes_fail_synthetic(monkeypatch):
    """If y_test has <10 classes, check fails."""
    import scripts.validate as val_mod
    from app.engine.data import split as real_split

    def fake_split(dataset, seed):
        X_tr, X_te, y_tr, y_te = real_split(dataset, seed)
        # Collapse to 5 classes only for digits
        if dataset == "digits":
            y_te = y_te % 5
        return X_tr, X_te, y_tr, y_te

    monkeypatch.setattr("scripts.validate.split", fake_split)
    result = _check_s2_digits_10_classes()
    assert not result["passed"]
    assert any("classes" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# S3: Digits zero-std columns unchanged
# ---------------------------------------------------------------------------

def test_s3_digits_zero_std_unchanged_pass():
    """Zero-std pixel columns must not change with Gaussian/outlier noise."""
    result = _check_s3_digits_zero_std_cols_unchanged()
    assert result["passed"], result["failures"]


def test_s3_digits_zero_std_fail_synthetic(monkeypatch):
    """If noise modifies zero-std columns, check fails."""
    from app.engine.data import split as real_split
    from app.engine.noise import apply_noise as real_apply_noise
    import scripts.validate as val_mod

    def noisy_apply_noise(X_tr, y_tr, combo, level, seed):
        X_noisy, y_noisy, stats = real_apply_noise(X_tr, y_tr, combo, level=level, seed=seed)
        # Corrupt ALL columns to simulate bad noise implementation
        X_noisy_bad = X_noisy + 999.0
        return X_noisy_bad, y_noisy, stats

    monkeypatch.setattr("scripts.validate.apply_noise", noisy_apply_noise)
    result = _check_s3_digits_zero_std_cols_unchanged()
    # Only fails if Digits actually has zero-std columns; if none, it passes vacuously
    from scripts.validate import clean_stats
    from app.engine.data import split as real_split2
    X_tr, _, _, _ = real_split2("digits", 0)
    _, std_clean_arr = clean_stats(X_tr)
    zero_std_cols = np.where(std_clean_arr == 0)[0]
    if len(zero_std_cols) > 0:
        assert not result["passed"]
    else:
        assert result["passed"]  # vacuously passes when no zero-std columns


# ---------------------------------------------------------------------------
# S4 & S5: Completeness and zero errors
# ---------------------------------------------------------------------------

def test_s4_s5_pass_full_csv(tmp_path):
    """Complete 1824-row CSV: S4 and S5 both pass."""
    csv_path = _make_full_stage2_csv(tmp_path)
    s4, s5 = _check_s4_s5_completeness(csv_path)
    assert s4["passed"], s4["failures"]
    assert s5["passed"], s5["failures"]


def test_s4_fails_missing_row(tmp_path):
    """CSV with one missing (dataset/model/seed/combo/level) row: S4 fails."""
    csv_path = _make_full_stage2_csv(tmp_path)
    df = pd.read_csv(csv_path)
    # Remove one noisy row
    mask = ~((df["dataset"] == _STAGE2_DATASETS[0]) &
             (df["model"] == _STAGE2_MODELS[0]) &
             (df["seed"] == 0) &
             (df["combo"] == "label") &
             (df["level"] == 1))
    df[mask].to_csv(csv_path, index=False)
    s4, s5 = _check_s4_s5_completeness(csv_path)
    assert not s4["passed"]


def test_s4_fails_missing_clean_row(tmp_path):
    """CSV without a clean row for one (dataset, model, seed): S4 fails."""
    csv_path = _make_full_stage2_csv(tmp_path)
    df = pd.read_csv(csv_path)
    mask = ~((df["combo"] == "clean") &
             (df["dataset"] == _STAGE2_DATASETS[0]) &
             (df["model"] == _STAGE2_MODELS[0]) &
             (df["seed"] == 0))
    df[mask].to_csv(csv_path, index=False)
    s4, _ = _check_s4_s5_completeness(csv_path)
    assert not s4["passed"]


def test_s4_fails_wrong_total(tmp_path):
    """CSV with too few ok rows: S4 fails."""
    csv_path = _make_full_stage2_csv(tmp_path)
    df = pd.read_csv(csv_path)
    df.iloc[:100].to_csv(csv_path, index=False)
    s4, _ = _check_s4_s5_completeness(csv_path)
    assert not s4["passed"]


def test_s5_fails_error_rows(tmp_path):
    """CSV with error rows: S5 fails."""
    csv_path = _make_full_stage2_csv(tmp_path)
    df = pd.read_csv(csv_path)
    df = df.astype({"error_msg": str})
    df.loc[0, "status"] = "error"
    df.loc[0, "error_msg"] = "some error"
    df.to_csv(csv_path, index=False)
    _, s5 = _check_s4_s5_completeness(csv_path)
    assert not s5["passed"]


def test_s4_fails_no_csv(tmp_path):
    """If CSV doesn't exist, S4 and S5 both fail with clear message."""
    missing = tmp_path / "nonexistent.csv"
    s4, s5 = _check_s4_s5_completeness(missing)
    assert not s4["passed"]
    assert not s5["passed"]


# ---------------------------------------------------------------------------
# S6: Compound order
# ---------------------------------------------------------------------------

def test_s6_pass_schema_only(tmp_path):
    """All combos from list_all_combos() have correct canonical order."""
    no_csv = tmp_path / "nonexistent.csv"
    result = _check_s6_compound_order(no_csv)
    assert result["passed"], result["failures"]


def test_s6_fail_wrong_order(tmp_path):
    """A CSV with a wrongly-ordered compound combo fails S6."""
    rows = [{
        "dataset": "breast_cancer", "model": "logreg", "seed": 0,
        "combo": "gaussian+label",  # wrong order
        "level": 1, "macro_f1": 0.8, "accuracy": 0.8,
        "status": "ok", "error_msg": "",
    }]
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    result = _check_s6_compound_order(csv_path)
    assert not result["passed"]
    assert any("gaussian+label" in f for f in result["failures"])


def test_s6_pass_full_csv(tmp_path):
    """Full Stage 2 CSV with correct combos passes S6."""
    csv_path = _make_full_stage2_csv(tmp_path)
    result = _check_s6_compound_order(csv_path)
    assert result["passed"], result["failures"]


# ---------------------------------------------------------------------------
# S7: Synergy components
# ---------------------------------------------------------------------------

def test_s7_pass_schema_only(tmp_path):
    """All combos from list_all_combos() include all components."""
    no_csv = tmp_path / "nonexistent.csv"
    result = _check_s7_synergy_components(no_csv)
    assert result["passed"], result["failures"]


def test_s7_fail_missing_component(tmp_path):
    """CSV with compound combo but missing a single-noise component fails S7."""
    # Only include label+gaussian but NOT gaussian alone
    rows = [
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label", "level": 1, "macro_f1": 0.8, "accuracy": 0.8,
         "status": "ok", "error_msg": ""},
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label+gaussian", "level": 1, "macro_f1": 0.75, "accuracy": 0.75,
         "status": "ok", "error_msg": ""},
        # gaussian single-noise is intentionally MISSING
    ]
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    result = _check_s7_synergy_components(csv_path)
    assert not result["passed"]
    assert any("gaussian" in f for f in result["failures"])


def test_s7_pass_full_csv(tmp_path):
    """Full Stage 2 CSV with all 15 combos passes S7."""
    csv_path = _make_full_stage2_csv(tmp_path)
    result = _check_s7_synergy_components(csv_path)
    assert result["passed"], result["failures"]
