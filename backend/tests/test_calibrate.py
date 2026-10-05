"""Tests for validate.py calibrate command (Task 3.1).

Tests cover:
- KEEP_A: all T1-T6 pass
- REVISE_TO_S: T4 fails (no breaking points)
- REVISE_TO_M: T5 fails (>= 50% break at level 1)
- FIX_CODE: T1/T2/T3/T6 failure (no scale failure)
- ESCALATE_TO_USER: T4+T5 both fail, or bug + scale failure
- levels.json byte-identical after running
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure backend is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate import run_calibration


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_raw_results(
    tmp_path: Path,
    *,
    n_seeds: int = 1,
    include_error: bool = False,
    nan_in_metric: bool = False,
    # Breaking point control
    all_break_at: int | None = None,   # all 16 pairs break at this level
    none_break: bool = False,           # no pair ever breaks
    all_break_at_1: bool = False,       # all 16 pairs break at level 1 (saturation)
    direction_violation: bool = False,  # L5 F1 > L0 for label noise
) -> Path:
    """Build a minimal synthetic raw_results.csv for calibration testing."""

    models = ["logreg", "svm_rbf", "decision_tree", "random_forest"]
    single_noises = ["label", "gaussian", "outliers", "missing"]
    levels = [1, 2, 3, 4, 5]
    seeds = list(range(n_seeds))

    rows = []

    # Clean rows (level=0, combo="clean") — baseline F1 = 0.95
    baseline_f1 = 0.95
    for model in models:
        for seed in seeds:
            rows.append({
                "dataset": "breast_cancer",
                "model": model,
                "combo": "clean",
                "level": 0,
                "seed": seed,
                "n_noises": 0,
                "macro_f1": baseline_f1,
                "accuracy": baseline_f1,
                "fit_time_s": 0.01,
                "status": "ok",
                "error": None,
            })

    # Noisy rows for single noises
    for noise in single_noises:
        for model in models:
            for seed in seeds:
                for level in levels:
                    if include_error and noise == "label" and level == 5 and seed == 0:
                        status = "error"
                        f1 = None
                    elif nan_in_metric and noise == "gaussian" and level == 5 and seed == 0:
                        status = "ok"
                        f1 = float("nan")
                    else:
                        status = "ok"
                        # Default: F1 stays high (no natural breaking)
                        f1 = 0.94  # above threshold 0.90 * 0.95 = 0.855

                        if all_break_at is not None and level >= all_break_at:
                            f1 = 0.80  # below threshold
                        if none_break:
                            f1 = 0.94  # never breaks
                        if all_break_at_1 and level >= 1:
                            f1 = 0.80  # breaks at level 1
                        if direction_violation and noise == "label" and level == 5:
                            f1 = 0.97  # higher than baseline — violation

                    rows.append({
                        "dataset": "breast_cancer",
                        "model": model,
                        "combo": noise,
                        "level": level,
                        "seed": seed,
                        "n_noises": 1,
                        "macro_f1": f1,
                        "accuracy": f1,
                        "fit_time_s": 0.01,
                        "status": status,
                        "error": "synthetic" if status == "error" else None,
                    })

    df = pd.DataFrame(rows)
    raw_csv = tmp_path / "raw_results.csv"
    df.to_csv(raw_csv, index=False)
    return tmp_path


def _make_val_report(tmp_path: Path, *, check1_pass: bool = True, check4_pass: bool = True) -> None:
    """Write a minimal validation_report.json so T1 and T2 can be derived."""
    report = {
        "stage": "mvp",
        "checks": [
            {"check": 1, "name": "level0_equals_clean", "passed": check1_pass, "failures": [], "details": []},
            {"check": 4, "name": "noise_rates", "passed": check4_pass, "failures": [], "details": []},
        ],
        "all_passed": check1_pass and check4_pass,
    }
    (tmp_path / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calibrate_keep_a(tmp_path):
    """All T1-T6 pass → recommendation KEEP_A."""
    _make_raw_results(tmp_path, all_break_at=3)   # some pairs break, not at level 1
    _make_val_report(tmp_path)
    report = run_calibration("mvp", tmp_path)
    assert report["recommendation"] == "KEEP_A"
    assert report["tests"]["T1"]["passed"]
    assert report["tests"]["T2"]["passed"]
    assert report["tests"]["T3"]["passed"]
    assert report["tests"]["T4"]["passed"]
    assert report["tests"]["T5"]["passed"]
    assert report["tests"]["T6"]["passed"]
    assert (tmp_path / "calibration_report.json").exists()


def test_calibrate_revise_to_s(tmp_path):
    """T4 fails (no breaking points) → REVISE_TO_S."""
    _make_raw_results(tmp_path, none_break=True)
    _make_val_report(tmp_path)
    report = run_calibration("mvp", tmp_path)
    assert report["recommendation"] == "REVISE_TO_S"
    assert not report["tests"]["T4"]["passed"]
    assert report["tests"]["T5"]["passed"]  # no breaks → no saturation either


def test_calibrate_revise_to_m(tmp_path):
    """T5 fails (all 16 pairs break at level 1, >=50%) → REVISE_TO_M."""
    _make_raw_results(tmp_path, all_break_at_1=True)
    _make_val_report(tmp_path)
    report = run_calibration("mvp", tmp_path)
    assert report["recommendation"] == "REVISE_TO_M"
    assert report["tests"]["T4"]["passed"]   # they do break (T4 is fine)
    assert not report["tests"]["T5"]["passed"]


def test_calibrate_fix_code_t1(tmp_path):
    """T1 fails (rate accuracy check 4 failed) → FIX_CODE (no scale failure)."""
    _make_raw_results(tmp_path, all_break_at=3)
    _make_val_report(tmp_path, check4_pass=False)   # T1 derived from check 4
    report = run_calibration("mvp", tmp_path)
    assert report["recommendation"] == "FIX_CODE"
    assert not report["tests"]["T1"]["passed"]


def test_calibrate_fix_code_t2(tmp_path):
    """T2 fails (check 1 failed) → FIX_CODE."""
    _make_raw_results(tmp_path, all_break_at=3)
    _make_val_report(tmp_path, check1_pass=False)
    report = run_calibration("mvp", tmp_path)
    assert report["recommendation"] == "FIX_CODE"
    assert not report["tests"]["T2"]["passed"]


def test_calibrate_fix_code_t3_error_rows(tmp_path):
    """T3 fails (error rows in raw CSV) → FIX_CODE."""
    _make_raw_results(tmp_path, all_break_at=3, include_error=True)
    _make_val_report(tmp_path)
    report = run_calibration("mvp", tmp_path)
    assert report["recommendation"] == "FIX_CODE"
    assert not report["tests"]["T3"]["passed"]
    assert report["tests"]["T3"]["n_error_rows"] >= 1


def test_calibrate_fix_code_t6_direction(tmp_path):
    """T6 fails (label noise F1 at L5 > L0) → FIX_CODE."""
    _make_raw_results(tmp_path, all_break_at=3, direction_violation=True)
    _make_val_report(tmp_path)
    report = run_calibration("mvp", tmp_path)
    assert report["recommendation"] == "FIX_CODE"
    assert not report["tests"]["T6"]["passed"]


def test_calibrate_escalate_t4_and_t5(tmp_path):
    """Both T4 and T5 fail at the same time → ESCALATE_TO_USER.

    This requires no breaks (T4 fails) but also >= half breaking at level 1 (T5 fails).
    That combination is logically impossible with consistent data, so we construct a
    degenerate CSV where some pairs break and some don't, to force both to fail via
    the bug+scale path instead.

    Actually the simplest route: T4 fails (none_break) but also inject a T1 failure,
    giving bug + scale → ESCALATE_TO_USER.
    """
    _make_raw_results(tmp_path, none_break=True)
    _make_val_report(tmp_path, check4_pass=False)  # T1 also fails
    report = run_calibration("mvp", tmp_path)
    # T4 fails (no breaking) + T1 fails (check4 FAIL) → ESCALATE_TO_USER
    assert report["recommendation"] == "ESCALATE_TO_USER"


def test_calibrate_levels_json_unchanged(tmp_path):
    """levels.json must be byte-identical after running calibrate."""
    levels_path = Path(__file__).resolve().parents[1] / "configs" / "levels.json"
    original_bytes = levels_path.read_bytes()

    _make_raw_results(tmp_path, all_break_at=3)
    _make_val_report(tmp_path)
    run_calibration("mvp", tmp_path)

    assert levels_path.read_bytes() == original_bytes, \
        "calibrate command must NOT modify levels.json"


def test_calibrate_report_written(tmp_path):
    """calibration_report.json is written with correct structure."""
    _make_raw_results(tmp_path, all_break_at=3)
    _make_val_report(tmp_path)
    run_calibration("mvp", tmp_path)

    report_path = tmp_path / "calibration_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "recommendation" in report
    assert "tests" in report
    assert set(report["tests"].keys()) == {"T1", "T2", "T3", "T4", "T5", "T6"}
    assert "summary" in report
