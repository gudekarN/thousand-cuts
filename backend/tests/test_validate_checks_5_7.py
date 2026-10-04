"""Tests for validate.py checks 5-7.

Each check is verified to PASS on correct synthetic data and FAIL when a bug is introduced.
Checks 6 and 7 use synthetic CSVs/data and temp dirs.
Check 5 uses two fresh runs to verify determinism (no real MVP results needed).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate import (
    check_reproducibility,
    check_performance_trend,
    check_runtime,
)


# ─── Check 5: Reproducibility ─────────────────────────────────────────────────

def test_check5_two_fresh_runs_identical():
    """Check 5 PASS: two fresh runs of the exact fixed subset give identical metrics."""
    result = check_reproducibility(mvp_csv_path=None)
    assert result["passed"], result["failures"]
    assert result["compare_source"] == "two_fresh_runs"
    # Confirm the detail message mentions the correct subset contents
    assert "clean L0" in result["details"][0]
    assert "noisy L3/L5" in result["details"][0]


def test_check5_subset_filters_correctly():
    """Check 5 PASS: _filter_repro_subset retains exactly clean L0 + noisy L3/L5 rows."""
    from scripts.validate import _filter_repro_subset
    import pandas as pd

    rows = [
        # Should be kept: clean level 0, seed 0, 4 models
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "clean", "level": 0, "status": "ok", "macro_f1": 0.95, "accuracy": 0.96},
        # Should be kept: label level 3
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label", "level": 3, "status": "ok", "macro_f1": 0.85, "accuracy": 0.86},
        # Should be kept: label+gaussian level 5
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label+gaussian", "level": 5, "status": "ok", "macro_f1": 0.75, "accuracy": 0.76},
        # Should be EXCLUDED: label level 1 (not in subset)
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label", "level": 1, "status": "ok", "macro_f1": 0.90, "accuracy": 0.91},
        # Should be EXCLUDED: different seed
        {"dataset": "breast_cancer", "model": "logreg", "seed": 1,
         "combo": "label", "level": 3, "status": "ok", "macro_f1": 0.82, "accuracy": 0.83},
        # Should be EXCLUDED: non-clean level 0 (does not exist in real data, but filtered out if present)
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label", "level": 0, "status": "ok", "macro_f1": 0.95, "accuracy": 0.96},
    ]
    df = pd.DataFrame(rows)
    filtered = _filter_repro_subset(df)

    assert len(filtered) == 3
    combos_kept = set(filtered["combo"].tolist())
    assert combos_kept == {"clean", "label", "label+gaussian"}
    levels_kept = set(filtered["level"].tolist())
    assert levels_kept == {0, 3, 5}
    # No non-clean level-0 rows
    assert not any((filtered["combo"] != "clean") & (filtered["level"] == 0))


def test_check5_fails_when_metrics_differ(monkeypatch):
    """Check 5 FAIL: even a tiny metric difference (strict ==) causes FAIL."""
    import app.engine.runner as runner_mod

    original_execute = runner_mod.execute_plan
    call_count = [0]

    def noisy_execute(plan, out_dir, run_meta, **kwargs):
        original_execute(plan, out_dir, run_meta, **kwargs)
        call_count[0] += 1
        # On the second call corrupt macro_f1 by the smallest representable float delta
        if call_count[0] == 2:
            csv = out_dir / "raw_results.csv"
            df = pd.read_csv(csv)
            # Add a tiny but non-zero delta that is detectable by strict ==
            import math
            df["macro_f1"] = df["macro_f1"].apply(
                lambda x: x + math.ulp(x)  # one ULP: smallest change for ==
            )
            df.to_csv(csv, index=False)

    monkeypatch.setattr(runner_mod, "execute_plan", noisy_execute)

    result = check_reproducibility(mvp_csv_path=None)
    assert not result["passed"], "Should FAIL when even a tiny metric difference exists"
    assert len(result["failures"]) > 0


# ─── Check 6: Performance trend ───────────────────────────────────────────────

def _make_trend_csv(tmp_path, f1_at_5_label=0.80, n_breaking=1):
    """Build a synthetic MVP CSV with a clear downward trend."""
    rows = []
    models = ["logreg", "svm_rbf", "decision_tree", "random_forest"]
    single_noises = ["label", "gaussian", "outliers", "missing"]
    seeds = [0, 1, 2]

    # Clean rows (baseline ~0.95)
    for model in models:
        for seed in seeds:
            rows.append({
                "dataset": "breast_cancer", "model": model, "seed": seed,
                "combo": "clean", "level": 0, "status": "ok",
                "macro_f1": 0.95, "accuracy": 0.96, "fit_time_s": 0.1,
            })

    # Noisy rows for each single noise, levels 1-5
    for noise in single_noises:
        for model in models:
            for seed in seeds:
                for level in range(1, 6):
                    # Gradual decrease; label at level 5 is configurable
                    if noise == "label" and level == 5:
                        f1 = f1_at_5_label
                    else:
                        f1 = max(0.50, 0.95 - level * 0.06)
                    rows.append({
                        "dataset": "breast_cancer", "model": model, "seed": seed,
                        "combo": noise, "level": level, "status": "ok",
                        "macro_f1": f1, "accuracy": f1 + 0.01, "fit_time_s": 0.1,
                    })

    # compound combo (label+gaussian) for completeness
    for model in models:
        for seed in seeds:
            for level in range(1, 6):
                rows.append({
                    "dataset": "breast_cancer", "model": model, "seed": seed,
                    "combo": "label+gaussian", "level": level, "status": "ok",
                    "macro_f1": max(0.40, 0.95 - level * 0.10), "accuracy": 0.5,
                    "fit_time_s": 0.1,
                })

    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_check6_passes_with_downward_trend(tmp_path):
    """Check 6 PASS: F1 at level 5 < level 0 for each single noise; at least 1 breaking pair."""
    csv_path = _make_trend_csv(tmp_path, f1_at_5_label=0.70)
    result = check_performance_trend(mvp_csv_path=csv_path)
    assert result["passed"], result["failures"]
    assert result["n_breaking_pairs"] >= 1
    assert len(result["trend_by_noise"]) == 4  # 4 single noises


def test_check6_fails_when_no_downward_trend(tmp_path):
    """Check 6 FAIL: if F1 at level 5 is equal to or above level 0."""
    rows = []
    models = ["logreg"]
    seeds = [0]

    # Clean rows (baseline 0.80)
    for model in models:
        for seed in seeds:
            rows.append({
                "dataset": "breast_cancer", "model": model, "seed": seed,
                "combo": "clean", "level": 0, "status": "ok",
                "macro_f1": 0.80, "accuracy": 0.81, "fit_time_s": 0.1,
            })

    # Label noise: F1 at level 5 is HIGHER than baseline (bug)
    for model in models:
        for seed in seeds:
            for level in range(1, 6):
                rows.append({
                    "dataset": "breast_cancer", "model": model, "seed": seed,
                    "combo": "label", "level": level, "status": "ok",
                    "macro_f1": 0.95,  # higher than clean — trend inverted
                    "accuracy": 0.95, "fit_time_s": 0.1,
                })

    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = check_performance_trend(mvp_csv_path=csv_path)
    assert not result["passed"], "Should FAIL when F1 does not decrease"
    assert any("NOT below level 0" in f for f in result["failures"])


def test_check6_fails_when_no_breaking_points(tmp_path):
    """Check 6 FAIL: trend is downward but no pair reaches the 90% threshold."""
    rows = []
    models = ["logreg", "svm_rbf", "decision_tree", "random_forest"]
    seeds = [0, 1, 2]

    for model in models:
        for seed in seeds:
            rows.append({
                "dataset": "breast_cancer", "model": model, "seed": seed,
                "combo": "clean", "level": 0, "status": "ok",
                "macro_f1": 0.95, "accuracy": 0.96, "fit_time_s": 0.1,
            })

    # Label noise: slight decrease but never below 90% threshold (0.855)
    for model in models:
        for seed in seeds:
            for level in range(1, 6):
                rows.append({
                    "dataset": "breast_cancer", "model": model, "seed": seed,
                    "combo": "label", "level": level, "status": "ok",
                    "macro_f1": 0.94 - (level * 0.005),  # stays above 0.90 * 0.95 = 0.855
                    "accuracy": 0.95, "fit_time_s": 0.1,
                })

    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = check_performance_trend(mvp_csv_path=csv_path)
    assert not result["passed"], "Should FAIL when no breaking point found"
    assert any("No (model, noise)" in f for f in result["failures"])


def test_check6_fails_when_csv_missing(tmp_path):
    """Check 6 FAIL: MVP CSV does not exist."""
    result = check_performance_trend(mvp_csv_path=tmp_path / "missing.csv")
    assert not result["passed"]
    assert any("not found" in f for f in result["failures"])


# ─── Check 7: Runtime ─────────────────────────────────────────────────────────

def test_check7_passes_no_mvp(tmp_path):
    """Check 7 PASS: no MVP CSV available; probe runs and estimate is recorded."""
    result = check_runtime(mvp_csv_path=None, probe_dataset="breast_cancer")
    assert result["passed"], result["failures"]
    assert result["mvp_total_fit_time_s"] is None
    assert result["probe_fits"] > 0
    assert result["full_run_estimate_s"] > 0
    assert result["full_run_estimate_min"] > 0
    # Verify estimate is recorded in details
    assert any("Full-run estimate" in d for d in result["details"])


def test_check7_passes_with_synthetic_mvp(tmp_path):
    """Check 7 PASS: MVP CSV exists; runtime is summed and reported."""
    rows = [
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "clean", "level": 0, "status": "ok",
         "macro_f1": 0.95, "accuracy": 0.96, "fit_time_s": 0.5},
        {"dataset": "breast_cancer", "model": "logreg", "seed": 0,
         "combo": "label", "level": 1, "status": "ok",
         "macro_f1": 0.90, "accuracy": 0.91, "fit_time_s": 0.6},
    ]
    csv_path = tmp_path / "raw_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = check_runtime(mvp_csv_path=csv_path, probe_dataset="breast_cancer")
    assert result["passed"]
    assert abs(result["mvp_total_fit_time_s"] - 1.1) < 0.01
    assert "logreg" in result["mean_fit_per_model_s"]
    assert result["full_run_estimate_s"] > 0


def test_check7_fails_when_probe_returns_zero_fits(monkeypatch):
    """Check 7 FAIL: timing probe produces no fit rows -> passed == False."""
    import app.engine.runner as runner_mod

    def empty_execute(plan, out_dir, run_meta, **kwargs):
        # Write an empty raw_results.csv with only headers but no data rows
        import csv as csv_mod
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "raw_results.csv", "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["dataset", "model", "combo", "level", "seed",
                             "macro_f1", "accuracy", "status", "fit_time_s"])
            # No data rows written

    monkeypatch.setattr(runner_mod, "execute_plan", empty_execute)

    result = check_runtime(mvp_csv_path=None, probe_dataset="breast_cancer")
    assert not result["passed"], "Should FAIL when timing probe returns 0 fits"
    assert result["probe_fits"] == 0
    assert any("0 fit results" in f for f in result["failures"])
