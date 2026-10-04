"""Validation script for the Thousand Cuts experiment.

Subcommands:
    checks --stage mvp   Run validation checks against experiment results.
    freeze               Freeze levels (not yet implemented here).

Checks 1-4 (Task 2.2):
    1. Level-0 equals clean: apply_noise at level 0 returns data identical to clean.
    2. Test set unchanged: hash X_test and y_test before and after noise + fit.
    3. Imputer/scaler statistics come from noisy train only.
    4. Noise rates match the active level table within tolerances.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Ensure backend dir is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.data import split
from app.engine.noise import apply_noise, list_all_combos, clean_stats
from app.engine.pipelines import build_pipeline
from app.core.config import DATASET_BREAST_CANCER, NOISE_LABEL, NOISE_GAUSSIAN, NOISE_OUTLIERS, NOISE_MISSING
from app.core.levels import get_params, load_levels
from app.storage.json_store import write_json_atomic


CHECKS_1_4 = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _safe_fit_pipeline(model_id: str, X_train: np.ndarray, y_train: np.ndarray, seed: int = 0):
    """Fit and return pipeline; returns None on failure."""
    try:
        pipe = build_pipeline(model_id, seed=seed)
        pipe.fit(X_train, y_train)
        return pipe
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Check 1: Level-0 identical to clean
# ---------------------------------------------------------------------------

def check_level0_equals_clean(dataset: str = DATASET_BREAST_CANCER, seeds=(0,)):
    """Check 1: apply_noise at level 0 returns data identical to clean input."""
    failures = []
    combos = list_all_combos()
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = split(dataset, seed)
        for combo in combos:
            X_noisy, y_noisy, _ = apply_noise(X_tr, y_tr, combo, level=0, seed=seed)
            if not np.array_equal(X_noisy, X_tr):
                failures.append(f"seed={seed} combo={combo}: X_train differs at level 0")
            if not np.array_equal(y_noisy, y_tr):
                failures.append(f"seed={seed} combo={combo}: y_train differs at level 0")
    passed = len(failures) == 0
    return {
        "check": 1,
        "name": "level0_equals_clean",
        "passed": passed,
        "details": failures if failures else "All combos return identical data at level 0",
    }


# ---------------------------------------------------------------------------
# Check 2: Test set unchanged before and after noise + fit
# ---------------------------------------------------------------------------

def check_test_set_unchanged(dataset: str = DATASET_BREAST_CANCER, seeds=(0, 1)):
    """Check 2: X_test and y_test hashes identical before and after noise+fit."""
    failures = []
    combos = ["label", "gaussian"]  # representative subset
    model_id = "logreg"

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = split(dataset, seed)
        hash_X_te_before = _array_hash(X_te)
        hash_y_te_before = _array_hash(y_te.astype(np.int64))

        for combo in combos:
            for level in [1, 3, 5]:
                X_noisy, y_noisy, _ = apply_noise(X_tr, y_tr, combo, level=level, seed=seed)
                pipe = _safe_fit_pipeline(model_id, X_noisy, y_noisy, seed=seed)
                if pipe is not None:
                    _ = pipe.predict(X_te)

                hash_X_te_after = _array_hash(X_te)
                hash_y_te_after = _array_hash(y_te.astype(np.int64))

                if hash_X_te_before != hash_X_te_after:
                    failures.append(f"seed={seed} combo={combo} L{level}: X_test was modified")
                if hash_y_te_before != hash_y_te_after:
                    failures.append(f"seed={seed} combo={combo} L{level}: y_test was modified")

    passed = len(failures) == 0
    return {
        "check": 2,
        "name": "test_set_unchanged",
        "passed": passed,
        "details": failures if failures else "Test set hashes unchanged across all fits",
    }


# ---------------------------------------------------------------------------
# Check 3: Imputer/scaler stats from noisy train only
# ---------------------------------------------------------------------------

def check_pipeline_stats_from_noisy_train(dataset: str = DATASET_BREAST_CANCER, seeds=(0,)):
    """Check 3: Pipeline imputer/scaler statistics match numpy values from noisy train."""
    results = []
    failures = []
    model_id = "logreg"

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = split(dataset, seed)
        # Use label noise (only label noise changes y, not X features, but still verify)
        # Use gaussian noise which does change X
        for combo in ["gaussian", "label+gaussian"]:
            for level in [1, 3]:
                X_noisy, y_noisy, _ = apply_noise(X_tr, y_tr, combo, level=level, seed=seed)

                pipe = build_pipeline(model_id, seed=seed)
                pipe.fit(X_noisy, y_noisy)

                imputer = pipe.named_steps["imputer"]
                scaler = pipe.named_steps["scaler"]

                # Compute expected imputer mean from noisy X
                expected_means = np.nanmean(X_noisy, axis=0)
                actual_means = imputer.statistics_

                if not np.allclose(expected_means, actual_means, atol=1e-10):
                    failures.append(
                        f"seed={seed} combo={combo} L{level}: imputer means mismatch"
                    )

                # Compute expected scaler mean/std from imputed X
                X_imputed = np.where(np.isnan(X_noisy), expected_means, X_noisy)
                expected_scaler_mean = np.mean(X_imputed, axis=0)
                actual_scaler_mean = scaler.mean_

                if not np.allclose(expected_scaler_mean, actual_scaler_mean, atol=1e-10):
                    failures.append(
                        f"seed={seed} combo={combo} L{level}: scaler mean mismatch"
                    )

                results.append(
                    f"seed={seed} combo={combo} L{level}: imputer OK, scaler OK"
                )

    passed = len(failures) == 0
    return {
        "check": 3,
        "name": "pipeline_stats_from_noisy_train",
        "passed": passed,
        "details": failures if failures else results,
    }


# ---------------------------------------------------------------------------
# Check 4: Noise rates match active level table
# ---------------------------------------------------------------------------

def _measure_label_rate(y_orig, y_noisy):
    return float(np.mean(y_orig != y_noisy))


def _measure_missing_rate(X_noisy):
    return float(np.mean(np.isnan(X_noisy)))


def _measure_outlier_rate(X_noisy, X_orig, mean_clean, std_clean):
    """Count cells set to approx mean ± 5*std."""
    eligible = std_clean > 0
    if not eligible.any():
        return 0.0
    hi = mean_clean + 5 * std_clean
    lo = mean_clean - 5 * std_clean
    X_e = X_noisy[:, eligible]
    X_o = X_orig[:, eligible]
    hi_e = hi[eligible]
    lo_e = lo[eligible]
    changed = X_e != X_o
    at_hi = np.abs(X_e - hi_e) < 1e-10
    at_lo = np.abs(X_e - lo_e) < 1e-10
    outlier_cells = changed & (at_hi | at_lo)
    return float(outlier_cells.sum()) / X_noisy.size


def _measure_gaussian_std_ratio(X_noisy, X_orig, std_clean):
    """Ratio of per-column noise std to k*std_clean (should be ~1.0)."""
    eligible = std_clean > 0
    if not eligible.any():
        return []
    diff = (X_noisy - X_orig)[:, eligible]
    actual_std = np.std(diff, axis=0, ddof=1)
    expected_std = std_clean[eligible]  # k*std_clean, but we compare ratio
    ratios = actual_std / expected_std
    return ratios.tolist()


def check_noise_rates(dataset: str = DATASET_BREAST_CANCER, seed: int = 0):
    """Check 4: actual noise rates match level table within tolerances."""
    X_tr, X_te, y_tr, y_te = split(dataset, seed)
    mean_clean, std_clean = clean_stats(X_tr)
    levels_data = load_levels()

    failures = []
    details = []

    for level in range(1, 6):
        params = get_params(level)

        # Label
        X_noisy_l, y_noisy_l, _ = apply_noise(X_tr, y_tr, "label", level=level, seed=seed)
        actual_label = _measure_label_rate(y_tr, y_noisy_l)
        expected_label = params["label_flip_rate"]
        # label must be exact (deterministic flip count via round())
        n = len(y_tr)
        n_expected = round(expected_label * n)
        n_actual = int(np.sum(y_tr != y_noisy_l))
        if n_actual != n_expected:
            failures.append(
                f"L{level} label: expected {n_expected} flips, got {n_actual}"
            )
        details.append(f"L{level} label: {n_actual}/{n} flips (expected {n_expected})")

        # Missing
        X_noisy_m, _, _ = apply_noise(X_tr, y_tr, "missing", level=level, seed=seed)
        actual_missing = _measure_missing_rate(X_noisy_m)
        expected_missing = params["missing_cell_rate"]
        tol_missing = 0.02
        if abs(actual_missing - expected_missing) > tol_missing:
            failures.append(
                f"L{level} missing: actual={actual_missing:.4f} expected={expected_missing:.4f} (tol={tol_missing})"
            )
        details.append(f"L{level} missing: actual={actual_missing:.4f} expected={expected_missing:.4f}")

        # Outliers
        X_noisy_o, _, _ = apply_noise(X_tr, y_tr, "outliers", level=level, seed=seed)
        actual_outlier = _measure_outlier_rate(X_noisy_o, X_tr, mean_clean, std_clean)
        expected_outlier = params["outlier_cell_rate"]
        tol_outlier = 0.02
        if abs(actual_outlier - expected_outlier) > tol_outlier:
            failures.append(
                f"L{level} outlier: actual={actual_outlier:.4f} expected={expected_outlier:.4f} (tol={tol_outlier})"
            )
        details.append(f"L{level} outlier: actual={actual_outlier:.4f} expected={expected_outlier:.4f}")

        # Gaussian: measure std ratio per eligible column
        X_noisy_g, _, _ = apply_noise(X_tr, y_tr, "gaussian", level=level, seed=seed)
        k = params["gaussian_k"]
        # diff std / (k * std_clean) should be ~1.0 per column
        eligible = std_clean > 0
        if eligible.any():
            diff = (X_noisy_g - X_tr)[:, eligible]
            actual_std = np.std(diff, axis=0, ddof=1)
            expected_std = k * std_clean[eligible]
            ratios = actual_std / expected_std
            mean_ratio = float(np.mean(ratios))
            tol_gaussian = 0.10
            if abs(mean_ratio - 1.0) > tol_gaussian:
                failures.append(
                    f"L{level} gaussian: mean std_ratio={mean_ratio:.3f} (expected ~1.0, tol={tol_gaussian})"
                )
            details.append(f"L{level} gaussian: mean_std_ratio={mean_ratio:.4f}")

    passed = len(failures) == 0
    return {
        "check": 4,
        "name": "noise_rates",
        "passed": passed,
        "details": failures if failures else details,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_checks_1_4(stage: str, out_dir: Path) -> dict:
    """Run checks 1-4 and write partial validation report."""
    print("Running check 1: level-0 equals clean...")
    c1 = check_level0_equals_clean()
    print(f"  Check 1: {'PASS' if c1['passed'] else 'FAIL'}")

    print("Running check 2: test set unchanged...")
    c2 = check_test_set_unchanged()
    print(f"  Check 2: {'PASS' if c2['passed'] else 'FAIL'}")

    print("Running check 3: pipeline stats from noisy train...")
    c3 = check_pipeline_stats_from_noisy_train()
    print(f"  Check 3: {'PASS' if c3['passed'] else 'FAIL'}")

    print("Running check 4: noise rates...")
    c4 = check_noise_rates()
    print(f"  Check 4: {'PASS' if c4['passed'] else 'FAIL'}")

    report = {
        "stage": stage,
        "checks_1_4": [c1, c2, c3, c4],
        "all_passed": all(c["passed"] for c in [c1, c2, c3, c4]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "validation_report.json"
    write_json_atomic(report_path, report)
    print(f"\nReport written to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Validation script for Thousand Cuts.")
    subparsers = parser.add_subparsers(dest="command")

    checks_parser = subparsers.add_parser("checks", help="Run validation checks")
    checks_parser.add_argument("--stage", required=True, choices=["mvp", "stage2", "full"])
    checks_parser.add_argument("--out", type=str, default=None,
                               help="Output directory (default: results/official/<stage>)")

    args = parser.parse_args()

    if args.command == "checks":
        if args.out:
            out_dir = Path(args.out)
        else:
            out_dir = Path("results/official") / args.stage

        report = run_checks_1_4(args.stage, out_dir)

        print("\nSummary:")
        for c in report["checks_1_4"]:
            status = "PASS" if c["passed"] else "FAIL"
            print(f"  Check {c['check']} ({c['name']}): {status}")

        if not report["all_passed"]:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
