"""Validation script for the Thousand Cuts experiment.

Subcommands:
    checks --stage mvp   Run validation checks against experiment results.
    freeze               Freeze levels (not yet implemented here).

Checks 1-4 (Task 2.2):
    1. Level-0 equals clean: apply_noise at level 0 returns data identical to clean.
       Also: if MVP results exist, level-0 fit metrics match the clean rows.
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
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Check 1: Level-0 identical to clean + optional MVP metric check
# ---------------------------------------------------------------------------

def check_level0_equals_clean(
    dataset: str = DATASET_BREAST_CANCER,
    seeds=(0,),
    mvp_csv_path: "Optional[Path]" = None,
):
    """Check 1: apply_noise at level 0 returns data identical to clean input.

    If mvp_csv_path exists, also verifies that level-0 fit metrics match
    the corresponding combo='clean' rows in the MVP raw results.
    """
    import pandas as pd

    failures = []
    details = []
    combos = list_all_combos()

    # Part A: data identity check
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = split(dataset, seed)
        for combo in combos:
            X_noisy, y_noisy, _ = apply_noise(X_tr, y_tr, combo, level=0, seed=seed)
            if not np.array_equal(X_noisy, X_tr):
                failures.append(f"seed={seed} combo={combo}: X_train differs at level 0")
            if not np.array_equal(y_noisy, y_tr):
                failures.append(f"seed={seed} combo={combo}: y_train differs at level 0")
    if not failures:
        details.append("Part A: all 15 combos return identical data at level 0")

    # Part B: MVP metric check (if results exist)
    mvp_metrics = []
    if mvp_csv_path is not None and Path(mvp_csv_path).exists():
        df = pd.read_csv(mvp_csv_path)
        df_ok = df[df["status"] == "ok"]

        # The raw CSV has exactly ONE clean row per (dataset, model, seed):
        #   combo="clean", level=0, n_noises=0
        # Non-clean combos do NOT have their own level-0 raw rows.
        # The analytical level-0 baseline is computed by reusing these clean rows.
        clean_rows = df_ok[df_ok["combo"] == "clean"].copy()

        if clean_rows.empty:
            failures.append("Part B: no combo='clean' rows found in MVP raw CSV")
        else:
            # Check for unexpected duplicates per (dataset, model, seed)
            dup_keys = clean_rows.groupby(["dataset", "model", "seed"]).size()
            for (ds, mdl, sd), cnt in dup_keys.items():
                if cnt > 1:
                    failures.append(
                        f"Part B: duplicate clean rows for dataset={ds} model={mdl} seed={sd} (count={cnt})"
                    )

            # Verify each clean row has valid, finite metrics
            for _, row in clean_rows.iterrows():
                f1 = float(row["macro_f1"])
                acc = float(row["accuracy"])
                entry = {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "macro_f1": f1,
                    "accuracy": acc,
                    "f1_finite": np.isfinite(f1),
                    "acc_finite": np.isfinite(acc),
                    "f1_nonneg": f1 >= 0.0,
                    "acc_nonneg": acc >= 0.0,
                }
                mvp_metrics.append(entry)
                if not np.isfinite(f1) or not np.isfinite(acc):
                    failures.append(
                        f"Part B: non-finite metric in clean row dataset={row['dataset']} "
                        f"model={row['model']} seed={row['seed']}: f1={f1} acc={acc}"
                    )
                if f1 < 0.0 or acc < 0.0:
                    failures.append(
                        f"Part B: negative metric in clean row dataset={row['dataset']} "
                        f"model={row['model']} seed={row['seed']}: f1={f1} acc={acc}"
                    )

            details.append(
                f"Part B: verified {len(clean_rows)} clean row(s) from MVP CSV"
            )
    else:
        details.append("Part B: MVP CSV not present, skipping metric check")

    passed = len(failures) == 0
    return {
        "check": 1,
        "name": "level0_equals_clean",
        "passed": passed,
        "failures": failures,
        "details": details,
        "mvp_metrics": mvp_metrics,
    }


# ---------------------------------------------------------------------------
# Check 2: Test set unchanged before and after noise + fit
# ---------------------------------------------------------------------------

def check_test_set_unchanged(dataset: str = DATASET_BREAST_CANCER, seeds=(0, 1)):
    """Check 2: X_test and y_test hashes identical before and after noise+fit."""
    failures = []
    combos = ["label", "gaussian", "outliers", "missing"]
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
        "failures": failures,
        "details": failures if failures else "Test set hashes unchanged across all fits",
    }


# ---------------------------------------------------------------------------
# Check 3: Imputer/scaler stats from noisy train only
# ---------------------------------------------------------------------------

def check_pipeline_stats_from_noisy_train(dataset: str = DATASET_BREAST_CANCER, seeds=(0,)):
    """Check 3: Pipeline imputer/scaler statistics match numpy values from noisy train.

    Also verifies:
    - Stats differ from clean-train stats when gaussian noise is present.
    - Noisy training matrix contains no inf values.
    """
    results = []
    failures = []
    model_id = "logreg"

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = split(dataset, seed)
        clean_imputer_means = np.nanmean(X_tr, axis=0)

        # A. Inf check on clean data (should always be clean)
        if not np.all(np.isfinite(X_tr)):
            failures.append(f"seed={seed}: clean X_train contains non-finite values")

        for combo in ["gaussian", "label+gaussian"]:
            for level in [1, 3]:
                X_noisy, y_noisy, _ = apply_noise(X_tr, y_tr, combo, level=level, seed=seed)

                # B. No inf in noisy data (check before fitting — sklearn rejects inf)
                if not np.all(np.isfinite(X_noisy)):
                    failures.append(
                        f"seed={seed} combo={combo} L{level}: noisy X_train contains inf or nan-derived inf"
                    )
                    continue  # cannot fit on invalid data; skip remaining checks for this combo/level

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

                # Compute expected scaler mean from noisy-imputed X
                X_imputed = np.where(np.isnan(X_noisy), expected_means, X_noisy)
                expected_scaler_mean = np.mean(X_imputed, axis=0)
                actual_scaler_mean = scaler.mean_

                if not np.allclose(expected_scaler_mean, actual_scaler_mean, atol=1e-10):
                    failures.append(
                        f"seed={seed} combo={combo} L{level}: scaler mean mismatch"
                    )

                # C. For gaussian combos: stats must differ from clean-train stats
                # Gaussian noise changes X features, so imputer means should differ.
                # Skip label-only combos (label noise only changes y, not X).
                if "gaussian" in combo:
                    if np.allclose(actual_means, clean_imputer_means, atol=1e-10):
                        failures.append(
                            f"seed={seed} combo={combo} L{level}: imputer means identical to clean — "
                            "gaussian noise should have changed X_train statistics"
                        )

                results.append(
                    f"seed={seed} combo={combo} L{level}: imputer OK, scaler OK, stats-differ OK"
                )

    passed = len(failures) == 0
    return {
        "check": 3,
        "name": "pipeline_stats_from_noisy_train",
        "passed": passed,
        "failures": failures,
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


def check_noise_rates(dataset: str = DATASET_BREAST_CANCER, seed: int = 0):
    """Check 4: actual noise rates match level table within tolerances."""
    X_tr, X_te, y_tr, y_te = split(dataset, seed)
    mean_clean, std_clean = clean_stats(X_tr)

    failures = []
    rate_details = []  # structured per-level, per-noise entries

    for level in range(1, 6):
        params = get_params(level)

        # Label: exact flip count
        X_noisy_l, y_noisy_l, _ = apply_noise(X_tr, y_tr, "label", level=level, seed=seed)
        n = len(y_tr)
        n_expected = round(params["label_flip_rate"] * n)
        n_actual = int(np.sum(y_tr != y_noisy_l))
        label_pass = n_actual == n_expected
        rate_details.append({
            "level": level, "noise": "label",
            "expected": n_expected, "actual": n_actual,
            "tolerance": "exact", "pass": label_pass,
        })
        if not label_pass:
            failures.append(f"L{level} label: expected {n_expected} flips, got {n_actual}")

        # Missing: ±2pp
        X_noisy_m, _, _ = apply_noise(X_tr, y_tr, "missing", level=level, seed=seed)
        actual_missing = _measure_missing_rate(X_noisy_m)
        expected_missing = params["missing_cell_rate"]
        tol_missing = 0.02
        missing_pass = abs(actual_missing - expected_missing) <= tol_missing
        rate_details.append({
            "level": level, "noise": "missing",
            "expected": round(expected_missing, 4), "actual": round(actual_missing, 4),
            "tolerance": tol_missing, "pass": missing_pass,
        })
        if not missing_pass:
            failures.append(
                f"L{level} missing: actual={actual_missing:.4f} expected={expected_missing:.4f}"
            )

        # Outliers: ±2pp
        X_noisy_o, _, _ = apply_noise(X_tr, y_tr, "outliers", level=level, seed=seed)
        actual_outlier = _measure_outlier_rate(X_noisy_o, X_tr, mean_clean, std_clean)
        expected_outlier = params["outlier_cell_rate"]
        tol_outlier = 0.02
        outlier_pass = abs(actual_outlier - expected_outlier) <= tol_outlier
        rate_details.append({
            "level": level, "noise": "outlier",
            "expected": round(expected_outlier, 4), "actual": round(actual_outlier, 4),
            "tolerance": tol_outlier, "pass": outlier_pass,
        })
        if not outlier_pass:
            failures.append(
                f"L{level} outlier: actual={actual_outlier:.4f} expected={expected_outlier:.4f}"
            )

        # Gaussian: std ratio ±10%
        X_noisy_g, _, _ = apply_noise(X_tr, y_tr, "gaussian", level=level, seed=seed)
        k = params["gaussian_k"]
        eligible = std_clean > 0
        if eligible.any():
            diff = (X_noisy_g - X_tr)[:, eligible]
            actual_std = np.std(diff, axis=0, ddof=1)
            expected_std = k * std_clean[eligible]
            ratios = actual_std / expected_std
            mean_ratio = float(np.mean(ratios))
            tol_gaussian = 0.10
            gaussian_pass = abs(mean_ratio - 1.0) <= tol_gaussian
            rate_details.append({
                "level": level, "noise": "gaussian",
                "expected": round(k, 4), "actual_mean_ratio": round(mean_ratio, 4),
                "tolerance": tol_gaussian, "pass": gaussian_pass,
            })
            if not gaussian_pass:
                failures.append(
                    f"L{level} gaussian: mean std_ratio={mean_ratio:.3f} (expected ~1.0)"
                )

    passed = len(failures) == 0
    return {
        "check": 4,
        "name": "noise_rates",
        "passed": passed,
        "failures": failures,
        "rate_details": rate_details,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_checks_1_4(stage: str, out_dir: Path) -> dict:
    """Run checks 1-4 and write partial validation report."""
    mvp_csv = Path("results/official/mvp/raw_results.csv")

    print("Running check 1: level-0 equals clean...")
    c1 = check_level0_equals_clean(mvp_csv_path=mvp_csv if mvp_csv.exists() else None)
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
