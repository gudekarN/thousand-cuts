"""Validation script for the Thousand Cuts experiment.

Subcommands:
    checks   --stage mvp   Run the 7 validation checks.
    calibrate --stage mvp  Evaluate T1-T6 calibration tests and write report.
    freeze                 Freeze levels (Task 3.2).

Checks 1-4 (Task 2.2):
    1. Level-0 equals clean: apply_noise at level 0 returns data identical to clean.
       Also: if MVP results exist, level-0 fit metrics match the clean rows.
    2. Test set unchanged: hash X_test and y_test before and after noise + fit.
    3. Imputer/scaler statistics come from noisy train only.
    4. Noise rates match the active level table within tolerances.

Calibration tests T1-T6 (Task 3.1, rules.md section 2.2):
    T1 Rate accuracy   — noise rates within tolerances.
    T2 Level 0         — level-0 equals clean baseline.
    T3 Numerical valid — no crashed fits, no NaN/inf metrics.
    T4 Dead range      — >= 1 of 16 single-noise pairs has a breaking point.
    T5 Saturation      — < 50% of 16 pairs already break at level 1.
    T6 Direction sanity— mean F1 at level 5 <= level 0 for each noise type.
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
from app.core.config import DATASET_BREAST_CANCER, NOISE_LABEL, NOISE_GAUSSIAN, NOISE_OUTLIERS, NOISE_MISSING, STAGE_MVP
from app.core.levels import get_params, load_levels
from app.storage.json_store import write_json_atomic
from app.storage.paths import official_dir


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
# Check 5: Exact reproducibility
# ---------------------------------------------------------------------------

_REPRO_SUBSET = {
    "dataset": DATASET_BREAST_CANCER,
    "models": ["logreg", "svm_rbf", "decision_tree", "random_forest"],
    "noises": ["label", "label+gaussian"],
    "levels": [0, 3, 5],
    "seed": 0,
}


def _filter_repro_subset(df):
    """Filter a raw_results DataFrame to the exact fixed reproducibility subset.

    Level 0 is represented by combo='clean' rows (one per dataset/model/seed).
    Levels 3 and 5 are noisy rows for the two specified combos.
    No raw rows exist with combo='label'/level=0 or combo='label+gaussian'/level=0.
    """
    subset = _REPRO_SUBSET

    # Level 0: clean rows
    clean_mask = (
        (df["dataset"] == subset["dataset"])
        & (df["model"].isin(subset["models"]))
        & (df["seed"] == subset["seed"])
        & (df["combo"] == "clean")
        & (df["level"] == 0)
    )

    # Levels 3 and 5: noisy rows for the two combos only
    noisy_mask = (
        (df["dataset"] == subset["dataset"])
        & (df["model"].isin(subset["models"]))
        & (df["seed"] == subset["seed"])
        & (df["combo"].isin(subset["noises"]))
        & (df["level"].isin([3, 5]))
    )

    return df[clean_mask | noisy_mask].copy()


def check_reproducibility(mvp_csv_path=None, run_meta=None):
    """Check 5: re-running the fixed subset gives EXACTLY identical metrics.

    Fixed subset:
      dataset=breast_cancer, seed=0, all 4 models,
      combos label and label+gaussian, levels 0 (clean), 3, 5.

    Level 0 is represented by combo='clean' rows; no separate non-clean level-0 rows exist.

    Comparison is STRICT equality (==) for macro_f1 and accuracy.

    If mvp_csv_path exists, compare one fresh run against those MVP results.
    Otherwise run twice and compare the two runs.
    """
    import tempfile
    import pandas as pd
    from app.engine.runner import build_plan, execute_plan
    from app.core.hashing import compute_config_hash
    from app.core.levels import load_levels

    subset = _REPRO_SUBSET
    custom_request = {
        "dataset": subset["dataset"],
        "models": subset["models"],
        "noises": subset["noises"],
        "mode": "sweep",
        "seed_count": 1,  # seed 0 only
    }
    lvl = load_levels()
    meta = run_meta or {
        "run_id": "repro-check",
        "run_type": "custom",
        "stage": "custom",
        "config_hash": compute_config_hash(),
        "methodology_version": "1.0",
        "levels_version": lvl.get("levels_version", ""),
        "levels_frozen": lvl.get("frozen", False),
        "requested_config": custom_request,
    }

    plan = build_plan(custom_request=custom_request)

    failures = []
    details = []
    compare_source = ""

    KEY_COLS = ["dataset", "model", "combo", "level", "seed"]

    def _run_and_filter(out_dir):
        execute_plan(plan, out_dir, meta)
        raw = pd.read_csv(out_dir / "raw_results.csv")
        return _filter_repro_subset(raw)

    if mvp_csv_path is not None and Path(mvp_csv_path).exists():
        # One fresh run compared against the existing MVP results
        compare_source = "mvp_csv"
        with tempfile.TemporaryDirectory() as td:
            fresh_sub = _run_and_filter(Path(td))
        ref_sub = _filter_repro_subset(pd.read_csv(mvp_csv_path))

        merged = fresh_sub.merge(ref_sub, on=KEY_COLS, suffixes=("_fresh", "_ref"))
        n_compared = len(merged)
        for col in ["macro_f1", "accuracy"]:
            mismatch = merged[merged[f"{col}_fresh"] != merged[f"{col}_ref"]]
            for _, row in mismatch.iterrows():
                failures.append(
                    f"Repro mismatch vs MVP: {col} model={row['model']} combo={row['combo']} "
                    f"L{row['level']} fresh={row[f'{col}_fresh']!r} ref={row[f'{col}_ref']!r}"
                )
        details.append(
            f"Compared {n_compared} subset rows (clean L0 + noisy L3/L5) against MVP CSV"
        )

    else:
        # Two fresh runs compared against each other
        compare_source = "two_fresh_runs"
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            df1 = _run_and_filter(Path(td1))
            df2 = _run_and_filter(Path(td2))

        df1s = df1.sort_values(KEY_COLS).reset_index(drop=True)
        df2s = df2.sort_values(KEY_COLS).reset_index(drop=True)

        for col in ["macro_f1", "accuracy", "status"]:
            mismatch_mask = df1s[col] != df2s[col]
            for i in mismatch_mask[mismatch_mask].index:
                row = df1s.iloc[i]
                failures.append(
                    f"Repro mismatch run1 vs run2: {col} model={row['model']} "
                    f"combo={row['combo']} L{row['level']}: "
                    f"{df1s.at[i, col]!r} vs {df2s.at[i, col]!r}"
                )
        details.append(
            f"Compared two fresh runs ({len(df1s)} subset rows each: clean L0 + noisy L3/L5)"
        )

    passed = len(failures) == 0
    return {
        "check": 5,
        "name": "reproducibility",
        "passed": passed,
        "compare_source": compare_source,
        "failures": failures,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Check 6: Performance trend and breaking points
# ---------------------------------------------------------------------------

def check_performance_trend(mvp_csv_path):
    """Check 6: from MVP results, mean F1 at level 5 < level 0 for each single noise;
    at least one (model, noise) pair has a breaking point."""
    import pandas as pd

    failures = []
    details = []

    if not Path(mvp_csv_path).exists():
        return {
            "check": 6,
            "name": "performance_trend",
            "passed": False,
            "failures": [f"MVP CSV not found: {mvp_csv_path}"],
            "details": [],
        }

    df = pd.read_csv(mvp_csv_path)
    df_ok = df[df["status"] == "ok"]

    # Baselines: mean F1 of clean rows per model
    clean_rows = df_ok[df_ok["combo"] == "clean"]
    baselines = clean_rows.groupby("model")["macro_f1"].mean()

    single_noises = ["label", "gaussian", "outliers", "missing"]
    n_breaking = 0
    trend_rows = []

    for noise in single_noises:
        noise_df = df_ok[df_ok["combo"] == noise]
        if noise_df.empty:
            details.append(f"{noise}: no rows found")
            continue

        # Mean F1 across all models at each level
        level_mean = noise_df.groupby("level")["macro_f1"].mean()
        f1_at_0 = float(baselines.mean())  # level 0 = clean baseline mean across models
        f1_at_5 = float(level_mean.get(5, float("nan")))

        trend_ok = f1_at_5 < f1_at_0
        trend_rows.append({
            "noise": noise,
            "mean_f1_level0": round(f1_at_0, 4),
            "mean_f1_level5": round(f1_at_5, 4),
            "trend_down": trend_ok,
        })
        if not trend_ok:
            failures.append(
                f"{noise}: mean F1 at level 5 ({f1_at_5:.4f}) is NOT below level 0 ({f1_at_0:.4f})"
            )
        details.append(f"{noise}: L0={f1_at_0:.4f} L5={f1_at_5:.4f} trend_down={trend_ok}")

        # Count breaking points per model
        for model, b_f1 in baselines.items():
            threshold = 0.90 * b_f1
            model_df = noise_df[noise_df["model"] == model]
            for level in [1, 2, 3, 4, 5]:
                ldf = model_df[model_df["level"] == level]
                if ldf.empty:
                    continue
                mean_f1_l = float(ldf["macro_f1"].mean())
                if mean_f1_l <= threshold + 1e-12:
                    n_breaking += 1
                    break

    if n_breaking == 0:
        failures.append(
            "No (model, noise) pair reaches its breaking point by level 5 "
            "(at least 1 required)"
        )

    details.append(f"Breaking-point pairs found: {n_breaking} / 16")
    passed = len(failures) == 0
    return {
        "check": 6,
        "name": "performance_trend",
        "passed": passed,
        "n_breaking_pairs": n_breaking,
        "trend_by_noise": trend_rows,
        "failures": failures,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Check 7: Runtime measurement and full-run estimate
# ---------------------------------------------------------------------------

def check_runtime(mvp_csv_path=None, probe_dataset="digits"):
    """Check 7: report MVP runtime and estimate full-run time.

    - If MVP results exist, reads fit_time_s from the CSV.
    - Runs a small timing probe (Digits, seed 0, label noise, levels 0 and 1)
      in a temp dir to get per-fit timing for Digits.
    - Estimates full-run time from mean fit time.
    """
    import tempfile
    import time as time_mod
    import pandas as pd
    from app.engine.runner import build_plan, execute_plan
    from app.core.hashing import compute_config_hash
    from app.core.levels import load_levels

    details = []
    failures = []
    mvp_runtime_s = None
    mean_fit_per_model = {}

    if mvp_csv_path is not None and Path(mvp_csv_path).exists():
        df = pd.read_csv(mvp_csv_path)
        df_ok = df[df["status"] == "ok"]
        mvp_runtime_s = float(df_ok["fit_time_s"].sum())
        mean_fit_per_model = (
            df_ok.groupby("model")["fit_time_s"].mean().round(4).to_dict()
        )
        details.append(f"MVP total fit_time_s sum: {mvp_runtime_s:.2f}s")
        details.append(f"Mean fit_time_s per model: {mean_fit_per_model}")
    else:
        details.append("MVP CSV not available — skipping MVP runtime summary")

    # Small timing probe with Digits dataset
    lvl = load_levels()
    probe_request = {
        "dataset": probe_dataset,
        "models": ["logreg", "random_forest"],
        "noises": ["label"],
        "mode": "sweep",
        "seed_count": 1,
    }
    probe_meta = {
        "run_id": "timing-probe",
        "run_type": "custom",
        "stage": "custom",
        "config_hash": compute_config_hash(),
        "methodology_version": "1.0",
        "levels_version": lvl.get("levels_version", ""),
        "levels_frozen": lvl.get("frozen", False),
        "requested_config": probe_request,
    }
    probe_plan = build_plan(custom_request=probe_request)

    with tempfile.TemporaryDirectory() as td:
        t0 = time_mod.time()
        execute_plan(probe_plan, Path(td), probe_meta)
        probe_elapsed = time_mod.time() - t0
        probe_df = pd.read_csv(Path(td) / "raw_results.csv")

    probe_fits = len(probe_df)
    mean_probe_fit_s = probe_elapsed / probe_fits if probe_fits else 0.0
    FULL_RUN_FITS = 6080
    full_run_estimate_s = mean_probe_fit_s * FULL_RUN_FITS
    full_run_estimate_min = full_run_estimate_s / 60.0

    # Failure condition: timing probe must produce at least one fit result
    if probe_fits == 0:
        failures.append(
            f"Timing probe produced 0 fit results on dataset={probe_dataset}: "
            "cannot estimate full-run time"
        )

    details.append(
        f"Timing probe: {probe_fits} fits in {probe_elapsed:.2f}s "
        f"(mean {mean_probe_fit_s:.3f}s/fit on {probe_dataset})"
    )
    details.append(
        f"Full-run estimate: {FULL_RUN_FITS} fits × {mean_probe_fit_s:.3f}s = "
        f"{full_run_estimate_s:.0f}s (~{full_run_estimate_min:.1f} min)"
    )

    passed = len(failures) == 0
    return {
        "check": 7,
        "name": "runtime",
        "passed": passed,
        "mvp_total_fit_time_s": mvp_runtime_s,
        "mean_fit_per_model_s": mean_fit_per_model,
        "probe_fits": probe_fits,
        "probe_elapsed_s": round(probe_elapsed, 3),
        "mean_probe_fit_s": round(mean_probe_fit_s, 4),
        "full_run_estimate_s": round(full_run_estimate_s, 1),
        "full_run_estimate_min": round(full_run_estimate_min, 2),
        "failures": failures,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Calibration: T1-T6  (rules.md section 2.2)
# ---------------------------------------------------------------------------

_SINGLE_NOISES = ["label", "gaussian", "outliers", "missing"]
_N_SINGLE_NOISE_PAIRS = 16  # 4 models x 4 single noises


def run_calibration(stage: str, out_dir: Path) -> dict:
    """Evaluate T1-T6 calibration tests on results from *stage*.

    Reads:
      - <out_dir>/raw_results.csv
      - <out_dir>/validation_report.json  (for T1, T2, T3)

    Writes:
      - <out_dir>/calibration_report.json  (READ-ONLY for levels.json)

    Returns the full calibration report dict.
    """
    import pandas as pd

    raw_csv = out_dir / "raw_results.csv"
    val_report_path = out_dir / "validation_report.json"

    # ── Load data ────────────────────────────────────────────────────────────
    if not raw_csv.exists():
        raise FileNotFoundError(f"raw_results.csv not found in {out_dir}")

    df = pd.read_csv(raw_csv)
    df_ok = df[df["status"] == "ok"]

    val_report = None
    if val_report_path.exists():
        val_report = json.loads(val_report_path.read_text(encoding="utf-8"))

    def _check_result(check_num: int) -> bool:
        """Return True (passed) for a check from the validation report."""
        if val_report is None:
            return False
        for c in val_report.get("checks", []):
            if c["check"] == check_num:
                return bool(c["passed"])
        return False

    tests = {}

    # ── T1: Rate accuracy (reuse Check 4 result) ─────────────────────────────
    t1_passed = _check_result(4)
    tests["T1"] = {
        "name": "Rate accuracy",
        "passed": t1_passed,
        "details": (
            "Rates within tolerances (check 4 PASS)"
            if t1_passed
            else "Rate check 4 FAILED — implementation bug"
        ),
    }

    # ── T2: Level 0 equals clean (reuse Check 1 result) ──────────────────────
    t2_passed = _check_result(1)
    tests["T2"] = {
        "name": "Level 0 equals clean",
        "passed": t2_passed,
        "details": (
            "Level-0 data equals clean (check 1 PASS)"
            if t2_passed
            else "Level-0 check 1 FAILED — implementation bug"
        ),
    }

    # ── T3: Numerical validity ────────────────────────────────────────────────
    # No crashed fits, no NaN/inf metrics, no inf in noisy training data.
    # We inspect the raw CSV directly for this.
    t3_failures = []

    n_error = int((df["status"] == "error").sum())
    if n_error > 0:
        t3_failures.append(f"{n_error} crashed (error) fit(s) found")

    for metric in ["macro_f1", "accuracy"]:
        if metric not in df_ok.columns:
            continue
        n_nan = int(df_ok[metric].isna().sum())
        n_inf = int(np.isinf(df_ok[metric]).sum())
        if n_nan:
            t3_failures.append(f"{n_nan} NaN value(s) in {metric}")
        if n_inf:
            t3_failures.append(f"{n_inf} inf value(s) in {metric}")

    t3_passed = len(t3_failures) == 0
    tests["T3"] = {
        "name": "Numerical validity",
        "passed": t3_passed,
        "n_error_rows": n_error,
        "details": t3_failures if t3_failures else [
            f"No crashed fits, no NaN/inf metrics ({len(df_ok)} ok rows)"
        ],
    }

    # ── T4: Dead range — at least 1 of 16 (model, noise) pairs breaks ────────
    clean_rows = df_ok[df_ok["combo"] == "clean"]
    baselines = clean_rows.groupby("model")["macro_f1"].mean()  # Series[model -> f1]

    n_breaking = 0
    breaking_pairs = []
    saturated_pairs = []  # pairs that break at level 1

    for noise in _SINGLE_NOISES:
        noise_df = df_ok[df_ok["combo"] == noise]
        for model, baseline_f1 in baselines.items():
            threshold = 0.90 * float(baseline_f1)
            model_df = noise_df[noise_df["model"] == model]
            broke = False
            broke_at_1 = False
            for level in [1, 2, 3, 4, 5]:
                ldf = model_df[model_df["level"] == level]
                if ldf.empty:
                    continue
                mean_f1 = float(ldf["macro_f1"].mean())
                if mean_f1 <= threshold + 1e-12:
                    if not broke:
                        n_breaking += 1
                        breaking_pairs.append(
                            {"model": model, "noise": noise, "breaking_level": level,
                             "mean_f1": round(mean_f1, 4), "threshold": round(threshold, 4)}
                        )
                        broke = True
                        if level == 1:
                            broke_at_1 = True
                    break
            if broke_at_1:
                saturated_pairs.append({"model": model, "noise": noise})

    t4_passed = n_breaking >= 1
    tests["T4"] = {
        "name": "Dead range",
        "passed": t4_passed,
        "n_breaking_pairs": n_breaking,
        "n_pairs_total": _N_SINGLE_NOISE_PAIRS,
        "breaking_pairs": breaking_pairs,
        "details": (
            f"{n_breaking}/{_N_SINGLE_NOISE_PAIRS} pairs have a breaking point"
        ),
    }

    # ── T5: Saturation — fewer than 50% already break at level 1 ─────────────
    n_saturated = len(saturated_pairs)
    saturation_pct = 100.0 * n_saturated / _N_SINGLE_NOISE_PAIRS
    t5_passed = n_saturated < _N_SINGLE_NOISE_PAIRS / 2
    tests["T5"] = {
        "name": "Saturation",
        "passed": t5_passed,
        "n_break_at_level_1": n_saturated,
        "n_pairs_total": _N_SINGLE_NOISE_PAIRS,
        "saturation_pct": round(saturation_pct, 1),
        "saturated_pairs": saturated_pairs,
        "details": (
            f"{n_saturated}/{_N_SINGLE_NOISE_PAIRS} pairs ({saturation_pct:.1f}%) "
            f"already break at level 1 (threshold: <50%)"
        ),
    }

    # ── T6: Direction sanity — mean F1 at L5 <= L0 for each noise type ────────
    t6_failures = []
    t6_rows = []
    baseline_mean = float(baselines.mean())

    for noise in _SINGLE_NOISES:
        noise_df = df_ok[df_ok["combo"] == noise]
        if noise_df.empty:
            t6_failures.append(f"{noise}: no rows")
            continue
        level_mean = noise_df.groupby("level")["macro_f1"].mean()
        f1_at_5 = float(level_mean.get(5, float("nan")))
        # Use per-noise baseline (same clean baseline for all)
        direction_ok = f1_at_5 <= baseline_mean + 1e-9
        t6_rows.append(
            {"noise": noise, "mean_f1_level0": round(baseline_mean, 4),
             "mean_f1_level5": round(f1_at_5, 4), "direction_ok": direction_ok}
        )
        if not direction_ok:
            t6_failures.append(
                f"{noise}: mean F1 at L5 ({f1_at_5:.4f}) > L0 ({baseline_mean:.4f})"
            )

    t6_passed = len(t6_failures) == 0
    tests["T6"] = {
        "name": "Direction sanity",
        "passed": t6_passed,
        "by_noise": t6_rows,
        "failures": t6_failures,
        "details": (
            "All noise types show non-increasing mean F1 L0→L5"
            if t6_passed
            else f"{len(t6_failures)} noise type(s) violate direction sanity"
        ),
    }

    # ── Recommendation logic ──────────────────────────────────────────────────
    bug_tests_fail = not (t1_passed and t2_passed and t3_passed and t6_passed)
    t4_fail = not t4_passed
    t5_fail = not t5_passed

    if bug_tests_fail and not (t4_fail or t5_fail):
        recommendation = "FIX_CODE"
    elif bug_tests_fail and (t4_fail or t5_fail):
        # Both bug-type and scale-type failures — unclear
        recommendation = "ESCALATE_TO_USER"
    elif t4_fail and t5_fail:
        recommendation = "ESCALATE_TO_USER"
    elif t4_fail:
        recommendation = "REVISE_TO_S"
    elif t5_fail:
        recommendation = "REVISE_TO_M"
    else:
        recommendation = "KEEP_A"

    # ── Print table ───────────────────────────────────────────────────────────
    print("\nCalibration report")
    print(f"  Stage: {stage}")
    print(f"  Raw results: {raw_csv}")
    print()
    print(f"  {'Test':<6} {'Name':<28} {'Pass?':<8} Details")
    print(f"  {'-'*6} {'-'*28} {'-'*8} {'-'*45}")
    for key, t in tests.items():
        status = "PASS" if t["passed"] else "FAIL"
        detail = t.get("details", "")
        if isinstance(detail, list):
            detail = detail[0] if detail else ""
        print(f"  {key:<6} {t['name']:<28} {status:<8} {str(detail)[:60]}")
    print()
    print(f"  Recommendation: {recommendation}")
    print()
    if recommendation == "KEEP_A":
        print("  All tests pass. Table A is technically usable. Proceed to freeze.")
    elif recommendation == "REVISE_TO_S":
        print("  T4 failed (dead range). No breaking points found. Switch to table S (stronger).")
    elif recommendation == "REVISE_TO_M":
        print("  T5 failed (saturation). >50% of pairs already break at level 1. Switch to table M (milder).")
    elif recommendation == "FIX_CODE":
        print("  T1/T2/T3/T6 failure indicates an implementation bug. Fix code; do not change levels.")
    else:
        print("  Conflicting or unclear failures. Stop and discuss with the user.")

    # ── Write report (READ-ONLY for levels.json) ──────────────────────────────
    report = {
        "stage": stage,
        "recommendation": recommendation,
        "tests": tests,
        "summary": {
            "n_ok_rows": len(df_ok),
            "n_error_rows": n_error,
            "n_breaking_pairs": n_breaking,
            "n_saturated_pairs": n_saturated,
            "saturation_pct": round(saturation_pct, 1),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "calibration_report.json"
    write_json_atomic(report_path, report)
    print(f"  Report written to: {report_path}")
    return report


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_checks(stage: str, out_dir: Path) -> dict:
    """Run all 7 checks and write the full validation report."""
    mvp_csv = official_dir(STAGE_MVP) / "raw_results.csv"

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

    print("Running check 5: reproducibility...")
    c5 = check_reproducibility(mvp_csv_path=mvp_csv if mvp_csv.exists() else None)
    print(f"  Check 5: {'PASS' if c5['passed'] else 'FAIL'}")

    print("Running check 6: performance trend...")
    c6 = check_performance_trend(mvp_csv_path=mvp_csv)
    print(f"  Check 6: {'PASS' if c6['passed'] else 'FAIL'}")

    print("Running check 7: runtime...")
    c7 = check_runtime(mvp_csv_path=mvp_csv if mvp_csv.exists() else None)
    print(f"  Check 7: {'PASS' if c7['passed'] else 'FAIL'}")

    all_checks = [c1, c2, c3, c4, c5, c6, c7]
    report = {
        "stage": stage,
        "checks": all_checks,
        "all_passed": all(c["passed"] for c in all_checks),
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

    calib_parser = subparsers.add_parser("calibrate", help="Run calibration tests T1-T6")
    calib_parser.add_argument("--stage", required=True, choices=["mvp", "stage2", "full"])
    calib_parser.add_argument("--out", type=str, default=None,
                              help="Output directory (default: results/official/<stage>)")

    args = parser.parse_args()

    if args.command == "checks":
        if args.out:
            out_dir = Path(args.out)
        else:
            out_dir = official_dir(args.stage)

        report = run_all_checks(args.stage, out_dir)

        print("\nSummary:")
        for c in report["checks"]:
            status = "PASS" if c["passed"] else "FAIL"
            print(f"  Check {c['check']} ({c['name']}): {status}")

        if not report["all_passed"]:
            sys.exit(1)

    elif args.command == "calibrate":
        if args.out:
            out_dir = Path(args.out)
        else:
            out_dir = official_dir(args.stage)

        run_calibration(args.stage, out_dir)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
