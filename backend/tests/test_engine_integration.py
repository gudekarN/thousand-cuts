"""Engine integration tests.

Tests the full pipeline: plan -> execute -> raw CSV -> write_all_outputs.
Uses a tiny custom plan with breast_cancer, 1 model, combos label and label+gaussian,
sweep levels, 2 seeds. All tests use a temp RESULTS_ROOT and never touch the
real results/ folder.
"""

import os
import csv
import hashlib
import tempfile
from pathlib import Path

import pytest
import pandas as pd

from app.engine.runner import build_plan, execute_plan, run_fit, FitSpec
from app.engine.analysis import write_all_outputs, compute_baselines, compute_summary


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _tiny_custom_request():
    """Return a tiny custom_request dict usable with build_plan."""
    return {
        "dataset": "breast_cancer",
        "models": ["logreg"],
        "noises": ["label", "label+gaussian"],
        "mode": "sweep",
        "seed_count": 2,  # seeds 0, 1
    }


def _run_meta(out_dir: Path) -> dict:
    from app.core.hashing import compute_config_hash
    from app.core.levels import load_levels
    lvl = load_levels()
    return {
        "run_id": "test-integration-001",
        "run_type": "custom",
        "stage": "custom",
        "config_hash": compute_config_hash(),
        "methodology_version": "1.0",
        "levels_version": lvl.get("levels_version", ""),
        "levels_frozen": lvl.get("frozen", False),
        "requested_config": _tiny_custom_request(),
    }


def _build_tiny_plan():
    return build_plan(custom_request=_tiny_custom_request())


# ─── Test 1: plan → execute → write_all_outputs ───────────────────────────────

def test_full_pipeline_files_and_columns(tmp_path):
    """plan -> execute -> raw CSV -> write_all_outputs produces all files with correct columns and row counts."""
    plan = _build_tiny_plan()
    out_dir = tmp_path / "run1"
    out_dir.mkdir()
    meta = _run_meta(out_dir)

    result = execute_plan(plan, out_dir, meta)
    assert result["status"] == "completed"

    raw_csv = out_dir / "raw_results.csv"
    assert raw_csv.exists()

    df = pd.read_csv(raw_csv)
    # noises=["label", "label+gaussian"] auto-expands to combos: clean, label, gaussian, label+gaussian
    # 2 seeds × (1 clean + 5 label + 5 gaussian + 5 label+gaussian) = 2 × 16 = 32 rows
    # (or 2 × (1 + 5 + 5) = 22 if gaussian is not auto-added)
    # Build plan is authoritative — just verify all rows have status ok
    assert len(df) == len(plan)
    assert (df["status"] == "ok").all()

    write_all_outputs(raw_csv, out_dir)

    # Verify all 5 files exist
    for fname in ["baselines.csv", "summary.csv", "breaking_points.csv", "synergy.csv", "robustness.csv"]:
        assert (out_dir / fname).exists(), f"{fname} missing"

    # Verify exact columns
    baselines_df = pd.read_csv(out_dir / "baselines.csv")
    assert list(baselines_df.columns) == ["dataset", "model", "f1_mean", "f1_std", "acc_mean", "acc_std", "threshold_f1"]

    summary_df = pd.read_csv(out_dir / "summary.csv")
    assert list(summary_df.columns) == ["dataset", "model", "combo", "level", "n_seeds", "f1_mean", "f1_std", "acc_mean", "acc_std", "fit_time_mean", "rel_f1", "drop_abs", "drop_rel"]

    breaking_df = pd.read_csv(out_dir / "breaking_points.csv")
    assert list(breaking_df.columns) == ["dataset", "model", "combo", "baseline_f1", "threshold_f1", "breaking_level", "reached", "noise_params", "f1_at_bp", "seeds_below_at_bp"]

    synergy_df = pd.read_csv(out_dir / "synergy.csv")
    assert list(synergy_df.columns) == ["dataset", "model", "combo", "level", "baseline_f1", "compound_f1", "drop_compound", "sum_single_drops", "synergy", "f1_floor_flag"]

    robustness_df = pd.read_csv(out_dir / "robustness.csv")
    assert list(robustness_df.columns) == ["dataset", "model", "scope", "BPI", "RS", "rank"]


# ─── Test 2: Determinism (same seeds → identical metrics) ─────────────────────

def test_two_runs_identical_metrics(tmp_path):
    """Two runs with the same seeds give identical metrics (ignoring timestamp and fit_time)."""
    plan = _build_tiny_plan()
    meta_cols_to_ignore = {"timestamp_utc", "fit_time_s"}

    for run_n in [1, 2]:
        out = tmp_path / f"run{run_n}"
        out.mkdir()
        meta = _run_meta(out)
        execute_plan(plan, out, meta)

    df1 = pd.read_csv(tmp_path / "run1" / "raw_results.csv")
    df2 = pd.read_csv(tmp_path / "run2" / "raw_results.csv")

    key_cols = ["dataset", "model", "combo", "level", "seed"]
    metric_cols = [c for c in df1.columns if c not in meta_cols_to_ignore and c not in {"run_id"}]

    df1s = df1.sort_values(key_cols).reset_index(drop=True)
    df2s = df2.sort_values(key_cols).reset_index(drop=True)

    for col in ["macro_f1", "accuracy", "status"]:
        assert df1s[col].equals(df2s[col]), f"Column {col} differs between runs"


# ─── Test 3: Test-set hash unchanged ──────────────────────────────────────────

def test_test_set_hash_unchanged(tmp_path):
    """Test-set hash is byte-identical before and after all fits."""
    from app.engine.data import split

    X_tr_before, X_te_before, y_tr_before, y_te_before = split("breast_cancer", 0)
    te_hash_before = hashlib.sha256(X_te_before.tobytes() + y_te_before.tobytes()).hexdigest()

    plan = _build_tiny_plan()
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    meta = _run_meta(out_dir)
    execute_plan(plan, out_dir, meta)

    X_tr_after, X_te_after, y_tr_after, y_te_after = split("breast_cancer", 0)
    te_hash_after = hashlib.sha256(X_te_after.tobytes() + y_te_after.tobytes()).hexdigest()

    assert te_hash_before == te_hash_after


# ─── Test 4: Interrupt and resume – no duplicates ─────────────────────────────

def test_resume_no_duplicates(tmp_path):
    """Interrupt after N fits, resume, and total rows == plan size with no duplicate keys."""
    plan = _build_tiny_plan()
    total = len(plan)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    meta = _run_meta(out_dir)

    # First partial run: cancel after 5 fits
    fits_done = [0]

    def cancel_after_5():
        return fits_done[0] >= 5

    def count_fit(done, total, spec):
        fits_done[0] = done

    result1 = execute_plan(plan, out_dir, meta, cancel_check=cancel_after_5, progress_cb=count_fit)
    assert result1["status"] == "cancelled"

    df_partial = pd.read_csv(out_dir / "raw_results.csv")
    assert len(df_partial) == 5

    # Resume: complete the rest
    result2 = execute_plan(plan, out_dir, meta)
    assert result2["status"] == "completed"

    df_full = pd.read_csv(out_dir / "raw_results.csv")
    assert len(df_full) == total

    # Verify no duplicate keys
    key_cols = ["dataset", "model", "combo", "level", "seed"]
    dupes = df_full.duplicated(subset=key_cols)
    assert not dupes.any(), f"Duplicate keys found:\n{df_full[dupes]}"


# ─── Test 5: Real results/ folder is untouched ────────────────────────────────

def test_real_results_untouched(tmp_path):
    """Running in a temp dir never touches the real results/ folder."""
    results_root = Path(__file__).parents[2] / "results"
    
    # Capture file listing before (if results/ exists)
    def list_files(d: Path):
        if not d.exists():
            return set()
        return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()}

    before = list_files(results_root)

    plan = _build_tiny_plan()
    out_dir = tmp_path / "safe_run"
    out_dir.mkdir()
    meta = _run_meta(out_dir)
    execute_plan(plan, out_dir, meta)

    after = list_files(results_root)

    new_files = after - before
    assert not new_files, f"Real results/ was modified. New files: {new_files}"
