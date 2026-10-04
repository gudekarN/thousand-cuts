"""Tests for execute_plan: resume, cancel, freeze guard, hash guard, progress callback."""

import json
import pytest

from app.core.config import DATASET_BREAST_CANCER, COMBO_CLEAN, STAGE_STAGE2, STAGE_FULL
from app.engine.runner import FitSpec, execute_plan, build_plan
from app.storage.csv_store import read_rows, RAW_COLUMNS
from app.storage.json_store import read_json


# Small 2-fit plan for fast testing
def _small_plan():
    return [
        FitSpec(dataset=DATASET_BREAST_CANCER, model="logreg", combo=COMBO_CLEAN, level=0, seed=0),
        FitSpec(dataset=DATASET_BREAST_CANCER, model="logreg", combo="label", level=1, seed=0),
    ]


def _run_meta(stage="mvp", run_type="custom", run_id="test_run"):
    return {"stage": stage, "run_type": run_type, "run_id": run_id}


def test_execute_plan_all_rows_have_columns(tmp_path):
    """Every row in raw_results.csv must contain all RAW_COLUMNS."""
    plan = _small_plan()
    result = execute_plan(plan, tmp_path, _run_meta())

    rows = read_rows(tmp_path / "raw_results.csv")
    assert len(rows) == 2
    for row in rows:
        for col in RAW_COLUMNS:
            assert col in row, f"Missing column: {col}"


def test_execute_plan_resume_no_duplicates(tmp_path):
    """Resume after a partial run completes without duplicates."""
    plan = _small_plan()

    # Run first fit only by cancelling after 1
    call_count = [0]
    def cancel_after_one():
        return call_count[0] >= 1

    def count_progress(done, total, spec):
        call_count[0] = done

    execute_plan(plan, tmp_path, _run_meta(), progress_cb=count_progress, cancel_check=cancel_after_one)

    rows_partial = read_rows(tmp_path / "raw_results.csv")
    assert len(rows_partial) == 1

    # Resume - should complete without duplicates
    result = execute_plan(plan, tmp_path, _run_meta())
    rows_final = read_rows(tmp_path / "raw_results.csv")
    assert len(rows_final) == 2  # No duplicates

    assert result["status"] == "completed"


def test_execute_plan_cancel_stops_cleanly(tmp_path):
    """Cancel stops cleanly and finalizes manifest with status=cancelled."""
    plan = _small_plan()

    call_count = [0]
    def cancel_after_one():
        return call_count[0] >= 1

    def count_progress(done, total, spec):
        call_count[0] = done

    result = execute_plan(plan, tmp_path, _run_meta(), progress_cb=count_progress, cancel_check=cancel_after_one)

    assert result["status"] == "cancelled"
    manifest = read_json(tmp_path / "manifest.json")
    assert manifest["status"] == "cancelled"


def test_execute_plan_stage2_refuses_when_not_frozen(tmp_path):
    """stage2 must refuse to run when levels are not frozen."""
    plan = _small_plan()
    meta = _run_meta(stage=STAGE_STAGE2, run_type="official")

    with pytest.raises(ValueError, match="frozen"):
        execute_plan(plan, tmp_path, meta)


def test_execute_plan_resume_refuses_on_hash_mismatch(tmp_path):
    """Resume must refuse if the stored config_hash in manifest doesn't match current hash."""
    plan = _small_plan()
    meta = _run_meta(run_type="official")

    # Write a manifest with a bogus config_hash
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"config_hash": "bogus_hash_1234567890"}))

    with pytest.raises(ValueError, match="hash mismatch"):
        execute_plan(plan, tmp_path, meta, resume=True)


def test_execute_plan_progress_callback_count(tmp_path):
    """Progress callback is called once per fit."""
    plan = _small_plan()
    calls = []

    def record_progress(done, total, spec):
        calls.append((done, total))

    execute_plan(plan, tmp_path, _run_meta(), progress_cb=record_progress)

    assert len(calls) == len(plan)
    assert calls[-1][0] == len(plan)
    assert calls[-1][1] == len(plan)


def test_execute_plan_error_rows_recorded_run_continues(tmp_path, monkeypatch):
    """A failed fit produces an error row but the run continues."""
    plan = _small_plan()

    # Patch evaluate to fail on the second call
    import app.engine.runner as runner_mod
    call_count = [0]
    original_run_fit = runner_mod.run_fit

    def patched_run_fit(spec, run_meta):
        call_count[0] += 1
        if call_count[0] == 2:
            return {
                **{col: "" for col in ["run_id", "run_type", "stage", "dataset", "model",
                                       "seed", "combo", "n_noises", "level", "noise_stats",
                                       "timestamp_utc", "config_hash", "methodology_version",
                                       "levels_version", "levels_frozen", "error_msg"]},
                "macro_f1": float("nan"),
                "accuracy": float("nan"),
                "fit_time_s": float("nan"),
                "n_train": 0,
                "n_test": 0,
                "status": "error",
                "error_msg": "Simulated failure",
            }
        return original_run_fit(spec, run_meta)

    monkeypatch.setattr(runner_mod, "run_fit", patched_run_fit)

    result = execute_plan(plan, tmp_path, _run_meta())

    rows = read_rows(tmp_path / "raw_results.csv")
    assert len(rows) == 2  # Both rows written
    statuses = {r["status"] for r in rows}
    assert "error" in statuses
    assert result["status"] == "completed"  # Run completes despite error
