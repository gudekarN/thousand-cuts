"""Tests for execute_plan: resume, cancel, freeze guard, hash guard, progress callback."""

import json
import pytest

from app.core.config import DATASET_BREAST_CANCER, COMBO_CLEAN, STAGE_STAGE2, STAGE_FULL, STAGE_MVP
from app.engine.runner import FitSpec, execute_plan, build_plan
from app.storage.csv_store import read_rows, RAW_COLUMNS
from app.storage.json_store import read_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_plan():
    """Two-fit plan: one clean, one noisy. Fast to execute."""
    return [
        FitSpec(dataset=DATASET_BREAST_CANCER, model="logreg", combo=COMBO_CLEAN, level=0, seed=0),
        FitSpec(dataset=DATASET_BREAST_CANCER, model="logreg", combo="label", level=1, seed=0),
    ]


def _run_meta(stage=STAGE_MVP, run_type="custom", run_id="test_run"):
    return {"stage": stage, "run_type": run_type, "run_id": run_id}


def _freeze_monkeypatch(monkeypatch, frozen: bool, hash_ok: bool = True):
    """Patch is_frozen and verify_frozen_hash in the runner module's local scope."""
    import app.engine.runner as runner_mod

    # We patch via the lazy imports inside execute_plan by patching in the source modules
    import app.core.levels as levels_mod
    import app.core.hashing as hashing_mod

    monkeypatch.setattr(levels_mod, "is_frozen", lambda path=None: frozen)

    if not frozen:
        def raise_not_frozen(path=None):
            raise ValueError("Levels are not frozen.")
        monkeypatch.setattr(hashing_mod, "verify_frozen_hash", raise_not_frozen)
    elif not hash_ok:
        def raise_hash_mismatch(path=None):
            raise ValueError("Config hash mismatch. Stored: aaa, Computed: bbb")
        monkeypatch.setattr(hashing_mod, "verify_frozen_hash", raise_hash_mismatch)
    else:
        monkeypatch.setattr(hashing_mod, "verify_frozen_hash", lambda path=None: None)


# ---------------------------------------------------------------------------
# 1. Stage 2 refuses when unfrozen
# ---------------------------------------------------------------------------

def test_execute_plan_stage2_refuses_when_not_frozen(tmp_path, monkeypatch):
    """stage2 must refuse to run when levels are not frozen (no fits executed)."""
    _freeze_monkeypatch(monkeypatch, frozen=False)
    plan = _small_plan()
    meta = _run_meta(stage=STAGE_STAGE2, run_type="official")

    with pytest.raises(ValueError, match="frozen"):
        execute_plan(plan, tmp_path, meta)

    # No fits must have been written
    assert not (tmp_path / "raw_results.csv").exists()


# ---------------------------------------------------------------------------
# 2. Full refuses when unfrozen
# ---------------------------------------------------------------------------

def test_execute_plan_full_refuses_when_not_frozen(tmp_path, monkeypatch):
    """full stage must refuse to run when levels are not frozen (no fits executed)."""
    _freeze_monkeypatch(monkeypatch, frozen=False)
    plan = _small_plan()
    meta = _run_meta(stage=STAGE_FULL, run_type="official")

    with pytest.raises(ValueError, match="frozen"):
        execute_plan(plan, tmp_path, meta)

    # No fits must have been written
    assert not (tmp_path / "raw_results.csv").exists()


# ---------------------------------------------------------------------------
# 3. Official MVP first run is allowed when unfrozen
# ---------------------------------------------------------------------------

def test_execute_plan_mvp_official_first_run_allowed_unfrozen(tmp_path, monkeypatch):
    """Official MVP first run proceeds even when levels are not frozen."""
    _freeze_monkeypatch(monkeypatch, frozen=False)
    plan = _small_plan()
    meta = _run_meta(stage=STAGE_MVP, run_type="official")

    # Should NOT raise; a manifest.json doesn't exist yet, so resume guard is skipped
    result = execute_plan(plan, tmp_path, meta)

    assert result["status"] == "completed"
    rows = read_rows(tmp_path / "raw_results.csv")
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# 4. Official MVP resume refuses when unfrozen
# ---------------------------------------------------------------------------

def test_execute_plan_mvp_official_resume_refuses_when_not_frozen(tmp_path, monkeypatch):
    """Any official resume (including MVP) must refuse when levels are not frozen."""
    _freeze_monkeypatch(monkeypatch, frozen=False)
    plan = _small_plan()
    meta = _run_meta(stage=STAGE_MVP, run_type="official")

    # Pre-create a manifest to simulate a partial run being resumed
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"config_hash": "some_hash"}))

    with pytest.raises(ValueError, match="frozen"):
        execute_plan(plan, tmp_path, meta, resume=True)


# ---------------------------------------------------------------------------
# 5. Official resume refuses on config-hash mismatch
# ---------------------------------------------------------------------------

def test_execute_plan_official_resume_refuses_on_hash_mismatch(tmp_path, monkeypatch):
    """Official resume must refuse if the stored manifest config_hash doesn't match current hash."""
    # Frozen, but compute_config_hash returns something different from the manifest
    _freeze_monkeypatch(monkeypatch, frozen=True, hash_ok=True)

    import app.core.hashing as hashing_mod
    # verify_frozen_hash passes (frozen=True, hash_ok=True), but current computed hash != stored
    real_compute = hashing_mod.compute_config_hash
    monkeypatch.setattr(hashing_mod, "compute_config_hash", lambda path=None: "current_hash_xyz")

    plan = _small_plan()
    meta = _run_meta(stage=STAGE_MVP, run_type="official")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"config_hash": "stored_hash_abc"}))

    with pytest.raises(ValueError, match="hash mismatch"):
        execute_plan(plan, tmp_path, meta, resume=True)


# ---------------------------------------------------------------------------
# 6. Official resume succeeds when frozen and hash matches
# ---------------------------------------------------------------------------

def test_execute_plan_official_resume_succeeds_when_frozen_and_hash_matches(tmp_path, monkeypatch):
    """Official resume proceeds when levels are frozen and config_hash matches manifest."""
    _freeze_monkeypatch(monkeypatch, frozen=True, hash_ok=True)

    import app.core.hashing as hashing_mod
    fixed_hash = "aabbcc1122334455667788990011223344556677889900112233445566778899"
    monkeypatch.setattr(hashing_mod, "compute_config_hash", lambda path=None: fixed_hash)

    plan = _small_plan()
    meta = _run_meta(stage=STAGE_MVP, run_type="official")

    # Pre-create manifest with a matching hash; write 1 ok row so resume skips it
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"config_hash": fixed_hash, "status": "started",
                                         "completed_fits": 0, "finished_utc": None}))
    csv_path = tmp_path / "raw_results.csv"
    # Write one already-completed row so we can verify skip behaviour
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerow({col: ("0" if col in ("seed", "level", "n_noises", "n_train", "n_test") else
                               "ok" if col == "status" else
                               COMBO_CLEAN if col == "combo" else
                               DATASET_BREAST_CANCER if col == "dataset" else
                               "logreg" if col == "model" else "")
                         for col in RAW_COLUMNS})

    result = execute_plan(plan, tmp_path, meta, resume=True)
    assert result["status"] == "completed"
    rows = read_rows(csv_path)
    # First row was pre-existing (skipped), second row gets added
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# 7. Existing resume / no-duplicate behaviour still passes
# ---------------------------------------------------------------------------

def test_execute_plan_resume_no_duplicates(tmp_path):
    """Resume after a partial run completes without duplicates (custom run, no freeze required)."""
    plan = _small_plan()

    call_count = [0]

    def cancel_after_one():
        return call_count[0] >= 1

    def count_progress(done, total, spec):
        call_count[0] = done

    execute_plan(plan, tmp_path, _run_meta(), progress_cb=count_progress, cancel_check=cancel_after_one)

    rows_partial = read_rows(tmp_path / "raw_results.csv")
    assert len(rows_partial) == 1

    result = execute_plan(plan, tmp_path, _run_meta())
    rows_final = read_rows(tmp_path / "raw_results.csv")
    assert len(rows_final) == 2  # No duplicates
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# 8. Existing cancellation behaviour still passes
# ---------------------------------------------------------------------------

def test_execute_plan_cancel_stops_cleanly(tmp_path):
    """Cancel stops cleanly and finalises manifest with status=cancelled."""
    plan = _small_plan()

    call_count = [0]

    def cancel_after_one():
        return call_count[0] >= 1

    def count_progress(done, total, spec):
        call_count[0] = done

    result = execute_plan(plan, tmp_path, _run_meta(), progress_cb=count_progress,
                          cancel_check=cancel_after_one)

    assert result["status"] == "cancelled"
    manifest = read_json(tmp_path / "manifest.json")
    assert manifest["status"] == "cancelled"


# ---------------------------------------------------------------------------
# 9. Existing error-row continuation behaviour still passes
# ---------------------------------------------------------------------------

def test_execute_plan_error_rows_recorded_run_continues(tmp_path, monkeypatch):
    """A failed fit produces an error row but the run continues to completion."""
    plan = _small_plan()

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


# ---------------------------------------------------------------------------
# Bonus: every row contains all RAW_COLUMNS
# ---------------------------------------------------------------------------

def test_execute_plan_all_rows_have_columns(tmp_path):
    """Every row in raw_results.csv must contain all RAW_COLUMNS."""
    plan = _small_plan()
    result = execute_plan(plan, tmp_path, _run_meta())

    rows = read_rows(tmp_path / "raw_results.csv")
    assert len(rows) == 2
    for row in rows:
        for col in RAW_COLUMNS:
            assert col in row, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Bonus: progress callback count
# ---------------------------------------------------------------------------

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
