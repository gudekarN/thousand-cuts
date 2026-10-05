"""Tests for validate.py freeze command (Task 3.2).

All tests operate on temporary copies of levels.json — the real file is NEVER touched.

Tests:
- refuses without --confirm
- refuses when validation checks failed
- refuses when calibration recommends a revision (not KEEP_A)
- refuses if already frozen
- successful freeze: verify_frozen_hash passes afterwards
- editing any level value afterwards makes verify_frozen_hash fail
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate import freeze_levels, FreezeRefused
from app.core.hashing import verify_frozen_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REAL_LEVELS_PATH = Path(__file__).resolve().parents[1] / "configs" / "levels.json"


def _copy_levels(tmp_path: Path) -> Path:
    """Copy the real levels.json to tmp_path, reset frozen state, and return the copy path.

    Tests operate on unfrozen temp copies so they remain independent of the
    real freeze state (which may be frozen=True after Task 3.6).
    """
    dst = tmp_path / "levels.json"
    shutil.copy2(REAL_LEVELS_PATH, dst)
    # Reset freeze fields so tests are self-contained
    data = json.loads(dst.read_text(encoding="utf-8"))
    data["frozen"] = False
    data["frozen_at"] = None
    data["frozen_reason"] = None
    data["config_hash"] = None
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dst


def _make_passing_val_report(out_dir: Path) -> None:
    """Write a validation_report.json where all 7 checks pass."""
    checks = [
        {"check": i, "name": f"check_{i}", "passed": True, "failures": [], "details": []}
        for i in range(1, 8)
    ]
    report = {"stage": "mvp", "checks": checks, "all_passed": True}
    (out_dir / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")


def _make_failing_val_report(out_dir: Path) -> None:
    """Write a validation_report.json where check 6 fails."""
    checks = [
        {"check": i, "name": f"check_{i}", "passed": (i != 6), "failures": [], "details": []}
        for i in range(1, 8)
    ]
    report = {"stage": "mvp", "checks": checks, "all_passed": False}
    (out_dir / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")


def _make_calib_report(out_dir: Path, recommendation: str = "KEEP_A") -> None:
    """Write a calibration_report.json with the given recommendation."""
    report = {
        "stage": "mvp",
        "recommendation": recommendation,
        "tests": {},
        "summary": {},
    }
    (out_dir / "calibration_report.json").write_text(json.dumps(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_freeze_refuses_without_confirm(tmp_path):
    """freeze_levels is the programmatic API and always requires --confirm via CLI.
    Test the CLI path: without --confirm, exit code is non-zero."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    _make_passing_val_report(out_dir)
    _make_calib_report(out_dir, "KEEP_A")

    result = subprocess.run(
        [sys.executable, "scripts/validate.py", "freeze",
         "--reason", "test",
         "--out", str(out_dir),
         "--levels-path", str(levels_path)],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode != 0
    assert "--confirm" in result.stderr or "confirm" in result.stderr.lower()


def test_freeze_refuses_when_checks_failed(tmp_path):
    """Refuses if validation_report shows a failed check."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    _make_failing_val_report(out_dir)
    _make_calib_report(out_dir, "KEEP_A")

    with pytest.raises(FreezeRefused, match="validation check"):
        freeze_levels("test reason", out_dir, levels_path)

    # levels.json must be untouched (not frozen)
    data = json.loads(levels_path.read_text(encoding="utf-8"))
    assert data["frozen"] is False


def test_freeze_refuses_when_calibration_not_keep_a(tmp_path):
    """Refuses if calibration recommendation is not KEEP_A."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    _make_passing_val_report(out_dir)
    _make_calib_report(out_dir, "REVISE_TO_S")

    with pytest.raises(FreezeRefused, match="REVISE_TO_S"):
        freeze_levels("test reason", out_dir, levels_path)

    data = json.loads(levels_path.read_text(encoding="utf-8"))
    assert data["frozen"] is False


def test_freeze_refuses_if_already_frozen(tmp_path):
    """Refuses if levels.json already has frozen=true."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    _make_passing_val_report(out_dir)
    _make_calib_report(out_dir, "KEEP_A")

    # Freeze it once
    freeze_levels("first freeze", out_dir, levels_path)

    # Attempt a second freeze
    with pytest.raises(FreezeRefused, match="already frozen"):
        freeze_levels("second freeze", out_dir, levels_path)


def test_freeze_success_verify_frozen_hash_passes(tmp_path):
    """Successful freeze: verify_frozen_hash passes, all fields are written."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    _make_passing_val_report(out_dir)
    _make_calib_report(out_dir, "KEEP_A")

    returned_hash = freeze_levels("calibration passed, ready for stage 2", out_dir, levels_path)

    data = json.loads(levels_path.read_text(encoding="utf-8"))
    assert data["frozen"] is True
    assert data["frozen_at"] is not None and "T" in data["frozen_at"]
    assert data["frozen_reason"] == "calibration passed, ready for stage 2"
    assert data["config_hash"] == returned_hash

    # verify_frozen_hash must pass on the temp copy
    verify_frozen_hash(levels_path)


def test_freeze_editing_level_makes_verify_fail(tmp_path):
    """After freeze, editing any level value makes verify_frozen_hash fail."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    _make_passing_val_report(out_dir)
    _make_calib_report(out_dir, "KEEP_A")

    freeze_levels("test", out_dir, levels_path)

    # Tamper: change one level value in the active table
    data = json.loads(levels_path.read_text(encoding="utf-8"))
    active = data["active_table"]
    data["tables"][active]["label_flip_rate"][3] = 0.99  # was 0.20
    levels_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_hash(levels_path)


def test_freeze_refuses_missing_val_report(tmp_path):
    """Refuses if validation_report.json does not exist."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    # No validation_report.json written
    _make_calib_report(out_dir, "KEEP_A")

    with pytest.raises(FreezeRefused, match="validation_report.json not found"):
        freeze_levels("test", out_dir, levels_path)


def test_freeze_refuses_missing_calib_report(tmp_path):
    """Refuses if calibration_report.json does not exist."""
    levels_path = _copy_levels(tmp_path)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    _make_passing_val_report(out_dir)
    # No calibration_report.json written

    with pytest.raises(FreezeRefused, match="calibration_report.json not found"):
        freeze_levels("test", out_dir, levels_path)
