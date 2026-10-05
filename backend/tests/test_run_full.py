"""Tests for the Full run script (Task 5.1).

Covers:
- dry-run prints exactly 6080 fits
- refuses when levels are not frozen
- refuses on config hash mismatch
- refuses --limit with official out dir
- --analyze-only on existing raw CSV
- tiny temp run produces deterministic metrics
- resume skips completed rows
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
REAL_LEVELS = BACKEND_DIR / "configs" / "levels.json"
SCRIPT = str(BACKEND_DIR / "scripts" / "run_full.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True,
        text=True,
        cwd=str(cwd or BACKEND_DIR),
    )


def _unfrozen_levels(tmp_path: Path) -> Path:
    """Copy real levels, set frozen=False."""
    p = tmp_path / "levels.json"
    data = json.loads(REAL_LEVELS.read_text())
    data["frozen"] = False
    p.write_text(json.dumps(data))
    return p


def _mismatched_levels(tmp_path: Path) -> Path:
    """Copy real levels, keep frozen=True but corrupt a value."""
    p = tmp_path / "levels.json"
    data = json.loads(REAL_LEVELS.read_text())
    data["frozen"] = True
    data["tables"]["A"]["label_flip_rate"][1] = 0.99  # mismatch
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dry_run_prints_6080():
    """--dry-run prints Total fits: 6080."""
    result = _run(["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "Total fits: 6080" in result.stdout
    assert "logreg" in result.stdout
    assert "label+gaussian" in result.stdout


def test_refuses_unfrozen(tmp_path):
    """Refuses when levels are not frozen."""
    levels_path = _unfrozen_levels(tmp_path)
    out_dir = tmp_path / "full_test"
    result = _run([
        "--limit", "2",
        "--out", str(out_dir),
        "--levels-path", str(levels_path),
    ])
    assert result.returncode == 1
    assert "Full run refused" in result.stderr
    assert "not frozen" in result.stderr.lower()


def test_refuses_hash_mismatch(tmp_path):
    """Refuses on config hash mismatch."""
    levels_path = _mismatched_levels(tmp_path)
    out_dir = tmp_path / "full_test"
    result = _run([
        "--limit", "2",
        "--out", str(out_dir),
        "--levels-path", str(levels_path),
    ])
    assert result.returncode == 1
    assert "Full run refused" in result.stderr


def test_refuses_limit_with_official_out():
    """--limit is refused if --out is the official directory."""
    result = _run([
        "--limit", "8",
        "--out", "results/official/full",
    ])
    assert result.returncode == 1
    assert "--limit is only allowed when --out is outside results/official" in result.stderr


def test_tiny_temp_run_deterministic(tmp_path):
    """A tiny temp run produces consistent metrics across two independent runs."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    for out_dir in (out1, out2):
        result = _run(["--limit", "4", "--out", str(out_dir)])
        assert result.returncode == 0, result.stderr
        assert (out_dir / "raw_results.csv").exists()

    df1 = pd.read_csv(out1 / "raw_results.csv")
    df2 = pd.read_csv(out2 / "raw_results.csv")

    # Same rows (same plan order, same seeds)
    assert len(df1) == len(df2) == 4
    assert list(df1["macro_f1"].round(8)) == list(df2["macro_f1"].round(8))


def test_resume_skips_completed_rows(tmp_path):
    """Resume after a partial run completes without duplicates."""
    out_dir = tmp_path / "full_resume"

    # First partial run: 4 fits
    r1 = _run(["--limit", "4", "--out", str(out_dir)])
    assert r1.returncode == 0, r1.stderr
    df_partial = pd.read_csv(out_dir / "raw_results.csv")
    assert len(df_partial) == 4

    # Second run with more fits — resume should skip the first 4
    r2 = _run(["--limit", "8", "--out", str(out_dir)])
    assert r2.returncode == 0, r2.stderr
    df_full = pd.read_csv(out_dir / "raw_results.csv")
    assert len(df_full) == 8  # no duplicates


def test_analyze_only(tmp_path):
    """--analyze-only re-runs derived outputs from existing raw_results.csv."""
    out_dir = tmp_path / "full_analyze"

    # Produce a raw CSV first
    r1 = _run(["--limit", "8", "--out", str(out_dir)])
    assert r1.returncode == 0, r1.stderr

    # Remove derived CSVs to verify they get regenerated
    for name in ["baselines.csv", "summary.csv", "breaking_points.csv"]:
        f = out_dir / name
        if f.exists():
            f.unlink()

    # --analyze-only should regenerate them
    r2 = _run(["--analyze-only", "--out", str(out_dir)])
    assert r2.returncode == 0, r2.stderr
    assert (out_dir / "baselines.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "breaking_points.csv").exists()


def test_analyze_only_missing_raw_fails(tmp_path):
    """--analyze-only fails cleanly if raw_results.csv does not exist."""
    out_dir = tmp_path / "empty_dir"
    out_dir.mkdir()
    result = _run(["--analyze-only", "--out", str(out_dir)])
    assert result.returncode == 1
    assert "raw_results.csv" in result.stderr
