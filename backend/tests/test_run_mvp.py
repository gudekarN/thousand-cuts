"""Tests for the MVP run script."""
import subprocess
import sys
from pathlib import Path
import pandas as pd

def test_run_mvp_dry_run():
    """--dry-run prints plan summary with exactly 312 fits."""
    cmd = [sys.executable, "scripts/run_mvp.py", "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Total fits: 312" in result.stdout
    assert "logreg" in result.stdout
    assert "label+gaussian" in result.stdout


def test_run_mvp_limit_temp_dir(tmp_path):
    """--limit 8 --out temp dir runs and produces files."""
    out_dir = tmp_path / "mvp_test"
    cmd = [
        sys.executable, "scripts/run_mvp.py", 
        "--limit", "8", 
        "--out", str(out_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Executing MVP plan with 8 fits" in result.stdout
    
    assert (out_dir / "raw_results.csv").exists()
    assert (out_dir / "baselines.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "breaking_points.csv").exists()
    assert (out_dir / "synergy.csv").exists()
    assert (out_dir / "robustness.csv").exists()
    
    df = pd.read_csv(out_dir / "raw_results.csv")
    assert len(df) == 8


def test_run_mvp_limit_official_dir_refused():
    """--limit is refused if --out is the official directory."""
    cmd = [
        sys.executable, "scripts/run_mvp.py",
        "--limit", "8",
        "--out", "results/official/mvp"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "ERROR: --limit is only allowed when --out is outside results/official" in result.stderr
