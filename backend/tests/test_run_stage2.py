"""Tests for the Stage 2 run script."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

def test_run_stage2_dry_run():
    """--dry-run prints plan summary with exactly 1824 fits."""
    cmd = [sys.executable, "scripts/run_stage2.py", "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Total fits: 1824" in result.stdout
    assert "logreg" in result.stdout
    assert "label+gaussian" in result.stdout


def test_run_stage2_limit_temp_dir(tmp_path):
    """--limit 8 --out temp dir runs and produces files."""
    out_dir = tmp_path / "stage2_test"
    cmd = [
        sys.executable, "scripts/run_stage2.py", 
        "--limit", "8", 
        "--out", str(out_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Executing Stage 2 plan with 8 fits" in result.stdout
    
    assert (out_dir / "raw_results.csv").exists()
    assert (out_dir / "baselines.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "breaking_points.csv").exists()
    assert (out_dir / "synergy.csv").exists()
    assert (out_dir / "robustness.csv").exists()
    
    df = pd.read_csv(out_dir / "raw_results.csv")
    assert len(df) == 8


def test_run_stage2_limit_official_dir_refused():
    """--limit is refused if --out is the official directory."""
    cmd = [
        sys.executable, "scripts/run_stage2.py",
        "--limit", "8",
        "--out", "results/official/stage2"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "ERROR: --limit is only allowed when --out is outside results/official" in result.stderr


def test_run_stage2_refuses_unfrozen(tmp_path):
    """Refuses if levels are not frozen."""
    out_dir = tmp_path / "stage2_test"
    levels_path = tmp_path / "levels.json"
    
    real_levels = Path(__file__).resolve().parent.parent / "configs" / "levels.json"
    shutil.copy2(real_levels, levels_path)
    
    # Tamper to make unfrozen
    data = json.loads(levels_path.read_text())
    data["frozen"] = False
    levels_path.write_text(json.dumps(data))
    
    cmd = [
        sys.executable, "scripts/run_stage2.py",
        "--limit", "2",
        "--out", str(out_dir),
        "--levels-path", str(levels_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "Stage 2 refused" in result.stderr
    assert "not frozen" in result.stderr.lower()


def test_run_stage2_refuses_hash_mismatch(tmp_path):
    """Refuses if config hash mismatches."""
    out_dir = tmp_path / "stage2_test"
    levels_path = tmp_path / "levels.json"
    
    real_levels = Path(__file__).resolve().parent.parent / "configs" / "levels.json"
    shutil.copy2(real_levels, levels_path)
    
    # Ensure it's frozen
    data = json.loads(levels_path.read_text())
    data["frozen"] = True
    
    # Change a level to trigger mismatch
    data["tables"]["A"]["label_flip_rate"][1] = 0.99
    levels_path.write_text(json.dumps(data))
    
    cmd = [
        sys.executable, "scripts/run_stage2.py",
        "--limit", "2",
        "--out", str(out_dir),
        "--levels-path", str(levels_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "Stage 2 refused" in result.stderr
    assert "mismatch" in result.stderr.lower()
