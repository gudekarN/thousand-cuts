import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from app.engine.analysis import compute_robustness, write_all_outputs

def test_compute_robustness():
    # Breaking points
    bp_data = [
        # M1 singles
        {"dataset": "D1", "model": "M1", "combo": "label", "reached": True, "breaking_level": 2},
        {"dataset": "D1", "model": "M1", "combo": "gaussian", "reached": False, "breaking_level": pd.NA},
        # BPI singles for M1 = (2 + 6) / 2 = 4.0
        
        # M1 compound
        {"dataset": "D1", "model": "M1", "combo": "label+gaussian", "reached": True, "breaking_level": 1},
        # BPI all for M1 = (2 + 6 + 1) / 3 = 3.0
        
        # M2 singles
        {"dataset": "D1", "model": "M2", "combo": "label", "reached": True, "breaking_level": 4},
        {"dataset": "D1", "model": "M2", "combo": "gaussian", "reached": True, "breaking_level": 4},
        # BPI singles for M2 = (4 + 4) / 2 = 4.0 (Tie with M1!)
        
        # M2 compound
        {"dataset": "D1", "model": "M2", "combo": "label+gaussian", "reached": True, "breaking_level": 1},
        # BPI all for M2 = (4 + 4 + 1) / 3 = 3.0 (Tie with M1!)
    ]
    breaking_points = pd.DataFrame(bp_data)
    
    # Summary
    sum_data = [
        # M1 singles
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 1, "rel_f1": 0.95},
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 2, "rel_f1": 0.80},
        {"dataset": "D1", "model": "M1", "combo": "gaussian", "level": 1, "rel_f1": 1.00},
        {"dataset": "D1", "model": "M1", "combo": "gaussian", "level": 2, "rel_f1": 0.95},
        # RS singles for M1 = mean(0.95, 0.80, 1.00, 0.95) = 0.925
        
        # M1 compound
        {"dataset": "D1", "model": "M1", "combo": "label+gaussian", "level": 1, "rel_f1": 0.70},
        # RS all for M1 = mean(0.95, 0.80, 1.00, 0.95, 0.70) = 4.4 / 5 = 0.88
        
        # M2 singles
        {"dataset": "D1", "model": "M2", "combo": "label", "level": 1, "rel_f1": 0.98},
        {"dataset": "D1", "model": "M2", "combo": "label", "level": 2, "rel_f1": 0.98},
        {"dataset": "D1", "model": "M2", "combo": "gaussian", "level": 1, "rel_f1": 0.99},
        {"dataset": "D1", "model": "M2", "combo": "gaussian", "level": 2, "rel_f1": 0.99},
        # RS singles for M2 = mean(0.98, 0.98, 0.99, 0.99) = 0.985
        
        # M2 compound
        {"dataset": "D1", "model": "M2", "combo": "label+gaussian", "level": 1, "rel_f1": 0.75},
        # RS all for M2 = mean(0.98, 0.98, 0.99, 0.99, 0.75) = 4.69 / 5 = 0.938
    ]
    summary = pd.DataFrame(sum_data)
    
    robustness = compute_robustness(breaking_points, summary)
    
    assert len(robustness) == 4 # (D1, M1, singles), (D1, M1, all), (D1, M2, singles), (D1, M2, all)
    
    rob_m1_s = robustness[(robustness["dataset"] == "D1") & (robustness["model"] == "M1") & (robustness["scope"] == "singles")].iloc[0]
    rob_m2_s = robustness[(robustness["dataset"] == "D1") & (robustness["model"] == "M2") & (robustness["scope"] == "singles")].iloc[0]
    
    assert np.isclose(rob_m1_s["BPI"], 4.0)
    assert np.isclose(rob_m2_s["BPI"], 4.0)
    assert np.isclose(rob_m1_s["RS"], 0.925)
    assert np.isclose(rob_m2_s["RS"], 0.985)
    
    # M2 has higher RS, so M2 rank = 1, M1 rank = 2
    assert rob_m2_s["rank"] == 1
    assert rob_m1_s["rank"] == 2
    
def test_write_all_outputs(monkeypatch, tmp_path):
    # Create a dummy raw CSV
    raw_path = tmp_path / "raw_results.csv"
    
    df = pd.DataFrame([
        {"dataset": "D1", "model": "M1", "combo": "clean", "level": 0, "seed": 0, "status": "ok", "macro_f1": 1.0, "accuracy": 1.0, "fit_time_s": 1.0},
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 1, "seed": 0, "status": "ok", "macro_f1": 0.5, "accuracy": 0.5, "fit_time_s": 1.0},
    ])
    df.to_csv(raw_path, index=False)
    
    def mock_get_params(level, path=None):
        return {"label_flip_rate": 0.1, "gaussian_k": 0.5, "outlier_cell_rate": 0.05, "missing_cell_rate": 0.02}
        
    import app.engine.analysis as analysis_mod
    monkeypatch.setattr(analysis_mod, "get_params", mock_get_params, raising=False)
    
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    
    write_all_outputs(raw_path, out_dir)
    
    assert (out_dir / "baselines.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "breaking_points.csv").exists()
    assert (out_dir / "synergy.csv").exists()
    assert (out_dir / "robustness.csv").exists()
    
    # Check that CSVs are readable
    pd.read_csv(out_dir / "robustness.csv")
