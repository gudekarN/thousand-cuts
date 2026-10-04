import pytest
import pandas as pd
import numpy as np
import json
from app.engine.analysis import compute_breaking_points

def test_compute_breaking_points(monkeypatch):
    # Mock get_params to return predictable values without levels.json
    def mock_get_params(level, path=None):
        return {
            "label_flip_rate": level * 0.1,
            "gaussian_k": level * 0.5,
            "outlier_cell_rate": level * 0.05,
            "missing_cell_rate": level * 0.02
        }
    
    import app.engine.analysis as analysis_mod
    monkeypatch.setattr(analysis_mod, "get_params", mock_get_params, raising=False)
    
    # Mock dataframes
    
    # Baselines: threshold is 0.90 for M1 and 0.80 for M2
    baselines_data = [
        {"dataset": "D1", "model": "M1", "f1_mean": 1.0, "threshold_f1": 0.90},
        {"dataset": "D1", "model": "M2", "f1_mean": 1.0, "threshold_f1": 0.80},
    ]
    baselines = pd.DataFrame(baselines_data)
    
    # Summary
    # M1 label breaks at level 3 (exactly 0.90 -> broken)
    # M1 label non-monotone: level 2 is 0.91 (not broken), level 3 is 0.90 (broken), level 4 is 0.95 (not broken)
    # M2 gaussian+missing not reached (never breaks)
    
    summary_data = [
        {"dataset": "D1", "model": "M1", "combo": "clean", "level": 0, "f1_mean": 1.0},
        
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 1, "f1_mean": 0.95},
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 2, "f1_mean": 0.91},
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 3, "f1_mean": 0.90}, # Breaks here
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 4, "f1_mean": 0.95},
        
        {"dataset": "D1", "model": "M2", "combo": "clean", "level": 0, "f1_mean": 1.0},
        
        {"dataset": "D1", "model": "M2", "combo": "gaussian+missing", "level": 1, "f1_mean": 0.90},
        {"dataset": "D1", "model": "M2", "combo": "gaussian+missing", "level": 2, "f1_mean": 0.85},
        {"dataset": "D1", "model": "M2", "combo": "gaussian+missing", "level": 3, "f1_mean": 0.81},
    ]
    summary = pd.DataFrame(summary_data)
    
    # df (raw): for counting seeds_below_at_bp
    # For M1, label, level 3, we'll have 3 seeds. 
    # Seed 0: 0.91 (not broken)
    # Seed 1: 0.89 (broken)
    # Seed 2: 0.90 (broken)
    # So 2 seeds below at bp.
    
    raw_data = [
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 3, "seed": 0, "status": "ok", "macro_f1": 0.91},
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 3, "seed": 1, "status": "ok", "macro_f1": 0.89},
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 3, "seed": 2, "status": "ok", "macro_f1": 0.90},
    ]
    df = pd.DataFrame(raw_data)
    
    bp = compute_breaking_points(df, summary, baselines)
    
    assert len(bp) == 2
    
    # Check M1 (breaks at level 3, value exactly at threshold counts as broken, non-monotone picks first)
    bp_m1 = bp[(bp["dataset"] == "D1") & (bp["model"] == "M1") & (bp["combo"] == "label")].iloc[0]
    assert bp_m1["reached"] == True
    assert bp_m1["breaking_level"] == 3
    assert np.isclose(bp_m1["f1_at_bp"], 0.90)
    assert bp_m1["seeds_below_at_bp"] == 2
    
    # Check noise_params JSON
    params = json.loads(bp_m1["noise_params"])
    assert "label_flip_rate" in params
    assert len(params) == 1
    assert np.isclose(params["label_flip_rate"], 3 * 0.1)
    
    # Check M2 (not reached, empty level)
    bp_m2 = bp[(bp["dataset"] == "D1") & (bp["model"] == "M2") & (bp["combo"] == "gaussian+missing")].iloc[0]
    assert bp_m2["reached"] == False
    assert pd.isna(bp_m2["breaking_level"])
    assert pd.isna(bp_m2["f1_at_bp"])
    assert pd.isna(bp_m2["seeds_below_at_bp"])
    assert bp_m2["noise_params"] == ""
    
    # Check noise_params for compound (if it had broken at level 2)
    s_df_mod = summary.copy()
    s_df_mod.loc[(s_df_mod["model"] == "M2") & (s_df_mod["level"] == 2), "f1_mean"] = 0.50
    bp2 = compute_breaking_points(df, s_df_mod, baselines)
    bp_m2_mod = bp2[(bp2["dataset"] == "D1") & (bp2["model"] == "M2") & (bp2["combo"] == "gaussian+missing")].iloc[0]
    assert bp_m2_mod["reached"] == True
    assert bp_m2_mod["breaking_level"] == 2
    params2 = json.loads(bp_m2_mod["noise_params"])
    assert "gaussian_k" in params2
    assert "missing_cell_rate" in params2
    assert "label_flip_rate" not in params2
    assert np.isclose(params2["gaussian_k"], 2 * 0.5)
    assert np.isclose(params2["missing_cell_rate"], 2 * 0.02)

