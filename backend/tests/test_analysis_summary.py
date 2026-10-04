import pytest
import pandas as pd
import numpy as np
from app.engine.analysis import compute_baselines, compute_summary

@pytest.fixture
def mock_df():
    """
    Hand-made fixture with known numbers.
    Dataset D1, Model M1.
    2 seeds (0, 1) for clean.
    1 seed (0) for noisy (one-seed case).
    1 error row.
    """
    data = [
        # Clean rows (combo="clean", level=0)
        {"dataset": "D1", "model": "M1", "combo": "clean", "level": 0, "seed": 0, "status": "ok", 
         "macro_f1": 0.8, "accuracy": 0.85, "fit_time_s": 1.0},
        {"dataset": "D1", "model": "M1", "combo": "clean", "level": 0, "seed": 1, "status": "ok", 
         "macro_f1": 0.9, "accuracy": 0.95, "fit_time_s": 2.0},
        
        # Noisy rows (combo="label", level=1) - single seed
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 1, "seed": 0, "status": "ok", 
         "macro_f1": 0.6, "accuracy": 0.70, "fit_time_s": 1.5},
        
        # Error row (status="error")
        {"dataset": "D1", "model": "M1", "combo": "label", "level": 1, "seed": 1, "status": "error", 
         "macro_f1": np.nan, "accuracy": np.nan, "fit_time_s": np.nan},
         
        # Another combo
        {"dataset": "D1", "model": "M1", "combo": "missing", "level": 2, "seed": 0, "status": "ok", 
         "macro_f1": 0.4, "accuracy": 0.5, "fit_time_s": 1.0},
    ]
    return pd.DataFrame(data)

def test_compute_baselines(mock_df):
    baselines, n_errors = compute_baselines(mock_df)
    
    assert n_errors == 1
    assert len(baselines) == 1
    
    row = baselines.iloc[0]
    assert row["dataset"] == "D1"
    assert row["model"] == "M1"
    assert np.isclose(row["f1_mean"], 0.85)  # (0.8 + 0.9)/2
    assert np.isclose(row["f1_std"], np.std([0.8, 0.9], ddof=1))
    assert np.isclose(row["acc_mean"], 0.90) # (0.85 + 0.95)/2
    assert np.isclose(row["threshold_f1"], 0.90 * 0.85)

def test_compute_summary(mock_df):
    baselines, _ = compute_baselines(mock_df)
    summary, n_errors = compute_summary(mock_df, baselines)
    
    assert n_errors == 1
    
    # We expect 5 rows in summary:
    # (D1, M1, clean, 0) - from original clean
    # (D1, M1, label, 0) - virtual from clean
    # (D1, M1, label, 1) - original noisy
    # (D1, M1, missing, 0) - virtual from clean
    # (D1, M1, missing, 2) - original noisy
    assert len(summary) == 5
    
    # Check virtual clean for "label"
    v_clean = summary[(summary["combo"] == "label") & (summary["level"] == 0)].iloc[0]
    assert v_clean["n_seeds"] == 2
    assert np.isclose(v_clean["f1_mean"], 0.85)
    assert np.isclose(v_clean["rel_f1"], 1.0)
    assert np.isclose(v_clean["drop_abs"], 0.0)
    assert np.isclose(v_clean["drop_rel"], 0.0)
    
    # Check noisy row for "label" (one seed case)
    noisy = summary[(summary["combo"] == "label") & (summary["level"] == 1)].iloc[0]
    assert noisy["n_seeds"] == 1
    assert np.isclose(noisy["f1_mean"], 0.6)
    assert np.isnan(noisy["f1_std"])  # One seed => NaN std
    assert np.isclose(noisy["rel_f1"], 0.6 / 0.85)
    assert np.isclose(noisy["drop_abs"], 0.85 - 0.6)
    assert np.isclose(noisy["drop_rel"], (0.85 - 0.6) / 0.85)

