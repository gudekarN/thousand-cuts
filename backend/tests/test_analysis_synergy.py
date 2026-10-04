import pytest
import pandas as pd
import numpy as np
import logging
from app.engine.analysis import compute_synergy

def test_compute_synergy(caplog):
    # Baselines
    baselines_data = [
        {"dataset": "D1", "model": "M1", "f1_mean": 1.0},
    ]
    baselines = pd.DataFrame(baselines_data)
    
    # Summary
    # clean: 1.0
    # single A: 0.9 (drop 0.1)
    # single B: 0.8 (drop 0.2)
    # single C: 0.4 (drop 0.6)
    # compound A+B level 1: f1 = 0.5 (drop = 0.5). sum_single_drops = 0.3. Synergy = 0.5 - 0.3 = +0.2 (Positive)
    # compound A+B level 2: f1 = 0.8 (drop = 0.2). sum_single_drops = 0.3. Synergy = 0.2 - 0.3 = -0.1 (Negative)
    # compound B+C level 1: f1 = 0.2 (drop = 0.8). sum_single_drops = 0.2 + 0.6 = 0.8. f1_floor_flag = False (0.8 < 1.0)
    # compound B+C level 2: single B is 0.2 (drop 0.8), single C is 0.1 (drop 0.9). sum_single_drops = 1.7 >= 1.0 (True)
    # compound A+D level 1: D is missing, should be skipped and logged.
    
    summary_data = [
        {"dataset": "D1", "model": "M1", "combo": "clean", "level": 0, "f1_mean": 1.0},
        
        {"dataset": "D1", "model": "M1", "combo": "A", "level": 1, "f1_mean": 0.9},
        {"dataset": "D1", "model": "M1", "combo": "B", "level": 1, "f1_mean": 0.8},
        {"dataset": "D1", "model": "M1", "combo": "C", "level": 1, "f1_mean": 0.4},
        
        {"dataset": "D1", "model": "M1", "combo": "A", "level": 2, "f1_mean": 0.9},
        {"dataset": "D1", "model": "M1", "combo": "B", "level": 2, "f1_mean": 0.2},
        {"dataset": "D1", "model": "M1", "combo": "C", "level": 2, "f1_mean": 0.1},
        
        {"dataset": "D1", "model": "M1", "combo": "A+B", "level": 1, "f1_mean": 0.5},
        {"dataset": "D1", "model": "M1", "combo": "A+B", "level": 2, "f1_mean": 0.8},
        
        {"dataset": "D1", "model": "M1", "combo": "B+C", "level": 1, "f1_mean": 0.2},
        {"dataset": "D1", "model": "M1", "combo": "B+C", "level": 2, "f1_mean": 0.1},
        
        {"dataset": "D1", "model": "M1", "combo": "A+D", "level": 1, "f1_mean": 0.5},
    ]
    summary = pd.DataFrame(summary_data)
    
    with caplog.at_level(logging.WARNING):
        synergy_df = compute_synergy(summary, baselines)
        
    assert len(synergy_df) == 4
    
    # Check positive synergy
    row_ab1 = synergy_df[(synergy_df["combo"] == "A+B") & (synergy_df["level"] == 1)].iloc[0]
    assert np.isclose(row_ab1["synergy"], 0.2)
    assert not row_ab1["f1_floor_flag"]
    
    # Check negative synergy
    row_ab2 = synergy_df[(synergy_df["combo"] == "A+B") & (synergy_df["level"] == 2)].iloc[0]
    assert np.isclose(row_ab2["synergy"], -0.7)
    assert not row_ab2["f1_floor_flag"]
    
    # Check floor flag True
    row_bc2 = synergy_df[(synergy_df["combo"] == "B+C") & (synergy_df["level"] == 2)].iloc[0]
    assert row_bc2["f1_floor_flag"] == True
    
    # Check missing component skipped
    assert len(synergy_df[synergy_df["combo"] == "A+D"]) == 0
    assert "Missing component 'D'" in caplog.text
