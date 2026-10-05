import pandas as pd
from pathlib import Path
import pytest
import os
from app.engine.figures import generate_all_figures

def test_generate_all_figures(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    
    # Tiny synthetic summary
    summary = pd.DataFrame({
        "dataset": ["test_data"] * 4,
        "model": ["logreg", "svm_rbf", "logreg", "svm_rbf"],
        "combo": ["label", "label", "clean", "clean"],
        "level": [1, 1, 0, 0],
        "f1_mean": [0.8, 0.7, 0.9, 0.85],
        "f1_std": [0.01, 0.02, 0.01, 0.02],
        "acc_mean": [0.8, 0.7, 0.9, 0.85],
        "acc_std": [0.01, 0.02, 0.01, 0.02]
    })
    
    baselines = pd.DataFrame({
        "dataset": ["test_data"] * 2,
        "model": ["logreg", "svm_rbf"],
        "f1_mean": [0.9, 0.85],
        "threshold_f1": [0.81, 0.765]
    })
    
    breaking_points = pd.DataFrame({
        "dataset": ["test_data"] * 2,
        "model": ["logreg", "svm_rbf"],
        "combo": ["label", "label"],
        "breaking_level": [1, 1],
        "reached": [True, True],
        "f1_at_bp": [0.8, 0.7]
    })
    
    synergy = pd.DataFrame({
        "dataset": ["test_data"] * 2,
        "model": ["logreg", "svm_rbf"],
        "combo": ["label+gaussian"] * 2,
        "level": [5, 5],
        "synergy": [-0.05, 0.02]
    })
    
    summary.to_csv(out_dir / "summary.csv", index=False)
    baselines.to_csv(out_dir / "baselines.csv", index=False)
    breaking_points.to_csv(out_dir / "breaking_points.csv", index=False)
    synergy.to_csv(out_dir / "synergy.csv", index=False)
    
    generate_all_figures(out_dir)
    
    figures_dir = out_dir / "figures"
    assert figures_dir.exists()
    
    expected_files = [
        "test_data_f1_vs_level.png",
        "test_data_acc_vs_level.png",
        "test_data_heatmap.png",
        "breaking_points_table.png",
        "test_data_synergy.png"
    ]
    
    for filename in expected_files:
        filepath = figures_dir / filename
        assert filepath.exists()
        assert filepath.stat().st_size > 0
