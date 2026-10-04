import json
import pytest
from app.core.levels import load_levels, get_params, is_frozen

def test_load_levels_valid(tmp_path):
    levels_data = {
        "levels_version": "1.0.0",
        "active_table": "A",
        "frozen": False,
        "level_indices": [0, 1, 2, 3, 4, 5],
        "tables": {
            "A": {"label_flip_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "S": {"label_flip_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "M": {"label_flip_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        }
    }
    path = tmp_path / "levels.json"
    with open(path, "w") as f:
        json.dump(levels_data, f)
        
    data = load_levels(path)
    assert data["active_table"] == "A"
    assert not is_frozen(path)

def test_load_levels_invalid_structure(tmp_path):
    levels_data = {
        "level_indices": [0, 1, 2, 3, 4],
        "tables": {"A": {}}
    }
    path = tmp_path / "levels.json"
    with open(path, "w") as f:
        json.dump(levels_data, f)
        
    with pytest.raises(ValueError, match="must have 6 levels"):
        load_levels(path)
        
def test_load_levels_nonzero_level0(tmp_path):
    levels_data = {
        "level_indices": [0, 1, 2, 3, 4, 5],
        "tables": {
            "A": {"label_flip_rate": [0.1, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "S": {"label_flip_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "M": {"label_flip_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        }
    }
    path = tmp_path / "levels.json"
    with open(path, "w") as f:
        json.dump(levels_data, f)
        
    with pytest.raises(ValueError, match="is not 0"):
        load_levels(path)

def test_get_params_table_A_level_3(tmp_path):
    levels_data = {
        "level_indices": [0, 1, 2, 3, 4, 5],
        "active_table": "A",
        "tables": {
            "A": {
                "label_flip_rate": [0, 1, 2, 3, 4, 5],
                "gaussian_k": [0, 1, 2, 3, 4, 5],
                "outlier_cell_rate": [0, 1, 2, 3, 4, 5],
                "missing_cell_rate": [0, 1, 2, 3, 4, 5]
            },
            "S": {}, "M": {}
        }
    }
    path = tmp_path / "levels.json"
    with open(path, "w") as f:
        json.dump(levels_data, f)
        
    params = get_params(3, path)
    assert params == {
        "label_flip_rate": 3,
        "gaussian_k": 3,
        "outlier_cell_rate": 3,
        "missing_cell_rate": 3
    }
    
def test_get_params_invalid_level(tmp_path):
    levels_data = {
        "level_indices": [0, 1, 2, 3, 4, 5],
        "active_table": "A",
        "tables": {"A": {}, "S": {}, "M": {}}
    }
    path = tmp_path / "levels.json"
    with open(path, "w") as f:
        json.dump(levels_data, f)
        
    with pytest.raises(ValueError, match="Invalid level: 6"):
        get_params(6, path)
