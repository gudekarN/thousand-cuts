import json
import pytest
from app.core.hashing import compute_config_hash, verify_frozen_hash

def get_base_levels():
    return {
        "levels_version": "1.0.0",
        "active_table": "A",
        "frozen": False,
        "config_hash": None,
        "level_indices": [0, 1, 2, 3, 4, 5],
        "tables": {
            "A": {"dummy": [0, 1, 2, 3, 4, 5]},
            "S": {"dummy": [0, 2, 4, 6, 8, 10]},
            "M": {"dummy": [0, 1, 1, 1, 1, 1]}
        }
    }

def test_hash_is_stable_and_independent_of_key_order(tmp_path):
    d1 = get_base_levels()
    path1 = tmp_path / "levels1.json"
    with open(path1, "w") as f:
        json.dump(d1, f)
        
    d2 = get_base_levels()
    d2["tables"] = {"S": d1["tables"]["S"], "A": d1["tables"]["A"], "M": d1["tables"]["M"]}
    path2 = tmp_path / "levels2.json"
    with open(path2, "w") as f:
        json.dump(d2, f)
        
    h1 = compute_config_hash(path1)
    h2 = compute_config_hash(path2)
    assert h1 == h2

def test_hash_changes_if_active_table_changes(tmp_path):
    d1 = get_base_levels()
    path1 = tmp_path / "levels1.json"
    with open(path1, "w") as f:
        json.dump(d1, f)
        
    d2 = get_base_levels()
    d2["active_table"] = "S"
    path2 = tmp_path / "levels2.json"
    with open(path2, "w") as f:
        json.dump(d2, f)
        
    h1 = compute_config_hash(path1)
    h2 = compute_config_hash(path2)
    assert h1 != h2

def test_verify_raises_when_frozen_false(tmp_path):
    d1 = get_base_levels()
    path1 = tmp_path / "levels.json"
    with open(path1, "w") as f:
        json.dump(d1, f)
        
    with pytest.raises(ValueError, match="Levels are not frozen"):
        verify_frozen_hash(path1)

def test_verify_raises_on_mismatch(tmp_path):
    d1 = get_base_levels()
    d1["frozen"] = True
    d1["config_hash"] = "wrong_hash"
    path1 = tmp_path / "levels.json"
    with open(path1, "w") as f:
        json.dump(d1, f)
        
    with pytest.raises(ValueError, match="Config hash mismatch"):
        verify_frozen_hash(path1)

def test_verify_passes_when_frozen_and_hash_equal(tmp_path):
    d1 = get_base_levels()
    path1 = tmp_path / "levels.json"
    with open(path1, "w") as f:
        json.dump(d1, f)
        
    h = compute_config_hash(path1)
    
    d1["frozen"] = True
    d1["config_hash"] = h
    with open(path1, "w") as f:
        json.dump(d1, f)
        
    # Should not raise
    verify_frozen_hash(path1)
