import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_LEVELS_PATH = Path(__file__).parent.parent.parent / "configs" / "levels.json"

def load_levels(path: str | Path = DEFAULT_LEVELS_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "level_indices" not in data or data["level_indices"] != [0, 1, 2, 3, 4, 5]:
        raise ValueError("Invalid levels: must have 6 levels 0-5")
    
    if "tables" not in data:
        raise ValueError("Invalid levels: missing 'tables'")
    
    for table_name in ["A", "S", "M"]:
        if table_name not in data["tables"]:
            raise ValueError(f"Invalid levels: missing table {table_name}")
            
    for table_name, table in data["tables"].items():
        for param_name, values in table.items():
            if values[0] != 0:
                raise ValueError(f"Invalid levels: level 0 for {param_name} in table {table_name} is not 0")
                
    return data

def get_params(level: int, path: str | Path = DEFAULT_LEVELS_PATH) -> Dict[str, float]:
    data = load_levels(path)
    if level not in data["level_indices"]:
        raise ValueError(f"Invalid level: {level}")
        
    active_table_name = data["active_table"]
    active_table = data["tables"][active_table_name]
    
    return {
        "label_flip_rate": active_table["label_flip_rate"][level],
        "gaussian_k": active_table["gaussian_k"][level],
        "outlier_cell_rate": active_table["outlier_cell_rate"][level],
        "missing_cell_rate": active_table["missing_cell_rate"][level]
    }

def is_frozen(path: str | Path = DEFAULT_LEVELS_PATH) -> bool:
    data = load_levels(path)
    return data.get("frozen", False)
