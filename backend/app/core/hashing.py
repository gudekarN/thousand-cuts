import json
import hashlib
from app.core.config import (
    METHODOLOGY_VERSION,
    MODEL_CONFIGS,
    NOISE_ORDER,
    TEST_SIZE,
    OUTLIER_SIGMA,
    BREAK_RATIO,
)
from app.core.levels import load_levels, DEFAULT_LEVELS_PATH

def compute_config_hash(levels_path=DEFAULT_LEVELS_PATH) -> str:
    levels_data = load_levels(levels_path)
    active_table_name = levels_data["active_table"]
    active_table_values = levels_data["tables"][active_table_name]
    
    config_dict = {
        "methodology": {
            "version": METHODOLOGY_VERSION,
            "test_size": TEST_SIZE,
            "outlier_sigma": OUTLIER_SIGMA,
            "break_ratio": BREAK_RATIO,
        },
        "active_table": active_table_values,
        "noise_order": list(NOISE_ORDER),
        "models": MODEL_CONFIGS
    }
    
    canonical_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

def verify_frozen_hash(levels_path=DEFAULT_LEVELS_PATH) -> None:
    levels_data = load_levels(levels_path)
    
    if not levels_data.get("frozen", False):
        raise ValueError("Levels are not frozen.")
        
    stored_hash = levels_data.get("config_hash")
    if not stored_hash:
        raise ValueError("Stored config_hash is missing.")
        
    computed_hash = compute_config_hash(levels_path)
    if stored_hash != computed_hash:
        raise ValueError(f"Config hash mismatch. Stored: {stored_hash}, Computed: {computed_hash}")
