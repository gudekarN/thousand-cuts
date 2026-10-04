"""CSV storage utilities for raw experiment results.

Guarantees exact column ordering per Architecture 6.2, atomic flushing,
and completed key extraction for resumable runs.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

# Exact column sequence per Architecture.md Section 6.2
RAW_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "run_type",
    "stage",
    "dataset",
    "model",
    "seed",
    "combo",
    "n_noises",
    "level",
    "macro_f1",
    "accuracy",
    "fit_time_s",
    "n_train",
    "n_test",
    "noise_stats",
    "timestamp_utc",
    "config_hash",
    "methodology_version",
    "levels_version",
    "levels_frozen",
    "status",
    "error_msg",
)


def append_row(path: Union[str, Path], row: Dict[str, Any]) -> None:
    """Append a single row to raw_results.csv.

    Creates parent directories and CSV header if the file does not exist or is empty.
    Flushes the file immediately after writing.

    Args:
        path: Path to the target CSV file.
        row: Dictionary mapping column names to values.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = file_path.exists() and file_path.stat().st_size > 0

    # Prepare row dictionary with serializations if needed
    row_to_write = dict(row)
    if isinstance(row_to_write.get("noise_stats"), (dict, list)):
        row_to_write["noise_stats"] = json.dumps(row_to_write["noise_stats"])

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_to_write)
        f.flush()


def read_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Read all rows from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        List of row dictionaries (all values as strings), or empty list if file doesn't exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def completed_keys(path: Union[str, Path]) -> Set[Tuple[str, str, str, int, int]]:
    """Extract set of completed execution keys from raw_results.csv.

    Filters for rows where status == 'ok'.

    Args:
        path: Path to the raw_results.csv file.

    Returns:
        Set of tuples: (dataset, model, combo, level, seed)
    """
    rows = read_rows(path)
    keys: Set[Tuple[str, str, str, int, int]] = set()

    for row in rows:
        if row.get("status") == "ok":
            try:
                dataset = str(row["dataset"])
                model = str(row["model"])
                combo = str(row["combo"])
                level = int(row["level"])
                seed = int(row["seed"])
                keys.add((dataset, model, combo, level, seed))
            except (KeyError, ValueError):
                continue

    return keys
