"""JSON storage utilities with atomic write guarantees.

Ensures that interruptions or serialization errors never leave corrupted or partial files.
"""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Union


def write_json_atomic(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically write data as JSON to the specified path.

    Writes to a temporary file in the target directory, then replaces
    the destination file via atomic filesystem rename (os.replace).
    If an error occurs during serialization or writing, the temporary
    file is deleted and the target destination remains untouched.

    Args:
        path: Target file path.
        data: JSON-serializable Python data structure.
        indent: Indentation for formatted output (default 2).
    """
    file_path = Path(path).resolve()
    parent = file_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Use NamedTemporaryFile in the destination directory to ensure same filesystem for os.replace
    tmp_file = tempfile.NamedTemporaryFile(
        "w",
        dir=parent,
        delete=False,
        encoding="utf-8",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_file.name)

    try:
        json.dump(data, tmp_file, indent=indent)
        tmp_file.flush()
        # Must close the handle before os.replace, especially on Windows
        tmp_file.close()
        os.replace(tmp_path, file_path)
    except Exception:
        # Ensure file handle is closed before attempting removal
        if not tmp_file.closed:
            tmp_file.close()
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def read_json(path: Union[str, Path]) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data structure.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
