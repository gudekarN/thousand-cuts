"""Custom-only datasets shared by the API and the experiment worker."""

import csv
import hashlib
import json
import re
import tempfile
import warnings
from functools import lru_cache
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype

from app.core.config import TEST_SIZE
from app.storage.paths import get_results_root

COVERTYPE_ID = "covertype"
MAX_CSV_BYTES = 128 * 1024 * 1024
_UPLOAD_ID = re.compile(r"csv-[a-f0-9]{64}")
_COVERTYPE = {
    "id": COVERTYPE_ID,
    "label": "UCI Covertype",
    "n_samples": 581_012,
    "n_features": 54,
    "n_classes": 7,
    "source_url": "https://archive.ics.uci.edu/dataset/31/covertype",
}


def dataset_dir() -> Path:
    return get_results_root() / "datasets"


def custom_dataset_info(dataset: str) -> dict:
    if dataset == COVERTYPE_ID:
        return _COVERTYPE.copy()
    if _UPLOAD_ID.fullmatch(dataset):
        path = dataset_dir() / f"{dataset}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                return json.loads(archive["metadata"].item())
    raise ValueError(f"Unknown dataset '{dataset}'. Select a dataset or upload a CSV.")


def list_custom_datasets() -> list[dict]:
    uploads = sorted(dataset_dir().glob("csv-*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_COVERTYPE.copy(), *(custom_dataset_info(path.stem) for path in uploads)]


def register_csv(path: Path, filename: str, target: str) -> dict:
    """Validate a CSV and atomically publish its arrays for the worker."""
    with path.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    dataset = "csv-" + hashlib.sha256(f"{sha256}:{target}".encode()).hexdigest()
    archive_path = dataset_dir() / f"{dataset}.npz"
    if archive_path.exists():
        return custom_dataset_info(dataset)

    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if len(header) < 2 or "" in header or len(set(header)) != len(header):
            raise ValueError("CSV needs unique, non-empty column names and at least two columns.")
        if target not in header:
            raise ValueError("The target column must exactly match a CSV column name.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(path, dtype={target: str}, index_col=False, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.ParserWarning, UnicodeError, csv.Error) as exc:
        raise ValueError("Upload a UTF-8, comma-separated CSV with a header and consistent columns.") from exc

    labels = frame.pop(target).str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("Every row needs a target label.")
    classes, y, counts = np.unique(labels.to_numpy(), return_inverse=True, return_counts=True)
    n_test = ceil(len(frame) * TEST_SIZE)
    if len(classes) < 2 or counts.min() < 2 or min(n_test, len(frame) - n_test) < len(classes):
        raise ValueError("The target needs at least two classes with enough rows for a stratified 70/30 split.")
    try:
        X = frame.to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Feature columns must contain finite numeric values. Resolve existing missing values before uploading.") from exc
    if not np.isfinite(X).all():
        raise ValueError("Feature columns must contain finite numeric values.")

    metadata = {
        "id": dataset,
        "label": filename.replace("\\", "/").rsplit("/", 1)[-1].strip()[:150] or "Uploaded CSV",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(classes),
        "target_column": target,
        "sha256": sha256,
    }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive_path.parent, suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        np.savez(tmp, X=X, y=y, metadata=json.dumps(metadata))
        tmp.replace(archive_path)
    finally:
        tmp.unlink(missing_ok=True)
    return custom_dataset_info(dataset)


@lru_cache(maxsize=1)
def _load_custom_dataset(root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == COVERTYPE_ID:
        data = fetch_covtype(data_home=root / "sklearn")
        return np.asarray(data.data, dtype=np.float64), np.asarray(data.target, dtype=np.int64)
    with np.load(root / f"{dataset}.npz", allow_pickle=False) as archive:
        return archive["X"], archive["y"]


def load_custom_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray, dict]:
    info = custom_dataset_info(dataset)
    X, y = _load_custom_dataset(dataset_dir(), dataset)
    return X, y, info
