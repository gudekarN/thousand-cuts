import hashlib
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from app.core.config import DATASET_IDS, TEST_SIZE
from app.engine.custom_data import load_custom_dataset

_CACHE = {}

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load dataset by name, cache it, and return a copy.
    Returns: X (float64), y (int), meta (dict with n_samples, n_features, n_classes)
    """
    X, y, meta = _dataset_arrays(name)
    return X.copy(), y.copy(), meta.copy()


def _dataset_arrays(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if name not in DATASET_IDS:
        return load_custom_dataset(name)

    if name not in _CACHE:
        if name == "breast_cancer":
            data = load_breast_cancer()
        else:
            data = load_digits()

        X = data.data.astype(np.float64)
        y = data.target.astype(int)

        meta = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y))
        }
        _CACHE[name] = (X, y, meta)

    return _CACHE[name]


def dataset_metadata(name: str) -> dict:
    return _dataset_arrays(name)[2].copy()

def split(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified 70/30 train/test split.
    """
    X, y, _ = _dataset_arrays(name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test

def split_hashes(name: str, seed: int) -> tuple[str, str]:
    """
    Return SHA-256 hashes of train and test indices for reproducibility checks.
    """
    _, y, meta = _dataset_arrays(name)
    n_samples = meta["n_samples"]
    indices = np.arange(n_samples)

    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    # Cast to int64 to ensure cross-platform hash consistency
    train_idx = train_idx.astype(np.int64)
    test_idx = test_idx.astype(np.int64)

    train_hash = hashlib.sha256(train_idx.tobytes()).hexdigest()
    test_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()

    return train_hash, test_hash
