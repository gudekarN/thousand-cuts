import hashlib
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

_CACHE = {}

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load dataset by name, cache it, and return a copy.
    Returns: X (float64), y (int), meta (dict with n_samples, n_features, n_classes)
    """
    if name not in ["breast_cancer", "digits"]:
        raise ValueError(f"Unknown dataset: {name}")

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

    X_cached, y_cached, meta_cached = _CACHE[name]
    return X_cached.copy(), y_cached.copy(), meta_cached.copy()

def split(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified 70/30 train/test split.
    """
    X, y, _ = load_dataset(name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test

def split_hashes(name: str, seed: int) -> tuple[str, str]:
    """
    Return SHA-256 hashes of train and test indices for reproducibility checks.
    """
    _, y, meta = load_dataset(name)
    n_samples = meta["n_samples"]
    indices = np.arange(n_samples)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, stratify=y, random_state=seed
    )

    # Cast to int64 to ensure cross-platform hash consistency
    train_idx = train_idx.astype(np.int64)
    test_idx = test_idx.astype(np.int64)

    train_hash = hashlib.sha256(train_idx.tobytes()).hexdigest()
    test_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()

    return train_hash, test_hash
