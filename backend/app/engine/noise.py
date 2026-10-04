"""Noise injection engine and common noise helpers for Death by a Thousand Cuts.

Pure numpy noise functions only. Never modify inputs in place.
RNG is always deterministic: np.random.default_rng(np.random.SeedSequence([seed, noise_id, level])).
No Python hash(), no global numpy random state.
"""

from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np

from app.core.config import (
    NOISE_IDS,
    NOISE_LABEL,
    NOISE_GAUSSIAN,
    NOISE_OUTLIERS,
    NOISE_MISSING,
)

__all__ = [
    "NOISE_IDS",
    "NOISE_LABEL",
    "NOISE_GAUSSIAN",
    "NOISE_OUTLIERS",
    "NOISE_MISSING",
    "make_rng",
    "clean_stats",
    "inject_label",
    "inject_gaussian",
    "inject_outliers",
    "inject_missing",
]


def make_rng(
    seed: int,
    noise: Union[str, int],
    level: int,
) -> np.random.Generator:
    """Create a deterministic NumPy Generator for a noise injection step.

    Uses np.random.default_rng(np.random.SeedSequence([seed, noise_id, level])).
    Strictly satisfies rules.md R-L4 and Architecture.md Section 4.5.

    Args:
        seed: Experiment seed integer.
        noise: Noise identifier, either string ("label", "gaussian", "outliers", "missing")
               or integer ID (1, 2, 3, 4).
        level: Perturbation severity level (0-5).

    Returns:
        np.random.Generator instance.
    """
    if isinstance(noise, str):
        if noise not in NOISE_IDS:
            raise ValueError(
                f"Unknown noise type: '{noise}'. Expected one of {list(NOISE_IDS.keys())}"
            )
        noise_id = NOISE_IDS[noise]
    elif isinstance(noise, (int, np.integer)):
        noise_id = int(noise)
        if noise_id not in NOISE_IDS.values():
            raise ValueError(
                f"Unknown noise id: {noise_id}. Expected one of {list(NOISE_IDS.values())}"
            )
    else:
        raise TypeError(f"noise must be str or int, got {type(noise).__name__}")

    return np.random.default_rng(np.random.SeedSequence([seed, noise_id, level]))


def clean_stats(
    X_train: np.ndarray,
    ddof: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean and standard deviation from the clean training set.

    Used strictly to size Gaussian and outlier noise (Architecture.md 4.5, rules.md R-L2).
    Never used as fitted Pipeline statistics.

    Args:
        X_train: 2D array of features of shape (n_samples, n_features).
        ddof: Delta degrees of freedom for sample standard deviation (default 1).

    Returns:
        Tuple of (mean, std) where each is a 1D float64 array of shape (n_features,).
    """
    X_arr = np.asarray(X_train, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X_train must be a 2D array, got shape {X_arr.shape}")

    mean = np.mean(X_arr, axis=0)
    std = np.std(X_arr, axis=0, ddof=ddof)
    return mean, std


def inject_label(
    y: np.ndarray,
    rate: float,
    rng: np.random.Generator,
    classes: Optional[Union[Sequence[Any], np.ndarray]] = None,
) -> Tuple[np.ndarray, int]:
    """Inject label noise by flipping labels to a different class.

    n_flip = round(rate * n)
    Chooses indices without replacement.
    New class = classes[(pos + rng.integers(1, n_classes)) % n_classes],
    guaranteeing that every flipped label is strictly different from the original.

    Never modifies input array in place.

    Args:
        y: 1D array of class labels.
        rate: Label flip rate in [0.0, 1.0].
        rng: NumPy Generator instance.
        classes: Optional sequence or array of unique classes. If None, uses np.unique(y).

    Returns:
        Tuple of (y_noisy, n_flipped) where y_noisy is a copy of y with flipped labels,
        and n_flipped is the integer count of flipped labels.
    """
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"rate must be in [0.0, 1.0], got {rate}")

    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y_arr.shape}")

    n = len(y_arr)
    n_flipped = int(round(rate * n))
    if n_flipped == 0:
        return y_arr.copy(), 0

    if n_flipped > n:
        n_flipped = n

    classes_arr = np.unique(y_arr) if classes is None else np.asarray(classes)
    n_classes = len(classes_arr)
    if n_classes < 2:
        raise ValueError(
            f"Cannot inject label noise with fewer than 2 classes (got {n_classes})"
        )

    class_to_pos = {c: i for i, c in enumerate(classes_arr)}
    y_noisy = y_arr.copy()
    flip_indices = rng.choice(n, size=n_flipped, replace=False)

    for idx in flip_indices:
        orig = y_noisy[idx]
        pos = class_to_pos[orig]
        offset = int(rng.integers(1, n_classes))
        y_noisy[idx] = classes_arr[(pos + offset) % n_classes]

    return y_noisy, n_flipped


def inject_gaussian(
    X: np.ndarray,
    k: float,
    std_clean: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """Inject Gaussian noise into feature matrix.

    Formula: X + rng.normal(0, 1, X.shape) * (k * std_clean)
    Columns with std_clean == 0 stay unchanged.
    k = 0 returns an identical copy.
    Never modifies input array in place.

    Args:
        X: 2D feature matrix of shape (n_samples, n_features).
        k: Gaussian noise severity multiplier (>= 0).
        std_clean: Per-column standard deviations from clean train set, shape (n_features,).
        rng: NumPy Generator instance.

    Returns:
        Tuple of (X_noisy, n_cells_perturbed).
        When k == 0, n_cells_perturbed is 0.
        When k > 0, n_cells_perturbed equals rows * non-zero-std columns.
    """
    if k < 0.0:
        raise ValueError(f"k must be non-negative, got {k}")

    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")

    std_arr = np.asarray(std_clean, dtype=np.float64)
    if std_arr.ndim != 1 or len(std_arr) != X_arr.shape[1]:
        raise ValueError(
            f"std_clean must be 1D with length matching X features ({X_arr.shape[1]}), got shape {std_arr.shape}"
        )

    if k == 0.0:
        return X_arr.copy(), 0

    nonzero_mask = std_arr > 0.0
    n_nonzero_cols = int(np.sum(nonzero_mask))
    n_cells_perturbed = int(X_arr.shape[0] * n_nonzero_cols)

    # Standard normal draw of shape X.shape scaled by (k * std_clean)
    noise = rng.normal(0.0, 1.0, size=X_arr.shape) * (k * std_arr)
    X_noisy = X_arr + noise

    # Ensure zero-std columns are strictly unchanged
    if not np.all(nonzero_mask):
        zero_cols = ~nonzero_mask
        X_noisy[:, zero_cols] = X_arr[:, zero_cols]

    return X_noisy, n_cells_perturbed


def inject_outliers(
    X: np.ndarray,
    rate: float,
    mean_clean: np.ndarray,
    std_clean: np.ndarray,
    rng: np.random.Generator,
    sigma: int = 5,
) -> Tuple[np.ndarray, int]:
    """Inject outlier noise into eligible cells.

    Eligible cells = columns with std_clean > 0.
    Chooses round(rate * eligible_cells) without replacement.
    Sets each to mean_clean + sign * sigma * std_clean, sign random +1/-1.
    Rate 0 returns identical copy.
    Never modifies input array in place.

    Args:
        X: 2D feature matrix of shape (n_samples, n_features).
        rate: Outlier rate in [0.0, 1.0].
        mean_clean: Per-column means from clean train set, shape (n_features,).
        std_clean: Per-column standard deviations from clean train set, shape (n_features,).
        rng: NumPy Generator instance.
        sigma: Standard deviation multiplier (default 5).

    Returns:
        Tuple of (X_noisy, n_cells_replaced).
    """
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"rate must be in [0.0, 1.0], got {rate}")

    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")

    mean_arr = np.asarray(mean_clean, dtype=np.float64)
    std_arr = np.asarray(std_clean, dtype=np.float64)

    n_samples, n_features = X_arr.shape
    if mean_arr.ndim != 1 or len(mean_arr) != n_features:
        raise ValueError("mean_clean must be 1D with length matching X features")
    if std_arr.ndim != 1 or len(std_arr) != n_features:
        raise ValueError("std_clean must be 1D with length matching X features")

    if rate == 0.0:
        return X_arr.copy(), 0

    valid_cols = np.where(std_arr > 0.0)[0]
    n_valid_cols = len(valid_cols)
    total_eligible = n_samples * n_valid_cols

    if total_eligible == 0:
        return X_arr.copy(), 0

    n_cells = int(round(rate * total_eligible))
    if n_cells == 0:
        return X_arr.copy(), 0
    if n_cells > total_eligible:
        n_cells = total_eligible

    # Sample exactly n_cells indices without replacement from 0 to total_eligible - 1
    flat_indices = rng.choice(total_eligible, size=n_cells, replace=False)

    row_indices = flat_indices // n_valid_cols
    valid_col_indices = flat_indices % n_valid_cols
    col_indices = valid_cols[valid_col_indices]

    signs = rng.choice([-1, 1], size=n_cells)
    outlier_values = mean_arr[col_indices] + signs * sigma * std_arr[col_indices]

    X_noisy = X_arr.copy()
    X_noisy[row_indices, col_indices] = outlier_values

    return X_noisy, n_cells


def inject_missing(
    X: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """Inject missing values (NaN) into feature matrix.

    Formula: mask = rng.random(X.shape) < p; X[mask] = NaN on a copy.
    Applies to all columns.
    p = 0 returns an identical copy and no NaN.
    Never modifies input array in place.

    Args:
        X: 2D feature matrix of shape (n_samples, n_features).
        p: Missingness probability in [0.0, 1.0].
        rng: NumPy Generator instance.

    Returns:
        Tuple of (X_noisy, n_cells).
        X_noisy is a copy of X with masked cells set to NaN.
        n_cells is the integer count of NaN cells injected.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in [0.0, 1.0], got {p}")

    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")

    if p == 0.0:
        return X_arr.copy(), 0

    mask = rng.random(X_arr.shape) < p
    n_cells = int(np.sum(mask))

    X_noisy = X_arr.copy()
    X_noisy[mask] = np.nan

    return X_noisy, n_cells

