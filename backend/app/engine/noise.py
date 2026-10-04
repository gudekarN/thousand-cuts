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
