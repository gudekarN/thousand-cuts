"""Unit tests for missing-value feature-noise injector (Task 1.6).

Verifies:
- on the real breast_cancer training matrix, NaN fraction is within 2 percentage points of target for p = 0.1 and 0.5 (T1 rate accuracy)
- deterministic for the same rng seed and level
- p=0 returns an identical copy and no NaN
- input matrix is never modified in place
- reported cell count equals the number of NaN
- p=1 sets all cells to NaN
- validation checks for p outside [0, 1] and non-2D arrays
- applies across all columns
"""

import numpy as np
import pytest

from app.engine.noise import (
    inject_missing,
    make_rng,
)
from app.engine.data import split


class TestInjectMissing:
    """Tests for inject_missing injector."""

    def test_p_zero_returns_identical_copy_and_no_nan(self):
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        rng = make_rng(seed=42, noise="missing", level=0)

        X_noisy, n_cells = inject_missing(X, p=0.0, rng=rng)

        assert n_cells == 0
        assert np.isnan(X_noisy).sum() == 0
        np.testing.assert_array_equal(X_noisy, X)
        # Verify it is an independent copy
        X_noisy[0, 0] = 999.0
        assert X[0, 0] == 1.0

    def test_input_not_modified_in_place(self):
        X = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
            [50.0, 60.0],
        ])
        X_original = X.copy()
        rng = make_rng(seed=42, noise="missing", level=2)

        inject_missing(X, p=0.5, rng=rng)
        np.testing.assert_array_equal(X, X_original)

    def test_reported_cell_count_equals_number_of_nan(self):
        X = np.ones((50, 20))
        rng = make_rng(seed=123, noise="missing", level=3)

        X_noisy, n_cells = inject_missing(X, p=0.25, rng=rng)

        actual_nans = int(np.isnan(X_noisy).sum())
        assert n_cells == actual_nans
        assert n_cells > 0

    def test_breast_cancer_target_rate_accuracy(self):
        """Technical test T1: on real breast_cancer train matrix, NaN fraction is within +/-2 pp."""
        X_train, _, _, _ = split("breast_cancer", seed=0)
        total_cells = X_train.size  # 398 * 30 = 11,940

        # Test p = 0.1
        rng_01 = make_rng(seed=0, noise="missing", level=1)
        X_noisy_01, n_cells_01 = inject_missing(X_train, p=0.1, rng=rng_01)
        fraction_01 = n_cells_01 / total_cells
        assert abs(fraction_01 - 0.1) <= 0.02, (
            f"p=0.1 yielded fraction {fraction_01:.4f}, expected within [0.08, 0.12]"
        )
        assert np.isnan(X_noisy_01).sum() == n_cells_01

        # Test p = 0.5
        rng_05 = make_rng(seed=0, noise="missing", level=5)
        X_noisy_05, n_cells_05 = inject_missing(X_train, p=0.5, rng=rng_05)
        fraction_05 = n_cells_05 / total_cells
        assert abs(fraction_05 - 0.5) <= 0.02, (
            f"p=0.5 yielded fraction {fraction_05:.4f}, expected within [0.48, 0.52]"
        )
        assert np.isnan(X_noisy_05).sum() == n_cells_05

    def test_deterministic_same_seed_and_level(self):
        X = np.ones((40, 10))

        rng1 = make_rng(seed=77, noise="missing", level=2)
        rng2 = make_rng(seed=77, noise="missing", level=2)

        X_noisy1, n_cells1 = inject_missing(X, p=0.3, rng=rng1)
        X_noisy2, n_cells2 = inject_missing(X, p=0.3, rng=rng2)

        assert n_cells1 == n_cells2
        assert np.array_equal(X_noisy1, X_noisy2, equal_nan=True)

    def test_different_level_gives_different_mask(self):
        X = np.ones((40, 10))

        rng_lvl1 = make_rng(seed=77, noise="missing", level=1)
        rng_lvl2 = make_rng(seed=77, noise="missing", level=2)

        X_noisy1, _ = inject_missing(X, p=0.3, rng=rng_lvl1)
        X_noisy2, _ = inject_missing(X, p=0.3, rng=rng_lvl2)

        mask1 = np.isnan(X_noisy1)
        mask2 = np.isnan(X_noisy2)
        assert not np.array_equal(mask1, mask2)

    def test_p_one_sets_all_cells_to_nan(self):
        X = np.ones((25, 8))
        rng = make_rng(seed=999, noise="missing", level=5)

        X_noisy, n_cells = inject_missing(X, p=1.0, rng=rng)

        assert n_cells == X.size
        assert np.all(np.isnan(X_noisy))

    def test_applies_to_all_columns(self):
        """Verify that missing values are not restricted to specific columns."""
        X = np.ones((200, 10))
        rng = make_rng(seed=42, noise="missing", level=3)

        X_noisy, _ = inject_missing(X, p=0.2, rng=rng)

        nan_mask = np.isnan(X_noisy)
        cols_with_nan = np.where(nan_mask.sum(axis=0) > 0)[0]
        # In a 200-row matrix with p=0.2, every column should have at least one NaN
        assert len(cols_with_nan) == 10

    def test_validation_errors(self):
        X = np.ones((10, 5))
        rng = make_rng(seed=0, noise="missing", level=1)

        with pytest.raises(ValueError, match="p must be in"):
            inject_missing(X, p=-0.05, rng=rng)

        with pytest.raises(ValueError, match="p must be in"):
            inject_missing(X, p=1.05, rng=rng)

        with pytest.raises(ValueError, match="must be a 2D array"):
            inject_missing(np.ones(10), p=0.2, rng=rng)

        with pytest.raises(ValueError, match="must be a 2D array"):
            inject_missing(np.ones((2, 3, 4)), p=0.2, rng=rng)
