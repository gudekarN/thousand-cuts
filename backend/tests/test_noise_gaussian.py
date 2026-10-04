"""Unit tests for Gaussian feature-noise injector (Task 1.4).

Verifies:
- On large synthetic matrix (20000 x 5) measured noise std / (k*std) is within 10% of 1 (T1 accuracy test)
- Zero-std columns remain strictly unchanged
- Input matrix is never modified in place
- Deterministic for the same rng seed and level
- k=0 returns an identical copy with 0 perturbed cells
- Perturbed cell count equals rows x non-zero-std columns
- Validation checks for negative k and dimensional mismatches
"""

import numpy as np
import pytest

from app.engine.noise import (
    clean_stats,
    inject_gaussian,
    make_rng,
)
from app.engine.data import split


class TestInjectGaussian:
    """Tests for inject_gaussian injector."""

    def test_k_zero_returns_identical_copy(self):
        X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        std_clean = np.array([2.45, 2.45, 2.45])
        rng = make_rng(seed=42, noise="gaussian", level=0)

        X_noisy, n_cells = inject_gaussian(X, k=0.0, std_clean=std_clean, rng=rng)

        assert n_cells == 0
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
        std_clean = np.array([16.33, 16.33])
        rng = make_rng(seed=42, noise="gaussian", level=2)

        inject_gaussian(X, k=0.2, std_clean=std_clean, rng=rng)
        np.testing.assert_array_equal(X, X_original)

    def test_cell_count_equals_rows_times_nonzero_std_cols(self):
        # 100 rows, 4 columns: 3 with std > 0, 1 with std == 0
        X = np.zeros((100, 4))
        std_clean = np.array([1.5, 0.0, 3.0, 4.5])
        rng = make_rng(seed=42, noise="gaussian", level=1)

        X_noisy, n_cells = inject_gaussian(X, k=0.1, std_clean=std_clean, rng=rng)

        # 100 rows x 3 non-zero-std columns = 300 cells
        assert n_cells == 300

    def test_zero_std_column_strictly_unchanged(self):
        # 50 rows, 3 columns; column 1 has std == 0
        rng_init = np.random.default_rng(100)
        X = np.empty((50, 3))
        X[:, 0] = rng_init.normal(5.0, 2.0, size=50)
        X[:, 1] = 42.0  # Constant column
        X[:, 2] = rng_init.normal(10.0, 1.0, size=50)

        _, std_clean = clean_stats(X)
        assert std_clean[1] == 0.0

        rng = make_rng(seed=42, noise="gaussian", level=3)
        X_noisy, n_cells = inject_gaussian(X, k=0.5, std_clean=std_clean, rng=rng)

        # Constant column must be strictly bit-level identical
        np.testing.assert_array_equal(X_noisy[:, 1], X[:, 1])
        # Non-zero columns must be perturbed
        assert not np.array_equal(X_noisy[:, 0], X[:, 0])
        assert not np.array_equal(X_noisy[:, 2], X[:, 2])

    def test_large_synthetic_matrix_measured_noise_ratio(self):
        """Technical test T1: noise std / (k * std_clean) is within 10% of 1.0."""
        n_rows = 20000
        n_cols = 5
        rng_data = np.random.default_rng(2026)

        # Create synthetic data with distinct standard deviations per column
        stds_target = np.array([1.0, 2.5, 5.0, 10.0, 20.0])
        X = rng_data.normal(loc=0.0, scale=stds_target, size=(n_rows, n_cols))

        mean_clean, std_clean = clean_stats(X)

        k = 0.25
        rng = make_rng(seed=42, noise="gaussian", level=2)
        X_noisy, n_cells = inject_gaussian(X, k=k, std_clean=std_clean, rng=rng)

        assert n_cells == n_rows * n_cols

        # Measure noise added
        noise = X_noisy - X
        measured_stds = np.std(noise, axis=0, ddof=1)
        expected_stds = k * std_clean

        ratios = measured_stds / expected_stds
        for col_idx, ratio in enumerate(ratios):
            assert 0.90 <= ratio <= 1.10, (
                f"Column {col_idx} ratio {ratio:.4f} is outside [0.90, 1.10]"
            )

    def test_deterministic_same_seed_and_level(self):
        X = np.random.default_rng(123).normal(size=(50, 4))
        std_clean = np.std(X, axis=0, ddof=1)

        rng1 = make_rng(seed=99, noise="gaussian", level=2)
        rng2 = make_rng(seed=99, noise="gaussian", level=2)

        X_noisy1, n_cells1 = inject_gaussian(X, k=0.1, std_clean=std_clean, rng=rng1)
        X_noisy2, n_cells2 = inject_gaussian(X, k=0.1, std_clean=std_clean, rng=rng2)

        assert n_cells1 == n_cells2
        np.testing.assert_array_equal(X_noisy1, X_noisy2)

    def test_different_level_gives_different_perturbations(self):
        X = np.random.default_rng(123).normal(size=(50, 4))
        std_clean = np.std(X, axis=0, ddof=1)

        rng_lvl1 = make_rng(seed=99, noise="gaussian", level=1)
        rng_lvl2 = make_rng(seed=99, noise="gaussian", level=2)

        X_noisy1, _ = inject_gaussian(X, k=0.1, std_clean=std_clean, rng=rng_lvl1)
        X_noisy2, _ = inject_gaussian(X, k=0.1, std_clean=std_clean, rng=rng_lvl2)

        assert not np.array_equal(X_noisy1, X_noisy2)

    def test_validation_errors(self):
        X = np.ones((10, 3))
        std = np.ones(3)
        rng = make_rng(seed=0, noise="gaussian", level=1)

        with pytest.raises(ValueError, match="k must be non-negative"):
            inject_gaussian(X, k=-0.1, std_clean=std, rng=rng)

        with pytest.raises(ValueError, match="must be a 2D array"):
            inject_gaussian(np.ones(10), k=0.1, std_clean=std, rng=rng)

        with pytest.raises(ValueError, match="matching X features"):
            inject_gaussian(X, k=0.1, std_clean=np.ones(2), rng=rng)

    def test_digits_constant_columns_remain_zero(self):
        """Verify on actual Digits dataset that corner zero-variance pixels remain unchanged."""
        X_train, _, _, _ = split("digits", seed=0)
        _, std_clean = clean_stats(X_train)

        zero_cols = np.where(std_clean == 0.0)[0]
        assert len(zero_cols) > 0  # Digits has constant 0 columns

        rng = make_rng(seed=0, noise="gaussian", level=3)
        X_noisy, n_cells = inject_gaussian(X_train, k=0.2, std_clean=std_clean, rng=rng)

        # Expected perturbed cells
        expected_cells = X_train.shape[0] * (X_train.shape[1] - len(zero_cols))
        assert n_cells == expected_cells

        # Constant columns in X_noisy must equal original
        for col in zero_cols:
            np.testing.assert_array_equal(X_noisy[:, col], X_train[:, col])
