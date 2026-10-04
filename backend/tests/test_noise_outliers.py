"""Unit tests for outlier feature-noise injector (Task 1.5).

Verifies:
- replaced-cell count is exactly round(rate * eligible_cells)
- every replaced value equals mean +/- 5*std of its column
- zero-std columns remain untouched
- the generated sign is always +1 or -1
- deterministic for the same rng seed and level
- rate 0 returns an identical copy
- input matrix is never modified in place
"""

import numpy as np
import pytest

from app.engine.noise import (
    clean_stats,
    inject_outliers,
    make_rng,
)
from app.engine.data import split


class TestInjectOutliers:
    """Tests for inject_outliers injector."""

    def test_rate_zero_returns_identical_copy(self):
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        mean_clean = np.array([2.0, 3.0])
        std_clean = np.array([1.414, 1.414])
        rng = make_rng(seed=42, noise="outliers", level=0)

        X_noisy, n_cells = inject_outliers(X, rate=0.0, mean_clean=mean_clean, std_clean=std_clean, rng=rng)

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
        mean_clean = np.array([30.0, 40.0])
        std_clean = np.array([20.0, 20.0])
        rng = make_rng(seed=42, noise="outliers", level=2)

        inject_outliers(X, rate=0.5, mean_clean=mean_clean, std_clean=std_clean, rng=rng)
        np.testing.assert_array_equal(X, X_original)

    def test_exact_replaced_cell_count(self):
        # 100 rows, 4 columns: 3 with std > 0, 1 with std == 0
        X = np.zeros((100, 4))
        mean_clean = np.zeros(4)
        std_clean = np.array([1.0, 0.0, 1.0, 1.0])
        rng = make_rng(seed=42, noise="outliers", level=1)

        # Eligible cells: 100 * 3 = 300
        # rate = 0.15 -> exactly 45 cells should be replaced
        X_noisy, n_cells = inject_outliers(X, rate=0.15, mean_clean=mean_clean, std_clean=std_clean, rng=rng)

        assert n_cells == 45
        diff = X_noisy != X
        assert np.sum(diff) == 45

    def test_zero_std_column_strictly_unchanged(self):
        # 50 rows, 3 columns; column 1 has std == 0
        rng_init = np.random.default_rng(100)
        X = np.empty((50, 3))
        X[:, 0] = rng_init.normal(5.0, 2.0, size=50)
        X[:, 1] = 42.0  # Constant column
        X[:, 2] = rng_init.normal(10.0, 1.0, size=50)

        mean_clean, std_clean = clean_stats(X)
        assert std_clean[1] == 0.0

        rng = make_rng(seed=42, noise="outliers", level=3)
        # Apply high rate to ensure many replacements, but column 1 must stay clean
        X_noisy, n_cells = inject_outliers(X, rate=0.8, mean_clean=mean_clean, std_clean=std_clean, rng=rng)

        # Constant column must be strictly bit-level identical
        np.testing.assert_array_equal(X_noisy[:, 1], X[:, 1])

        # Other columns should have many outliers
        diff_col0 = np.sum(X_noisy[:, 0] != X[:, 0])
        diff_col2 = np.sum(X_noisy[:, 2] != X[:, 2])
        assert diff_col0 > 0
        assert diff_col2 > 0

    def test_replaced_values_are_mean_plus_minus_sigma_std(self):
        X = np.zeros((200, 2))
        mean_clean = np.array([10.0, -5.0])
        std_clean = np.array([2.0, 3.0])
        rng = make_rng(seed=42, noise="outliers", level=1)
        sigma = 5

        X_noisy, n_cells = inject_outliers(X, rate=0.10, mean_clean=mean_clean, std_clean=std_clean, rng=rng, sigma=sigma)

        # Eligible cells = 400. 10% = 40.
        assert n_cells == 40
        
        diff = X_noisy != X
        diff_indices = np.where(diff)

        for row, col in zip(*diff_indices):
            val = X_noisy[row, col]
            expected_plus = mean_clean[col] + sigma * std_clean[col]
            expected_minus = mean_clean[col] - sigma * std_clean[col]

            # The value must exactly match one of the two
            assert np.isclose(val, expected_plus) or np.isclose(val, expected_minus), (
                f"Value {val} at {row},{col} is not mean +/- 5*std"
            )

    def test_both_signs_are_generated(self):
        """Verify that both positive and negative outlier shifts occur."""
        X = np.zeros((1000, 1))
        mean_clean = np.array([0.0])
        std_clean = np.array([1.0])
        rng = make_rng(seed=42, noise="outliers", level=3)
        sigma = 5

        # 1000 eligible cells. rate=0.2 means 200 cells replaced.
        # With 200 uniform sign draws, the chance of all being + or all being - is 2^(-199), effectively 0.
        X_noisy, n_cells = inject_outliers(X, rate=0.20, mean_clean=mean_clean, std_clean=std_clean, rng=rng, sigma=sigma)
        
        assert n_cells == 200
        diff_mask = X_noisy != X
        replaced_values = X_noisy[diff_mask]

        has_plus = np.any(np.isclose(replaced_values, sigma * std_clean[0]))
        has_minus = np.any(np.isclose(replaced_values, -sigma * std_clean[0]))
        
        assert has_plus, "Expected at least one positive outlier sign (+5*std)"
        assert has_minus, "Expected at least one negative outlier sign (-5*std)"


    def test_deterministic_same_seed_and_level(self):
        X = np.random.default_rng(123).normal(size=(50, 4))
        mean_clean, std_clean = clean_stats(X)

        rng1 = make_rng(seed=99, noise="outliers", level=2)
        rng2 = make_rng(seed=99, noise="outliers", level=2)

        X_noisy1, n_cells1 = inject_outliers(X, rate=0.2, mean_clean=mean_clean, std_clean=std_clean, rng=rng1)
        X_noisy2, n_cells2 = inject_outliers(X, rate=0.2, mean_clean=mean_clean, std_clean=std_clean, rng=rng2)

        assert n_cells1 == n_cells2
        np.testing.assert_array_equal(X_noisy1, X_noisy2)

    def test_different_level_gives_different_perturbations(self):
        X = np.random.default_rng(123).normal(size=(50, 4))
        mean_clean, std_clean = clean_stats(X)

        rng_lvl1 = make_rng(seed=99, noise="outliers", level=1)
        rng_lvl2 = make_rng(seed=99, noise="outliers", level=2)

        X_noisy1, _ = inject_outliers(X, rate=0.2, mean_clean=mean_clean, std_clean=std_clean, rng=rng_lvl1)
        X_noisy2, _ = inject_outliers(X, rate=0.2, mean_clean=mean_clean, std_clean=std_clean, rng=rng_lvl2)

        assert not np.array_equal(X_noisy1, X_noisy2)

    def test_validation_errors(self):
        X = np.ones((10, 3))
        mean_clean = np.ones(3)
        std_clean = np.ones(3)
        rng = make_rng(seed=0, noise="outliers", level=1)

        with pytest.raises(ValueError, match="rate must be in"):
            inject_outliers(X, rate=-0.1, mean_clean=mean_clean, std_clean=std_clean, rng=rng)

        with pytest.raises(ValueError, match="must be a 2D array"):
            inject_outliers(np.ones(10), rate=0.1, mean_clean=mean_clean, std_clean=std_clean, rng=rng)

        with pytest.raises(ValueError, match="matching X features"):
            inject_outliers(X, rate=0.1, mean_clean=np.ones(2), std_clean=std_clean, rng=rng)

        with pytest.raises(ValueError, match="matching X features"):
            inject_outliers(X, rate=0.1, mean_clean=mean_clean, std_clean=np.ones(2), rng=rng)
