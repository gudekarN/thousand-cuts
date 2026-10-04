"""Unit tests for noise helpers and label-noise injector (Task 1.3).

Verifies:
- make_rng determinism, parameter acceptance, and invalid input handling
- clean_stats mean and std computation, ddof, and immutability
- inject_label exact flip count, class divergence on flips, immutability,
  identical output for same (seed, level), divergence for different level/seed,
  rate=0 identity, and support for 2 and 10 classes.
"""

import numpy as np
import pytest

from app.core.config import NOISE_IDS
from app.engine.noise import (
    NOISE_LABEL,
    NOISE_GAUSSIAN,
    NOISE_OUTLIERS,
    NOISE_MISSING,
    clean_stats,
    inject_label,
    make_rng,
)


class TestMakeRng:
    """Tests for make_rng helper."""

    def test_make_rng_with_string_noise(self):
        rng = make_rng(seed=42, noise="label", level=1)
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_with_int_noise(self):
        rng = make_rng(seed=42, noise=1, level=1)
        assert isinstance(rng, np.random.Generator)

    def test_make_rng_string_and_int_equivalent(self):
        rng_str = make_rng(seed=42, noise="label", level=1)
        rng_int = make_rng(seed=42, noise=NOISE_IDS["label"], level=1)
        draw_str = rng_str.integers(0, 1000, size=20)
        draw_int = rng_int.integers(0, 1000, size=20)
        np.testing.assert_array_equal(draw_str, draw_int)

    def test_make_rng_deterministic_same_seed_noise_level(self):
        rng1 = make_rng(seed=123, noise="gaussian", level=3)
        rng2 = make_rng(seed=123, noise="gaussian", level=3)
        np.testing.assert_array_equal(
            rng1.normal(size=15),
            rng2.normal(size=15),
        )

    def test_make_rng_differs_across_seeds(self):
        rng1 = make_rng(seed=0, noise="label", level=1)
        rng2 = make_rng(seed=1, noise="label", level=1)
        assert not np.array_equal(rng1.integers(0, 1000, size=20), rng2.integers(0, 1000, size=20))

    def test_make_rng_differs_across_noises(self):
        rng1 = make_rng(seed=42, noise="label", level=1)
        rng2 = make_rng(seed=42, noise="gaussian", level=1)
        assert not np.array_equal(rng1.integers(0, 1000, size=20), rng2.integers(0, 1000, size=20))

    def test_make_rng_differs_across_levels(self):
        rng1 = make_rng(seed=42, noise="label", level=1)
        rng2 = make_rng(seed=42, noise="label", level=2)
        assert not np.array_equal(rng1.integers(0, 1000, size=20), rng2.integers(0, 1000, size=20))

    def test_make_rng_invalid_noise_string_raises(self):
        with pytest.raises(ValueError, match="Unknown noise type"):
            make_rng(seed=42, noise="quantum_flux", level=1)

    def test_make_rng_invalid_noise_id_raises(self):
        with pytest.raises(ValueError, match="Unknown noise id"):
            make_rng(seed=42, noise=999, level=1)

    def test_make_rng_invalid_type_raises(self):
        with pytest.raises(TypeError, match="noise must be str or int"):
            make_rng(seed=42, noise=None, level=1)  # type: ignore


class TestCleanStats:
    """Tests for clean_stats helper."""

    def test_clean_stats_basic_computation(self):
        X = np.array([
            [1.0, 10.0, 5.0],
            [3.0, 20.0, 5.0],
            [5.0, 30.0, 5.0],
        ])
        mean, std = clean_stats(X, ddof=1)
        np.testing.assert_allclose(mean, [3.0, 20.0, 5.0])
        # ddof=1: sample std of [1, 3, 5] is 2.0; [10, 20, 30] is 10.0; [5, 5, 5] is 0.0
        np.testing.assert_allclose(std, [2.0, 10.0, 0.0])

    def test_clean_stats_zero_variance_column(self):
        X = np.ones((50, 4), dtype=np.float64) * 3.14
        mean, std = clean_stats(X)
        np.testing.assert_allclose(mean, [3.14, 3.14, 3.14, 3.14])
        np.testing.assert_allclose(std, [0.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_clean_stats_does_not_modify_input(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_copy = X.copy()
        clean_stats(X)
        np.testing.assert_array_equal(X, X_copy)

    def test_clean_stats_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="must be a 2D array"):
            clean_stats(np.array([1.0, 2.0, 3.0]))


class TestInjectLabel:
    """Tests for inject_label function."""

    def test_rate_zero_returns_identical_copy(self):
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        rng = make_rng(seed=42, noise="label", level=1)
        y_noisy, n_flipped = inject_label(y, rate=0.0, rng=rng)

        assert n_flipped == 0
        np.testing.assert_array_equal(y_noisy, y)
        # Verify it is an independent copy
        y_noisy[0] = 99
        assert y[0] == 0

    def test_exact_flip_count_binary(self):
        # 100 samples, rate = 0.15 -> round(0.15 * 100) = 15 flips
        rng_init = np.random.default_rng(999)
        y = rng_init.integers(0, 2, size=100)
        rng = make_rng(seed=42, noise="label", level=1)
        rate = 0.15

        y_noisy, n_flipped = inject_label(y, rate=rate, rng=rng)
        assert n_flipped == 15
        # In binary, every flip changes to the other class, so exactly 15 elements change
        diff_count = int(np.sum(y_noisy != y))
        assert diff_count == 15

    def test_exact_flip_count_rounding(self):
        # 10 samples, rate = 0.14 -> round(1.4) = 1; rate = 0.16 -> round(1.6) = 2
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        rng1 = make_rng(seed=42, noise="label", level=1)
        rng2 = make_rng(seed=42, noise="label", level=1)
        _, n_flipped1 = inject_label(y, rate=0.14, rng=rng1)
        _, n_flipped2 = inject_label(y, rate=0.16, rng=rng2)
        assert n_flipped1 == 1
        assert n_flipped2 == 2

    def test_every_flipped_label_differs_from_original(self):
        # 10 classes, rate 0.50, 200 samples
        classes = np.arange(10)
        rng_init = np.random.default_rng(123)
        y = rng_init.choice(classes, size=200)

        rng = make_rng(seed=42, noise="label", level=2)
        y_noisy, n_flipped = inject_label(y, rate=0.50, rng=rng, classes=classes)

        # Because new_class is guaranteed != original class, diff_count must strictly equal n_flipped
        diff_count = int(np.sum(y_noisy != y))
        assert n_flipped == 100
        assert diff_count == 100

        # Also verify that for every position that changed, y_noisy != y
        for i in range(len(y)):
            if y_noisy[i] != y[i]:
                assert y_noisy[i] in classes
                assert y_noisy[i] != y[i]

    def test_unflipped_labels_remain_unchanged(self):
        rng_init = np.random.default_rng(777)
        y = rng_init.integers(0, 5, size=150)
        rng = make_rng(seed=42, noise="label", level=1)

        y_noisy, n_flipped = inject_label(y, rate=0.20, rng=rng)
        assert n_flipped == 30

        # Positions where y_noisy == y must have exactly 150 - 30 = 120 elements
        unchanged_count = int(np.sum(y_noisy == y))
        assert unchanged_count == 120

    def test_input_not_modified_in_place(self):
        y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
        y_original = y.copy()
        rng = make_rng(seed=42, noise="label", level=3)

        inject_label(y, rate=0.5, rng=rng)
        np.testing.assert_array_equal(y, y_original)

    def test_deterministic_same_seed_and_level(self):
        y = np.repeat(np.arange(10), 20)  # 200 samples, 10 classes
        rng1 = make_rng(seed=7, noise="label", level=2)
        rng2 = make_rng(seed=7, noise="label", level=2)

        y_noisy1, n_flipped1 = inject_label(y, rate=0.25, rng=rng1)
        y_noisy2, n_flipped2 = inject_label(y, rate=0.25, rng=rng2)

        assert n_flipped1 == n_flipped2
        np.testing.assert_array_equal(y_noisy1, y_noisy2)

    def test_different_level_gives_different_flips(self):
        y = np.repeat(np.arange(10), 30)  # 300 samples
        rng_lvl1 = make_rng(seed=7, noise="label", level=1)
        rng_lvl2 = make_rng(seed=7, noise="label", level=2)

        y_noisy1, _ = inject_label(y, rate=0.20, rng=rng_lvl1)
        y_noisy2, _ = inject_label(y, rate=0.20, rng=rng_lvl2)

        assert not np.array_equal(y_noisy1, y_noisy2)

    def test_different_seed_gives_different_flips(self):
        y = np.repeat(np.arange(10), 30)
        rng_s0 = make_rng(seed=0, noise="label", level=1)
        rng_s1 = make_rng(seed=1, noise="label", level=1)

        y_noisy0, _ = inject_label(y, rate=0.20, rng=rng_s0)
        y_noisy1, _ = inject_label(y, rate=0.20, rng=rng_s1)

        assert not np.array_equal(y_noisy0, y_noisy1)

    def test_works_for_2_classes(self):
        # Binary case (e.g. breast_cancer labels {0, 1})
        y = np.array([0] * 50 + [1] * 50)
        rng = make_rng(seed=42, noise="label", level=1)
        classes = np.array([0, 1])

        y_noisy, n_flipped = inject_label(y, rate=0.10, rng=rng, classes=classes)
        assert n_flipped == 10
        # In binary classification, flipping 0 MUST give 1, and flipping 1 MUST give 0
        flipped_mask = (y_noisy != y)
        assert np.sum(flipped_mask) == 10
        # Check that original 0s became 1s and original 1s became 0s
        assert np.all(y_noisy[flipped_mask & (y == 0)] == 1)
        assert np.all(y_noisy[flipped_mask & (y == 1)] == 0)

    def test_works_for_10_classes(self):
        # 10 classes (e.g. digits labels {0..9})
        classes = np.arange(10)
        y = np.tile(classes, 20)  # 200 samples, 20 per class
        rng = make_rng(seed=42, noise="label", level=3)

        y_noisy, n_flipped = inject_label(y, rate=0.30, rng=rng, classes=classes)
        assert n_flipped == 60
        assert np.sum(y_noisy != y) == 60
        # All classes in noisy array are valid classes in 0..9
        assert np.all(np.isin(y_noisy, classes))

    def test_works_without_explicit_classes_argument(self):
        y = np.array([10, 20, 30, 10, 20, 30, 10, 20, 30, 10])
        rng = make_rng(seed=42, noise="label", level=1)
        y_noisy, n_flipped = inject_label(y, rate=0.40, rng=rng)
        assert n_flipped == 4
        assert np.sum(y_noisy != y) == 4
        assert np.all(np.isin(y_noisy, [10, 20, 30]))

    def test_invalid_rates_raise(self):
        y = np.array([0, 1, 0, 1])
        rng = make_rng(seed=42, noise="label", level=1)
        with pytest.raises(ValueError, match="rate must be in"):
            inject_label(y, rate=-0.1, rng=rng)
        with pytest.raises(ValueError, match="rate must be in"):
            inject_label(y, rate=1.5, rng=rng)

    def test_invalid_dimension_raises(self):
        y_2d = np.array([[0, 1], [1, 0]])
        rng = make_rng(seed=42, noise="label", level=1)
        with pytest.raises(ValueError, match="must be a 1D array"):
            inject_label(y_2d, rate=0.1, rng=rng)

    def test_single_class_raises(self):
        y_single = np.array([0, 0, 0, 0])
        rng = make_rng(seed=42, noise="label", level=1)
        with pytest.raises(ValueError, match="fewer than 2 classes"):
            inject_label(y_single, rate=0.5, rng=rng)
