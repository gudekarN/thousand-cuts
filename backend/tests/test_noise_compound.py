"""Unit tests for compound noise logic (Task 1.7).

Verifies:
- 15 combos (4 singles, 6 pairs, 4 triples, 1 four-way)
- canonical_combo with shuffled input gives the same canonical name and output
- the label flips inside a compound equal the flips of the single label run at the same level and seed
- the NaN positions inside a compound equal the single missing run at the same level and seed
- level 0 equals the input for every combo
- y is unchanged when label is not in the combo
- inputs not modified
"""

import numpy as np
import pytest

from app.engine.noise import (
    canonical_combo,
    list_all_combos,
    apply_noise,
    make_rng,
    inject_label,
    inject_missing,
)
from app.core.config import NOISE_ORDER
from app.core.levels import get_params


class TestCompoundNoise:

    def test_list_all_combos(self):
        combos = list_all_combos()
        assert len(combos) == 15
        
        # 4 singles
        singles = [c for c in combos if "+" not in c]
        assert len(singles) == 4
        
        # 6 pairs
        pairs = [c for c in combos if c.count("+") == 1]
        assert len(pairs) == 6
        
        # 4 triples
        triples = [c for c in combos if c.count("+") == 2]
        assert len(triples) == 4
        
        # 1 four-way
        four = [c for c in combos if c.count("+") == 3]
        assert len(four) == 1
        
        assert four[0] == "+".join(NOISE_ORDER)

    def test_canonical_combo(self):
        assert canonical_combo("gaussian") == "gaussian"
        assert canonical_combo("label+gaussian") == "label+gaussian"
        assert canonical_combo("gaussian+label") == "label+gaussian"
        
        # list input
        assert canonical_combo(["missing", "label", "outliers"]) == "label+outliers+missing"
        
        with pytest.raises(ValueError, match="Unknown noise type"):
            canonical_combo("label+invalid")
            
        with pytest.raises(ValueError, match="Empty combo"):
            canonical_combo([])

    def test_level_0_returns_unchanged(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        X_orig = X.copy()
        y_orig = y.copy()
        
        for combo in list_all_combos():
            X_noisy, y_noisy, stats = apply_noise(X, y, combo, level=0, seed=42)
            
            assert stats["label_flipped"] == 0
            assert stats["gaussian_cells"] == 0
            assert stats["outlier_cells"] == 0
            assert stats["missing_cells"] == 0
            
            np.testing.assert_array_equal(X_noisy, X_orig)
            np.testing.assert_array_equal(y_noisy, y_orig)
            
            # ensure independent copies
            X_noisy[0, 0] = 999.0
            assert X[0, 0] == 1.0

    def test_y_unchanged_without_label_noise(self):
        X = np.random.default_rng(1).normal(size=(50, 4))
        y = np.random.default_rng(2).integers(0, 2, size=50)
        
        for combo in list_all_combos():
            if "label" not in combo:
                X_noisy, y_noisy, stats = apply_noise(X, y, combo, level=3, seed=123)
                assert stats["label_flipped"] == 0
                np.testing.assert_array_equal(y_noisy, y)

    def test_label_flips_equal_single_label_run(self):
        X = np.random.default_rng(1).normal(size=(50, 4))
        y = np.random.default_rng(2).integers(0, 2, size=50)
        level = 4
        seed = 42
        
        params = get_params(level)
        rate = params["label_flip_rate"]
        
        rng_single = make_rng(seed, "label", level)
        y_single, n_flipped = inject_label(y, rate, rng_single)
        
        for combo in list_all_combos():
            if "label" in combo:
                X_noisy, y_comp, stats = apply_noise(X, y, combo, level=level, seed=seed)
                np.testing.assert_array_equal(y_comp, y_single)
                assert stats["label_flipped"] == n_flipped

    def test_missing_nan_positions_equal_single_missing_run(self):
        X = np.random.default_rng(1).normal(size=(50, 4))
        y = np.random.default_rng(2).integers(0, 2, size=50)
        level = 5
        seed = 99
        
        params = get_params(level)
        p = params["missing_cell_rate"]
        
        # We need a pristine X for missing alone, because compound noise might apply 
        # missing on top of gaussian and outliers. The NaNs applied should still be in the exact same cells!
        rng_single = make_rng(seed, "missing", level)
        X_single_missing, n_single_nan = inject_missing(X, p, rng_single)
        single_nan_mask = np.isnan(X_single_missing)
        
        for combo in list_all_combos():
            if "missing" in combo:
                X_comp, y_comp, stats = apply_noise(X, y, combo, level=level, seed=seed)
                comp_nan_mask = np.isnan(X_comp)
                
                assert np.array_equal(comp_nan_mask, single_nan_mask)
                assert stats["missing_cells"] == n_single_nan

    def test_inputs_not_modified(self):
        X = np.random.default_rng(1).normal(size=(20, 2))
        y = np.random.default_rng(2).integers(0, 2, size=20)
        X_orig = X.copy()
        y_orig = y.copy()
        
        apply_noise(X, y, "label+gaussian+outliers+missing", level=5, seed=77)
        
        np.testing.assert_array_equal(X, X_orig)
        np.testing.assert_array_equal(y, y_orig)

    def test_shuffled_input_gives_same_output(self):
        X = np.random.default_rng(1).normal(size=(20, 2))
        y = np.random.default_rng(2).integers(0, 2, size=20)
        
        X1, y1, stats1 = apply_noise(X, y, "gaussian+label", level=3, seed=42)
        X2, y2, stats2 = apply_noise(X, y, "label+gaussian", level=3, seed=42)
        
        assert np.array_equal(X1, X2, equal_nan=True)
        np.testing.assert_array_equal(y1, y2)
        assert stats1 == stats2
