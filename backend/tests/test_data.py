import numpy as np
import pytest
from app.engine.data import load_dataset, split, split_hashes

def test_load_dataset_invalid():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("invalid_dataset")

def test_load_dataset_shapes_and_types():
    # Breast cancer
    X_bc, y_bc, meta_bc = load_dataset("breast_cancer")
    assert X_bc.shape == (569, 30)
    assert X_bc.dtype == np.float64
    assert y_bc.shape == (569,)
    assert y_bc.dtype == int or y_bc.dtype == np.int64 or y_bc.dtype == np.int32
    assert meta_bc == {"n_samples": 569, "n_features": 30, "n_classes": 2}

    # Digits
    X_dig, y_dig, meta_dig = load_dataset("digits")
    assert X_dig.shape == (1797, 64)
    assert meta_dig == {"n_samples": 1797, "n_features": 64, "n_classes": 10}

def test_dataset_isolation():
    X1, y1, _ = load_dataset("breast_cancer")
    X1[0, 0] = 999.0
    y1[0] = 999

    X2, y2, _ = load_dataset("breast_cancer")
    assert X2[0, 0] != 999.0
    assert y2[0] != 999

def test_split_sizes():
    # breast cancer: 569 -> 398 / 171
    X_train, X_test, y_train, y_test = split("breast_cancer", seed=42)
    assert X_train.shape[0] == 398
    assert X_test.shape[0] == 171

    # digits: 1797 -> 1257 / 540
    X_train, X_test, y_train, y_test = split("digits", seed=42)
    assert X_train.shape[0] == 1257
    assert X_test.shape[0] == 540

def test_stratification():
    _, y, _ = load_dataset("digits")
    _, _, y_train, y_test = split("digits", seed=42)
    
    # calculate proportions
    _, orig_counts = np.unique(y, return_counts=True)
    _, train_counts = np.unique(y_train, return_counts=True)
    _, test_counts = np.unique(y_test, return_counts=True)
    
    orig_props = orig_counts / len(y)
    
    # "class proportions preserved (within 1 sample per class)"
    expected_train_counts = orig_props * len(y_train)
    assert np.all(np.abs(train_counts - expected_train_counts) <= 1.0)
    
    expected_test_counts = orig_props * len(y_test)
    assert np.all(np.abs(test_counts - expected_test_counts) <= 1.0)

def test_reproducibility():
    X_train1, X_test1, y_train1, y_test1 = split("breast_cancer", seed=10)
    X_train2, X_test2, y_train2, y_test2 = split("breast_cancer", seed=10)
    
    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(y_train1, y_train2)

    h_train1, h_test1 = split_hashes("breast_cancer", seed=10)
    h_train2, h_test2 = split_hashes("breast_cancer", seed=10)
    assert h_train1 == h_train2
    assert h_test1 == h_test2

    # Different seed
    X_train3, _, _, _ = split("breast_cancer", seed=11)
    assert not np.array_equal(X_train1, X_train3)
    
    h_train3, _ = split_hashes("breast_cancer", seed=11)
    assert h_train1 != h_train3
