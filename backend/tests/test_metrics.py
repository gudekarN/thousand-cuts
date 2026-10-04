"""Tests for evaluation metrics (Macro F1 and Accuracy)."""

import numpy as np
import pytest

from app.engine.metrics import evaluate


def test_metrics_keys_and_types():
    """Verify return format has 'macro_f1' and 'accuracy' as Python floats."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    result = evaluate(y_true, y_pred)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"macro_f1", "accuracy"}
    assert isinstance(result["macro_f1"], float)
    assert isinstance(result["accuracy"], float)


def test_hand_computed_binary_example():
    """Hand-computed binary test case.

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    Class 0: TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5 -> F1=0.5
    Class 1: TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5 -> F1=0.5
    Macro F1 = 0.5
    Accuracy = 2 / 4 = 0.5
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    res = evaluate(y_true, y_pred)

    assert pytest.approx(res["macro_f1"]) == 0.5
    assert pytest.approx(res["accuracy"]) == 0.5


def test_hand_computed_multiclass_example():
    """Hand-computed 3-class test case.

    y_true = [0, 1, 2]
    y_pred = [0, 2, 1]
    Class 0: TP=1, FP=0, FN=0 -> F1=1.0
    Class 1: TP=0, FP=1, FN=1 -> F1=0.0
    Class 2: TP=0, FP=1, FN=1 -> F1=0.0
    Macro F1 = (1.0 + 0.0 + 0.0) / 3 = 1/3
    Accuracy = 1 / 3
    """
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 1])
    res = evaluate(y_true, y_pred)

    assert pytest.approx(res["macro_f1"]) == 1.0 / 3.0
    assert pytest.approx(res["accuracy"]) == 1.0 / 3.0


def test_perfect_prediction():
    """Perfect prediction gives 1.0 for both macro_f1 and accuracy."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    res = evaluate(y_true, y_pred)

    assert res["macro_f1"] == 1.0
    assert res["accuracy"] == 1.0


def test_all_wrong_prediction():
    """All-wrong prediction gives 0.0 for both macro_f1 and accuracy."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    res = evaluate(y_true, y_pred)

    assert res["macro_f1"] == 0.0
    assert res["accuracy"] == 0.0


def test_unpredicted_class_zero_division():
    """Unpredicted class does not crash and gets zero F1 with zero_division=0.

    y_true = [0, 1, 1]
    y_pred = [0, 0, 0]
    Class 0: TP=1, FP=2, FN=0 -> Prec=1/3, Rec=1.0 -> F1=0.5
    Class 1: TP=0, FP=0, FN=2 -> zero_division=0 -> F1=0.0
    Macro F1 = (0.5 + 0.0) / 2 = 0.25
    Accuracy = 1 / 3
    """
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 0])
    res = evaluate(y_true, y_pred)

    assert pytest.approx(res["macro_f1"]) == 0.25
    assert pytest.approx(res["accuracy"]) == 1.0 / 3.0


def test_ten_class_support():
    """Evaluate on 10-class arrays (matching Digits dataset)."""
    # 10 classes 0..9, 2 samples each = 20 samples
    y_true = np.repeat(np.arange(10), 2)
    # Perfect on first 8 classes (16 samples), misclassify last 2 classes
    y_pred = y_true.copy()
    y_pred[16:18] = 9  # true class 8 predicted as 9
    y_pred[18:20] = 8  # true class 9 predicted as 8

    res = evaluate(y_true, y_pred)

    # Classes 0..7: F1 = 1.0 each (8 classes)
    # Classes 8 and 9: TP=0, F1 = 0.0 each (2 classes)
    # Macro F1 = (8 * 1.0 + 2 * 0.0) / 10 = 0.8
    # Accuracy = 16 / 20 = 0.8
    assert pytest.approx(res["macro_f1"]) == 0.8
    assert pytest.approx(res["accuracy"]) == 0.8


def test_ten_class_perfect():
    """Verify 10-class perfect predictions yield 1.0."""
    y_true = np.random.default_rng(42).integers(0, 10, size=100)
    y_pred = y_true.copy()
    res = evaluate(y_true, y_pred)

    assert res["macro_f1"] == 1.0
    assert res["accuracy"] == 1.0
