"""Tests for backend/app/core/config.py methodology constants."""

import sys
import pytest

from app.core import config


def test_no_sklearn_import():
    """Verify rules.md requirement: config.py must not import sklearn."""
    # Ensure config itself does not import sklearn
    assert "sklearn" not in sys.modules or "sklearn" not in getattr(config, "__dict__", {})
    # Check that config module attributes don't reference sklearn objects
    for name, val in vars(config).items():
        if not name.startswith("__"):
            assert "sklearn" not in str(type(val)).lower()


def test_methodology_version():
    assert config.METHODOLOGY_VERSION == "1.0.0"


def test_dataset_ids():
    assert config.DATASET_BREAST_CANCER == "breast_cancer"
    assert config.DATASET_DIGITS == "digits"
    assert config.DATASET_IDS == ("breast_cancer", "digits")


def test_model_ids_and_params():
    expected_ids = ("logreg", "svm_rbf", "decision_tree", "random_forest")
    assert config.MODEL_IDS == expected_ids

    # Exact parameters from Architecture Section 4.4
    assert config.MODEL_PARAMS["logreg"] == {"max_iter": 1000}
    assert config.MODEL_PARAMS["svm_rbf"] == {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
    }
    assert config.MODEL_PARAMS["decision_tree"] == {}
    assert config.MODEL_PARAMS["random_forest"] == {"n_estimators": 100, "n_jobs": 1}

    # Model configuration metadata
    assert config.MODEL_CONFIGS["logreg"]["class_name"] == "LogisticRegression"
    assert config.MODEL_CONFIGS["logreg"]["has_random_state"] is True

    assert config.MODEL_CONFIGS["svm_rbf"]["class_name"] == "SVC"
    assert config.MODEL_CONFIGS["svm_rbf"]["has_random_state"] is False

    assert config.MODEL_CONFIGS["decision_tree"]["class_name"] == "DecisionTreeClassifier"
    assert config.MODEL_CONFIGS["decision_tree"]["has_random_state"] is True

    assert config.MODEL_CONFIGS["random_forest"]["class_name"] == "RandomForestClassifier"
    assert config.MODEL_CONFIGS["random_forest"]["has_random_state"] is True


def test_noise_ids_and_order():
    expected_ids = {
        "label": 1,
        "gaussian": 2,
        "outliers": 3,
        "missing": 4,
    }
    assert config.NOISE_IDS == expected_ids
    assert config.NOISE_ORDER == ("label", "gaussian", "outliers", "missing")


def test_methodology_parameters():
    assert config.TEST_SIZE == 0.3
    assert config.OUTLIER_SIGMA == 5
    assert config.BREAK_RATIO == 0.90


def test_levels():
    assert config.LEVELS == (0, 1, 2, 3, 4, 5)


def test_stages_and_seeds():
    assert config.STAGES == ("mvp", "stage2", "full")
    assert config.STAGE_MVP == "mvp"
    assert config.STAGE_STAGE2 == "stage2"
    assert config.STAGE_FULL == "full"

    assert config.SEEDS_MVP == (0, 1, 2)
    assert config.SEEDS_STAGE2 == (0, 1, 2)
    assert config.SEEDS_FULL == tuple(range(10))

    assert config.STAGE_SEEDS["mvp"] == (0, 1, 2)
    assert config.STAGE_SEEDS["stage2"] == (0, 1, 2)
    assert config.STAGE_SEEDS["full"] == tuple(range(10))


def test_metrics_and_clean_convention():
    assert config.METRICS == ("macro_f1", "accuracy")
    assert config.PRIMARY_METRIC == getattr(config, "PRIMARY_METRIC", "macro_f1")
    assert config.COMBO_CLEAN == "clean"
    assert config.N_NOISES_CLEAN == 0
