"""Methodology constants for Death by a Thousand Cuts.

Pure configuration constants only: NO logic, NO sklearn imports.
Values match PRD Section 4 and Architecture.md Sections 4.4, 4.5, 4.6.
"""

from typing import Any, Dict, Tuple

# Versioning
METHODOLOGY_VERSION: str = "1.0.0"

# Dataset IDs (PRD Section 4, Architecture Section 4.1)
DATASET_BREAST_CANCER: str = "breast_cancer"
DATASET_DIGITS: str = "digits"
DATASET_IDS: Tuple[str, ...] = (
    DATASET_BREAST_CANCER,
    DATASET_DIGITS,
)

# Model IDs and parameter specifications (Architecture Section 4.4)
MODEL_LOGREG: str = "logreg"
MODEL_SVM_RBF: str = "svm_rbf"
MODEL_DECISION_TREE: str = "decision_tree"
MODEL_RANDOM_FOREST: str = "random_forest"
MODEL_IDS: Tuple[str, ...] = (
    MODEL_LOGREG,
    MODEL_SVM_RBF,
    MODEL_DECISION_TREE,
    MODEL_RANDOM_FOREST,
)

# Exact model parameters from Architecture Section 4.4 (no hyperparameter tuning)
MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    MODEL_LOGREG: {
        "max_iter": 1000,
    },
    MODEL_SVM_RBF: {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
    },
    MODEL_DECISION_TREE: {},
    MODEL_RANDOM_FOREST: {
        "n_estimators": 100,
        "n_jobs": 1,
    },
}

# Metadata on model classes and seed requirements
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    MODEL_LOGREG: {
        "class_name": "LogisticRegression",
        "params": {"max_iter": 1000},
        "has_random_state": True,
    },
    MODEL_SVM_RBF: {
        "class_name": "SVC",
        "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "has_random_state": False,
    },
    MODEL_DECISION_TREE: {
        "class_name": "DecisionTreeClassifier",
        "params": {},
        "has_random_state": True,
    },
    MODEL_RANDOM_FOREST: {
        "class_name": "RandomForestClassifier",
        "params": {"n_estimators": 100, "n_jobs": 1},
        "has_random_state": True,
    },
}

# Noise IDs and canonical application order (Architecture Section 4.5)
NOISE_LABEL: str = "label"
NOISE_GAUSSIAN: str = "gaussian"
NOISE_OUTLIERS: str = "outliers"
NOISE_MISSING: str = "missing"

NOISE_IDS: Dict[str, int] = {
    NOISE_LABEL: 1,
    NOISE_GAUSSIAN: 2,
    NOISE_OUTLIERS: 3,
    NOISE_MISSING: 4,
}

NOISE_ORDER: Tuple[str, ...] = (
    NOISE_LABEL,
    NOISE_GAUSSIAN,
    NOISE_OUTLIERS,
    NOISE_MISSING,
)

# Splitting and noise parameters (PRD Section 4, Architecture Section 4.1, 4.5)
TEST_SIZE: float = 0.3
OUTLIER_SIGMA: int = 5
BREAK_RATIO: float = 0.90

# Perturbation levels (0-5)
LEVELS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# Stage names
STAGE_MVP: str = "mvp"
STAGE_STAGE2: str = "stage2"
STAGE_FULL: str = "full"
STAGES: Tuple[str, ...] = (
    STAGE_MVP,
    STAGE_STAGE2,
    STAGE_FULL,
)

# Seeds per stage (rules.md R-M9, PRD Section 4)
SEEDS_MVP: Tuple[int, ...] = (0, 1, 2)
SEEDS_STAGE2: Tuple[int, ...] = (0, 1, 2)
SEEDS_FULL: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

STAGE_SEEDS: Dict[str, Tuple[int, ...]] = {
    STAGE_MVP: SEEDS_MVP,
    STAGE_STAGE2: SEEDS_STAGE2,
    STAGE_FULL: SEEDS_FULL,
}

# Metrics (PRD Section 4, Architecture Section 4.7)
PRIMARY_METRIC: str = "macro_f1"
SECONDARY_METRIC: str = "accuracy"
METRIC_MACRO_F1: str = PRIMARY_METRIC
METRIC_ACCURACY: str = SECONDARY_METRIC
METRICS: Tuple[str, ...] = (PRIMARY_METRIC, SECONDARY_METRIC)

# Clean baseline convention (D-020, rules.md 1.1)
COMBO_CLEAN: str = "clean"
N_NOISES_CLEAN: int = 0
