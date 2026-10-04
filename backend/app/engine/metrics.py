"""Evaluation metrics for model predictions.

Primary metric: Macro F1 (average='macro', zero_division=0).
Secondary metric: Accuracy.
"""

from typing import Any, Mapping
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute Macro F1 and Accuracy for true and predicted labels.

    Args:
        y_true: Ground truth target values (array-like).
        y_pred: Estimated targets returned by a classifier (array-like).

    Returns:
        Dictionary with keys 'macro_f1' and 'accuracy', mapped to float values.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }
