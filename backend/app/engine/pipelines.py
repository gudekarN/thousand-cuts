"""Model pipelines for Death by a Thousand Cuts.

Pure sklearn Pipelines. No fitting logic outside tests.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from app.core.config import MODEL_CONFIGS

__all__ = ["build_pipeline"]


def build_pipeline(model_id: str, seed: int) -> Pipeline:
    """Build the exact sklearn Pipeline for the requested model.
    
    Pipeline consists of:
    - imputer: SimpleImputer(strategy="mean", keep_empty_features=True)
    - scaler: StandardScaler()
    - model: The specified classifier with exact parameters and random_state=seed (if supported)
    
    Args:
        model_id: One of the supported model IDs (e.g. "logreg", "svm_rbf")
        seed: Random seed for the model's random_state (where supported)
        
    Returns:
        A ready-to-fit sklearn Pipeline.
    """
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_id: {model_id}")
        
    config = MODEL_CONFIGS[model_id]
    class_name = config["class_name"]
    params = config["params"].copy()
    
    if config.get("has_random_state", False):
        params["random_state"] = seed
        
    if class_name == "LogisticRegression":
        model = LogisticRegression(**params)
    elif class_name == "SVC":
        model = SVC(**params)
    elif class_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(**params)
    elif class_name == "RandomForestClassifier":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model class: {class_name}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", model)
    ])
