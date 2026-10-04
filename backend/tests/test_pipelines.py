"""Unit tests for model pipelines (Task 1.8).

Verifies:
- parameters match Architecture 4.4
- fit works on data containing NaN
- LEAKAGE: after fit on noisy train, imputer.statistics_ equals np.nanmean 
  of the noisy train and scaler.mean_ equals the mean of the imputed noisy train
- calling predict on a test set does not change those statistics
- an all-NaN column keeps the column count
- unknown model raises
- all 4 models fit and predict on both datasets (smoke)
"""

import numpy as np
import pytest

from app.engine.pipelines import build_pipeline
from app.core.config import MODEL_IDS, DATASET_IDS, MODEL_CONFIGS
from app.engine.data import split


class TestPipelines:

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model_id: invalid_model"):
            build_pipeline("invalid_model", seed=42)

    def test_parameters_match_architecture(self):
        seed = 42
        for model_id in MODEL_IDS:
            pipeline = build_pipeline(model_id, seed)
            
            # Check imputer
            imputer = pipeline.named_steps["imputer"]
            assert imputer.strategy == "mean"
            assert imputer.keep_empty_features is True
            
            # Check model parameters
            model = pipeline.named_steps["model"]
            config = MODEL_CONFIGS[model_id]
            expected_params = config["params"]
            
            # Verify exact params match
            for k, v in expected_params.items():
                assert getattr(model, k) == v
                
            if config.get("has_random_state", False):
                assert getattr(model, "random_state") == seed

    def test_fit_works_on_data_containing_nan(self):
        X_train = np.array([
            [1.0, 2.0],
            [np.nan, 4.0],
            [5.0, 6.0],
            [7.0, np.nan],
        ])
        y_train = np.array([0, 1, 0, 1])
        
        # Should fit without error
        pipeline = build_pipeline("logreg", seed=1)
        pipeline.fit(X_train, y_train)

    def test_leakage_statistics_and_predict_invariance(self):
        # Create a noisy train set
        X_train = np.array([
            [1.0, 2.0],
            [np.nan, 4.0],
            [5.0, 6.0],
            [7.0, np.nan],
        ])
        y_train = np.array([0, 1, 0, 1])
        
        pipeline = build_pipeline("logreg", seed=1)
        pipeline.fit(X_train, y_train)
        
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        
        # 1. imputer.statistics_ equals np.nanmean of noisy train
        expected_imputer_stats = np.nanmean(X_train, axis=0)
        np.testing.assert_array_almost_equal(imputer.statistics_, expected_imputer_stats)
        
        # 2. scaler.mean_ equals the mean of the imputed noisy train
        X_imputed = np.where(np.isnan(X_train), expected_imputer_stats, X_train)
        expected_scaler_mean = np.mean(X_imputed, axis=0)
        np.testing.assert_array_almost_equal(scaler.mean_, expected_scaler_mean)
        
        # 3. calling predict on a test set does not change those statistics
        X_test = np.array([
            [2.0, 3.0],
            [np.nan, 5.0],
        ])
        
        # Save exact arrays to ensure they are not modified in place or replaced
        imputer_stats_before = imputer.statistics_.copy()
        scaler_mean_before = scaler.mean_.copy()
        
        pipeline.predict(X_test)
        
        np.testing.assert_array_equal(imputer.statistics_, imputer_stats_before)
        np.testing.assert_array_equal(scaler.mean_, scaler_mean_before)

    def test_all_nan_column_keeps_column_count(self):
        X_train = np.array([
            [1.0, np.nan, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, np.nan, 9.0],
        ])
        y_train = np.array([0, 1, 0])
        
        pipeline = build_pipeline("decision_tree", seed=42)
        pipeline.fit(X_train, y_train)
        
        # Model should receive 3 features, even though one was all NaN
        model = pipeline.named_steps["model"]
        assert model.n_features_in_ == 3
        
        # Imputer statistics should have 0 or NaN for the all-NaN column depending on sklearn version,
        # but keep_empty_features=True ensures it stays in the pipeline
        
        X_test = np.array([
            [2.0, 5.0, 8.0],
        ])
        
        # Predict should not fail due to feature shape mismatch
        pred = pipeline.predict(X_test)
        assert len(pred) == 1

    @pytest.mark.parametrize("dataset", DATASET_IDS)
    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_all_models_fit_and_predict_smoke(self, dataset, model_id):
        """Smoke test: all 4 models fit and predict on both datasets."""
        X_train, X_test, y_train, y_test = split(dataset, seed=0)
        
        pipeline = build_pipeline(model_id, seed=0)
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        
        assert len(preds) == len(y_test)
        # Predictions should have exactly the shape of y_test
        assert preds.shape == y_test.shape
