# src/services/inference.py

import os
import logging
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import load as joblib_load
import mlflow
import mlflow.xgboost

# --- FIX 1: Prevent Celery Memory Leaks ---
import matplotlib

matplotlib.use('Agg')  # Force non-interactive, memory-safe backend
import matplotlib.pyplot as plt

from src.config.settings import settings
from src.services.model_wrappers import SklearnPyTorchWrapper

log = logging.getLogger(__name__)


class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.preproc = None
        self.explainer = None
        self.feature_names = None
        # Load configurable threshold (defaults to 0.5 if not set in .env)
        self.threshold = getattr(settings, "FRAUD_THRESHOLD", 0.5)
        self._load_model()

    def _load_model(self):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        try:
            log.info(f"Connecting to MLflow at {tracking_uri}...")
            self.model = mlflow.xgboost.load_model("models:/FraudDetectionModel/Production")
            log.info("✅ MLflow 'Production' model loaded successfully.")
        except Exception as e:
            log.warning(f"⚠️ MLflow load failed (Fallback to local model). Reason: {e}")
            self.model = xgb.Booster()
            self.model.load_model(settings.MODEL_PATH)

        self.preproc = joblib_load(settings.PREPROC_PATH)

        if not os.path.exists(settings.FEATNAMES_PATH):
            raise FileNotFoundError(
                f"{settings.FEATNAMES_PATH} missing – this Booster requires exact column names."
            )
        _names = np.load(settings.FEATNAMES_PATH, allow_pickle=True)
        self.feature_names = [str(x) for x in (_names.tolist() if hasattr(_names, "tolist") else list(_names))]

        # Initialize explainer once
        self.explainer = shap.TreeExplainer(self.model)

    def _prepare_dmatrix(self, raw_features_df: pd.DataFrame) -> xgb.DMatrix:
        """Helper to DRY up the transformation and validation logic."""
        X_trans = self.preproc.transform(raw_features_df)
        if X_trans.shape[1] != len(self.feature_names):
            raise ValueError(f"Column mismatch: X={X_trans.shape[1]} vs feature_names={len(self.feature_names)}")

        # --- FIX 2: Zero-Copy DMatrix ---
        # Pass the NumPy array (X_trans) directly instead of wrapping in Pandas
        return xgb.DMatrix(X_trans, feature_names=self.feature_names), X_trans

    def predict(self, raw_features_df: pd.DataFrame) -> tuple[float, int, str]:
        dmat, _ = self._prepare_dmatrix(raw_features_df)
        prob = float(self.model.predict(dmat)[0])

        # --- FIX 3: Dynamic Threshold ---
        is_fraud = 1 if prob >= self.threshold else 0
        decision = "BLOCKED" if is_fraud else "SUCCESS"

        return prob, is_fraud, decision

    def predict_batch(self, raw_features_df: pd.DataFrame) -> list[tuple[float, int, str]]:
        dmat, _ = self._prepare_dmatrix(raw_features_df)
        probs = self.model.predict(dmat).astype(float)

        results = []
        for prob in probs:
            is_fraud = 1 if prob >= self.threshold else 0
            decision = "BLOCKED" if is_fraud else "SUCCESS"
            results.append((float(prob), is_fraud, decision))
        return results

    def predict_proba(self, raw_features_df: pd.DataFrame) -> np.ndarray:
        dmat, _ = self._prepare_dmatrix(raw_features_df)
        return self.model.predict(dmat).astype(float)

    def save_shap_barplot(self, raw_features_df: pd.DataFrame, out_path: str):
        _, X_trans = self._prepare_dmatrix(raw_features_df)
        shap_values = self.explainer.shap_values(X_trans)

        # Explicit memory management for background plots
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_trans, feature_names=self.feature_names, show=False, plot_type="bar")

        fig.savefig(out_path, dpi=150, bbox_inches='tight')

        # Force garbage collection of the figure
        fig.clf()
        plt.close(fig)