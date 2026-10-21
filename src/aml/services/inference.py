import os
from typing import Any

import matplotlib
import mlflow
import mlflow.xgboost
import numpy as np
import polars as pl
import shap
import xgboost as xgb
from joblib import load as joblib_load
from loguru import logger as log

matplotlib.use("Agg")  # Force non-interactive, memory-safe backend
import matplotlib.pyplot as plt

from aml.config.settings import settings


class FraudDetectionModel:
    def __init__(self):
        self.model: Any = None
        self.preproc: Any = None
        self.explainer: Any = None
        self.feature_names: list[str] = []
        self.threshold: float = settings.FRAUD_THRESHOLD
        self._load_model()

    def _load_model(self):
        tracking_uri = settings.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(tracking_uri)

        loaded_from_mlflow = False
        # Try modern MLflow alias @champion first, then legacy Production stage
        for model_uri in (
            "models:/FraudDetectionModel@champion",
            "models:/FraudDetectionModel/Production",
        ):
            try:
                log.info(f"Attempting to load model from MLflow: {model_uri}...")
                self.model = mlflow.xgboost.load_model(model_uri)
                log.info(f"MLflow model loaded successfully from {model_uri}.")
                loaded_from_mlflow = True
                break
            except Exception as e:
                log.debug(f"Could not load '{model_uri}': {e}")

        if not loaded_from_mlflow:
            log.warning(
                f"MLflow load failed (Fallback to local model). Target: {settings.MODEL_PATH}"
            )
            self.model = xgb.Booster()
            self.model.load_model(str(settings.MODEL_PATH))

        self.preproc = joblib_load(settings.PREPROC_PATH)

        if not settings.FEATNAMES_PATH.exists():
            raise FileNotFoundError(
                f"{settings.FEATNAMES_PATH} missing - this Booster requires exact column names."
            )
        _names = np.load(settings.FEATNAMES_PATH, allow_pickle=True)
        self.feature_names = [
            str(x) for x in (_names.tolist() if hasattr(_names, "tolist") else list(_names))
        ]

        # Initialize explainer once
        self.explainer = shap.TreeExplainer(self.model)

    def _prepare_dmatrix(self, raw_features_df: pl.DataFrame) -> tuple[xgb.DMatrix, np.ndarray]:
        """Helper to DRY up the transformation and validation logic."""
        X_trans = self.preproc.transform(raw_features_df)
        if X_trans.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Column mismatch: X={X_trans.shape[1]} vs feature_names={len(self.feature_names)}"
            )

        # Pass the NumPy array (X_trans) directly instead of wrapping in a DataFrame
        return xgb.DMatrix(X_trans, feature_names=self.feature_names), X_trans

    def predict(self, raw_features_df: pl.DataFrame) -> tuple[float, int, str]:
        dmat, _ = self._prepare_dmatrix(raw_features_df)
        prob = float(self.model.predict(dmat)[0])

        is_fraud = 1 if prob >= self.threshold else 0
        decision = "BLOCKED" if is_fraud else "SUCCESS"

        return prob, is_fraud, decision

    def predict_batch(self, raw_features_df: pl.DataFrame) -> list[tuple[float, int, str]]:
        dmat, _ = self._prepare_dmatrix(raw_features_df)
        probs = self.model.predict(dmat).astype(float)

        results = []
        for prob in probs:
            is_fraud = 1 if prob >= self.threshold else 0
            decision = "BLOCKED" if is_fraud else "SUCCESS"
            results.append((float(prob), is_fraud, decision))
        return results

    def predict_proba(self, raw_features_df: pl.DataFrame) -> np.ndarray:
        dmat, _ = self._prepare_dmatrix(raw_features_df)
        return self.model.predict(dmat).astype(float)

    def save_shap_barplot(self, raw_features_df: pl.DataFrame, out_path: str):
        _, X_trans = self._prepare_dmatrix(raw_features_df)
        shap_values = self.explainer.shap_values(X_trans)

        # Ensure target directory exists before saving
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Explicit memory management for background plots
        fig, _ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_trans,
            feature_names=self.feature_names,
            show=False,
            plot_type="bar",
        )

        fig.savefig(out_path, dpi=150, bbox_inches="tight")

        # Force garbage collection of the figure
        fig.clf()
        plt.close(fig)
