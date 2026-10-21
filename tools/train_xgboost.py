"""
Production-grade XGBoost Training & Hyperparameter Tuning Pipeline
===================================================================
Implements an end-to-end Optuna-optimized training workflow for XGBoost.
Eliminates training-serving skew by sharing `create_features` and schema
contracts directly from `aml.services.feature_engineering`.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Any

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from loguru import logger as log

try:
    from optuna_integration.xgboost import XGBoostPruningCallback
except (ImportError, ModuleNotFoundError):
    try:
        from optuna.integration.xgboost import XGBoostPruningCallback
    except (ImportError, ModuleNotFoundError):
        XGBoostPruningCallback = None  # Optional pruning callback
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Ensure project root is in sys.path when invoked directly as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aml.config.settings import settings
from aml.models.dataframes import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURE_COLUMNS,
)
from aml.services.feature_engineering import create_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def check_gpu_availability() -> str:
    """Checks for NVIDIA CUDA support; falls back to CPU if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def generate_benchmark_data(num_rows: int = 1500) -> pl.DataFrame:
    """Generates synthetic AML transaction stream if local dataset is missing."""
    log.info(f"Generating {num_rows} synthetic benchmark transactions for XGBoost training...")
    rng = np.random.default_rng(42)

    steps = rng.integers(1, 10, size=num_rows)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], size=num_rows)
    amounts = rng.exponential(scale=5000.0, size=num_rows)

    is_fraud = np.zeros(num_rows, dtype=np.int64)
    fraud_mask = (amounts > 12000) & np.isin(types, ["TRANSFER", "CASH_OUT"])
    is_fraud[fraud_mask] = 1

    orig_bal = rng.uniform(100.0, 50000.0, size=num_rows)
    new_orig = np.maximum(0.0, orig_bal - amounts)
    dest_bal = rng.uniform(0.0, 30000.0, size=num_rows)
    new_dest = dest_bal + amounts

    from datetime import datetime, timedelta

    base_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=int(s)) for s in steps]

    return pl.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amounts,
            "nameOrig": [f"C{i:05d}" for i in rng.integers(1, 250, size=num_rows)],
            "nameDest": [f"M{i:05d}" for i in rng.integers(1, 250, size=num_rows)],
            "oldbalanceOrg": orig_bal,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": dest_bal,
            "newbalanceDest": new_dest,
            "timestamp": timestamps,
            "isFraud": is_fraud,
        }
    )


def resolve_mlflow_uri(uri: str | None = None) -> str:
    candidate = uri or settings.MLFLOW_TRACKING_URI
    if candidate.startswith("http://") or candidate.startswith("https://"):
        try:
            import urllib.request

            urllib.request.urlopen(candidate, timeout=0.6)
            return candidate
        except Exception:
            db_path = (settings.PROJECT_ROOT / "mlflow.db").as_posix()
            fallback = f"sqlite:///{db_path}"
            log.info(
                f"MLflow server at '{candidate}' unreachable. Using SQLite registry store: '{fallback}'"
            )
            return fallback
    return candidate


def train_xgboost(
    data_path: str | None = None,
    n_trials: int = 15,
    device: str | None = None,
    tracking_uri: str = settings.MLFLOW_TRACKING_URI,
    experiment_name: str = "AML_Fraud_Detection",
    register_as: str | None = "FraudDetectionModel",
    save_local_artifacts: bool = False,
) -> dict[str, Any]:
    """
    Executes end-to-end Optuna hyperparameter optimization, model training,
    and MLflow registration.
    """
    chosen_device = device or check_gpu_availability()
    log.info(f"Using compute device: {chosen_device}")

    resolved_uri = resolve_mlflow_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)

    # 1. Load data
    resolved_path = data_path or os.path.join(settings.PROJECT_ROOT, "data", "training.csv")
    if os.path.exists(resolved_path):
        log.info(f"Loading raw dataset from {resolved_path}...")
        raw_df = pl.read_csv(resolved_path)
    else:
        log.warning(f"Dataset not found at {resolved_path}. Generating synthetic benchmark data.")
        raw_df = generate_benchmark_data(num_rows=2000)

    # 2. Schema-validated Feature Engineering (Zero-Skew)
    log.info("Computing features via shared aml.services.feature_engineering pipeline...")
    enriched_df = create_features(raw_df)

    # 3. Align with schema columns
    target_series = raw_df["isFraud"].to_numpy()
    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in enriched_df.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    # Preprocessor (OrdinalEncoder for categorical, passthrough for numeric)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    pandas_df = enriched_df.select(feature_cols).to_pandas()
    X_processed = preprocessor.fit_transform(pandas_df)
    final_feature_names = cat_cols + num_cols

    y = target_series

    # 4. Stratified Train / Val / Test Splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_processed, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    log.info(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # 5. Optuna Hyperparameter Optimization
    log.info(f"Starting Optuna optimization ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 30.0),
        }

        callbacks = (
            [XGBoostPruningCallback(trial, "validation_0-aucpr")]
            if XGBoostPruningCallback is not None
            else None
        )

        clf = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device=chosen_device,
            early_stopping_rounds=15,
            callbacks=callbacks,
            random_state=42,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = clf.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, preds))

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials)
    log.info(f"Best Trial Val PR-AUC: {study.best_value:.4f}")

    # 6. Final Model Retraining (Train + Val combined)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    best_params = study.best_params
    final_clf = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=chosen_device,
        random_state=42,
    )
    final_clf.fit(X_train_full, y_train_full, verbose=False)

    # 7. Evaluate on Independent Test Set
    t0 = time.perf_counter()
    test_preds = final_clf.predict_proba(X_test)[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000.0 / len(y_test)

    test_pr_auc = float(average_precision_score(y_test, test_preds))
    test_roc_auc = float(roc_auc_score(y_test, test_preds))
    test_brier = float(brier_score_loss(y_test, test_preds))

    log.info(f"Test PR-AUC:  {test_pr_auc:.4f}")
    log.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    log.info(f"Latency:      {latency_ms:.2f} ms / sample")

    # 8. MLflow Logging & Model Registration
    with mlflow.start_run(run_name="XGBoost_Optuna_Best") as run:
        mlflow.log_params(best_params)
        mlflow.log_params(
            {
                "algorithm": "xgboost",
                "device": chosen_device,
                "n_trials": n_trials,
                "num_features": len(final_feature_names),
            }
        )
        mlflow.log_metrics(
            {
                "pr_auc": test_pr_auc,
                "roc_auc": test_roc_auc,
                "brier_score": test_brier,
                "latency_ms": latency_ms,
            }
        )

        model_info = mlflow.xgboost.log_model(
            xgb_model=final_clf.get_booster(),
            artifact_path="xgboost-model",
            registered_model_name=register_as,
        )
        run_id = run.info.run_id
        log.info(f"XGBoost run logged to MLflow [Run ID: {run_id}]")

    # 9. Optionally update local ml_models directory
    if save_local_artifacts:
        os.makedirs(settings.PROJECT_ROOT / "ml_models", exist_ok=True)
        final_clf.get_booster().save_model(str(settings.MODEL_PATH))
        joblib.dump(preprocessor, str(settings.PREPROC_PATH))
        np.save(str(settings.FEATNAMES_PATH), np.array(final_feature_names))
        log.info(f"Synchronized local model artifacts to {settings.PROJECT_ROOT / 'ml_models'}")

    return {
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "pr_auc": test_pr_auc,
        "roc_auc": test_roc_auc,
        "brier_score": test_brier,
        "latency_ms": latency_ms,
        "model_name": register_as,
        "feature_names": final_feature_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and optimize XGBoost model for AML.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw training CSV")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--tracking-uri", type=str, default=settings.MLFLOW_TRACKING_URI)
    parser.add_argument(
        "--sync-local", action="store_true", help="Save artifacts to local ml_models/"
    )
    args = parser.parse_args()

    results = train_xgboost(
        data_path=args.data_path,
        n_trials=args.trials,
        device=args.device,
        tracking_uri=args.tracking_uri,
        save_local_artifacts=args.sync_local,
    )
    print("\n--- Training Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
