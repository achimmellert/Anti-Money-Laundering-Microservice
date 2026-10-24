"""
Kumo Tabular (NVIDIA Foundation Model) Training & In-Context Learning Pipeline
==============================================================================
Implements NVIDIA's KumoTabular tabular foundation model for Anti-Money Laundering.

KumoTabular performs in-context learning on structured transaction data:
it ingests a labeled context set (historical verified fraud & clean transactions)
and predicts probabilities for query transactions in a single forward pass
without requiring task-specific backpropagation or gradient updates.

This script:
1. Loads transaction data and applies schema-verified feature engineering from `aml.services`.
2. Encodes data into TableTensors and splits into reference context and query evaluation sets.
3. Runs KumoTabular inference or calibrated zero-shot evaluation.
4. Packages the in-context model as an MLflow PyFunc model bundle.
5. Logs metrics (PR-AUC, ROC-AUC, Brier score, latency) and registers to MLflow.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import polars as pl
from loguru import logger as log
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path when invoked directly as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aml.config.settings import settings
from aml.models.dataframes import (
    NUMERIC_FEATURES,
    RAW_FEATURE_COLUMNS,
)
from aml.services.feature_engineering import create_features

# Check for native NVIDIA structured-data-models (sdm)
try:
    import sdm
    import torch

    SDM_AVAILABLE = True
except ImportError:
    sdm = None
    torch = None
    SDM_AVAILABLE = False


class KumoTabularPyFuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for serving NVIDIA KumoTabular in production.
    Encapsulates the reference context set and dispatches queries to the foundation model.
    """

    def __init__(
        self,
        context_df: pl.DataFrame,
        feature_names: list[str],
        target_col: str = "isFraud",
        model_size: str = "small",
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.context_records = context_df.to_dicts()
        self.feature_names = feature_names
        self.target_col = target_col
        self.model_size = model_size
        self.device = device
        self.threshold = threshold
        self.model: Any = None
        self.table_context: Any = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext | None = None) -> None:
        """Initializes the KumoTabular foundation model and prepares context tensor."""
        if SDM_AVAILABLE and sdm is not None:
            log.info(f"Loading native KumoTabular ({self.model_size}) on device '{self.device}'...")
            self.model = sdm.models.KumoTabular(
                task="classification",
                size=self.model_size,
                pretrained=True,
                device=self.device,
            )
            df_context = pl.DataFrame(self.context_records).to_pandas()
            stypes = sdm.infer_stypes(df_context, overrides={self.target_col: "categorical"})
            self.table_context = sdm.TableTensor.from_pandas(
                df=df_context, stypes=stypes, device=self.device
            )
        else:
            log.warning(
                "NVIDIA 'structured-data-models' (sdm) is not installed. "
                "Running in calibrated in-context similarity emulator mode."
            )
            self.model = None

    def predict(
        self,
        context,
        model_input,
    ) -> np.ndarray:
        """
        Executes in-context inference against stored context transactions.
        Returns predicted fraud probability (1D float array).
        """
        if isinstance(model_input, np.ndarray):
            query_df = pl.DataFrame(model_input, schema=self.feature_names)
        elif hasattr(model_input, "to_pandas"):
            query_df = pl.from_pandas(model_input.to_pandas())
        else:
            query_df = pl.DataFrame(model_input)

        if SDM_AVAILABLE and self.model is not None and sdm is not None:
            df_query = query_df.select(self.feature_names).to_pandas()
            stypes = sdm.infer_stypes(df_query)
            table_query = sdm.TableTensor.from_pandas(
                df=df_query, stypes=stypes, device=self.device
            )

            x_ctx = self.table_context.drop_columns(self.target_col)
            y_ctx = self.table_context[:, self.target_col]

            probs = self.model(
                x_context=x_ctx,
                y_context=y_ctx,
                x_query=table_query,
                num_estimators=8,
            )
            return np.asarray(probs, dtype=np.float64)[:, 1]

        # Emulated in-context score when sdm package is absent (lightweight testing)
        ctx_df = pl.DataFrame(self.context_records)
        num_cols = [c for c in NUMERIC_FEATURES if c in query_df.columns]
        q_vals = query_df.select(num_cols).to_numpy()
        c_vals = ctx_df.select(num_cols).to_numpy()
        c_labels = ctx_df[self.target_col].to_numpy()

        # Distance-weighted nearest neighbor probability
        c_norm = c_vals / (np.std(c_vals, axis=0, keepdims=True) + 1e-5)
        q_norm = q_vals / (np.std(c_vals, axis=0, keepdims=True) + 1e-5)

        # Pairwise distance to context
        dists = np.linalg.norm(q_norm[:, None, :] - c_norm[None, :, :], axis=2)
        weights = np.exp(-dists / 10.0)
        weights /= np.sum(weights, axis=1, keepdims=True) + 1e-7
        probs = np.sum(weights * c_labels[None, :], axis=1)
        return np.clip(probs, 0.001, 0.999)


def generate_benchmark_data(num_rows: int = 1000) -> pl.DataFrame:
    """Generates synthetic AML transaction stream if local dataset is missing."""
    log.info(f"Generating {num_rows} synthetic benchmark transactions...")
    rng = np.random.default_rng(42)

    steps = rng.integers(1, 10, size=num_rows)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], size=num_rows)
    amounts = rng.exponential(scale=5000.0, size=num_rows)

    is_fraud = np.zeros(num_rows, dtype=np.int64)
    # Fraud heuristic: high transfer amounts
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
            "nameOrig": [f"C{i:05d}" for i in rng.integers(1, 200, size=num_rows)],
            "nameDest": [f"M{i:05d}" for i in rng.integers(1, 200, size=num_rows)],
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


def train_kumo_tabular(
    data_path: str | None = None,
    context_size: int = 500,
    model_size: str = "small",
    device: str = "cpu",
    mlflow_tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "AML_Fraud_Detection",
    register_as: str | None = "FraudDetectionModel",
) -> dict[str, Any]:
    """Executes the full KumoTabular training, evaluation, and MLflow logging cycle."""
    log.info("=== Starting Kumo Tabular Pipeline ===")
    resolved_uri = resolve_mlflow_uri(mlflow_tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)

    # 1. Load Data
    resolved_path = data_path or os.path.join(settings.PROJECT_ROOT, "data", "training.csv")
    if os.path.exists(resolved_path):
        log.info(f"Loading raw dataset from {resolved_path}...")
        raw_df = pl.read_csv(resolved_path)
    else:
        log.warning(f"Dataset not found at {resolved_path}. Using benchmark generator.")
        raw_df = generate_benchmark_data(num_rows=1200)

    # 2. Unified Feature Engineering (Using single source of truth in aml.services)
    log.info("Applying core feature engineering pipeline...")
    features_df = create_features(raw_df)

    # Extract target and features
    target = raw_df["isFraud"].to_numpy()
    available_features = [col for col in RAW_FEATURE_COLUMNS if col in features_df.columns]

    X = features_df.select(available_features)
    y = pl.Series("isFraud", target)

    # 3. Stratified Context / Evaluation Split
    X_context, X_query, y_context, y_query = train_test_split(
        X.to_pandas(), y.to_pandas(), train_size=context_size, random_state=42, stratify=target
    )

    context_full = pl.from_pandas(X_context).with_columns(pl.Series("isFraud", y_context))
    query_full = pl.from_pandas(X_query)
    y_test = np.asarray(y_query)

    # 4. Initialize MLflow PyFunc Wrapper
    pyfunc_model = KumoTabularPyFuncModel(
        context_df=context_full,
        feature_names=available_features,
        target_col="isFraud",
        model_size=model_size,
        device=device,
        threshold=settings.FRAUD_THRESHOLD,
    )
    pyfunc_model.load_context()

    # 5. Evaluation & Inference Benchmark
    t0 = time.perf_counter()
    y_pred_proba = pyfunc_model.predict(context=None, model_input=query_full)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / len(y_test)

    pr_auc = float(average_precision_score(y_test, y_pred_proba))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    brier = float(brier_score_loss(y_test, y_pred_proba))

    log.info(f"Evaluation PR-AUC:  {pr_auc:.4f}")
    log.info(f"Evaluation ROC-AUC: {roc_auc:.4f}")
    log.info(f"Per-sample Latency: {latency_ms:.2f} ms")

    # 6. Log Artifacts to MLflow
    with mlflow.start_run(run_name=f"KumoTabular_{model_size.upper()}") as run:
        mlflow.log_params(
            {
                "algorithm": "kumo-tabular",
                "model_size": model_size,
                "context_size": context_size,
                "device": device,
                "sdm_native": SDM_AVAILABLE,
                "num_features": len(available_features),
            }
        )
        mlflow.log_metrics(
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
                "latency_ms": latency_ms,
            }
        )

        # Log PyFunc model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="kumo-tabular-model",
            python_model=pyfunc_model,
            registered_model_name=register_as,
        )

        run_id = run.info.run_id
        log.info(f"KumoTabular run logged to MLflow [Run ID: {run_id}]")

    return {
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "latency_ms": latency_ms,
        "model_name": register_as,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate NVIDIA KumoTabular foundation model."
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw training CSV")
    parser.add_argument(
        "--context-size", type=int, default=400, help="Number of labeled context rows"
    )
    parser.add_argument(
        "--size", type=str, default="small", choices=["small", "medium", "large"], help="Model size"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--tracking-uri", type=str, default=settings.MLFLOW_TRACKING_URI, help="MLflow URI"
    )
    args = parser.parse_args()

    results = train_kumo_tabular(
        data_path=args.data_path,
        context_size=args.context_size,
        model_size=args.size,
        device=args.device,
        mlflow_tracking_uri=args.tracking_uri,
    )
    print("\n--- Training Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
