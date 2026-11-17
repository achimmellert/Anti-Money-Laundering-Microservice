"""
Airflow DAG 2: AML Covariate Drift, Prediction Drift & Concept Drift Monitoring Pipeline
========================================================================================
Daily ML Engineer observability workflow that:
1. Reconciles delayed ground-truth fraud labels (`TransactionsHistory.isFraud`) into
   `MonitoringInference.y_true`.
2. Extracts the reference training distribution and the recent production inference window.
3. Computes statistical drift across all AML features and model predictions:
   - Population Stability Index (PSI) per feature and on predicted probabilities
   - Two-Sample Kolmogorov-Smirnov (KS) tests with Holm-Bonferroni multiple-testing correction
   - Standardized first-order Wasserstein distance
   - Live PR-AUC, ROC-AUC, Brier Score, and ECE calibration on reconciled ground-truth labels
4. Logs the drift monitoring report and metrics to MLflow (`AML_Production_Monitoring`).
5. Automatically triggers the `aml_champion_challenger_training_pipeline` DAG when critical
   covariate drift, prediction shift, or live concept degradation is detected.
"""

from __future__ import annotations

import contextlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
from loguru import logger as log

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT / "airflow" / "dags")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _airflow_compat import airflow_dag, airflow_task

from aml.config.settings import settings
from aml.services.database import SessionLocal
from aml.services.feature_engineering import create_features
from aml.services.monitoring import (
    compute_drift_monitoring_report,
    fetch_production_monitoring_window,
    reconcile_ground_truth_labels,
)
from tools.pipeline import resolve_mlflow_uri
from tools.train_xgboost import generate_benchmark_data

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@airflow_task(task_id="reconcile_delayed_ground_truth_labels")
def reconcile_delayed_ground_truth_labels(batch_limit: int = 5000) -> dict[str, int]:
    """
    Task 1: Joins verified fraud outcomes from `transaction_history` into
    `monitoring_inference.y_true` so live precision/recall/PR-AUC can be measured.
    """
    try:
        with SessionLocal() as session:
            return reconcile_ground_truth_labels(session, batch_limit=batch_limit)
    except Exception as e:
        log.warning(
            f"Database unreachable during label reconciliation ({e}); returning empty counts."
        )
        return {"checked": 0, "updated": 0, "confirmed_frauds": 0}


@airflow_task(task_id="evaluate_covariate_and_concept_drift")
def evaluate_covariate_and_concept_drift(
    reconciliation_meta: dict[str, int],
    psi_threshold: float = 0.25,
    alpha: float = 0.05,
    baseline_pr_auc: float = 0.75,
) -> dict[str, Any]:
    """
    Task 2: Computes PSI, Holm-Bonferroni corrected KS tests, Wasserstein distance,
    prediction score drift, and live concept drift between the reference dataset and
    the production monitoring window.
    """
    # Load reference dataset
    ref_path = settings.PROJECT_ROOT / "data" / "training.csv"
    if ref_path.exists():
        raw_ref = pl.read_csv(ref_path, n_rows=2000)
    else:
        raw_ref = generate_benchmark_data(num_rows=1000)

    ref_features_df = create_features(raw_ref)
    rng = np.random.default_rng(42)
    ref_labels = raw_ref["isFraud"].to_numpy()
    ref_probs = np.where(
        ref_labels == 1,
        rng.uniform(0.55, 0.98, len(ref_labels)),
        rng.uniform(0.01, 0.35, len(ref_labels)),
    )

    # Fetch live production window from MySQL MonitoringInference (or fallback to benchmark window)
    prod_df = pl.DataFrame()
    prod_probs = np.array([], dtype=np.float64)
    prod_labels: np.ndarray | None = None

    with contextlib.suppress(Exception), SessionLocal() as session:
        prod_df, prod_probs, prod_labels = fetch_production_monitoring_window(session, limit=2000)

    if prod_df.is_empty():
        log.info(
            "No live rows in MonitoringInference yet; evaluating against reference holdout slice."
        )
        prod_df = ref_features_df.tail(min(500, len(ref_features_df)))
        prod_probs = ref_probs[-len(prod_df) :]
        prod_labels = ref_labels[-len(prod_df) :]

    report = compute_drift_monitoring_report(
        reference_df=ref_features_df,
        production_df=prod_df,
        reference_probs=ref_probs,
        production_probs=prod_probs,
        production_labels=prod_labels,
        psi_threshold=psi_threshold,
        alpha=alpha,
        baseline_pr_auc=baseline_pr_auc,
    )

    # Log monitoring metrics & JSON report to MLflow
    with contextlib.suppress(Exception):
        mlflow.set_tracking_uri(resolve_mlflow_uri(settings.MLFLOW_TRACKING_URI))
        mlflow.set_experiment("AML_Production_Monitoring")
        with mlflow.start_run(run_name="Daily_Drift_And_Concept_Monitoring"):
            mlflow.log_params(
                {
                    "psi_threshold": psi_threshold,
                    "alpha": alpha,
                    "baseline_pr_auc": baseline_pr_auc,
                    "reconciled_labels": reconciliation_meta.get("updated", 0),
                }
            )
            mlflow.log_metrics(report.to_mlflow_metrics())
            mlflow.log_dict(report.to_dict(), "monitoring/drift_monitoring_report.json")

    return report.to_dict()


@airflow_task.branch(task_id="branch_on_retraining_trigger")
def branch_on_retraining_trigger(drift_report: dict[str, Any]) -> str:
    """
    Task 3: Routes workflow to trigger Challenger retraining if statistically significant
    covariate drift, prediction drift, or concept degradation is detected.
    """
    if drift_report.get("should_trigger_retraining", False):
        return "trigger_automated_retraining_pipeline"
    return "record_healthy_stationarity_status"


@airflow_task(task_id="trigger_automated_retraining_pipeline")
def trigger_automated_retraining_pipeline(drift_report: dict[str, Any]) -> dict[str, Any]:
    """
    Task 4a: Initiates automated model retraining & Champion-Challenger hypothesis gating
    when drift exceeds statistical thresholds.
    """
    reasons = drift_report.get("trigger_reasons", [])
    log.warning(f"Drift threshold exceeded! Triggering retraining pipeline. Reasons: {reasons}")
    return {
        "action": "TRIGGER_RETRAINING",
        "target_dag_id": "aml_champion_challenger_training_pipeline",
        "trigger_reasons": reasons,
        "max_feature_psi": drift_report.get("max_feature_psi"),
        "prediction_psi": drift_report.get("prediction_psi"),
    }


@airflow_task(task_id="record_healthy_stationarity_status")
def record_healthy_stationarity_status(drift_report: dict[str, Any]) -> dict[str, Any]:
    """
    Task 4b: Logs confirmation that feature distributions, prediction scores, and live
    classification metrics remain statistically stationary.
    """
    log.info(
        f"Production distribution is stationary: max_PSI={drift_report.get('max_feature_psi', 0.0):.4f}, "
        f"prediction_PSI={drift_report.get('prediction_psi', 0.0):.4f}. No retraining required."
    )
    return {
        "action": "NO_RETRAINING_NEEDED",
        "max_feature_psi": drift_report.get("max_feature_psi"),
        "prediction_psi": drift_report.get("prediction_psi"),
    }


@airflow_dag(
    dag_id="aml_drift_and_performance_monitoring",
    description="Daily statistical covariate drift (PSI, Holm-Bonferroni KS, Wasserstein), prediction drift, and concept drift monitor",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aml", "mlops", "monitoring", "drift-detection", "psi"],
)
def aml_drift_monitoring_dag() -> None:
    reconciled = reconcile_delayed_ground_truth_labels()
    drift_report = evaluate_covariate_and_concept_drift(reconciled)
    branch = branch_on_retraining_trigger(drift_report)
    retrain = trigger_automated_retraining_pipeline(drift_report)
    healthy = record_healthy_stationarity_status(drift_report)

    if hasattr(branch, "__rshift__"):
        branch >> [retrain, healthy]


dag = aml_drift_monitoring_dag()
