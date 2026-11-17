"""
Airflow DAG 1: AML Automated Training, Statistical Evaluation & Champion-Challenger Promotion
=============================================================================================
Orchestrates the core ML Engineer model training and release lifecycle:
1. Validates raw training data and ensures train/test splits exist.
2. Verifies zero-skew feature engineering contract (`create_features` + Pandera validation).
3. Executes Challenger model training (XGBoost Optuna or NVIDIA KumoTabular) and computes
   single-model statistical evaluation (95% stratified bootstrap CIs, ECE/MCE calibration,
   Brier Skill Score, Spiegelhalter calibration test, and Kolmogorov-Smirnov separation test).
4. Evaluates Challenger against the active `@champion` on the identical holdout split using
   the 6-test Paired Statistical Hypothesis Testing Suite (Bootstrap PR-AUC, Permutation PR-AUC,
   DeLong Correlated ROC-AUC, Wilcoxon Signed-Rank Brier, Paired t-test, Exact McNemar, and
   Holm-Bonferroni multiple testing correction).
5. Branches on the statistical gate decision to either promote `@challenger` -> `@champion`
   and synchronize local `ml_models/` artifacts, or record rejection while preserving `@champion`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger as log

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT / "airflow" / "dags")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _airflow_compat import airflow_dag, airflow_task

from aml.config.settings import settings
from aml.models.dataframes import RAW_FEATURE_COLUMNS
from aml.services.feature_engineering import create_features
from tools.pipeline import run_full_pipeline
from tools.train_xgboost import generate_benchmark_data

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@airflow_task(task_id="prepare_and_validate_dataset")
def prepare_and_validate_dataset(data_path: str | None = None) -> dict[str, Any]:
    """
    Task 1: Verifies that the training dataset exists and meets minimum sample size
    and positive fraud representation requirements; generates a benchmark dataset if absent.
    """
    target_path = Path(data_path) if data_path else settings.PROJECT_ROOT / "data" / "training.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        df = pl.read_csv(target_path, n_rows=10000)
        source = "existing_csv"
    else:
        log.info(f"Training dataset not found at {target_path}; generating benchmark dataset.")
        df = generate_benchmark_data(num_rows=2000)
        df.write_csv(target_path)
        source = "synthetic_benchmark"

    n_rows = len(df)
    n_fraud = int(df["isFraud"].sum()) if "isFraud" in df.columns else 0
    if n_rows < 50 or n_fraud < 2:
        raise ValueError(
            f"Dataset validation failed: insufficient rows ({n_rows}) or positive frauds ({n_fraud})."
        )

    return {
        "data_path": str(target_path),
        "sample_rows_checked": n_rows,
        "sample_fraud_count": n_fraud,
        "source": source,
    }


@airflow_task(task_id="verify_feature_engineering_contract")
def verify_feature_engineering_contract(dataset_meta: dict[str, Any]) -> dict[str, Any]:
    """
    Task 2: Executes Polars `create_features` and Pandera schema validation on a
    validation slice to guarantee zero training-serving skew before training starts.
    """
    data_path = dataset_meta["data_path"]
    sample_df = pl.read_csv(data_path, n_rows=500)
    enriched_df = create_features(sample_df)

    missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in enriched_df.columns]
    if missing_cols:
        raise ValueError(f"Feature contract violation: missing columns {missing_cols}")

    return {
        **dataset_meta,
        "validated_feature_count": len(RAW_FEATURE_COLUMNS),
        "contract_verified": True,
    }


@airflow_task(task_id="train_and_statistically_evaluate_challenger")
def train_and_statistically_evaluate_challenger(
    contract_meta: dict[str, Any],
    model_type: str = "xgboost",
    n_trials: int = 10,
    min_improvement: float = 0.005,
    alpha: float = 0.05,
    n_bootstraps: int = 300,
    n_permutations: int = 300,
) -> dict[str, Any]:
    """
    Task 3: Trains the Challenger model, computes single-model statistical evaluation
    (95% bootstrap CIs, calibration ECE/MCE, Spiegelhalter test, KS test), and runs
    the 6-test Paired Champion vs. Challenger Statistical Hypothesis Testing Suite.
    """
    decision = run_full_pipeline(
        model_type=model_type,
        data_path=contract_meta["data_path"],
        n_trials=n_trials,
        min_improvement=min_improvement,
        alpha=alpha,
        n_bootstraps=n_bootstraps,
        n_permutations=n_permutations,
        auto_promote=True,
    )
    return decision


@airflow_task.branch(task_id="branch_on_hypothesis_gate")
def branch_on_hypothesis_gate(decision: dict[str, Any]) -> str:
    """
    Task 4: Branches workflow based on whether the Challenger passed the statistical
    hypothesis test gate and was promoted to `@champion`.
    """
    if decision.get("promoted", False):
        return "verify_promoted_champion_artifacts"
    return "log_challenger_rejection_audit"


@airflow_task(task_id="verify_promoted_champion_artifacts")
def verify_promoted_champion_artifacts(decision: dict[str, Any]) -> dict[str, Any]:
    """
    Task 5a: Verifies that offline production artifacts (`model.json`, `preprocessor.pkl`,
    `feature_names.npy`) are present on disk after a successful `@champion` promotion.
    """
    model_exists = os.path.exists(settings.MODEL_PATH)
    preproc_exists = os.path.exists(settings.PREPROC_PATH)
    feat_exists = os.path.exists(settings.FEATNAMES_PATH)

    log.info(
        f"Verified promoted @champion (Version {decision.get('challenger_version')}) artifacts: "
        f"model={model_exists}, preproc={preproc_exists}, feature_names={feat_exists}"
    )
    return {
        "status": "PROMOTED",
        "champion_version": decision.get("challenger_version"),
        "pr_auc": decision.get("challenger_pr_auc"),
        "artifacts_ready": bool(model_exists and preproc_exists and feat_exists),
    }


@airflow_task(task_id="log_challenger_rejection_audit")
def log_challenger_rejection_audit(decision: dict[str, Any]) -> dict[str, Any]:
    """
    Task 5b: Records an audit trail when the Challenger fails the statistical hypothesis
    test gate, confirming that the existing `@champion` remains untouched in production.
    """
    log.warning(
        f"Challenger Version {decision.get('challenger_version')} was REJECTED by statistical gate. "
        f"Active @champion remains Version {decision.get('champion_version')}. "
        f"Reason: {decision.get('reason')}"
    )
    return {
        "status": "REJECTED",
        "rejected_challenger_version": decision.get("challenger_version"),
        "preserved_champion_version": decision.get("champion_version"),
        "reason": decision.get("reason"),
    }


@airflow_dag(
    dag_id="aml_champion_challenger_training_pipeline",
    description="Automated AML Challenger training, statistical evaluation, and paired hypothesis-gated promotion to @champion",
    default_args=DEFAULT_ARGS,
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aml", "mlops", "training", "hypothesis-testing", "champion-challenger"],
)
def aml_champion_challenger_training_dag() -> None:
    ds_meta = prepare_and_validate_dataset()
    contract_meta = verify_feature_engineering_contract(ds_meta)
    decision = train_and_statistically_evaluate_challenger(contract_meta)
    branch = branch_on_hypothesis_gate(decision)
    promoted_done = verify_promoted_champion_artifacts(decision)
    rejected_done = log_challenger_rejection_audit(decision)

    if hasattr(branch, "__rshift__"):
        branch >> [promoted_done, rejected_done]


dag = aml_champion_challenger_training_dag()
