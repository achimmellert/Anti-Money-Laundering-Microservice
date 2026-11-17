import logging
import os
import sys
import time
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import polars as pl
import xgboost as xgb

try:
    from optuna_integration.xgboost import XGBoostPruningCallback
except (ImportError, ModuleNotFoundError):
    try:
        from optuna.integration.xgboost import XGBoostPruningCallback
    except (ImportError, ModuleNotFoundError):
        XGBoostPruningCallback = None

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Ensure project root is in sys.path when invoked directly as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aml.config.settings import settings
from aml.services.evaluation import (
    compare_challenger_vs_champion,
    evaluate_model_statistically,
)
from aml.services.feature_engineering import create_features
from aml.services.model_wrappers import SklearnPyTorchWrapper

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_DIR = str(settings.PROJECT_ROOT)
DATA_PATH = os.path.join(BASE_DIR, "data", "training.csv")
MLFLOW_URI = settings.MLFLOW_TRACKING_URI


# === 2. MAIN TRAINING PIPELINE ===
def main():
    mlflow.set_tracking_uri(MLFLOW_URI)

    if not os.path.exists(DATA_PATH):
        log.error(f"Data not found at: {DATA_PATH}")
        return

    # Load and process features
    log.info("Loading raw data...")
    df_raw = pl.read_csv(DATA_PATH)
    df = create_features(df_raw)

    drop_cols = ["step", "nameOrig", "nameDest", "timestamp", "isFlaggedFraud"]
    df = df.drop([c for c in drop_cols if c in df.columns])

    df = df.with_columns(
        type=pl.Series(OrdinalEncoder().fit_transform(df.select("type").to_numpy()).ravel()).cast(
            pl.Int64
        )
    )

    X = df.drop("isFraud")
    y = df["isFraud"]

    categorical_cols = ["type"]
    continuous_cols = [col for col in X.columns if col not in categorical_cols]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    # === EXPERIMENT 1: XGBoost Optimization ===
    log.info("--- Starting Experiment 1: XGBoost Optimization ---")
    mlflow.set_experiment("AML_XGBoost_Standalone")

    def objective_xgb(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
            "device": "cuda",
            "tree_method": "hist",
        }
        callbacks = (
            [XGBoostPruningCallback(trial, "validation_0-aucpr")]
            if XGBoostPruningCallback is not None
            else []
        )
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
            verbose=False,
        )
        return average_precision_score(y_val, clf.predict_proba(X_val)[:, 1])

    pruner = optuna.pruners.HyperbandPruner()
    study_xgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_xgb.optimize(objective_xgb, n_trials=20)
    best_xgb = xgb.XGBClassifier(
        **study_xgb.best_params, random_state=42, device="cuda", tree_method="hist"
    )
    best_xgb.fit(X_train, y_train)

    # === EXPERIMENT 2: PyTorch Tabular Optimization ===
    log.info("--- Starting Experiment 2: PyTorch Tabular Optimization ---")

    def objective_tabular(trial):
        layers = trial.suggest_categorical("layers", ["32-16", "64-32", "128-64"])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Using the imported wrapper
        model = SklearnPyTorchWrapper(
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            layers=layers,
            dropout=dropout,
            learning_rate=lr,
            epochs=15,
            trial=trial,
        )
        model.fit(X_train, y_train, X_val, y_val)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study_tabular = optuna.create_study(direction="maximize", pruner=pruner)
    study_tabular.optimize(objective_tabular, n_trials=15)

    best_tabular = SklearnPyTorchWrapper(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        **study_tabular.best_params,
        epochs=20,
    )
    best_tabular.fit(X_train, y_train)

    # === EXPERIMENT 3: Stacking Ensemble & Statistical Hypothesis Testing ===
    log.info("--- Starting Experiment 3: Stacking Ensemble ---")
    mlflow.set_experiment("AML_Stacking_Ensemble")

    estimators = [("xgb", best_xgb), ("pytorch_tabular", best_tabular)]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced"),
        cv=3,
        passthrough=False,
    )

    stacking_clf.fit(X_train, y_train)

    y_test_np = np.asarray(y_test, dtype=np.int64)

    t0_xgb = time.perf_counter()
    xgb_preds = best_xgb.predict_proba(X_test)[:, 1]
    xgb_latency_ms = (time.perf_counter() - t0_xgb) * 1000.0 / max(1, len(y_test_np))

    t0_stack = time.perf_counter()
    test_preds_stack = stacking_clf.predict_proba(X_test)[:, 1]
    stack_latency_ms = (time.perf_counter() - t0_stack) * 1000.0 / max(1, len(y_test_np))

    stack_eval = evaluate_model_statistically(
        y_true=y_test_np,
        y_prob=test_preds_stack,
        latency_ms=stack_latency_ms,
        threshold=settings.FRAUD_THRESHOLD,
    )
    comparison = compare_challenger_vs_champion(
        y_true=y_test_np,
        y_prob_challenger=test_preds_stack,
        y_prob_champion=xgb_preds,
        challenger_latency_ms=stack_latency_ms,
        champion_latency_ms=xgb_latency_ms,
        threshold=settings.FRAUD_THRESHOLD,
    )

    with mlflow.start_run(run_name="Stacking_XGB_PyTorchTabular"):
        mlflow.log_params({f"xgb_{k}": v for k, v in study_xgb.best_params.items()})
        mlflow.log_params({f"tabular_{k}": v for k, v in study_tabular.best_params.items()})

        mlflow.log_metric("test_pr_auc", stack_eval.pr_auc)
        mlflow.log_metric("test_roc_auc", stack_eval.roc_auc)
        mlflow.log_metrics(comparison.to_mlflow_metrics())
        mlflow.log_dict(
            comparison.to_dict(), "evaluation/stacking_vs_xgboost_hypothesis_report.json"
        )

        mlflow.sklearn.log_model(
            stacking_clf,
            "stacking-model",
            registered_model_name="Modern_FraudDetection_Stacking",
        )
        log.info(
            f"Stacking Ensemble Test PR-AUC: {stack_eval.pr_auc:.4f} "
            f"(vs Standalone XGBoost: {comparison.champion_eval.pr_auc:.4f}, "
            f"Bootstrap p={comparison.hypothesis_tests['bootstrap_pr_auc'].p_value:.4f})"
        )
        log.info(f"Hypothesis Gate Summary: {comparison.decision_summary}")


if __name__ == "__main__":
    main()
