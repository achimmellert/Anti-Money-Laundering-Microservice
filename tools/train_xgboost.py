"""
Production-grade XGBoost Training, Statistical Evaluation & Hypothesis-Gated Promotion
======================================================================================
Implements an end-to-end Optuna-optimized training workflow for XGBoost with:
1. Zero training-serving skew via shared `aml.services.feature_engineering.create_features`.
2. Comprehensive statistical evaluation on an independent stratified holdout test split:
   - Stratified Bootstrap 95% Confidence Intervals & Standard Errors (PR-AUC, ROC-AUC, Brier, Log-Loss, F1, MCC)
   - Calibration analysis (Brier Skill Score, ECE, MCE, Spiegelhalter's z-test)
   - Distributional separation & ranking tests (Two-Sample Kolmogorov-Smirnov test, Mann-Whitney U test, Gini, Recall@Top-K%)
3. Paired Champion vs. Challenger Statistical Hypothesis Testing Gate:
   - Evaluates the newly trained Challenger against the active `@champion` on the identical holdout test split
   - Executes Paired Stratified Bootstrap PR-AUC test, Paired Permutation PR-AUC test, Fast DeLong's Correlated ROC-AUC test,
     Wilcoxon Signed-Rank Brier loss test, Paired Student's t-test (Diebold-Mariano), Exact McNemar's test, and Holm-Bonferroni correction
   - Promotes Challenger to `@champion` and updates offline `ml_models/` ONLY when statistically superior.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from loguru import logger as log
from mlflow.tracking import MlflowClient

try:
    from optuna_integration.xgboost import XGBoostPruningCallback
except (ImportError, ModuleNotFoundError):
    try:
        from optuna.integration.xgboost import XGBoostPruningCallback
    except (ImportError, ModuleNotFoundError):
        XGBoostPruningCallback = None  # Optional pruning callback

if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
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
from aml.services.evaluation import (
    ChampionChallengerComparisonReport,
    compare_challenger_vs_champion,
    evaluate_model_statistically,
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


def _load_existing_champion(
    model_name: str,
) -> tuple[Any, str | None]:
    """Loads the active @champion model from MLflow prior to registering a new candidate."""
    try:
        client = MlflowClient()
        try:
            meta = client.get_model_version_by_alias(model_name, "champion")
            version = str(meta.version)
            model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
            log.info(f"Found existing '@champion' model [Version {version}] in MLflow.")
            return model, version
        except Exception:
            prod = client.get_latest_versions(model_name, stages=["Production"])
            if prod:
                version = str(prod[0].version)
                model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
                log.info(f"Found existing Production champion [Version {version}] in MLflow.")
                return model, version
    except Exception:
        pass
    return None, None


def _predict_champion(
    champion_model: Any,
    X_test: np.ndarray,
    X_test_df: pl.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, float]:
    """Executes inference with the loaded Champion model on the holdout test split."""
    t0 = time.perf_counter()
    if isinstance(champion_model, xgb.Booster):
        try:
            dmat = xgb.DMatrix(X_test, feature_names=feature_names)
            preds = champion_model.predict(dmat)
        except Exception:
            dmat = xgb.DMatrix(X_test)
            preds = champion_model.predict(dmat)
    elif hasattr(champion_model, "predict_proba"):
        probs = np.asarray(champion_model.predict_proba(X_test), dtype=np.float64)
        preds = probs[:, 1] if probs.ndim > 1 else probs
    else:
        try:
            preds = champion_model.predict(X_test)
        except Exception:
            preds = champion_model.predict(X_test_df)
        if hasattr(preds, "ndim") and isinstance(preds.ndim, int) and preds.ndim > 1:
            preds = preds[:, 1]

    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X_test))
    return np.asarray(preds, dtype=np.float64).ravel(), latency_ms


def train_xgboost(
    data_path: str | None = None,
    n_trials: int = 15,
    sample_size: int = 250000,
    device: str | None = None,
    tracking_uri: str = settings.MLFLOW_TRACKING_URI,
    experiment_name: str = "AML_Fraud_Detection",
    register_as: str | None = "FraudDetectionModel",
    save_local_artifacts: bool = False,
    n_bootstraps: int = 500,
    n_permutations: int = 500,
    alpha: float = 0.05,
    min_improvement: float = 0.005,
    min_pr_auc: float = 0.50,
    max_latency_ms: float = 50.0,
    gate_promotion: bool = True,
) -> dict[str, Any]:
    """
    Executes end-to-end Optuna hyperparameter optimization, model training,
    comprehensive statistical evaluation (with bootstrap CIs, calibration, and KS tests),
    and statistically-gated Champion vs. Challenger MLflow promotion.
    """
    chosen_device = device or check_gpu_availability()
    log.info(f"Using compute device: {chosen_device}")

    resolved_uri = resolve_mlflow_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)

    # Query existing Champion BEFORE training/registering candidate
    champion_model: Any = None
    champion_version: str | None = None
    if register_as and gate_promotion:
        champion_model, champion_version = _load_existing_champion(register_as)

    # 1. Load data
    resolved_path = data_path or os.path.join(settings.PROJECT_ROOT, "data", "training.csv")
    if os.path.exists(resolved_path):
        log.info(f"Loading raw dataset from {resolved_path}...")
        raw_df = pl.read_csv(resolved_path)
        if sample_size and sample_size > 0 and len(raw_df) > sample_size:
            log.info(
                f"Subsampling {sample_size} records while preserving all positive fraud cases..."
            )
            if "isFraud" in raw_df.columns:
                frauds = raw_df.filter(pl.col("isFraud") == 1)
                n_non_fraud = max(1, sample_size - len(frauds))
                non_frauds = raw_df.filter(pl.col("isFraud") == 0).sample(
                    n=min(n_non_fraud, len(raw_df.filter(pl.col("isFraud") == 0))), seed=42
                )
                raw_df = pl.concat([frauds, non_frauds]).sample(fraction=1.0, shuffle=True, seed=42)
            else:
                raw_df = raw_df.sample(n=sample_size, seed=42)
            log.info(
                f"Sampled dataset: {len(raw_df)} rows ({raw_df['isFraud'].sum() if 'isFraud' in raw_df.columns else 0} frauds)."
            )
    else:
        log.warning(f"Dataset not found at {resolved_path}. Generating synthetic benchmark data.")
        raw_df = generate_benchmark_data(num_rows=2000)

    # 2. Schema-validated Feature Engineering (Zero-Skew)
    log.info("Computing features via shared aml.services.feature_engineering pipeline...")
    raw_df = raw_df.sort("step")
    enriched_df = create_features(raw_df)

    # 3. Align with schema columns
    target_series = raw_df["isFraud"].to_numpy()
    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in enriched_df.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    # 4. Stratified Train / Val / Test Splits (on indices to prevent data leakage and preserve pairing)
    y = target_series
    all_indices = np.arange(len(y))
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        all_indices, y, test_size=0.15, stratify=y, random_state=42
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    # Fit preprocessor strictly on training split (idx_temp = train + val) to prevent test leakage
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    features_only_df = enriched_df.select(feature_cols)
    pandas_df = features_only_df.to_pandas()
    preprocessor.fit(pandas_df.iloc[idx_temp])
    X_processed = preprocessor.transform(pandas_df)
    final_feature_names = cat_cols + num_cols

    X_train = X_processed[idx_train]
    X_val = X_processed[idx_val]
    X_test = X_processed[idx_test]
    X_test_df = features_only_df[idx_test]

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

    # 7. Statistical Evaluation on Independent Holdout Test Set
    t0 = time.perf_counter()
    test_preds = final_clf.predict_proba(X_test)[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000.0 / len(y_test)

    eval_report = evaluate_model_statistically(
        y_true=y_test,
        y_prob=test_preds,
        latency_ms=latency_ms,
        threshold=settings.FRAUD_THRESHOLD,
        n_bootstraps=n_bootstraps,
        confidence_level=1.0 - alpha,
    )

    pr_ci = eval_report.bootstrap_cis["pr_auc"]
    roc_ci = eval_report.bootstrap_cis["roc_auc"]
    brier_ci = eval_report.bootstrap_cis["brier_score"]

    log.info(
        f"Test PR-AUC:      {eval_report.pr_auc:.4f} "
        f"(95% CI: [{pr_ci.ci_lower:.4f}, {pr_ci.ci_upper:.4f}], SE: {pr_ci.std_error:.4f})"
    )
    log.info(
        f"Test ROC-AUC:     {eval_report.roc_auc:.4f} "
        f"(95% CI: [{roc_ci.ci_lower:.4f}, {roc_ci.ci_upper:.4f}], Gini: {eval_report.ranking.gini_coefficient:.4f})"
    )
    log.info(
        f"Test Brier Score: {eval_report.brier_score:.4f} "
        f"(95% CI: [{brier_ci.ci_lower:.4f}, {brier_ci.ci_upper:.4f}], "
        f"BSS: {eval_report.calibration.brier_skill_score:.4f}, "
        f"ECE: {eval_report.calibration.expected_calibration_error:.4f})"
    )
    log.info(
        f"Separation Tests: KS={eval_report.ranking.ks_statistic:.4f} (p={eval_report.ranking.ks_pvalue:.3e}), "
        f"MCC={eval_report.threshold_metrics.mcc:.4f}, "
        f"Recall@Top5%={eval_report.ranking.recall_at_top_5pct:.4f}"
    )
    log.info(f"Latency:          {latency_ms:.2f} ms / sample")

    # 8. Paired Champion vs. Challenger Hypothesis Test (when gating promotion directly)
    comparison_report: ChampionChallengerComparisonReport | None = None
    promoted = False
    promotion_reason = "Candidate registered without automatic promotion."

    if gate_promotion:
        if champion_model is not None:
            try:
                champ_preds, champ_latency_ms = _predict_champion(
                    champion_model, X_test, X_test_df, final_feature_names
                )
                comparison_report = compare_challenger_vs_champion(
                    y_true=y_test,
                    y_prob_challenger=test_preds,
                    y_prob_champion=champ_preds,
                    challenger_latency_ms=latency_ms,
                    champion_latency_ms=champ_latency_ms,
                    threshold=settings.FRAUD_THRESHOLD,
                    alpha=alpha,
                    min_improvement=min_improvement,
                    min_pr_auc=min_pr_auc,
                    max_latency_ms=max_latency_ms,
                    n_bootstraps=n_bootstraps,
                    n_permutations=n_permutations,
                )
                promoted = comparison_report.promote_challenger
                promotion_reason = comparison_report.decision_summary
                log.info(f"Champion vs. Challenger Hypothesis Test: {promotion_reason}")
            except Exception as e:
                log.warning(f"Champion evaluation failed ({e}); evaluating against baseline floor.")

        if comparison_report is None:
            passed_baseline = (
                eval_report.pr_auc >= min_pr_auc
                and eval_report.ranking.ks_pvalue < alpha
                and latency_ms <= max_latency_ms
            )
            promoted = passed_baseline
            promotion_reason = (
                f"Inaugural champion passed statistical baseline quality gate "
                f"(PR-AUC={eval_report.pr_auc:.4f}, KS p={eval_report.ranking.ks_pvalue:.3e})."
                if passed_baseline
                else f"Inaugural candidate rejected by statistical quality floor "
                f"(PR-AUC={eval_report.pr_auc:.4f}, KS p={eval_report.ranking.ks_pvalue:.3e})."
            )

    # 9. MLflow Logging & Statistically-Gated Model Registration
    run_id = None
    model_uri = None
    model_version: str | None = None
    try:
        with mlflow.start_run(run_name="XGBoost_Optuna_Best") as run:
            run_id = run.info.run_id
            mlflow.log_params(best_params)
            mlflow.log_params(
                {
                    "algorithm": "xgboost",
                    "device": chosen_device,
                    "n_trials": n_trials,
                    "num_features": len(final_feature_names),
                    "n_bootstraps": n_bootstraps,
                    "alpha": alpha,
                }
            )
            mlflow.log_metrics(eval_report.to_mlflow_metrics())
            mlflow.log_dict(eval_report.to_dict(), "evaluation/statistical_evaluation_report.json")
            if comparison_report is not None:
                mlflow.log_metrics(comparison_report.to_mlflow_metrics())
                mlflow.log_dict(
                    comparison_report.to_dict(),
                    "evaluation/champion_vs_challenger_report.json",
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                preproc_tmp = os.path.join(tmp_dir, "preprocessor.pkl")
                feat_tmp = os.path.join(tmp_dir, "feature_names.npy")
                joblib.dump(preprocessor, preproc_tmp)
                np.save(feat_tmp, np.array(final_feature_names))
                mlflow.log_artifact(preproc_tmp, artifact_path="preprocessing")
                mlflow.log_artifact(feat_tmp, artifact_path="preprocessing")

            model_info = mlflow.xgboost.log_model(
                xgb_model=final_clf.get_booster(),
                artifact_path="xgboost-model",
                registered_model_name=register_as,
            )
            model_uri = model_info.model_uri
            log.info(f"XGBoost run logged to MLflow [Run ID: {run_id}]")

            if register_as:
                try:
                    client = MlflowClient()
                    reg_ver = getattr(model_info, "registered_model_version", None)
                    if reg_ver is not None:
                        model_version = str(reg_ver)
                    else:
                        latest_v = client.get_latest_versions(register_as)[-1].version
                        model_version = str(latest_v)

                    # Always register new candidate as @challenger first
                    client.set_registered_model_alias(register_as, "challenger", model_version)

                    if gate_promotion:
                        if promoted:
                            if champion_version and champion_version != model_version:
                                with contextlib.suppress(Exception):
                                    client.set_registered_model_alias(
                                        register_as, "previous_champion", champion_version
                                    )
                            client.set_registered_model_alias(
                                register_as, "champion", model_version
                            )
                            client.set_model_version_tag(
                                register_as, model_version, "promotion_status", "PROMOTED"
                            )
                            client.set_model_version_tag(
                                register_as,
                                model_version,
                                "promotion_reason",
                                promotion_reason[:250],
                            )
                            with contextlib.suppress(Exception):
                                client.transition_model_version_stage(
                                    name=register_as,
                                    version=model_version,
                                    stage="Production",
                                    archive_existing_versions=True,
                                )
                            log.info(
                                f"MLflow model '{register_as}' v{model_version} passed statistical hypothesis gate and was promoted to @champion."
                            )
                        else:
                            client.set_model_version_tag(
                                register_as, model_version, "promotion_status", "REJECTED"
                            )
                            client.set_model_version_tag(
                                register_as,
                                model_version,
                                "rejection_reason",
                                promotion_reason[:250],
                            )
                            log.warning(
                                f"MLflow model '{register_as}' v{model_version} did NOT pass statistical gate; keeping existing @champion. Reason: {promotion_reason}"
                            )
                except Exception as e:
                    log.warning(f"Could not update MLflow model alias/stage: {e}")
    except Exception as e:
        log.warning(f"MLflow logging encountered a non-fatal error: {e}")

    # 10. Update local ml_models directory ONLY if requested AND (not gated OR promoted)
    if save_local_artifacts and (not gate_promotion or promoted):
        os.makedirs(settings.PROJECT_ROOT / "ml_models", exist_ok=True)
        final_clf.get_booster().save_model(str(settings.MODEL_PATH))
        joblib.dump(preprocessor, str(settings.PREPROC_PATH))
        np.save(str(settings.FEATNAMES_PATH), np.array(final_feature_names))
        log.info(f"Synchronized local model artifacts to {settings.PROJECT_ROOT / 'ml_models'}")
    elif save_local_artifacts and gate_promotion and not promoted:
        log.warning(
            "Skipped overwriting local ml_models/ because Challenger did not pass statistical hypothesis gate."
        )

    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "model_version": model_version,
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "pr_auc": eval_report.pr_auc,
        "pr_auc_ci_lower": pr_ci.ci_lower,
        "pr_auc_ci_upper": pr_ci.ci_upper,
        "roc_auc": eval_report.roc_auc,
        "roc_auc_ci_lower": roc_ci.ci_lower,
        "roc_auc_ci_upper": roc_ci.ci_upper,
        "brier_score": eval_report.brier_score,
        "brier_skill_score": eval_report.calibration.brier_skill_score,
        "ece": eval_report.calibration.expected_calibration_error,
        "ks_statistic": eval_report.ranking.ks_statistic,
        "ks_pvalue": eval_report.ranking.ks_pvalue,
        "mcc": eval_report.threshold_metrics.mcc,
        "f1_score": eval_report.threshold_metrics.f1_score,
        "latency_ms": latency_ms,
        "model_name": register_as,
        "feature_names": final_feature_names,
        "statistical_evaluation": eval_report.to_dict(),
        "hypothesis_testing": comparison_report.to_dict() if comparison_report else None,
        "eval_report": eval_report,
        "X_test": X_test,
        "X_test_df": X_test_df,
        "y_test": y_test,
        "y_pred_proba": test_preds,
        "preprocessor": preprocessor,
        "booster": final_clf.get_booster(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and optimize XGBoost model for AML.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw training CSV")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250000,
        help="Max samples to train on while preserving all frauds (0 for full dataset)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--tracking-uri", type=str, default=settings.MLFLOW_TRACKING_URI)
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level alpha for hypothesis tests"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help="Minimum PR-AUC lift required to promote over Champion",
    )
    parser.add_argument(
        "--sync-local", action="store_true", help="Save artifacts to local ml_models/ if promoted"
    )
    args = parser.parse_args()

    results = train_xgboost(
        data_path=args.data_path,
        n_trials=args.trials,
        sample_size=args.sample_size,
        device=args.device,
        tracking_uri=args.tracking_uri,
        save_local_artifacts=args.sync_local,
        alpha=args.alpha,
        min_improvement=args.min_improvement,
    )
    excluded_keys = {
        "X_test",
        "X_test_df",
        "y_test",
        "y_pred_proba",
        "preprocessor",
        "booster",
        "eval_report",
        "statistical_evaluation",
        "hypothesis_testing",
    }
    print("\n--- Training & Statistical Evaluation Results ---")
    for k, v in results.items():
        if k not in excluded_keys:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
