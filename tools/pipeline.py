"""
Modern Automated MLOps Training, Statistical Evaluation & Deployment Pipeline
=============================================================================
Implements an automated, statistically-gated Champion vs. Challenger deployment
lifecycle for Anti-Money Laundering (AML) fraud detection models.

Key Capabilities:
1. Unified Feature Pipeline: Uses `aml.services.feature_engineering` directly,
   eliminating training-serving skew.
2. Multi-Model Support: Trains either XGBoost or NVIDIA KumoTabular foundation models.
3. Comprehensive Statistical Model Evaluation:
   - Stratified Non-Parametric Bootstrap (95% Confidence Intervals & Standard Errors)
     for PR-AUC, ROC-AUC, Brier Score, Log-Loss, F1-Score, and Matthews Correlation (MCC).
   - Probability Calibration: Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
     Brier Skill Score (BSS), and Spiegelhalter's z-test for calibration.
   - Distributional Separation & Ranking: Two-sample Kolmogorov-Smirnov (KS) test,
     Mann-Whitney U test, Gini coefficient, and Alert Budget Recall@Top-1%/5%/10%.
4. Paired Champion vs. Challenger Statistical Hypothesis Testing Gate:
   - Evaluates Challenger and active Champion on the IDENTICAL holdout test split.
   - Executes Paired Stratified Bootstrap PR-AUC test, Paired Permutation PR-AUC test,
     Fast DeLong's Correlated ROC-AUC test, Wilcoxon Signed-Rank Brier loss test,
     Paired Student's t-test (Diebold-Mariano), Exact McNemar's classification test,
     and Holm-Bonferroni multiple testing correction.
5. Safe Promotion & Instant Rollback: Promotes to `@champion` and syncs local artifacts
   ONLY after statistical superiority and operational SLA gates pass.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger as log
from mlflow.tracking import MlflowClient

# Ensure project root is in sys.path when invoked directly as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aml.config.settings import settings
from aml.services.evaluation import (
    ChampionChallengerComparisonReport,
    compare_challenger_vs_champion,
    evaluate_model_statistically,
)
from tools.train_kumo_tabular import train_kumo_tabular
from tools.train_xgboost import train_xgboost


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


MODEL_NAME = "FraudDetectionModel"
MLFLOW_URI = resolve_mlflow_uri(settings.MLFLOW_TRACKING_URI)


def get_mlflow_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_URI)
    return MlflowClient()


def load_champion_model(
    client: MlflowClient, model_name: str = MODEL_NAME
) -> tuple[Any, str | None]:
    """
    Loads the current active @champion model from MLflow Model Registry.
    Falls back to legacy 'Production' stage if aliases are not yet set.
    """
    # 1. Try modern Model Alias @champion
    try:
        alias_meta = client.get_model_version_by_alias(model_name, "champion")
        version = str(alias_meta.version)
        model_uri = f"models:/{model_name}@champion"
        log.info(f"Discovered active Champion model [Version {version}] via alias '@champion'.")
        model = mlflow.pyfunc.load_model(model_uri)
        return model, version
    except Exception:
        log.debug(
            "No model registered under alias '@champion'. Checking legacy stage 'Production'..."
        )

    # 2. Try legacy stage 'Production'
    try:
        prod_models = client.get_latest_versions(model_name, stages=["Production"])
        if prod_models:
            version = str(prod_models[0].version)
            model_uri = f"models:/{model_name}/{version}"
            log.info(
                f"Discovered active Champion model [Version {version}] via legacy stage 'Production'."
            )
            model = mlflow.pyfunc.load_model(model_uri)
            return model, version
    except Exception:
        log.debug("No model found in legacy stage 'Production'.")

    log.info(
        "No active Champion found in MLflow. The candidate will be evaluated as inaugural Champion."
    )
    return None, None


def predict_model_probabilities(
    model: Any,
    X_test: np.ndarray | pl.DataFrame,
    X_test_df: pl.DataFrame | None = None,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Executes model inference and returns (1D predicted fraud probabilities, per_sample_latency_ms).
    Supports XGBoost Booster, Scikit-Learn / XGBClassifier, and MLflow PyFunc models.
    """
    n_samples = len(X_test)
    t0 = time.perf_counter()

    if isinstance(model, xgb.Booster):
        try:
            dmat = (
                xgb.DMatrix(X_test, feature_names=feature_names)
                if feature_names is not None
                else xgb.DMatrix(X_test)
            )
            y_pred = model.predict(dmat)
        except Exception:
            dmat = xgb.DMatrix(np.asarray(X_test))
            y_pred = model.predict(dmat)
    elif hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(X_test), dtype=np.float64)
            if probs.size == 0 and n_samples > 0:
                raise ValueError("Empty output from predict_proba")
            y_pred = probs[:, 1] if probs.ndim > 1 else probs
        except Exception:
            y_pred = model.predict(X_test)
            if hasattr(y_pred, "ndim") and isinstance(y_pred.ndim, int) and y_pred.ndim > 1:
                y_pred = y_pred[:, 1]
    elif hasattr(model, "predict"):
        try:
            y_pred = model.predict(X_test)
        except Exception:
            if X_test_df is not None:
                y_pred = model.predict(X_test_df)
            else:
                raise
        if hasattr(y_pred, "ndim") and isinstance(y_pred.ndim, int) and y_pred.ndim > 1:
            y_pred = y_pred[:, 1]
    else:
        raise ValueError(f"Model {type(model)} has no predict or predict_proba method.")

    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    per_sample_latency_ms = total_latency_ms / max(1, n_samples)
    return np.asarray(y_pred, dtype=np.float64).ravel(), per_sample_latency_ms


def evaluate_model_on_test_data(
    model: Any,
    X_test: np.ndarray | pl.DataFrame,
    y_test: np.ndarray,
    X_test_df: pl.DataFrame | None = None,
    feature_names: list[str] | None = None,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """
    Computes comprehensive statistical production-gating metrics on the holdout evaluation set:
    - Stratified Bootstrap 95% CIs & Standard Errors (PR-AUC, ROC-AUC, Brier, Log-Loss, F1, MCC)
    - Calibration (Brier Skill Score, ECE, MCE, Spiegelhalter's z-test)
    - Separation & Ranking (Kolmogorov-Smirnov test, Mann-Whitney U test, Gini, Recall@Top-K%)
    - Operating-Point Metrics (Precision, Recall, Specificity, F1, F2, MCC, Optimal Thresholds)
    """
    y_pred, per_sample_latency_ms = predict_model_probabilities(
        model=model,
        X_test=X_test,
        X_test_df=X_test_df,
        feature_names=feature_names,
    )

    eval_report = evaluate_model_statistically(
        y_true=y_test,
        y_prob=y_pred,
        latency_ms=per_sample_latency_ms,
        threshold=settings.FRAUD_THRESHOLD,
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
    )

    pr_ci = eval_report.bootstrap_cis["pr_auc"]
    roc_ci = eval_report.bootstrap_cis["roc_auc"]
    brier_ci = eval_report.bootstrap_cis["brier_score"]

    return {
        "pr_auc": eval_report.pr_auc,
        "pr_auc_ci_lower": pr_ci.ci_lower,
        "pr_auc_ci_upper": pr_ci.ci_upper,
        "pr_auc_se": pr_ci.std_error,
        "roc_auc": eval_report.roc_auc,
        "roc_auc_ci_lower": roc_ci.ci_lower,
        "roc_auc_ci_upper": roc_ci.ci_upper,
        "roc_auc_se": roc_ci.std_error,
        "gini_coefficient": eval_report.ranking.gini_coefficient,
        "brier_score": eval_report.brier_score,
        "brier_score_ci_lower": brier_ci.ci_lower,
        "brier_score_ci_upper": brier_ci.ci_upper,
        "brier_skill_score": eval_report.calibration.brier_skill_score,
        "log_loss": eval_report.calibration.log_loss,
        "ece": eval_report.calibration.expected_calibration_error,
        "mce": eval_report.calibration.max_calibration_error,
        "spiegelhalter_z": eval_report.calibration.spiegelhalter_z,
        "spiegelhalter_pvalue": eval_report.calibration.spiegelhalter_pvalue,
        "is_well_calibrated": eval_report.calibration.is_well_calibrated,
        "ks_statistic": eval_report.ranking.ks_statistic,
        "ks_pvalue": eval_report.ranking.ks_pvalue,
        "mann_whitney_u": eval_report.ranking.mann_whitney_u,
        "mann_whitney_pvalue": eval_report.ranking.mann_whitney_pvalue,
        "recall_at_top_1pct": eval_report.ranking.recall_at_top_1pct,
        "recall_at_top_5pct": eval_report.ranking.recall_at_top_5pct,
        "recall_at_top_10pct": eval_report.ranking.recall_at_top_10pct,
        "precision_at_top_5pct": eval_report.ranking.precision_at_top_5pct,
        "lift_at_top_5pct": eval_report.ranking.lift_at_top_5pct,
        "precision": eval_report.threshold_metrics.precision,
        "recall": eval_report.threshold_metrics.recall,
        "specificity": eval_report.threshold_metrics.specificity,
        "f1_score": eval_report.threshold_metrics.f1_score,
        "f2_score": eval_report.threshold_metrics.f2_score,
        "balanced_accuracy": eval_report.threshold_metrics.balanced_accuracy,
        "mcc": eval_report.threshold_metrics.mcc,
        "cohens_kappa": eval_report.threshold_metrics.cohens_kappa,
        "optimal_f1_threshold": eval_report.threshold_metrics.optimal_f1_threshold,
        "max_f1_score": eval_report.threshold_metrics.max_f1_score,
        "optimal_youden_threshold": eval_report.threshold_metrics.optimal_youden_threshold,
        "youden_j_statistic": eval_report.threshold_metrics.youden_j_statistic,
        "latency_ms": per_sample_latency_ms,
        "statistical_report": eval_report.to_dict(),
        "eval_report": eval_report,
        "y_pred_proba": y_pred,
    }


def promote_model_to_champion(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    metrics: dict[str, Any],
    run_id: str | None = None,
    comparison_report: ChampionChallengerComparisonReport | None = None,
) -> None:
    """
    Safely promotes a statistically validated candidate model to @champion.
    Demotes the previous champion to @previous_champion to enable zero-downtime rollback.
    """
    # 1. Fetch current champion to demote
    prev_version: str | None = None
    try:
        curr_champ = client.get_model_version_by_alias(model_name, "champion")
        prev_version = str(curr_champ.version)
    except Exception:
        pass

    if prev_version and prev_version != str(new_version):
        client.set_registered_model_alias(model_name, "previous_champion", prev_version)
        log.info(f"Demoted previous champion (Version {prev_version}) to '@previous_champion'.")

    # 2. Assign @champion alias to new version
    client.set_registered_model_alias(model_name, "champion", str(new_version))
    log.info(f"Promoted Version {new_version} to '@champion'!")

    # 3. Add statistical audit tags
    client.set_model_version_tag(model_name, str(new_version), "promotion_status", "PROMOTED")
    client.set_model_version_tag(
        model_name, str(new_version), "eval_pr_auc", f"{metrics.get('pr_auc', 0.0):.4f}"
    )
    if "pr_auc_ci_lower" in metrics and "pr_auc_ci_upper" in metrics:
        client.set_model_version_tag(
            model_name,
            str(new_version),
            "eval_pr_auc_ci_95",
            f"[{metrics['pr_auc_ci_lower']:.4f}, {metrics['pr_auc_ci_upper']:.4f}]",
        )
    client.set_model_version_tag(
        model_name, str(new_version), "eval_roc_auc", f"{metrics.get('roc_auc', 0.0):.4f}"
    )
    client.set_model_version_tag(
        model_name, str(new_version), "eval_brier_score", f"{metrics.get('brier_score', 0.0):.4f}"
    )
    client.set_model_version_tag(
        model_name, str(new_version), "eval_latency_ms", f"{metrics.get('latency_ms', 0.0):.2f}"
    )
    if comparison_report is not None:
        client.set_model_version_tag(
            model_name,
            str(new_version),
            "eval_pr_auc_lift",
            f"{comparison_report.pr_auc_lift:+.4f}",
        )
        client.set_model_version_tag(
            model_name,
            str(new_version),
            "bootstrap_pr_auc_pvalue",
            f"{comparison_report.hypothesis_tests['bootstrap_pr_auc'].p_value:.4f}",
        )
        client.set_model_version_tag(
            model_name,
            str(new_version),
            "permutation_pr_auc_pvalue",
            f"{comparison_report.hypothesis_tests['permutation_pr_auc'].p_value:.4f}",
        )
        client.set_model_version_tag(
            model_name,
            str(new_version),
            "delong_roc_pvalue",
            f"{comparison_report.hypothesis_tests['delong_roc_auc'].p_value:.4f}",
        )
    client.set_model_version_tag(
        model_name,
        str(new_version),
        "promoted_at",
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # 4. Graceful backward compatibility with older stage-based readers
    with contextlib.suppress(Exception):
        client.transition_model_version_stage(
            name=model_name,
            version=str(new_version),
            stage="Production",
            archive_existing_versions=True,
        )


def sync_champion_to_local_models(
    client: MlflowClient,
    model_name: str = MODEL_NAME,
    target_dir: str | None = None,
    candidate_artifacts: dict[str, Any] | None = None,
) -> None:
    """
    Downloads/exports the current @champion artifacts to `ml_models/`
    so FastAPI and Celery workers can initialize with zero external dependencies.
    """
    out_dir = target_dir or str(settings.PROJECT_ROOT / "ml_models")
    os.makedirs(out_dir, exist_ok=True)

    dest_model = os.path.join(out_dir, "model.json")
    preproc_file = os.path.join(out_dir, "preprocessor.pkl")
    featnames_file = os.path.join(out_dir, "feature_names.npy")

    # Fast-path if in-memory candidate artifacts were just promoted
    if candidate_artifacts is not None and candidate_artifacts.get("booster") is not None:
        try:
            candidate_artifacts["booster"].save_model(dest_model)
            if candidate_artifacts.get("preprocessor") is not None:
                joblib.dump(candidate_artifacts["preprocessor"], preproc_file)
            if candidate_artifacts.get("feature_names") is not None:
                np.save(featnames_file, np.array(candidate_artifacts["feature_names"]))
            log.info(f"Synchronized promoted @champion artifacts directly to {out_dir}.")
            return
        except Exception as e:
            log.warning(f"Direct artifact export failed, falling back to MLflow download: {e}")

    try:
        champ_meta = client.get_model_version_by_alias(model_name, "champion")
        version = champ_meta.version
        run_id = champ_meta.run_id
        log.info(f"Synchronizing @champion (Version {version}, Run ID: {run_id}) to {out_dir}...")

        # Download model artifacts from run
        client.download_artifacts(run_id, "xgboost-model", out_dir)
        sub_model = os.path.join(out_dir, "xgboost-model", "model.xgb")
        sub_json = os.path.join(out_dir, "xgboost-model", "model.json")

        if os.path.exists(sub_json):
            with open(sub_json, "rb") as f_src, open(dest_model, "wb") as f_dst:
                f_dst.write(f_src.read())
            log.info(f"Updated {dest_model} from MLflow run.")
        elif os.path.exists(sub_model):
            booster = mlflow.xgboost.load_model(f"models:/{model_name}@champion")
            booster.save_model(dest_model)
            log.info(f"Exported booster to {dest_model}.")

        # Download preprocessing artifacts if logged in the run
        with contextlib.suppress(Exception):
            client.download_artifacts(run_id, "preprocessing", out_dir)
            sub_preproc = os.path.join(out_dir, "preprocessing", "preprocessor.pkl")
            sub_feats = os.path.join(out_dir, "preprocessing", "feature_names.npy")
            if os.path.exists(sub_preproc):
                shutil.copyfile(sub_preproc, preproc_file)
            if os.path.exists(sub_feats):
                shutil.copyfile(sub_feats, featnames_file)

    except Exception as e:
        log.warning(f"Could not automatically sync artifacts from MLflow run: {e}")
        log.info("Checking if existing local artifacts in ml_models/ are intact...")

    if (
        not os.path.exists(dest_model)
        or not os.path.exists(preproc_file)
        or not os.path.exists(featnames_file)
    ):
        log.warning(
            "Some local offline fallback files are missing. Generating baseline artifacts..."
        )
        train_xgboost(n_trials=3, save_local_artifacts=True)


def rollback_champion(client: MlflowClient, model_name: str = MODEL_NAME) -> bool:
    """
    Instantly rolls back production traffic to @previous_champion.
    """
    try:
        prev_champ = client.get_model_version_by_alias(model_name, "previous_champion")
        prev_version = str(prev_champ.version)
    except Exception:
        log.error("Rollback failed: No '@previous_champion' found in MLflow.")
        return False

    curr_champ_version: str | None = None
    try:
        curr_champ = client.get_model_version_by_alias(model_name, "champion")
        curr_champ_version = str(curr_champ.version)
    except Exception:
        pass

    # Swap aliases
    client.set_registered_model_alias(model_name, "champion", prev_version)
    if curr_champ_version:
        client.set_registered_model_alias(model_name, "previous_champion", curr_champ_version)

    log.info(f"Rollback successful: Version {prev_version} is now '@champion'!")
    sync_champion_to_local_models(client, model_name)
    return True


def run_full_pipeline(
    model_type: str = "xgboost",
    data_path: str | None = None,
    n_trials: int = 15,
    min_improvement: float = 0.005,
    min_pr_auc: float = 0.50,
    max_latency_ms: float = 50.0,
    alpha: float = 0.05,
    n_bootstraps: int = 500,
    n_permutations: int = 500,
    auto_promote: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Executes the complete statistically-gated MLOps workflow:
    1. Query active `@champion` model prior to candidate training.
    2. Train Challenger model (`xgboost` or `kumo-tabular`) WITHOUT overwriting local production files.
    3. Register Challenger to MLflow under `@challenger` and compute single-model statistical metrics
       (95% Stratified Bootstrap CIs, ECE/MCE calibration, Brier Skill Score, Spiegelhalter test, KS test).
    4. Evaluate active `@champion` on the IDENTICAL holdout test set `(X_test, y_test)` and execute
       Paired Statistical Hypothesis Tests:
       - Paired Stratified Bootstrap Superiority Test on PR-AUC
       - Paired Monte-Carlo Permutation Test on PR-AUC
       - Fast DeLong's Test on Correlated ROC Curves
       - Wilcoxon Signed-Rank Test & Paired t-Test on Per-Sample Brier Losses
       - Exact Binomial & Continuity-Corrected McNemar's Test
       - Holm-Bonferroni Step-Down Multiple Testing Correction
    5. Promote Challenger to `@champion` and synchronize offline artifacts ONLY if the statistical
       hypothesis test gate and operational SLA checks pass.
    """
    client = get_mlflow_client()
    log.info("==============================================================")
    log.info("   MLOps Statistical Evaluation & Hypothesis Testing Pipeline ")
    log.info(f"   Target Model:  {MODEL_NAME:<42}")
    log.info(f"   Architecture:  {model_type.upper():<42}")
    log.info(f"   Significance:  alpha={alpha:.3f}, min_lift={min_improvement:+.4f}          ")
    log.info("==============================================================")

    # 1. Query Existing Champion BEFORE training Challenger
    champion_model, champion_version = load_champion_model(client, MODEL_NAME)

    # 2. Train Candidate (Challenger) without overwriting local production artifacts
    if model_type == "kumo-tabular":
        log.info("Step 1: Training NVIDIA KumoTabular Foundation Model (Challenger)...")
        train_res = train_kumo_tabular(
            data_path=data_path,
            context_size=500,
            model_size="small",
            device=device or "cpu",
            mlflow_tracking_uri=MLFLOW_URI,
            register_as=MODEL_NAME,
            n_bootstraps=n_bootstraps,
        )
    elif model_type == "xgboost":
        log.info(f"Step 1: Training XGBoost Challenger with Optuna ({n_trials} trials)...")
        train_res = train_xgboost(
            data_path=data_path,
            n_trials=n_trials,
            device=device,
            tracking_uri=MLFLOW_URI,
            register_as=MODEL_NAME,
            save_local_artifacts=False,  # Never overwrite production artifacts before passing gate!
            n_bootstraps=n_bootstraps,
            gate_promotion=False,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")

    # Retrieve registered version of Challenger
    challenger_version = train_res.get("model_version")
    if not challenger_version:
        latest_versions = client.get_latest_versions(MODEL_NAME)
        challenger_version = str(latest_versions[0].version)

    log.info(f"Registered Challenger Model: Version {challenger_version} (alias '@challenger')")
    client.set_registered_model_alias(MODEL_NAME, "challenger", str(challenger_version))
    client.set_model_version_tag(
        MODEL_NAME, str(challenger_version), "promotion_status", "EVALUATING"
    )

    chal_eval_report = train_res["eval_report"]
    X_test = train_res["X_test"]
    X_test_df = train_res["X_test_df"]
    y_test = train_res["y_test"]
    chal_preds = train_res["y_pred_proba"]
    feat_names = train_res.get("feature_names")

    # 3. Statistical Decision Gate: Inaugural vs. Paired Champion-Challenger Hypothesis Testing
    decision: dict[str, Any] = {
        "challenger_version": str(challenger_version),
        "champion_version": champion_version,
        "challenger_pr_auc": float(train_res["pr_auc"]),
        "challenger_pr_auc_ci_95": [
            float(train_res["pr_auc_ci_lower"]),
            float(train_res["pr_auc_ci_upper"]),
        ],
        "challenger_roc_auc": float(train_res["roc_auc"]),
        "challenger_roc_auc_ci_95": [
            float(train_res["roc_auc_ci_lower"]),
            float(train_res["roc_auc_ci_upper"]),
        ],
        "challenger_brier_score": float(train_res["brier_score"]),
        "challenger_brier_skill_score": float(train_res["brier_skill_score"]),
        "challenger_ece": float(train_res["ece"]),
        "challenger_ks_statistic": float(train_res["ks_statistic"]),
        "challenger_ks_pvalue": float(train_res["ks_pvalue"]),
        "challenger_mcc": float(train_res["mcc"]),
        "challenger_latency_ms": float(train_res["latency_ms"]),
        "alpha": alpha,
        "promoted": False,
        "reason": "",
        "statistical_evaluation": chal_eval_report.to_dict(),
        "hypothesis_testing": None,
    }

    comparison_report: ChampionChallengerComparisonReport | None = None

    if (
        champion_model is None
        or champion_version is None
        or str(champion_version) == str(challenger_version)
    ):
        log.info(
            "Step 2: No distinct active Champion exists. Running Inaugural Statistical Quality Gate..."
        )
        passed_pr_floor = chal_eval_report.pr_auc >= min_pr_auc
        passed_ks_sep = chal_eval_report.ranking.ks_pvalue < alpha
        passed_mw_sep = chal_eval_report.ranking.mann_whitney_pvalue < alpha
        passed_latency = chal_eval_report.latency_ms <= max_latency_ms

        if passed_pr_floor and passed_ks_sep and passed_mw_sep and passed_latency:
            decision["promoted"] = True
            decision["reason"] = (
                f"Inaugural champion passed statistical baseline validation: "
                f"PR-AUC={chal_eval_report.pr_auc:.4f} (95% CI [{train_res['pr_auc_ci_lower']:.4f}, {train_res['pr_auc_ci_upper']:.4f}]), "
                f"KS={chal_eval_report.ranking.ks_statistic:.4f} (p={chal_eval_report.ranking.ks_pvalue:.3e} < {alpha}), "
                f"Mann-Whitney p={chal_eval_report.ranking.mann_whitney_pvalue:.3e} < {alpha}, "
                f"latency={chal_eval_report.latency_ms:.2f}ms <= {max_latency_ms}ms."
            )
        else:
            decision["promoted"] = False
            reasons = []
            if not passed_pr_floor:
                reasons.append(f"PR-AUC {chal_eval_report.pr_auc:.4f} < floor {min_pr_auc:.4f}")
            if not passed_ks_sep or not passed_mw_sep:
                reasons.append(
                    f"Failed statistical class separation (KS p={chal_eval_report.ranking.ks_pvalue:.3e}, "
                    f"MW p={chal_eval_report.ranking.mann_whitney_pvalue:.3e}, alpha={alpha})"
                )
            if not passed_latency:
                reasons.append(
                    f"Latency {chal_eval_report.latency_ms:.2f}ms > max {max_latency_ms:.2f}ms"
                )
            decision["reason"] = (
                "Inaugural candidate rejected by statistical baseline gate: " + "; ".join(reasons)
            )
    else:
        log.info(
            f"Step 2: Evaluating active Champion [Version {champion_version}] vs. "
            f"Challenger [Version {challenger_version}] on identical holdout test split (n={len(y_test)})..."
        )
        champ_preds, champ_latency_ms = predict_model_probabilities(
            model=champion_model,
            X_test=X_test,
            X_test_df=X_test_df,
            feature_names=feat_names,
        )

        comparison_report = compare_challenger_vs_champion(
            y_true=y_test,
            y_prob_challenger=chal_preds,
            y_prob_champion=champ_preds,
            challenger_latency_ms=float(train_res["latency_ms"]),
            champion_latency_ms=champ_latency_ms,
            threshold=settings.FRAUD_THRESHOLD,
            alpha=alpha,
            min_improvement=min_improvement,
            min_pr_auc=min_pr_auc,
            max_latency_ms=max_latency_ms,
            n_bootstraps=n_bootstraps,
            n_permutations=n_permutations,
        )

        champ_eval = comparison_report.champion_eval
        champ_pr_ci = champ_eval.bootstrap_cis["pr_auc"]
        log.info(
            f"Champion [v{champion_version}] Holdout PR-AUC:   {champ_eval.pr_auc:.4f} "
            f"(95% CI: [{champ_pr_ci.ci_lower:.4f}, {champ_pr_ci.ci_upper:.4f}]), "
            f"ROC-AUC: {champ_eval.roc_auc:.4f}, Brier: {champ_eval.brier_score:.4f}"
        )
        log.info(
            f"Challenger [v{challenger_version}] Holdout PR-AUC: {chal_eval_report.pr_auc:.4f} "
            f"(95% CI: [{train_res['pr_auc_ci_lower']:.4f}, {train_res['pr_auc_ci_upper']:.4f}]), "
            f"ROC-AUC: {chal_eval_report.roc_auc:.4f}, Brier: {chal_eval_report.brier_score:.4f}"
        )

        for _test_key, test_res in comparison_report.hypothesis_tests.items():
            log.info(
                f"  [Hypothesis Test] {test_res.test_name:<45} | "
                f"Effect={test_res.effect_size:+.4f} | Stat={test_res.statistic:8.4f} | "
                f"p={test_res.p_value:.4f} (Holm adj p={test_res.adjusted_p_value:.4f}) | "
                f"Sig={'YES' if test_res.significant else 'NO'}"
            )

        decision["champion_pr_auc"] = champ_eval.pr_auc
        decision["champion_roc_auc"] = champ_eval.roc_auc
        decision["champion_brier_score"] = champ_eval.brier_score
        decision["champion_latency_ms"] = champ_latency_ms
        decision["pr_auc_lift"] = comparison_report.pr_auc_lift
        decision["roc_auc_lift"] = comparison_report.roc_auc_lift
        decision["brier_reduction"] = comparison_report.brier_reduction
        decision["is_statistically_superior"] = comparison_report.is_statistically_superior
        decision["holm_bonferroni_significant_count"] = (
            comparison_report.holm_bonferroni_significant_count
        )
        decision["hypothesis_testing"] = {
            k: v.to_dict() for k, v in comparison_report.hypothesis_tests.items()
        }
        decision["promoted"] = comparison_report.promote_challenger
        decision["reason"] = comparison_report.decision_summary

        # Log paired comparison report and hypothesis test p-values to the Challenger's MLflow run
        run_id = train_res.get("run_id")
        if run_id:
            with contextlib.suppress(Exception), mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(comparison_report.to_mlflow_metrics())
                mlflow.log_dict(
                    comparison_report.to_dict(),
                    "evaluation/champion_vs_challenger_report.json",
                )

    # 4. Execute Promotion ONLY if approved by Statistical Hypothesis Gate
    if decision["promoted"] and auto_promote:
        log.info(f"Statistical Gating Decision: APPROVED ({decision['reason']})")
        promote_model_to_champion(
            client=client,
            model_name=MODEL_NAME,
            new_version=str(challenger_version),
            metrics=train_res,
            run_id=train_res.get("run_id"),
            comparison_report=comparison_report,
        )
        sync_champion_to_local_models(
            client=client,
            model_name=MODEL_NAME,
            candidate_artifacts=train_res if model_type == "xgboost" else None,
        )
        log.info(
            "Pipeline execution successfully completed with statistically validated promotion!"
        )
    else:
        log.info(f"Statistical Gating Decision: NOT PROMOTED ({decision['reason']}).")
        client.set_model_version_tag(
            MODEL_NAME, str(challenger_version), "promotion_status", "REJECTED"
        )
        client.set_model_version_tag(
            MODEL_NAME, str(challenger_version), "rejection_reason", decision["reason"][:250]
        )

    return decision


def main():
    parser = argparse.ArgumentParser(
        description="Modern MLOps Training, Statistical Evaluation & Deployment Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command to run")

    # Command: run (end-to-end)
    run_parser = subparsers.add_parser(
        "run", help="Run full pipeline: Train -> Statistical Hypothesis Gate -> Promote -> Sync"
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "kumo-tabular"],
        help="Architecture",
    )
    run_parser.add_argument("--data-path", type=str, default=None, help="Path to training CSV")
    run_parser.add_argument("--trials", type=int, default=15, help="Optuna trials for XGBoost")
    run_parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help="Minimum PR-AUC lift required to promote",
    )
    run_parser.add_argument(
        "--min-pr-auc",
        type=float,
        default=0.50,
        help="Minimum absolute PR-AUC quality floor",
    )
    run_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance level alpha for hypothesis tests (default: 0.05)",
    )
    run_parser.add_argument(
        "--bootstraps",
        type=int,
        default=500,
        help="Number of stratified bootstrap resamples for CIs and hypothesis testing",
    )
    run_parser.add_argument(
        "--permutations",
        type=int,
        default=500,
        help="Number of Monte-Carlo permutations for randomization test",
    )
    run_parser.add_argument(
        "--max-latency", type=float, default=50.0, help="Maximum allowed latency in ms"
    )
    run_parser.add_argument(
        "--no-auto-promote", action="store_true", help="Skip automatic promotion"
    )
    run_parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")

    # Command: rollback
    subparsers.add_parser(
        "rollback", help="Instantly rollback production model to @previous_champion"
    )

    # Command: sync
    subparsers.add_parser(
        "sync", help="Synchronize current @champion model to local ml_models/ directory"
    )

    # Command: promote
    promote_parser = subparsers.add_parser(
        "promote", help="Manually promote a specific version to @champion"
    )
    promote_parser.add_argument(
        "--version", type=str, required=True, help="Model version number to promote"
    )

    args = parser.parse_args()

    if not args.command or args.command == "run":
        res = run_full_pipeline(
            model_type=getattr(args, "model", "xgboost"),
            data_path=getattr(args, "data_path", None),
            n_trials=getattr(args, "trials", 15),
            min_improvement=getattr(args, "min_improvement", 0.005),
            min_pr_auc=getattr(args, "min_pr_auc", 0.50),
            max_latency_ms=getattr(args, "max_latency", 50.0),
            alpha=getattr(args, "alpha", 0.05),
            n_bootstraps=getattr(args, "bootstraps", 500),
            n_permutations=getattr(args, "permutations", 500),
            auto_promote=not getattr(args, "no_auto_promote", False),
            device=getattr(args, "device", None),
        )
        print("\n=== Pipeline Statistical Decision Summary ===")
        print(json.dumps(res, indent=2))

    elif args.command == "rollback":
        client = get_mlflow_client()
        success = rollback_champion(client)
        sys.exit(0 if success else 1)

    elif args.command == "sync":
        client = get_mlflow_client()
        sync_champion_to_local_models(client)
        print("Local artifact synchronization complete.")

    elif args.command == "promote":
        client = get_mlflow_client()
        promote_model_to_champion(client, MODEL_NAME, args.version, {})
        sync_champion_to_local_models(client)
        print(f"Version {args.version} promoted to @champion.")


if __name__ == "__main__":
    main()
