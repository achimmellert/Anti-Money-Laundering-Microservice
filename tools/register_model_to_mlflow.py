"""
Statistically-Gated MLflow Model Registration & Champion-Challenger Promotion
=============================================================================
Replaces naive "push-and-overwrite" model registration with a statistically
validated evaluation and hypothesis-testing gate before promoting any model
to `@champion`.

Workflow:
1. Loads the candidate local XGBoost model and prepares a stratified holdout test set.
2. Evaluates candidate performance using stratified bootstrap 95% confidence intervals,
   calibration tests (ECE, MCE, Brier Skill Score, Spiegelhalter's z-test), and
   distributional separation tests (Kolmogorov-Smirnov, Mann-Whitney U, Gini).
3. Registers the candidate version in MLflow Model Registry under `@challenger`.
4. If an active `@champion` exists, evaluates both models on the identical holdout
   test set and executes paired statistical hypothesis tests:
   - Paired Stratified Bootstrap PR-AUC Superiority Test
   - Paired Monte-Carlo Permutation PR-AUC Test
   - Fast DeLong's Test for Correlated ROC Curves
   - Wilcoxon Signed-Rank & Paired t-Test on Per-Transaction Brier Losses
   - Exact Binomial & Continuity-Corrected McNemar's Test
   - Holm-Bonferroni Multiple Testing Correction
5. Promotes `@challenger` -> `@champion` (and demotes old `@champion` -> `@previous_champion`)
   ONLY when the statistical gate passes (or when no prior champion exists and baseline
   statistical quality is met, or when `force_promote=True`).
"""

from __future__ import annotations

import argparse
import contextlib
import os
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
from sklearn.compose import ColumnTransformer
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
    compare_challenger_vs_champion,
    evaluate_model_statistically,
)
from aml.services.feature_engineering import create_features
from tools.train_xgboost import generate_benchmark_data


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


MLFLOW_URI = resolve_mlflow_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_URI)

BASE_DIR = str(settings.PROJECT_ROOT)
MODEL_PATH = str(settings.MODEL_PATH)


def _prepare_evaluation_holdout(
    data_path: str | None = None,
    preproc_path: str | None = None,
) -> tuple[np.ndarray, pl.DataFrame, np.ndarray, list[str]]:
    """
    Loads or generates a holdout test split with schema-validated engineered features
    so candidate and champion models can be compared on identical transactions.
    """
    resolved_data = data_path or os.path.join(settings.PROJECT_ROOT, "data", "training.csv")
    if os.path.exists(resolved_data):
        raw_df = pl.read_csv(resolved_data)
    else:
        raw_df = generate_benchmark_data(num_rows=1500)

    raw_sorted = raw_df.sort("step")
    enriched_df = create_features(raw_df)
    y = raw_sorted["isFraud"].to_numpy()

    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in enriched_df.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    all_idx = np.arange(len(y))
    idx_train, idx_test, _, y_test = train_test_split(
        all_idx, y, test_size=0.20, stratify=y, random_state=42
    )

    features_df = enriched_df.select(feature_cols)
    pandas_df = features_df.to_pandas()

    resolved_preproc = preproc_path or str(settings.PREPROC_PATH)
    preprocessor = None
    if os.path.exists(resolved_preproc):
        with contextlib.suppress(Exception):
            preprocessor = joblib.load(resolved_preproc)

    if preprocessor is not None:
        try:
            X_test = preprocessor.transform(pandas_df.iloc[idx_test])
            feat_names = list(RAW_FEATURE_COLUMNS)
            if settings.FEATNAMES_PATH.exists():
                loaded_names = np.load(settings.FEATNAMES_PATH, allow_pickle=True)
                feat_names = [str(x) for x in loaded_names.tolist()]
            return X_test, features_df[idx_test], y_test, feat_names
        except Exception:
            pass

    fallback_preproc = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    fallback_preproc.fit(pandas_df.iloc[idx_train])
    X_test = fallback_preproc.transform(pandas_df.iloc[idx_test])
    return X_test, features_df[idx_test], y_test, cat_cols + num_cols


def _predict_booster_or_pyfunc(
    model: Any,
    X_test: np.ndarray,
    X_test_df: pl.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, float]:
    """Executes inference on a Booster, sklearn estimator, or MLflow PyFunc model."""
    t0 = time.perf_counter()
    if isinstance(model, xgb.Booster):
        try:
            dmat = xgb.DMatrix(X_test, feature_names=feature_names)
            preds = model.predict(dmat)
        except Exception:
            dmat = xgb.DMatrix(X_test)
            preds = model.predict(dmat)
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        preds = probs[:, 1] if hasattr(probs, "ndim") and probs.ndim > 1 else probs
    elif hasattr(model, "predict"):
        try:
            preds = model.predict(X_test)
        except Exception:
            preds = model.predict(X_test_df)
        if hasattr(preds, "ndim") and preds.ndim > 1:
            preds = preds[:, 1]
    else:
        raise ValueError(f"Unsupported model object for evaluation: {type(model)}")

    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X_test))
    return np.asarray(preds, dtype=np.float64).ravel(), latency_ms


def register_model_to_mlflow(
    model_path: str = MODEL_PATH,
    model_name: str = "FraudDetectionModel",
    alias: str = "champion",
    data_path: str | None = None,
    alpha: float = 0.05,
    min_improvement: float = 0.0,
    min_pr_auc: float = 0.50,
    max_latency_ms: float = 50.0,
    n_bootstraps: int = 300,
    n_permutations: int = 300,
    force_promote: bool = False,
) -> str | None:
    """
    Evaluates a local XGBoost model using modern statistical methods, registers it in MLflow
    as `@challenger`, and runs paired Champion vs. Challenger hypothesis tests before promoting
    to `@champion`.
    """
    if not os.path.exists(model_path):
        log.error(f"Model file not found at {model_path}.")
        return None

    booster = xgb.Booster()
    booster.load_model(model_path)
    log.info(f"Loaded candidate local model from {model_path}.")

    experiment_name = "AML_Fraud_Detection"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # 1. Discover existing Champion BEFORE registering candidate
    prev_champion_version: str | None = None
    champion_model: Any = None
    try:
        model_meta = client.get_model_version_by_alias(model_name, alias)
        prev_champion_version = str(model_meta.version)
        log.info(f"Current '@{alias}' is version {prev_champion_version}.")
        with contextlib.suppress(Exception):
            champion_model = mlflow.xgboost.load_model(f"models:/{model_name}@{alias}")
        if champion_model is None:
            with contextlib.suppress(Exception):
                champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    except Exception:
        log.info(f"No existing '@{alias}' found for model '{model_name}'.")

    # 2. Evaluate Candidate Statistically on Holdout Set
    X_test, X_test_df, y_test, feat_names = _prepare_evaluation_holdout(data_path=data_path)
    chal_preds, chal_latency_ms = _predict_booster_or_pyfunc(booster, X_test, X_test_df, feat_names)

    chal_eval = evaluate_model_statistically(
        y_true=y_test,
        y_prob=chal_preds,
        latency_ms=chal_latency_ms,
        threshold=settings.FRAUD_THRESHOLD,
        n_bootstraps=n_bootstraps,
        confidence_level=1.0 - alpha,
    )
    pr_ci = chal_eval.bootstrap_cis["pr_auc"]
    log.info(
        f"Candidate Statistical Evaluation: PR-AUC={chal_eval.pr_auc:.4f} "
        f"(95% CI: [{pr_ci.ci_lower:.4f}, {pr_ci.ci_upper:.4f}]), "
        f"ROC-AUC={chal_eval.roc_auc:.4f}, Brier={chal_eval.brier_score:.4f}, "
        f"ECE={chal_eval.calibration.expected_calibration_error:.4f}, "
        f"KS={chal_eval.ranking.ks_statistic:.4f} (p={chal_eval.ranking.ks_pvalue:.3e})"
    )

    # 3. Compare against existing Champion via Hypothesis Testing if one exists
    comparison_report = None
    if champion_model is not None:
        try:
            champ_preds, champ_latency_ms = _predict_booster_or_pyfunc(
                champion_model, X_test, X_test_df, feat_names
            )
            comparison_report = compare_challenger_vs_champion(
                y_true=y_test,
                y_prob_challenger=chal_preds,
                y_prob_champion=champ_preds,
                challenger_latency_ms=chal_latency_ms,
                champion_latency_ms=champ_latency_ms,
                threshold=settings.FRAUD_THRESHOLD,
                alpha=alpha,
                min_improvement=min_improvement,
                min_pr_auc=min_pr_auc,
                max_latency_ms=max_latency_ms,
                n_bootstraps=n_bootstraps,
                n_permutations=n_permutations,
            )
            log.info(
                f"Champion vs. Challenger Hypothesis Test: {comparison_report.decision_summary}"
            )
        except Exception as e:
            log.warning(f"Could not evaluate existing champion for paired comparison: {e}")

    # 4. Log Model & Statistical Artifacts to MLflow and Register Candidate
    with mlflow.start_run(run_name="Model_Statistical_Evaluation_And_Registration") as run:
        mlflow.log_params(
            {
                "source_model_path": model_path,
                "alpha": alpha,
                "min_improvement": min_improvement,
                "n_bootstraps": n_bootstraps,
                "n_permutations": n_permutations,
                "force_promote": force_promote,
            }
        )
        if comparison_report is not None:
            mlflow.log_metrics(comparison_report.to_mlflow_metrics())
            mlflow.log_dict(
                comparison_report.to_dict(),
                "evaluation/champion_vs_challenger_report.json",
            )
        else:
            mlflow.log_metrics(chal_eval.to_mlflow_metrics())
            mlflow.log_dict(
                chal_eval.to_dict(),
                "evaluation/statistical_evaluation_report.json",
            )

        mlflow.xgboost.log_model(xgb_model=booster, artifact_path="xgboost-model")
        model_uri = f"runs:/{run.info.run_id}/xgboost-model"
        reg_result = mlflow.register_model(model_uri=model_uri, name=model_name)
        new_version = str(reg_result.version)

    # Always assign @challenger to the newly registered candidate first
    with contextlib.suppress(Exception):
        client.set_registered_model_alias(model_name, "challenger", new_version)

    # 5. Statistical Promotion Decision Gate
    should_promote = False
    decision_reason = ""
    if force_promote:
        should_promote = True
        decision_reason = "Force promotion explicitly requested."
    elif comparison_report is not None:
        should_promote = comparison_report.promote_challenger
        decision_reason = comparison_report.decision_summary
    else:
        # Inaugural model quality check
        passed_baseline = (
            chal_eval.pr_auc >= min_pr_auc
            and chal_eval.ranking.ks_pvalue < alpha
            and chal_latency_ms <= max_latency_ms
        )
        should_promote = passed_baseline
        decision_reason = (
            f"Inaugural champion passed baseline statistical gate (PR-AUC={chal_eval.pr_auc:.4f}, KS p={chal_eval.ranking.ks_pvalue:.3e})."
            if passed_baseline
            else f"Inaugural candidate failed statistical quality floor (PR-AUC={chal_eval.pr_auc:.4f}, KS p={chal_eval.ranking.ks_pvalue:.3e})."
        )

    if not should_promote:
        log.warning(
            f"Model version {new_version} did NOT pass statistical promotion gate and will stay '@challenger'. "
            f"Reason: {decision_reason}"
        )
        with contextlib.suppress(Exception):
            client.set_model_version_tag(model_name, new_version, "promotion_status", "REJECTED")
            client.set_model_version_tag(
                model_name, new_version, "rejection_reason", decision_reason
            )
        return new_version

    # 6. Promote to @champion and demote previous champion to @previous_champion
    if prev_champion_version and prev_champion_version != new_version:
        try:
            client.set_registered_model_alias(
                model_name, "previous_champion", prev_champion_version
            )
            log.info(f"Assigned '@previous_champion' to version {prev_champion_version}.")
        except Exception as e:
            log.warning(f"Could not assign '@previous_champion': {e}")

    client.set_registered_model_alias(model_name, alias, new_version)
    log.info(f"Model version {new_version} statistically validated and assigned alias '@{alias}'!")

    with contextlib.suppress(Exception):
        client.set_model_version_tag(model_name, new_version, "promotion_status", "PROMOTED")
        client.set_model_version_tag(model_name, new_version, "promotion_reason", decision_reason)
        client.set_model_version_tag(
            model_name, new_version, "eval_pr_auc", f"{chal_eval.pr_auc:.4f}"
        )
        client.set_model_version_tag(
            model_name,
            new_version,
            "eval_pr_auc_ci_95",
            f"[{pr_ci.ci_lower:.4f}, {pr_ci.ci_upper:.4f}]",
        )

    # Backward compatibility with legacy systems still checking stage="Production"
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        log.debug(f"Stage transition skipped (modern MLflow prefers aliases): {e}")

    return new_version


def main():
    parser = argparse.ArgumentParser(
        description="Statistically evaluate and register a local model to MLflow."
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to model.json")
    parser.add_argument("--data-path", type=str, default=None, help="Path to evaluation CSV")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level alpha")
    parser.add_argument(
        "--min-improvement", type=float, default=0.0, help="Minimum PR-AUC lift margin"
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Bypass statistical hypothesis test gate and force promotion",
    )
    args = parser.parse_args()
    register_model_to_mlflow(
        model_path=args.model_path,
        data_path=args.data_path,
        alpha=args.alpha,
        min_improvement=args.min_improvement,
        force_promote=args.force_promote,
    )


if __name__ == "__main__":
    main()
