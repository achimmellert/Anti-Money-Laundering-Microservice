"""
MLOps Statistical Drift Detection, Concept Drift & Production Monitoring Service
================================================================================
Provides core statistical monitoring and data-reconciliation primitives executed
by the Apache Airflow ML Engineer pipelines:

1. Covariate / Feature Drift Detection:
   - Population Stability Index (PSI) with quantile binning and Laplace smoothing
   - Two-Sample Kolmogorov-Smirnov (KS) test per feature with Holm-Bonferroni FWER correction
   - First-order Wasserstein (Earth Mover's) distance normalized by reference standard deviation

2. Prediction Score & Concept Drift Detection:
   - Predicted probability distribution shift (PSI & KS test on `y_pred_proba`)
   - Alert / Block rate shift monitoring against compliance capacity SLAs
   - Delayed ground-truth concept drift evaluation (`evaluate_model_statistically` on reconciled labels)

3. Ground-Truth Label Reconciliation & Dead-Letter Queue (DLQ) Auditing:
   - Joins confirmed `TransactionsHistory.isFraud` outcomes into `MonitoringInference.y_true`
   - Audits `DeadLetterQueue` task failures to surface schema or feature-pipeline regressions
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
from loguru import logger as log
from scipy import stats
from sqlalchemy import func
from sqlalchemy.orm import Session

from aml.config.settings import settings
from aml.models.dataframes import NUMERIC_FEATURES
from aml.models.db_models import (
    DeadLetterQueue,
    MonitoringInference,
    TransactionsHistory,
)
from aml.services.evaluation import evaluate_model_statistically


@dataclass(frozen=True)
class FeatureDriftMetric:
    """Statistical drift metrics for a single feature between reference and production windows."""

    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    adjusted_pvalue: float
    wasserstein_distance: float
    reference_mean: float
    production_mean: float
    drift_severity: str  # "STABLE" | "MODERATE" | "CRITICAL"
    is_drifted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftMonitoringReport:
    """Comprehensive covariate, prediction, and concept drift report for Airflow MLOps gating."""

    reference_samples: int
    production_samples: int
    evaluated_at: str
    feature_drift: dict[str, FeatureDriftMetric]
    drifted_features_count: int
    critical_psi_features: list[str]
    max_feature_psi: float
    mean_feature_psi: float
    prediction_psi: float
    prediction_ks_statistic: float
    prediction_ks_pvalue: float
    reference_alert_rate: float
    production_alert_rate: float
    alert_rate_shift: float
    labeled_sample_count: int
    live_pr_auc: float | None = None
    live_roc_auc: float | None = None
    live_brier_score: float | None = None
    live_ece: float | None = None
    should_trigger_retraining: bool = False
    trigger_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_samples": self.reference_samples,
            "production_samples": self.production_samples,
            "evaluated_at": self.evaluated_at,
            "drifted_features_count": self.drifted_features_count,
            "critical_psi_features": self.critical_psi_features,
            "max_feature_psi": self.max_feature_psi,
            "mean_feature_psi": self.mean_feature_psi,
            "prediction_psi": self.prediction_psi,
            "prediction_ks_statistic": self.prediction_ks_statistic,
            "prediction_ks_pvalue": self.prediction_ks_pvalue,
            "reference_alert_rate": self.reference_alert_rate,
            "production_alert_rate": self.production_alert_rate,
            "alert_rate_shift": self.alert_rate_shift,
            "labeled_sample_count": self.labeled_sample_count,
            "live_pr_auc": self.live_pr_auc,
            "live_roc_auc": self.live_roc_auc,
            "live_brier_score": self.live_brier_score,
            "live_ece": self.live_ece,
            "should_trigger_retraining": self.should_trigger_retraining,
            "trigger_reasons": self.trigger_reasons,
            "feature_drift": {k: v.to_dict() for k, v in self.feature_drift.items()},
        }

    def to_mlflow_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "drift_reference_samples": float(self.reference_samples),
            "drift_production_samples": float(self.production_samples),
            "drift_drifted_features_count": float(self.drifted_features_count),
            "drift_max_feature_psi": float(self.max_feature_psi),
            "drift_mean_feature_psi": float(self.mean_feature_psi),
            "drift_prediction_psi": float(self.prediction_psi),
            "drift_prediction_ks_stat": float(self.prediction_ks_statistic),
            "drift_prediction_ks_pvalue": float(self.prediction_ks_pvalue),
            "drift_reference_alert_rate": float(self.reference_alert_rate),
            "drift_production_alert_rate": float(self.production_alert_rate),
            "drift_alert_rate_shift": float(self.alert_rate_shift),
            "drift_should_retrain": 1.0 if self.should_trigger_retraining else 0.0,
        }
        if self.live_pr_auc is not None and math.isfinite(self.live_pr_auc):
            metrics["live_pr_auc"] = float(self.live_pr_auc)
        if self.live_roc_auc is not None and math.isfinite(self.live_roc_auc):
            metrics["live_roc_auc"] = float(self.live_roc_auc)
        if self.live_brier_score is not None and math.isfinite(self.live_brier_score):
            metrics["live_brier_score"] = float(self.live_brier_score)
        if self.live_ece is not None and math.isfinite(self.live_ece):
            metrics["live_ece"] = float(self.live_ece)
        return metrics


def compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Computes the Population Stability Index (PSI) between a reference distribution
    and a production distribution using reference quantile edges.

    Interpretation thresholds (industry standard in credit risk & AML):
    - PSI < 0.10:  No significant population shift ("STABLE")
    - 0.10 <= PSI < 0.25: Moderate shift ("MODERATE")
    - PSI >= 0.25: Critical distributional shift requiring action ("CRITICAL")
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    prod = np.asarray(production, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    prod = prod[np.isfinite(prod)]

    if len(ref) < 2 or len(prod) < 2:
        return 0.0

    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.unique(np.percentile(ref, quantiles))
    if len(bin_edges) <= 2:
        # Constant or near-constant feature; fallback to uniform range
        min_val = min(float(np.min(ref)), float(np.min(prod)))
        max_val = max(float(np.max(ref)), float(np.max(prod)))
        if abs(max_val - min_val) < 1e-12:
            return 0.0
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    # Extend outer boundaries to capture out-of-range production values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    prod_counts, _ = np.histogram(prod, bins=bin_edges)

    ref_pct = np.clip(ref_counts / len(ref), eps, 1.0)
    prod_pct = np.clip(prod_counts / len(prod), eps, 1.0)

    psi_val = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    return max(0.0, psi_val)


def compute_drift_monitoring_report(
    reference_df: pl.DataFrame,
    production_df: pl.DataFrame,
    reference_probs: np.ndarray,
    production_probs: np.ndarray,
    production_labels: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
    psi_threshold: float = 0.25,
    alpha: float = 0.05,
    baseline_pr_auc: float = 0.75,
    max_pr_auc_drop: float = 0.05,
    threshold: float = settings.FRAUD_THRESHOLD,
) -> DriftMonitoringReport:
    """
    Executes statistical covariate drift, prediction score drift, and delayed concept
    drift analysis to determine whether automated model retraining should be triggered.
    """
    cols = feature_columns or [
        c for c in NUMERIC_FEATURES if c in reference_df.columns and c in production_df.columns
    ]

    raw_pvalues: list[float] = []
    interim_metrics: list[dict[str, Any]] = []

    for col in cols:
        ref_arr = reference_df[col].to_numpy().astype(np.float64)
        prod_arr = production_df[col].to_numpy().astype(np.float64)
        ref_clean = ref_arr[np.isfinite(ref_arr)]
        prod_clean = prod_arr[np.isfinite(prod_arr)]

        if len(ref_clean) < 2 or len(prod_clean) < 2:
            psi_val = 0.0
            ks_stat, ks_pval = 0.0, 1.0
            w_dist = 0.0
            ref_mean, prod_mean = 0.0, 0.0
        else:
            psi_val = compute_psi(ref_clean, prod_clean)
            ks_res = stats.ks_2samp(ref_clean, prod_clean, alternative="two-sided")
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)
            ref_std = max(float(np.std(ref_clean)), 1e-6)
            w_dist = float(stats.wasserstein_distance(ref_clean, prod_clean) / ref_std)
            ref_mean = float(np.mean(ref_clean))
            prod_mean = float(np.mean(prod_clean))

        raw_pvalues.append(ks_pval)
        interim_metrics.append(
            {
                "feature_name": col,
                "psi": psi_val,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "wasserstein_distance": w_dist,
                "reference_mean": ref_mean,
                "production_mean": prod_mean,
            }
        )

    # Apply Holm-Bonferroni correction across feature KS tests
    m = len(raw_pvalues)
    adjusted_p = np.ones(m, dtype=np.float64)
    if m > 0:
        p_arr = np.asarray(raw_pvalues, dtype=np.float64)
        order = np.argsort(p_arr)
        running_max = 0.0
        for rank_idx, orig_idx in enumerate(order):
            adj = min(1.0, float(p_arr[orig_idx] * (m - rank_idx)))
            running_max = max(running_max, adj)
            adjusted_p[orig_idx] = running_max

    feature_drift: dict[str, FeatureDriftMetric] = {}
    critical_psi_features: list[str] = []
    drifted_count = 0

    for idx, item in enumerate(interim_metrics):
        psi_val = item["psi"]
        adj_p = float(adjusted_p[idx])
        if psi_val >= psi_threshold:
            severity = "CRITICAL"
            critical_psi_features.append(item["feature_name"])
        elif psi_val >= 0.10:
            severity = "MODERATE"
        else:
            severity = "STABLE"

        is_drifted = bool(psi_val >= psi_threshold or (adj_p < alpha and psi_val >= 0.10))
        if is_drifted:
            drifted_count += 1

        feature_drift[item["feature_name"]] = FeatureDriftMetric(
            feature_name=item["feature_name"],
            psi=psi_val,
            ks_statistic=item["ks_statistic"],
            ks_pvalue=item["ks_pvalue"],
            adjusted_pvalue=adj_p,
            wasserstein_distance=item["wasserstein_distance"],
            reference_mean=item["reference_mean"],
            production_mean=item["production_mean"],
            drift_severity=severity,
            is_drifted=is_drifted,
        )

    psi_values = [m.psi for m in feature_drift.values()]
    max_psi = float(max(psi_values)) if psi_values else 0.0
    mean_psi = float(np.mean(psi_values)) if psi_values else 0.0

    # Prediction score drift
    ref_p = np.asarray(reference_probs, dtype=np.float64).ravel()
    prod_p = np.asarray(production_probs, dtype=np.float64).ravel()
    pred_psi = compute_psi(ref_p, prod_p)
    if len(ref_p) >= 2 and len(prod_p) >= 2:
        pred_ks = stats.ks_2samp(ref_p, prod_p, alternative="two-sided")
        pred_ks_stat = float(pred_ks.statistic)
        pred_ks_pval = float(pred_ks.pvalue)
    else:
        pred_ks_stat, pred_ks_pval = 0.0, 1.0

    ref_alert_rate = float(np.mean(ref_p >= threshold)) if len(ref_p) > 0 else 0.0
    prod_alert_rate = float(np.mean(prod_p >= threshold)) if len(prod_p) > 0 else 0.0
    alert_shift = prod_alert_rate - ref_alert_rate

    # Concept drift (when ground-truth labels are available)
    labeled_count = 0
    live_pr_auc: float | None = None
    live_roc_auc: float | None = None
    live_brier: float | None = None
    live_ece: float | None = None

    if production_labels is not None:
        y_live = np.asarray(production_labels, dtype=np.int64).ravel()
        if len(y_live) == len(prod_p) and len(y_live) >= 10 and len(np.unique(y_live)) > 1:
            labeled_count = len(y_live)
            live_eval = evaluate_model_statistically(
                y_true=y_live,
                y_prob=prod_p,
                threshold=threshold,
                n_bootstraps=100,
            )
            live_pr_auc = live_eval.pr_auc
            live_roc_auc = live_eval.roc_auc
            live_brier = live_eval.brier_score
            live_ece = live_eval.calibration.expected_calibration_error

    # Evaluate retraining triggers
    trigger_reasons: list[str] = []
    if len(critical_psi_features) > 0:
        trigger_reasons.append(
            f"Critical covariate drift (PSI >= {psi_threshold}) on {len(critical_psi_features)} feature(s): "
            f"{', '.join(critical_psi_features[:5])}"
        )
    if pred_psi >= psi_threshold:
        trigger_reasons.append(
            f"Prediction score distribution drift: PSI={pred_psi:.4f} >= {psi_threshold} "
            f"(KS={pred_ks_stat:.4f}, p={pred_ks_pval:.3e})"
        )
    if live_pr_auc is not None and live_pr_auc < (baseline_pr_auc - max_pr_auc_drop):
        trigger_reasons.append(
            f"Concept drift / live PR-AUC degradation: {live_pr_auc:.4f} < "
            f"threshold {baseline_pr_auc - max_pr_auc_drop:.4f} (baseline={baseline_pr_auc:.4f})"
        )

    return DriftMonitoringReport(
        reference_samples=len(reference_df),
        production_samples=len(production_df),
        evaluated_at=datetime.now(UTC).isoformat(),
        feature_drift=feature_drift,
        drifted_features_count=drifted_count,
        critical_psi_features=critical_psi_features,
        max_feature_psi=max_psi,
        mean_feature_psi=mean_psi,
        prediction_psi=pred_psi,
        prediction_ks_statistic=pred_ks_stat,
        prediction_ks_pvalue=pred_ks_pval,
        reference_alert_rate=ref_alert_rate,
        production_alert_rate=prod_alert_rate,
        alert_rate_shift=alert_shift,
        labeled_sample_count=labeled_count,
        live_pr_auc=live_pr_auc,
        live_roc_auc=live_roc_auc,
        live_brier_score=live_brier,
        live_ece=live_ece,
        should_trigger_retraining=len(trigger_reasons) > 0,
        trigger_reasons=trigger_reasons,
    )


def reconcile_ground_truth_labels(session: Session, batch_limit: int = 5000) -> dict[str, int]:
    """
    Reconciles verified fraud outcomes from `TransactionsHistory.isFraud` into
    `MonitoringInference.y_true` for records where `y_true` is still NULL.
    Enables continuous evaluation of live precision/recall/PR-AUC and retraining curation.
    """
    unlabeled_rows = (
        session.query(MonitoringInference)
        .filter(MonitoringInference.y_true.is_(None))
        .limit(batch_limit)
        .all()
    )

    if not unlabeled_rows:
        return {"checked": 0, "updated": 0, "confirmed_frauds": 0}

    event_ids = [r.event_id for r in unlabeled_rows]
    history_rows = (
        session.query(TransactionsHistory.event_id, TransactionsHistory.isFraud)
        .filter(TransactionsHistory.event_id.in_(event_ids))
        .all()
    )
    label_map = {eid: bool(is_fraud) for eid, is_fraud in history_rows}

    updated = 0
    confirmed_frauds = 0
    for row in unlabeled_rows:
        if row.event_id in label_map:
            row.y_true = label_map[row.event_id]
            updated += 1
            if row.y_true:
                confirmed_frauds += 1

    if updated > 0:
        session.commit()
    log.info(
        f"Ground-truth reconciliation complete: checked={len(unlabeled_rows)}, "
        f"updated={updated}, confirmed_frauds={confirmed_frauds}"
    )
    return {
        "checked": len(unlabeled_rows),
        "updated": updated,
        "confirmed_frauds": confirmed_frauds,
    }


def audit_dead_letter_queue(
    session: Session,
    lookback_hours: int = 24,
    alert_threshold: int = 10,
) -> dict[str, Any]:
    """
    Audits the `DeadLetterQueue` table over the recent lookback window to detect
    feature-engineering, schema-validation, or inference worker failures.
    """
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=lookback_hours)
    rows = (
        session.query(DeadLetterQueue.task_name, func.count(DeadLetterQueue.id))
        .filter(DeadLetterQueue.created_at >= cutoff)
        .group_by(DeadLetterQueue.task_name)
        .all()
    )
    counts_by_task = {str(task_name): int(cnt) for task_name, cnt in rows}
    total_failures = sum(counts_by_task.values())

    total_processed = (
        session.query(func.count(MonitoringInference.id))
        .filter(MonitoringInference.timestamp >= cutoff)
        .scalar()
        or 0
    )
    failure_rate = float(total_failures / max(1, total_failures + int(total_processed)))

    return {
        "lookback_hours": lookback_hours,
        "total_dlq_failures": total_failures,
        "total_processed": int(total_processed),
        "failure_rate": failure_rate,
        "failures_by_task": counts_by_task,
        "healthy": total_failures < alert_threshold,
    }


def fetch_production_monitoring_window(
    session: Session,
    limit: int = 2000,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray | None]:
    """
    Extracts the most recent production inference records (`features_json`, `y_pred_proba`, `y_true`)
    from `MonitoringInference` for drift and live performance analysis.
    """
    records = (
        session.query(MonitoringInference)
        .order_by(MonitoringInference.timestamp.desc())
        .limit(limit)
        .all()
    )
    if not records:
        return pl.DataFrame(), np.array([], dtype=np.float64), None

    feature_dicts: list[dict[str, Any]] = []
    probs: list[float] = []
    labels: list[int] = []
    all_labeled = True

    for rec in records:
        if isinstance(rec.features_json, dict):
            feature_dicts.append(rec.features_json)
            probs.append(float(rec.y_pred_proba))
            if rec.y_true is not None:
                labels.append(1 if rec.y_true else 0)
            else:
                all_labeled = False

    if not feature_dicts:
        return pl.DataFrame(), np.array([], dtype=np.float64), None

    df = pl.DataFrame(feature_dicts)
    prob_arr = np.asarray(probs, dtype=np.float64)
    label_arr = (
        np.asarray(labels, dtype=np.int64) if (all_labeled and len(labels) == len(probs)) else None
    )
    return df, prob_arr, label_arr
