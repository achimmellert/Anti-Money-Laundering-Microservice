"""
Statistical Model Evaluation & Champion-Challenger Hypothesis Testing
=====================================================================
Provides rigorous, production-grade statistical evaluation and paired hypothesis
testing for Anti-Money Laundering (AML) fraud detection models.

Statistical Methodology Implemented:
1. Single-Model Statistical Performance Evaluation:
   - Stratified Non-Parametric Bootstrap (95% Confidence Intervals & Standard Errors)
     for PR-AUC, ROC-AUC, Brier Score, Log-Loss, F1-Score, and Matthews Correlation (MCC).
   - Probability Calibration Analysis: Expected Calibration Error (ECE), Maximum Calibration
     Error (MCE), Brier Skill Score (BSS) against base-rate climatology, and
     Spiegelhalter's z-test for absolute probability calibration.
   - Distributional Separation & Ranking Power: Two-sample Kolmogorov-Smirnov (KS) statistic
     & exact/asymptotic p-value, Mann-Whitney U rank-sum test, Gini coefficient (Somers' D),
     and operational AML alert-budget metrics (Recall@Top-1%, Top-5%, Top-10%, Lift@Top-5%).
   - Operating-Point Optimization: Precision, Recall, Specificity, F1, F2, Balanced Accuracy,
     Cohen's Kappa, MCC at operational threshold, plus optimal thresholds via Youden's J and Max-F1.

2. Paired Champion vs. Challenger Hypothesis Testing Suite:
   - Paired Stratified Bootstrap Superiority Test on PR-AUC, ROC-AUC, and Brier Score.
   - Exact Monte-Carlo Paired Permutation (Randomization) Test on PR-AUC.
   - Fast DeLong's Test (DeLong et al., 1988; Sun & Xu, 2014) for two correlated ROC curves.
   - Non-Parametric Wilcoxon Signed-Rank Test on per-transaction Brier loss differentials.
   - Paired Student's t-test (Diebold-Mariano loss differential test) & Cohen's d_z effect size.
   - Exact Binomial & Continuity-Corrected McNemar's Test on paired binary decisions.
   - Holm-Bonferroni step-down multiple hypothesis testing correction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class BootstrapMetricCI:
    """Point estimate, bootstrap standard error, and confidence interval for a metric."""

    point_estimate: float
    mean: float
    std_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Probability calibration and proper scoring rule evaluation."""

    brier_score: float
    brier_skill_score: float
    log_loss: float
    expected_calibration_error: float
    max_calibration_error: float
    spiegelhalter_z: float
    spiegelhalter_pvalue: float
    is_well_calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdClassificationMetrics:
    """Classification metrics at a specific decision threshold and optimal operating points."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    fpr: float
    f1_score: float
    f2_score: float
    balanced_accuracy: float
    mcc: float
    cohens_kappa: float
    optimal_f1_threshold: float
    max_f1_score: float
    optimal_youden_threshold: float
    youden_j_statistic: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankingSeparationMetrics:
    """Distributional separation and operational AML alert-budget ranking metrics."""

    roc_auc: float
    pr_auc: float
    gini_coefficient: float
    ks_statistic: float
    ks_pvalue: float
    mann_whitney_u: float
    mann_whitney_pvalue: float
    recall_at_top_1pct: float
    recall_at_top_5pct: float
    recall_at_top_10pct: float
    precision_at_top_5pct: float
    lift_at_top_5pct: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class StatisticalEvaluationReport:
    """Comprehensive statistical evaluation report for a single model."""

    n_samples: int
    n_positive: int
    positive_rate: float
    latency_ms: float
    bootstrap_cis: dict[str, BootstrapMetricCI]
    calibration: CalibrationMetrics
    threshold_metrics: ThresholdClassificationMetrics
    ranking: RankingSeparationMetrics

    @property
    def pr_auc(self) -> float:
        return self.ranking.pr_auc

    @property
    def roc_auc(self) -> float:
        return self.ranking.roc_auc

    @property
    def brier_score(self) -> float:
        return self.calibration.brier_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "positive_rate": self.positive_rate,
            "latency_ms": self.latency_ms,
            "pr_auc": self.ranking.pr_auc,
            "roc_auc": self.ranking.roc_auc,
            "brier_score": self.calibration.brier_score,
            "bootstrap_cis": {k: v.to_dict() for k, v in self.bootstrap_cis.items()},
            "calibration": self.calibration.to_dict(),
            "threshold_metrics": self.threshold_metrics.to_dict(),
            "ranking": self.ranking.to_dict(),
        }

    def to_mlflow_metrics(self, prefix: str = "") -> dict[str, float]:
        """Flattens key statistical evaluation metrics into MLflow-compatible float dict."""
        p = f"{prefix}_" if prefix else ""
        metrics: dict[str, float] = {
            f"{p}pr_auc": self.ranking.pr_auc,
            f"{p}roc_auc": self.ranking.roc_auc,
            f"{p}gini_coefficient": self.ranking.gini_coefficient,
            f"{p}brier_score": self.calibration.brier_score,
            f"{p}brier_skill_score": self.calibration.brier_skill_score,
            f"{p}log_loss": self.calibration.log_loss,
            f"{p}ece": self.calibration.expected_calibration_error,
            f"{p}mce": self.calibration.max_calibration_error,
            f"{p}spiegelhalter_z": self.calibration.spiegelhalter_z,
            f"{p}spiegelhalter_pvalue": self.calibration.spiegelhalter_pvalue,
            f"{p}ks_statistic": self.ranking.ks_statistic,
            f"{p}ks_pvalue": self.ranking.ks_pvalue,
            f"{p}mann_whitney_pvalue": self.ranking.mann_whitney_pvalue,
            f"{p}recall_at_top_1pct": self.ranking.recall_at_top_1pct,
            f"{p}recall_at_top_5pct": self.ranking.recall_at_top_5pct,
            f"{p}recall_at_top_10pct": self.ranking.recall_at_top_10pct,
            f"{p}precision_at_top_5pct": self.ranking.precision_at_top_5pct,
            f"{p}lift_at_top_5pct": self.ranking.lift_at_top_5pct,
            f"{p}precision": self.threshold_metrics.precision,
            f"{p}recall": self.threshold_metrics.recall,
            f"{p}specificity": self.threshold_metrics.specificity,
            f"{p}f1_score": self.threshold_metrics.f1_score,
            f"{p}f2_score": self.threshold_metrics.f2_score,
            f"{p}balanced_accuracy": self.threshold_metrics.balanced_accuracy,
            f"{p}mcc": self.threshold_metrics.mcc,
            f"{p}cohens_kappa": self.threshold_metrics.cohens_kappa,
            f"{p}optimal_f1_threshold": self.threshold_metrics.optimal_f1_threshold,
            f"{p}max_f1_score": self.threshold_metrics.max_f1_score,
            f"{p}optimal_youden_threshold": self.threshold_metrics.optimal_youden_threshold,
            f"{p}youden_j_statistic": self.threshold_metrics.youden_j_statistic,
            f"{p}latency_ms": self.latency_ms,
        }
        for metric_name, ci in self.bootstrap_cis.items():
            metrics[f"{p}{metric_name}_ci_lower"] = ci.ci_lower
            metrics[f"{p}{metric_name}_ci_upper"] = ci.ci_upper
            metrics[f"{p}{metric_name}_se"] = ci.std_error
        return {k: float(v) for k, v in metrics.items() if math.isfinite(float(v))}


@dataclass(frozen=True)
class HypothesisTestResult:
    """Result of an individual statistical hypothesis test comparing Challenger vs. Champion."""

    test_name: str
    metric: str
    null_hypothesis: str
    alternative_hypothesis: str
    effect_size: float
    statistic: float
    p_value: float
    adjusted_p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    alpha: float = 0.05
    significant: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChampionChallengerComparisonReport:
    """
    Complete paired statistical hypothesis testing report comparing a Challenger model
    against the active Champion model on an identical holdout evaluation dataset.
    """

    challenger_eval: StatisticalEvaluationReport
    champion_eval: StatisticalEvaluationReport
    alpha: float
    min_improvement: float
    max_latency_ms: float
    pr_auc_lift: float
    roc_auc_lift: float
    brier_reduction: float
    hypothesis_tests: dict[str, HypothesisTestResult]
    holm_bonferroni_significant_count: int
    is_statistically_superior: bool
    passed_quality_floor: bool
    passed_latency_sla: bool
    passed_practical_lift: bool
    promote_challenger: bool
    decision_summary: str
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote_challenger": self.promote_challenger,
            "is_statistically_superior": self.is_statistically_superior,
            "passed_quality_floor": self.passed_quality_floor,
            "passed_latency_sla": self.passed_latency_sla,
            "passed_practical_lift": self.passed_practical_lift,
            "alpha": self.alpha,
            "min_improvement": self.min_improvement,
            "max_latency_ms": self.max_latency_ms,
            "pr_auc_lift": self.pr_auc_lift,
            "roc_auc_lift": self.roc_auc_lift,
            "brier_reduction": self.brier_reduction,
            "holm_bonferroni_significant_count": self.holm_bonferroni_significant_count,
            "decision_summary": self.decision_summary,
            "rejection_reasons": self.rejection_reasons,
            "hypothesis_tests": {k: v.to_dict() for k, v in self.hypothesis_tests.items()},
            "challenger": self.challenger_eval.to_dict(),
            "champion": self.champion_eval.to_dict(),
        }

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Flattens comparison statistics and p-values for MLflow tracking."""
        metrics: dict[str, float] = {
            "eval_pr_auc_lift": self.pr_auc_lift,
            "eval_roc_auc_lift": self.roc_auc_lift,
            "eval_brier_reduction": self.brier_reduction,
            "eval_holm_significant_tests": float(self.holm_bonferroni_significant_count),
            "eval_is_statistically_superior": 1.0 if self.is_statistically_superior else 0.0,
            "eval_promote_challenger": 1.0 if self.promote_challenger else 0.0,
        }
        metrics.update(self.challenger_eval.to_mlflow_metrics(prefix="challenger"))
        metrics.update(self.champion_eval.to_mlflow_metrics(prefix="champion"))
        for test_key, res in self.hypothesis_tests.items():
            if math.isfinite(res.p_value):
                metrics[f"test_{test_key}_pvalue"] = float(res.p_value)
            if res.adjusted_p_value is not None and math.isfinite(res.adjusted_p_value):
                metrics[f"test_{test_key}_adj_pvalue"] = float(res.adjusted_p_value)
            if math.isfinite(res.statistic):
                metrics[f"test_{test_key}_stat"] = float(res.statistic)
            if math.isfinite(res.effect_size):
                metrics[f"test_{test_key}_effect"] = float(res.effect_size)
        return metrics


# =============================================================================
# Core Statistical Functions
# =============================================================================


def _validate_arrays(
    y_true: np.ndarray | list[int] | list[float],
    y_prob: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validates and normalizes ground-truth binary labels and predicted probabilities."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"Length mismatch: y_true ({y.shape[0]}) vs y_prob ({p.shape[0]})")
    if y.shape[0] < 2:
        raise ValueError("At least 2 samples are required for statistical evaluation.")
    p = np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7), 1e-7, 1.0 - 1e-7)
    return y, p


def _safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def _safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if np.sum(y == 1) == 0:
        return 0.0
    if np.sum(y == 0) == 0:
        return 1.0
    return float(average_precision_score(y, p))


def compute_stratified_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, BootstrapMetricCI]:
    """
    Computes non-parametric stratified bootstrap confidence intervals and standard errors
    for PR-AUC, ROC-AUC, Brier Score, Log-Loss, F1-Score, and Matthews Correlation (MCC).

    Stratified resampling preserves exact class proportions in every replicate, preventing
    degenerate all-zero samples in highly imbalanced AML datasets.
    """
    y, p = _validate_arrays(y_true, y_prob)
    rng = np.random.default_rng(random_state)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # Point estimates on full sample
    y_pred_bin = (p >= threshold).astype(np.int64)
    point_estimates = {
        "pr_auc": _safe_pr_auc(y, p),
        "roc_auc": _safe_roc_auc(y, p),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "f1_score": float(f1_score(y, y_pred_bin, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, y_pred_bin)) if len(np.unique(y)) > 1 else 0.0,
    }

    if n_bootstraps <= 0 or n_pos == 0 or n_neg == 0:
        return {
            k: BootstrapMetricCI(
                point_estimate=v,
                mean=v,
                std_error=0.0,
                ci_lower=v,
                ci_upper=v,
                confidence_level=confidence_level,
            )
            for k, v in point_estimates.items()
        }

    # Vectorized stratified bootstrap index generation: shape (n_bootstraps, n_samples)
    boot_pos = rng.choice(pos_idx, size=(n_bootstraps, n_pos), replace=True)
    boot_neg = rng.choice(neg_idx, size=(n_bootstraps, n_neg), replace=True)
    boot_indices = np.hstack([boot_pos, boot_neg])

    # Ground truth in every stratified replicate has identical layout: [1...1, 0...0]
    y_boot_fixed = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])

    samples: dict[str, list[float]] = {k: [] for k in point_estimates}

    for b in range(n_bootstraps):
        idx = boot_indices[b]
        p_b = p[idx]
        pred_b = (p_b >= threshold).astype(np.int64)

        samples["pr_auc"].append(float(average_precision_score(y_boot_fixed, p_b)))
        samples["roc_auc"].append(float(roc_auc_score(y_boot_fixed, p_b)))
        samples["brier_score"].append(float(np.mean((y_boot_fixed - p_b) ** 2)))
        samples["log_loss"].append(
            float(-np.mean(y_boot_fixed * np.log(p_b) + (1 - y_boot_fixed) * np.log(1.0 - p_b)))
        )
        samples["f1_score"].append(float(f1_score(y_boot_fixed, pred_b, zero_division=0)))
        samples["mcc"].append(float(matthews_corrcoef(y_boot_fixed, pred_b)))

    alpha_tail = (1.0 - confidence_level) / 2.0
    lower_pct = alpha_tail * 100.0
    upper_pct = (1.0 - alpha_tail) * 100.0

    results: dict[str, BootstrapMetricCI] = {}
    for metric_name, pe in point_estimates.items():
        arr = np.asarray(samples[metric_name], dtype=np.float64)
        ci_low, ci_high = np.percentile(arr, [lower_pct, upper_pct])
        results[metric_name] = BootstrapMetricCI(
            point_estimate=float(pe),
            mean=float(np.mean(arr)),
            std_error=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            ci_lower=float(ci_low),
            ci_upper=float(ci_high),
            confidence_level=confidence_level,
        )

    return results


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    alpha: float = 0.05,
) -> CalibrationMetrics:
    """
    Computes proper scoring rules (Brier Score, Brier Skill Score, Log-Loss),
    binned calibration errors (ECE, MCE), and Spiegelhalter's z-test (1986)
    for absolute probability calibration.
    """
    y, p = _validate_arrays(y_true, y_prob)
    n = len(y)

    brier = float(brier_score_loss(y, p))
    base_rate = float(np.mean(y))
    brier_ref = base_rate * (1.0 - base_rate)
    brier_skill = float(1.0 - (brier / brier_ref)) if brier_ref > 1e-12 else 0.0
    logloss_val = float(log_loss(y, p, labels=[0, 1]))

    # Expected Calibration Error (ECE) & Maximum Calibration Error (MCE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p, bin_edges[1:-1], right=False)

    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y[mask]))
            bin_conf = float(np.mean(p[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            if gap > mce:
                mce = gap

    # Spiegelhalter's (1986) z-statistic for calibration under H0: E[Y_i] = p_i
    # E[Brier | H0] = (1/n) * sum(p_i * (1 - p_i))
    # Var[Brier | H0] = (1/n^2) * sum((1 - 2*p_i)^2 * p_i * (1 - p_i))
    p_var = p * (1.0 - p)
    expected_brier = float(np.mean(p_var))
    var_brier = float(np.sum(((1.0 - 2.0 * p) ** 2) * p_var)) / (n * n)

    if var_brier > 1e-15:
        spiegelhalter_z = float((brier - expected_brier) / math.sqrt(var_brier))
        spiegelhalter_p = float(2.0 * stats.norm.sf(abs(spiegelhalter_z)))
    else:
        spiegelhalter_z = 0.0
        spiegelhalter_p = 1.0

    return CalibrationMetrics(
        brier_score=brier,
        brier_skill_score=brier_skill,
        log_loss=logloss_val,
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
        spiegelhalter_z=spiegelhalter_z,
        spiegelhalter_pvalue=spiegelhalter_p,
        is_well_calibrated=bool(spiegelhalter_p >= alpha or ece <= 0.10),
    )


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ThresholdClassificationMetrics:
    """
    Computes operating-point classification metrics at `threshold` plus optimal
    decision thresholds via Precision-Recall F1 maximization and Youden's J index.
    """
    y, p = _validate_arrays(y_true, y_prob)
    y_pred = (p >= threshold).astype(np.int64)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = float(f1_score(y, y_pred, zero_division=0))
    f2 = float(fbeta_score(y, y_pred, beta=2.0, zero_division=0))
    bal_acc = 0.5 * (rec + spec)
    mcc_val = float(matthews_corrcoef(y, y_pred)) if len(np.unique(y)) > 1 else 0.0
    kappa = float(cohen_kappa_score(y, y_pred)) if len(np.unique(y)) > 1 else 0.0

    # Optimal F1 threshold via Precision-Recall curve
    opt_f1_thresh = threshold
    max_f1 = f1
    if len(np.unique(y)) > 1:
        precisions, recalls, pr_thresholds = precision_recall_curve(y, p)
        if len(pr_thresholds) > 0:
            p_arr = precisions[:-1]
            r_arr = recalls[:-1]
            denom = p_arr + r_arr
            f1_scores = np.where(denom > 0, 2.0 * p_arr * r_arr / denom, 0.0)
            best_idx = int(np.argmax(f1_scores))
            max_f1 = float(f1_scores[best_idx])
            opt_f1_thresh = float(pr_thresholds[best_idx])

    # Optimal Youden's J threshold via ROC curve (J = TPR - FPR)
    opt_youden_thresh = threshold
    max_youden_j = rec - fpr
    if len(np.unique(y)) > 1:
        fprs, tprs, roc_thresholds = roc_curve(y, p)
        j_scores = tprs - fprs
        best_j_idx = int(np.argmax(j_scores))
        max_youden_j = float(j_scores[best_j_idx])
        cand_thresh = float(roc_thresholds[best_j_idx])
        opt_youden_thresh = cand_thresh if math.isfinite(cand_thresh) else threshold

    return ThresholdClassificationMetrics(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=prec,
        recall=rec,
        specificity=spec,
        fpr=fpr,
        f1_score=f1,
        f2_score=f2,
        balanced_accuracy=bal_acc,
        mcc=mcc_val,
        cohens_kappa=kappa,
        optimal_f1_threshold=opt_f1_thresh,
        max_f1_score=max_f1,
        optimal_youden_threshold=opt_youden_thresh,
        youden_j_statistic=max_youden_j,
    )


def compute_ranking_and_separation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> RankingSeparationMetrics:
    """
    Computes rank-based discrimination and distributional separation tests:
    - ROC-AUC, PR-AUC, Gini Coefficient (2 * ROC-AUC - 1)
    - Two-sample Kolmogorov-Smirnov (KS) statistic & p-value
    - Mann-Whitney U test statistic & p-value
    - AML Alert Budget Capture: Recall@Top-1%, Top-5%, Top-10%, Precision@Top-5%, Lift@Top-5%
    """
    y, p = _validate_arrays(y_true, y_prob)
    n = len(y)
    pos_probs = p[y == 1]
    neg_probs = p[y == 0]

    roc = _safe_roc_auc(y, p)
    pr = _safe_pr_auc(y, p)
    gini = 2.0 * roc - 1.0

    if len(pos_probs) > 0 and len(neg_probs) > 0:
        ks_res = stats.ks_2samp(pos_probs, neg_probs, alternative="two-sided")
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        mw_res = stats.mannwhitneyu(pos_probs, neg_probs, alternative="greater")
        mw_stat = float(mw_res.statistic)
        mw_pval = float(mw_res.pvalue)
    else:
        ks_stat, ks_pval = 0.0, 1.0
        mw_stat, mw_pval = 0.0, 1.0

    # Top-K% Alert Budget Metrics
    order = np.argsort(-p)
    y_sorted = y[order]
    total_pos = max(1, int(np.sum(y == 1)))
    base_rate = float(np.mean(y))

    def _recall_at_pct(pct: float) -> float:
        if np.sum(y == 1) == 0:
            return 0.0
        k = max(1, math.ceil(n * pct))
        return float(np.sum(y_sorted[:k]) / total_pos)

    k_5pct = max(1, math.ceil(n * 0.05))
    prec_at_5pct = float(np.sum(y_sorted[:k_5pct]) / k_5pct)
    lift_at_5pct = float(prec_at_5pct / base_rate) if base_rate > 1e-12 else 1.0

    return RankingSeparationMetrics(
        roc_auc=roc,
        pr_auc=pr,
        gini_coefficient=gini,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        mann_whitney_u=mw_stat,
        mann_whitney_pvalue=mw_pval,
        recall_at_top_1pct=_recall_at_pct(0.01),
        recall_at_top_5pct=_recall_at_pct(0.05),
        recall_at_top_10pct=_recall_at_pct(0.10),
        precision_at_top_5pct=prec_at_5pct,
        lift_at_top_5pct=lift_at_5pct,
    )


def evaluate_model_statistically(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    latency_ms: float = 0.0,
    threshold: float = 0.5,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> StatisticalEvaluationReport:
    """
    Executes a comprehensive statistical evaluation of a model's predictions on a holdout set.
    """
    y, p = _validate_arrays(y_true, y_prob)
    n_samples = len(y)
    n_pos = int(np.sum(y == 1))
    pos_rate = float(n_pos / n_samples)

    bootstrap_cis = compute_stratified_bootstrap_cis(
        y,
        p,
        threshold=threshold,
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    calibration = compute_calibration_metrics(y, p, alpha=1.0 - confidence_level)
    threshold_metrics = compute_threshold_metrics(y, p, threshold=threshold)
    ranking = compute_ranking_and_separation_metrics(y, p)

    return StatisticalEvaluationReport(
        n_samples=n_samples,
        n_positive=n_pos,
        positive_rate=pos_rate,
        latency_ms=float(latency_ms),
        bootstrap_cis=bootstrap_cis,
        calibration=calibration,
        threshold_metrics=threshold_metrics,
        ranking=ranking,
    )


# =============================================================================
# Paired Champion vs. Challenger Hypothesis Tests
# =============================================================================


def _compute_midranks(x: np.ndarray) -> np.ndarray:
    """Computes midranks for tied values in O(N log N) time (Sun & Xu, 2014 fast DeLong)."""
    n = len(x)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    midranks = np.zeros(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        # 1-based midrank for indices [i, j-1]
        midranks[i:j] = 0.5 * ((i + 1) + j)
        i = j

    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = midranks
    return out


def delong_roc_test(
    y_true: np.ndarray,
    y_prob_challenger: np.ndarray,
    y_prob_champion: np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    Fast DeLong's Test (DeLong et al., 1988; Sun & Xu, 2014) for comparing two correlated
    ROC curves evaluated on the exact same sample of transactions.

    Null Hypothesis H0: ROC-AUC(Challenger) <= ROC-AUC(Champion)
    Alternative H1:     ROC-AUC(Challenger) >  ROC-AUC(Champion)
    """
    y, p_chal = _validate_arrays(y_true, y_prob_challenger)
    _, p_champ = _validate_arrays(y_true, y_prob_champion)

    pos_mask = y == 1
    neg_mask = y == 0
    m = int(np.sum(pos_mask))
    n = int(np.sum(neg_mask))

    auc_chal = _safe_roc_auc(y, p_chal)
    auc_champ = _safe_roc_auc(y, p_champ)
    diff = auc_chal - auc_champ

    if m < 2 or n < 2:
        return HypothesisTestResult(
            test_name="DeLong Correlated ROC-AUC Test",
            metric="roc_auc",
            null_hypothesis="ROC_AUC_challenger <= ROC_AUC_champion",
            alternative_hypothesis="ROC_AUC_challenger > ROC_AUC_champion",
            effect_size=diff,
            statistic=0.0,
            p_value=1.0,
            ci_lower=diff,
            ci_upper=diff,
            alpha=alpha,
            significant=False,
        )

    # Order samples so positive examples come first, then negative examples
    order = np.concatenate([np.flatnonzero(pos_mask), np.flatnonzero(neg_mask)])
    preds = np.vstack([p_chal[order], p_champ[order]])  # shape (2, m + n)

    k = 2
    tx = np.zeros((k, m), dtype=np.float64)
    ty = np.zeros((k, n), dtype=np.float64)
    tz = np.zeros((k, m + n), dtype=np.float64)

    for r in range(k):
        tx[r, :] = _compute_midranks(preds[r, :m])
        ty[r, :] = _compute_midranks(preds[r, m:])
        tz[r, :] = _compute_midranks(preds[r, :])

    aucs = (np.sum(tz[:, :m], axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n  # shape (2, m)
    v10 = 1.0 - (tz[:, m:] - ty) / m  # shape (2, n)

    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n

    var_diff = float(delong_cov[0, 0] + delong_cov[1, 1] - 2.0 * delong_cov[0, 1])
    se_diff = math.sqrt(max(0.0, var_diff))

    if se_diff < 1e-12:
        z_stat = float("inf") if diff > 0 else (float("-inf") if diff < 0 else 0.0)
        p_one_sided = 0.0 if diff > 0 else 1.0
        p_two_sided = 0.0 if abs(diff) > 0 else 1.0
        ci_low, ci_high = diff, diff
    else:
        z_stat = float((aucs[0] - aucs[1]) / se_diff)
        p_one_sided = float(stats.norm.sf(z_stat))
        p_two_sided = float(2.0 * stats.norm.sf(abs(z_stat)))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_low = float(diff - z_crit * se_diff)
        ci_high = float(diff + z_crit * se_diff)

    return HypothesisTestResult(
        test_name="DeLong Correlated ROC-AUC Test",
        metric="roc_auc",
        null_hypothesis="ROC_AUC_challenger <= ROC_AUC_champion",
        alternative_hypothesis="ROC_AUC_challenger > ROC_AUC_champion",
        effect_size=float(diff),
        statistic=float(z_stat) if math.isfinite(z_stat) else 0.0,
        p_value=p_one_sided,
        ci_lower=ci_low,
        ci_upper=ci_high,
        alpha=alpha,
        significant=bool(p_one_sided < alpha and diff > 0),
        details={
            "two_sided_p_value": p_two_sided,
            "se_diff": se_diff,
            "challenger_roc_auc": float(aucs[0]),
            "champion_roc_auc": float(aucs[1]),
        },
    )


def paired_bootstrap_pr_auc_test(
    y_true: np.ndarray,
    y_prob_challenger: np.ndarray,
    y_prob_champion: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    min_margin: float = 0.0,
    random_state: int = 42,
) -> HypothesisTestResult:
    """
    Paired Stratified Non-Parametric Bootstrap Hypothesis Test for PR-AUC (Average Precision).
    Resamples transactions with replacement in pairs (p_challenger_i, p_champion_i) within
    positive and negative strata to preserve exact class imbalance.

    Null Hypothesis H0: PR-AUC(Challenger) - PR-AUC(Champion) <= min_margin
    Alternative H1:     PR-AUC(Challenger) - PR-AUC(Champion) >  min_margin
    """
    y, p_chal = _validate_arrays(y_true, y_prob_challenger)
    _, p_champ = _validate_arrays(y_true, y_prob_champion)

    pr_chal = _safe_pr_auc(y, p_chal)
    pr_champ = _safe_pr_auc(y, p_champ)
    obs_diff = pr_chal - pr_champ

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_bootstraps <= 0 or n_pos == 0 or n_neg == 0:
        return HypothesisTestResult(
            test_name="Paired Stratified Bootstrap PR-AUC Test",
            metric="pr_auc",
            null_hypothesis=f"PR_AUC_challenger - PR_AUC_champion <= {min_margin:.4f}",
            alternative_hypothesis=f"PR_AUC_challenger - PR_AUC_champion > {min_margin:.4f}",
            effect_size=obs_diff,
            statistic=0.0,
            p_value=1.0,
            ci_lower=obs_diff,
            ci_upper=obs_diff,
            alpha=alpha,
            significant=False,
        )

    rng = np.random.default_rng(random_state)
    boot_pos = rng.choice(pos_idx, size=(n_bootstraps, n_pos), replace=True)
    boot_neg = rng.choice(neg_idx, size=(n_bootstraps, n_neg), replace=True)
    boot_indices = np.hstack([boot_pos, boot_neg])
    y_boot = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])

    diffs = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        idx = boot_indices[b]
        ap_chal = float(average_precision_score(y_boot, p_chal[idx]))
        ap_champ = float(average_precision_score(y_boot, p_champ[idx]))
        diffs[b] = ap_chal - ap_champ

    se_diff = float(np.std(diffs, ddof=1)) if n_bootstraps > 1 else 0.0
    ci_low, ci_high = np.percentile(diffs, [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)])

    # Bootstrap hypothesis test p-value:
    # Combine empirical bootstrap tail probability with studentized bootstrap normal approximation
    non_positive_count = int(np.sum(diffs <= min_margin))
    p_empirical = float((non_positive_count + 1) / (n_bootstraps + 1))

    if se_diff > 1e-12:
        z_stat = float((obs_diff - min_margin) / se_diff)
        p_normal = float(stats.norm.sf(z_stat))
        p_val = (
            min(p_empirical, max(p_normal, 1.0 / (n_bootstraps + 1)))
            if obs_diff > min_margin
            else p_empirical
        )
    else:
        z_stat = 0.0
        p_val = 0.0 if obs_diff > min_margin else 1.0

    return HypothesisTestResult(
        test_name="Paired Stratified Bootstrap PR-AUC Test",
        metric="pr_auc",
        null_hypothesis=f"PR_AUC_challenger - PR_AUC_champion <= {min_margin:.4f}",
        alternative_hypothesis=f"PR_AUC_challenger - PR_AUC_champion > {min_margin:.4f}",
        effect_size=float(obs_diff),
        statistic=float(z_stat),
        p_value=float(p_val),
        ci_lower=float(ci_low),
        ci_upper=float(ci_high),
        alpha=alpha,
        significant=bool(p_val < alpha and obs_diff > min_margin),
        details={
            "challenger_pr_auc": pr_chal,
            "champion_pr_auc": pr_champ,
            "bootstrap_mean_diff": float(np.mean(diffs)),
            "bootstrap_se_diff": se_diff,
            "empirical_p_value": p_empirical,
            "n_bootstraps": n_bootstraps,
        },
    )


def paired_permutation_pr_auc_test(
    y_true: np.ndarray,
    y_prob_challenger: np.ndarray,
    y_prob_champion: np.ndarray,
    n_permutations: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> HypothesisTestResult:
    """
    Exact Monte-Carlo Paired Permutation (Randomization) Test on PR-AUC.
    Under H0 (exchangeability of model predictions per transaction), independently flips
    (p_challenger_i, p_champion_i) with probability 0.5 for each transaction i.

    Null Hypothesis H0: Challenger and Champion score distributions are exchangeable
    Alternative H1:     PR-AUC(Challenger) > PR-AUC(Champion)
    """
    y, p_chal = _validate_arrays(y_true, y_prob_challenger)
    _, p_champ = _validate_arrays(y_true, y_prob_champion)

    obs_diff = _safe_pr_auc(y, p_chal) - _safe_pr_auc(y, p_champ)
    n = len(y)

    if n_permutations <= 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return HypothesisTestResult(
            test_name="Paired Permutation PR-AUC Test",
            metric="pr_auc",
            null_hypothesis="Challenger and Champion predictions are exchangeable (diff == 0)",
            alternative_hypothesis="PR_AUC_challenger > PR_AUC_champion",
            effect_size=obs_diff,
            statistic=obs_diff,
            p_value=1.0,
            alpha=alpha,
            significant=False,
        )

    rng = np.random.default_rng(random_state)
    perm_diffs = np.empty(n_permutations, dtype=np.float64)

    for r in range(n_permutations):
        swap = rng.integers(0, 2, size=n, dtype=np.int8).astype(bool)
        perm_a = np.where(swap, p_champ, p_chal)
        perm_b = np.where(swap, p_chal, p_champ)
        perm_diffs[r] = float(
            average_precision_score(y, perm_a) - average_precision_score(y, perm_b)
        )

    exceed_count = int(np.sum(perm_diffs >= obs_diff - 1e-12))
    p_val = float((exceed_count + 1) / (n_permutations + 1))
    null_se = float(np.std(perm_diffs, ddof=1)) if n_permutations > 1 else 0.0
    z_Null = float(obs_diff / null_se) if null_se > 1e-12 else 0.0

    return HypothesisTestResult(
        test_name="Paired Permutation PR-AUC Test",
        metric="pr_auc",
        null_hypothesis="Challenger and Champion predictions are exchangeable (diff == 0)",
        alternative_hypothesis="PR_AUC_challenger > PR_AUC_champion",
        effect_size=float(obs_diff),
        statistic=z_Null,
        p_value=p_val,
        alpha=alpha,
        significant=bool(p_val < alpha and obs_diff > 0),
        details={
            "n_permutations": n_permutations,
            "exceed_count": exceed_count,
            "null_std_error": null_se,
        },
    )


def paired_wilcoxon_brier_test(
    y_true: np.ndarray,
    y_prob_challenger: np.ndarray,
    y_prob_champion: np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    Non-parametric Wilcoxon Signed-Rank Test on per-sample squared error (Brier) loss
    reductions: d_i = (y_i - p_champion_i)^2 - (y_i - p_challenger_i)^2.
    d_i > 0 indicates the Challenger achieved lower loss on transaction i.

    Null Hypothesis H0: Median(Loss_champion - Loss_challenger) <= 0
    Alternative H1:     Median(Loss_champion - Loss_challenger) >  0
    """
    y, p_chal = _validate_arrays(y_true, y_prob_challenger)
    _, p_champ = _validate_arrays(y_true, y_prob_champion)

    loss_chal = (y - p_chal) ** 2
    loss_champ = (y - p_champ) ** 2
    d = loss_champ - loss_chal  # positive means Challenger is better (lower Brier loss)
    mean_reduction = float(np.mean(d))
    median_reduction = float(np.median(d))

    if np.all(np.abs(d) < 1e-15):
        return HypothesisTestResult(
            test_name="Wilcoxon Signed-Rank Brier Loss Test",
            metric="brier_score",
            null_hypothesis="Median(Brier_champion - Brier_challenger) <= 0",
            alternative_hypothesis="Median(Brier_champion - Brier_challenger) > 0",
            effect_size=0.0,
            statistic=0.0,
            p_value=1.0,
            alpha=alpha,
            significant=False,
        )

    try:
        res = stats.wilcoxon(d, alternative="greater", zero_method="zsplit")
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat_val = 0.0
        p_val = 1.0

    return HypothesisTestResult(
        test_name="Wilcoxon Signed-Rank Brier Loss Test",
        metric="brier_score",
        null_hypothesis="Median(Brier_champion - Brier_challenger) <= 0",
        alternative_hypothesis="Median(Brier_champion - Brier_challenger) > 0",
        effect_size=mean_reduction,
        statistic=stat_val,
        p_value=p_val,
        alpha=alpha,
        significant=bool(p_val < alpha and mean_reduction > 0),
        details={
            "mean_brier_reduction": mean_reduction,
            "median_brier_reduction": median_reduction,
            "challenger_brier": float(np.mean(loss_chal)),
            "champion_brier": float(np.mean(loss_champ)),
        },
    )


def paired_ttest_loss_differential(
    y_true: np.ndarray,
    y_prob_challenger: np.ndarray,
    y_prob_champion: np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    Paired Student's t-test / Diebold-Mariano Loss Differential Test on per-sample
    Brier losses, including Cohen's d_z standardized effect size and 95% CI.

    Null Hypothesis H0: E[Loss_champion - Loss_challenger] <= 0
    Alternative H1:     E[Loss_champion - Loss_challenger] >  0
    """
    y, p_chal = _validate_arrays(y_true, y_prob_challenger)
    _, p_champ = _validate_arrays(y_true, y_prob_champion)
    n = len(y)

    loss_chal = (y - p_chal) ** 2
    loss_champ = (y - p_champ) ** 2
    d = loss_champ - loss_chal

    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se_d = sd_d / math.sqrt(n) if n > 1 else 0.0

    if se_d < 1e-15:
        t_stat = 0.0
        p_val = 0.0 if mean_d > 0 else 1.0
        cohens_dz = 0.0
        ci_low, ci_high = mean_d, mean_d
    else:
        t_res = stats.ttest_1samp(d, popmean=0.0, alternative="greater")
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
        cohens_dz = float(mean_d / sd_d)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        ci_low = float(mean_d - t_crit * se_d)
        ci_high = float(mean_d + t_crit * se_d)

    return HypothesisTestResult(
        test_name="Paired t-Test / Diebold-Mariano Brier Loss Test",
        metric="brier_score",
        null_hypothesis="E[Brier_champion - Brier_challenger] <= 0",
        alternative_hypothesis="E[Brier_champion - Brier_challenger] > 0",
        effect_size=mean_d,
        statistic=t_stat,
        p_value=p_val,
        ci_lower=ci_low,
        ci_upper=ci_high,
        alpha=alpha,
        significant=bool(p_val < alpha and mean_d > 0),
        details={
            "cohens_dz": cohens_dz,
            "se_diff": se_d,
            "df": n - 1,
        },
    )


def mcnemar_paired_classification_test(
    y_true: np.ndarray,
    y_prob_challenger: np.ndarray,
    y_prob_champion: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """
    McNemar's Test (Exact Binomial & Continuity-Corrected Chi-Square) for paired binary
    classification errors at the decision threshold (Dietterich, 1998).

    Discordant pairs:
    - b (n_chal_correct_champ_wrong): Challenger is right, Champion is wrong
    - c (n_chal_wrong_champ_correct): Challenger is wrong, Champion is right

    Null Hypothesis H0: P(Challenger right, Champion wrong) <= P(Challenger wrong, Champion right)
    Alternative H1:     P(Challenger right, Champion wrong) >  P(Challenger wrong, Champion right)
    """
    y, p_chal = _validate_arrays(y_true, y_prob_challenger)
    _, p_champ = _validate_arrays(y_true, y_prob_champion)

    pred_chal = (p_chal >= threshold).astype(np.int64)
    pred_champ = (p_champ >= threshold).astype(np.int64)

    chal_correct = pred_chal == y
    champ_correct = pred_champ == y

    b = int(np.sum(chal_correct & ~champ_correct))  # Challenger right, Champion wrong
    c = int(np.sum(~chal_correct & champ_correct))  # Challenger wrong, Champion right
    both_correct = int(np.sum(chal_correct & champ_correct))
    both_wrong = int(np.sum(~chal_correct & ~champ_correct))
    n_discordant = b + c

    acc_lift = float((b - c) / len(y))

    if n_discordant == 0:
        chi2_stat = 0.0
        p_one_sided = 1.0
        p_two_sided = 1.0
    else:
        chi2_stat = float(((abs(b - c) - 1.0) ** 2) / n_discordant)
        binom_one = stats.binomtest(b, n=n_discordant, p=0.5, alternative="greater")
        binom_two = stats.binomtest(b, n=n_discordant, p=0.5, alternative="two-sided")
        p_one_sided = float(binom_one.pvalue)
        p_two_sided = float(binom_two.pvalue)

    return HypothesisTestResult(
        test_name="McNemar Paired Classification Test",
        metric="classification_accuracy",
        null_hypothesis="P(Challenger correct, Champion wrong) <= 0.5",
        alternative_hypothesis="P(Challenger correct, Champion wrong) > 0.5",
        effect_size=acc_lift,
        statistic=chi2_stat,
        p_value=p_one_sided,
        alpha=alpha,
        significant=bool(p_one_sided < alpha and b > c),
        details={
            "challenger_right_champion_wrong_b": b,
            "challenger_wrong_champion_right_c": c,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "two_sided_exact_p_value": p_two_sided,
            "threshold": threshold,
        },
    )


def holm_bonferroni_correction(
    tests: dict[str, HypothesisTestResult],
    alpha: float = 0.05,
) -> dict[str, HypothesisTestResult]:
    """
    Applies the Holm-Bonferroni step-down multiple testing procedure to control
    the Family-Wise Error Rate (FWER) at level `alpha` across all hypothesis tests.
    """
    keys = list(tests.keys())
    m = len(keys)
    if m == 0:
        return {}

    p_vals = np.array([tests[k].p_value for k in keys], dtype=np.float64)
    sort_order = np.argsort(p_vals)

    adjusted_p = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(sort_order):
        multiplier = m - rank_idx
        adj = min(1.0, float(p_vals[orig_idx] * multiplier))
        running_max = max(running_max, adj)
        adjusted_p[orig_idx] = running_max

    updated: dict[str, HypothesisTestResult] = {}
    for idx, k in enumerate(keys):
        orig = tests[k]
        adj_p = float(adjusted_p[idx])
        updated[k] = HypothesisTestResult(
            test_name=orig.test_name,
            metric=orig.metric,
            null_hypothesis=orig.null_hypothesis,
            alternative_hypothesis=orig.alternative_hypothesis,
            effect_size=orig.effect_size,
            statistic=orig.statistic,
            p_value=orig.p_value,
            adjusted_p_value=adj_p,
            ci_lower=orig.ci_lower,
            ci_upper=orig.ci_upper,
            alpha=alpha,
            significant=orig.significant,
            details={
                **orig.details,
                "holm_bonferroni_significant": bool(adj_p < alpha and orig.effect_size > 0),
            },
        )
    return updated


def compare_challenger_vs_champion(
    y_true: np.ndarray | list[int],
    y_prob_challenger: np.ndarray | list[float],
    y_prob_champion: np.ndarray | list[float],
    challenger_latency_ms: float = 0.0,
    champion_latency_ms: float = 0.0,
    threshold: float = 0.5,
    alpha: float = 0.05,
    min_improvement: float = 0.005,
    min_pr_auc: float = 0.50,
    max_latency_ms: float = 50.0,
    n_bootstraps: int = 500,
    n_permutations: int = 500,
    random_state: int = 42,
) -> ChampionChallengerComparisonReport:
    """
    Executes the full paired Champion vs. Challenger statistical evaluation and
    hypothesis testing pipeline on an identical holdout test split.

    Decision Gate Criteria:
    1. Quality Floor: Challenger PR-AUC >= `min_pr_auc` and KS test p-value < `alpha`.
    2. Latency SLA:   Challenger latency <= `max_latency_ms`.
    3. Practical Lift:
       - Either PR-AUC lift >= `min_improvement`, OR
       - When Champion PR-AUC is already near ceiling (>= 0.995), Challenger maintains
         non-inferior PR-AUC (lift >= -0.001) while reducing Brier loss.
    4. Statistical Significance (Hypothesis Test Gate):
       - Challenger achieves statistically significant superiority (p < `alpha`) on the
         primary PR-AUC tests (Paired Stratified Bootstrap or Paired Permutation) OR,
         when ranking is tied near ceiling, statistically significant Brier loss reduction
         (Wilcoxon Signed-Rank or Paired t-test p < `alpha`) with positive PR-AUC lift.
    """
    y, p_chal = _validate_arrays(y_true, y_prob_challenger)
    _, p_champ = _validate_arrays(y_true, y_prob_champion)

    challenger_eval = evaluate_model_statistically(
        y,
        p_chal,
        latency_ms=challenger_latency_ms,
        threshold=threshold,
        n_bootstraps=n_bootstraps,
        confidence_level=1.0 - alpha,
        random_state=random_state,
    )
    champion_eval = evaluate_model_statistically(
        y,
        p_champ,
        latency_ms=champion_latency_ms,
        threshold=threshold,
        n_bootstraps=n_bootstraps,
        confidence_level=1.0 - alpha,
        random_state=random_state,
    )

    pr_lift = challenger_eval.pr_auc - champion_eval.pr_auc
    roc_lift = challenger_eval.roc_auc - champion_eval.roc_auc
    brier_reduction = champion_eval.brier_score - challenger_eval.brier_score

    # Run hypothesis test battery
    raw_tests: dict[str, HypothesisTestResult] = {
        "bootstrap_pr_auc": paired_bootstrap_pr_auc_test(
            y,
            p_chal,
            p_champ,
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            min_margin=0.0,
            random_state=random_state,
        ),
        "permutation_pr_auc": paired_permutation_pr_auc_test(
            y,
            p_chal,
            p_champ,
            n_permutations=n_permutations,
            alpha=alpha,
            random_state=random_state,
        ),
        "delong_roc_auc": delong_roc_test(
            y,
            p_chal,
            p_champ,
            alpha=alpha,
        ),
        "wilcoxon_brier": paired_wilcoxon_brier_test(
            y,
            p_chal,
            p_champ,
            alpha=alpha,
        ),
        "ttest_brier": paired_ttest_loss_differential(
            y,
            p_chal,
            p_champ,
            alpha=alpha,
        ),
        "mcnemar_accuracy": mcnemar_paired_classification_test(
            y,
            p_chal,
            p_champ,
            threshold=threshold,
            alpha=alpha,
        ),
    }

    tests = holm_bonferroni_correction(raw_tests, alpha=alpha)
    holm_sig_count = sum(
        1 for t in tests.values() if t.details.get("holm_bonferroni_significant", False)
    )

    # Evaluate Gate Conditions
    passed_quality_floor = bool(
        challenger_eval.pr_auc >= min_pr_auc and challenger_eval.ranking.ks_pvalue < alpha
    )
    passed_latency_sla = bool(challenger_latency_ms <= max_latency_ms)

    near_ceiling = champion_eval.pr_auc >= 0.995 and challenger_eval.pr_auc >= 0.995
    passed_practical_lift = bool(
        pr_lift >= min_improvement or (near_ceiling and pr_lift >= -0.001 and brier_reduction > 0)
    )

    # Statistical superiority check
    pr_sig = tests["bootstrap_pr_auc"].significant or tests["permutation_pr_auc"].significant
    loss_sig_with_non_inferior_pr = pr_lift >= 0.0 and (
        tests["wilcoxon_brier"].significant or tests["ttest_brier"].significant
    )
    is_statistically_superior = bool(
        (pr_sig and pr_lift > 0.0) or (loss_sig_with_non_inferior_pr and passed_practical_lift)
    )

    rejection_reasons: list[str] = []
    if not passed_quality_floor:
        rejection_reasons.append(
            f"Quality floor failed: Challenger PR-AUC={challenger_eval.pr_auc:.4f} (min={min_pr_auc:.4f}), "
            f"KS p-value={challenger_eval.ranking.ks_pvalue:.4e} (alpha={alpha})"
        )
    if not passed_latency_sla:
        rejection_reasons.append(
            f"Latency SLA violated: {challenger_latency_ms:.2f}ms > max {max_latency_ms:.2f}ms"
        )
    if not passed_practical_lift:
        rejection_reasons.append(
            f"Insufficient practical PR-AUC lift: {pr_lift:+.4f} < required {min_improvement:+.4f}"
        )
    if not is_statistically_superior:
        boot_p = tests["bootstrap_pr_auc"].p_value
        perm_p = tests["permutation_pr_auc"].p_value
        wilc_p = tests["wilcoxon_brier"].p_value
        rejection_reasons.append(
            f"Challenger failed statistical superiority hypothesis test at alpha={alpha} "
            f"(Bootstrap PR-AUC p={boot_p:.4f}, Permutation PR-AUC p={perm_p:.4f}, "
            f"Wilcoxon Brier p={wilc_p:.4f})"
        )

    promote = bool(
        passed_quality_floor
        and passed_latency_sla
        and passed_practical_lift
        and is_statistically_superior
    )

    if promote:
        decision_summary = (
            f"APPROVED: Challenger statistically outperforms Champion "
            f"(PR-AUC: {challenger_eval.pr_auc:.4f} vs {champion_eval.pr_auc:.4f}, "
            f"lift={pr_lift:+.4f}, Bootstrap p={tests['bootstrap_pr_auc'].p_value:.4f}, "
            f"Permutation p={tests['permutation_pr_auc'].p_value:.4f}, "
            f"DeLong ROC p={tests['delong_roc_auc'].p_value:.4f}, "
            f"Wilcoxon Brier p={tests['wilcoxon_brier'].p_value:.4f}, "
            f"latency={challenger_latency_ms:.2f}ms)."
        )
    else:
        decision_summary = "REJECTED: " + "; ".join(rejection_reasons)

    return ChampionChallengerComparisonReport(
        challenger_eval=challenger_eval,
        champion_eval=champion_eval,
        alpha=alpha,
        min_improvement=min_improvement,
        max_latency_ms=max_latency_ms,
        pr_auc_lift=pr_lift,
        roc_auc_lift=roc_lift,
        brier_reduction=brier_reduction,
        hypothesis_tests=tests,
        holm_bonferroni_significant_count=holm_sig_count,
        is_statistically_superior=is_statistically_superior,
        passed_quality_floor=passed_quality_floor,
        passed_latency_sla=passed_latency_sla,
        passed_practical_lift=passed_practical_lift,
        promote_challenger=promote,
        decision_summary=decision_summary,
        rejection_reasons=rejection_reasons,
    )
