from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

import tools.pipeline as pipeline_mod
import tools.register_model_to_mlflow as register_mod
from aml.services.evaluation import (
    compare_challenger_vs_champion,
    compute_calibration_metrics,
    compute_ranking_and_separation_metrics,
    compute_stratified_bootstrap_cis,
    compute_threshold_metrics,
    delong_roc_test,
    evaluate_model_statistically,
    holm_bonferroni_correction,
    mcnemar_paired_classification_test,
    paired_bootstrap_pr_auc_test,
    paired_permutation_pr_auc_test,
    paired_ttest_loss_differential,
    paired_wilcoxon_brier_test,
)


@pytest.fixture
def synthetic_aml_holdout() -> dict[str, np.ndarray]:
    """
    Generates a realistic imbalanced AML holdout dataset (n=800, ~10% fraud rate)
    with paired predictions from:
    - `p_champion`: a moderate baseline model
    - `p_challenger_better`: a genuinely superior model with higher discrimination & better calibration
    - `p_challenger_noisy`: a model indistinguishable from Champion (random noise within SE)
    - `p_challenger_worse`: an inferior model
    """
    rng = np.random.default_rng(12345)
    n = 800
    y = np.zeros(n, dtype=np.int64)
    y[:80] = 1  # 10% positive fraud class
    rng.shuffle(y)

    # Champion scores: moderate separation
    champ_logits = np.where(y == 1, rng.normal(0.3, 1.1, n), rng.normal(-1.8, 1.1, n))
    p_champion = 1.0 / (1.0 + np.exp(-champ_logits))

    # Genuinely superior Challenger: strong separation and well-calibrated
    chal_better_logits = np.where(y == 1, rng.normal(1.8, 0.9, n), rng.normal(-2.8, 0.9, n))
    p_challenger_better = 1.0 / (1.0 + np.exp(-chal_better_logits))

    # Noisy Challenger: virtually identical to Champion with tiny zero-mean jitter
    p_challenger_noisy = np.clip(p_champion + rng.normal(0.0, 0.002, n), 1e-5, 1.0 - 1e-5)

    # Inferior Challenger: weak separation
    chal_worse_logits = np.where(y == 1, rng.normal(-0.5, 1.3, n), rng.normal(-1.2, 1.3, n))
    p_challenger_worse = 1.0 / (1.0 + np.exp(-chal_worse_logits))

    return {
        "y": y,
        "p_champion": p_champion,
        "p_challenger_better": p_challenger_better,
        "p_challenger_noisy": p_challenger_noisy,
        "p_challenger_worse": p_challenger_worse,
    }


def test_stratified_bootstrap_cis_imbalanced_data(synthetic_aml_holdout):
    """
    Verifies that stratified bootstrap confidence intervals bracket point estimates
    and remain numerically stable under AML class imbalance.
    """
    y = synthetic_aml_holdout["y"]
    p = synthetic_aml_holdout["p_challenger_better"]

    cis = compute_stratified_bootstrap_cis(
        y_true=y,
        y_prob=p,
        threshold=0.5,
        n_bootstraps=250,
        confidence_level=0.95,
        random_state=42,
    )

    expected_metrics = {"pr_auc", "roc_auc", "brier_score", "log_loss", "f1_score", "mcc"}
    assert set(cis.keys()) == expected_metrics

    for name, ci in cis.items():
        assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper, (
            f"{name} point estimate {ci.point_estimate} outside 95% CI [{ci.ci_lower}, {ci.ci_upper}]"
        )
        assert ci.std_error > 0.0
        assert ci.ci_upper > ci.ci_lower


def test_calibration_and_spiegelhalter_test():
    """
    Verifies that calibration metrics (ECE, MCE, Brier Skill Score, Spiegelhalter z-test)
    distinguish well-calibrated probabilities from severely miscalibrated ones.
    """
    rng = np.random.default_rng(99)
    n = 1000
    true_probs = rng.uniform(0.05, 0.85, size=n)
    y = rng.binomial(1, true_probs).astype(np.int64)

    # Well-calibrated model predicts true generating probabilities
    cal_good = compute_calibration_metrics(y, true_probs)
    # Severely miscalibrated model predicts inverted / extreme probabilities
    miscal_probs = np.clip(1.0 - true_probs, 0.05, 0.95)
    cal_bad = compute_calibration_metrics(y, miscal_probs)

    assert cal_good.brier_score < cal_bad.brier_score
    assert cal_good.brier_skill_score > 0.15
    assert cal_bad.brier_skill_score < 0.0
    assert cal_good.expected_calibration_error < cal_bad.expected_calibration_error
    assert cal_good.is_well_calibrated is True
    assert cal_bad.spiegelhalter_pvalue < 0.001
    assert cal_bad.is_well_calibrated is False


def test_ranking_separation_and_threshold_metrics(synthetic_aml_holdout):
    """
    Verifies KS two-sample test, Mann-Whitney U test, Gini coefficient,
    alert-budget Recall@Top-K%, and optimal operating thresholds.
    """
    y = synthetic_aml_holdout["y"]
    p = synthetic_aml_holdout["p_challenger_better"]

    ranking = compute_ranking_and_separation_metrics(y, p)
    assert 0.80 < ranking.roc_auc <= 1.0
    assert ranking.gini_coefficient == pytest.approx(2.0 * ranking.roc_auc - 1.0)
    assert ranking.ks_statistic > 0.50
    assert ranking.ks_pvalue < 1e-10
    assert ranking.mann_whitney_pvalue < 1e-10
    assert ranking.recall_at_top_1pct <= ranking.recall_at_top_5pct <= ranking.recall_at_top_10pct
    assert ranking.lift_at_top_5pct > 3.0

    thresh = compute_threshold_metrics(y, p, threshold=0.5)
    assert thresh.tp + thresh.fp + thresh.tn + thresh.fn == len(y)
    assert thresh.max_f1_score >= thresh.f1_score - 1e-9
    assert 0.0 < thresh.optimal_f1_threshold < 1.0
    assert 0.0 < thresh.optimal_youden_threshold < 1.0
    assert thresh.youden_j_statistic > 0.50


def test_evaluate_model_statistically_mlflow_export(synthetic_aml_holdout):
    """
    Verifies that `evaluate_model_statistically` produces a complete report and
    flattens cleanly into finite float metrics for MLflow logging.
    """
    y = synthetic_aml_holdout["y"]
    p = synthetic_aml_holdout["p_challenger_better"]

    report = evaluate_model_statistically(
        y_true=y,
        y_prob=p,
        latency_ms=1.25,
        threshold=0.5,
        n_bootstraps=100,
    )
    mlflow_metrics = report.to_mlflow_metrics(prefix="test")

    assert "test_pr_auc" in mlflow_metrics
    assert "test_pr_auc_ci_lower" in mlflow_metrics
    assert "test_pr_auc_ci_upper" in mlflow_metrics
    assert "test_ks_statistic" in mlflow_metrics
    assert "test_ece" in mlflow_metrics
    assert "test_brier_skill_score" in mlflow_metrics
    assert all(isinstance(v, float) and np.isfinite(v) for v in mlflow_metrics.values())


def test_hypothesis_tests_detect_genuinely_better_challenger(synthetic_aml_holdout):
    """
    Verifies that all 6 paired statistical hypothesis tests (Bootstrap PR-AUC,
    Permutation PR-AUC, DeLong ROC-AUC, Wilcoxon Brier, Paired t-test Brier,
    and McNemar Classification) detect a genuinely superior Challenger model.
    """
    y = synthetic_aml_holdout["y"]
    p_champ = synthetic_aml_holdout["p_champion"]
    p_chal = synthetic_aml_holdout["p_challenger_better"]

    boot_res = paired_bootstrap_pr_auc_test(y, p_chal, p_champ, n_bootstraps=300, alpha=0.05)
    assert boot_res.effect_size > 0.15
    assert boot_res.ci_lower is not None and boot_res.ci_lower > 0.0
    assert boot_res.p_value < 0.01
    assert boot_res.significant is True

    perm_res = paired_permutation_pr_auc_test(y, p_chal, p_champ, n_permutations=300, alpha=0.05)
    assert perm_res.effect_size > 0.15
    assert perm_res.p_value < 0.01
    assert perm_res.significant is True

    delong_res = delong_roc_test(y, p_chal, p_champ, alpha=0.05)
    assert delong_res.effect_size > 0.05
    assert delong_res.statistic > 2.0
    assert delong_res.p_value < 0.01
    assert delong_res.significant is True

    wilc_res = paired_wilcoxon_brier_test(y, p_chal, p_champ, alpha=0.05)
    assert wilc_res.effect_size > 0.0
    assert wilc_res.p_value < 0.01
    assert wilc_res.significant is True

    ttest_res = paired_ttest_loss_differential(y, p_chal, p_champ, alpha=0.05)
    assert ttest_res.effect_size > 0.0
    assert ttest_res.details["cohens_dz"] > 0.1
    assert ttest_res.p_value < 0.01
    assert ttest_res.significant is True

    mcnemar_res = mcnemar_paired_classification_test(y, p_chal, p_champ, threshold=0.5, alpha=0.05)
    assert (
        mcnemar_res.details["challenger_right_champion_wrong_b"]
        > mcnemar_res.details["challenger_wrong_champion_right_c"]
    )
    assert mcnemar_res.p_value < 0.01
    assert mcnemar_res.significant is True


def test_compare_challenger_vs_champion_approves_genuine_improvement(synthetic_aml_holdout):
    """
    Verifies that `compare_challenger_vs_champion` approves promotion when the Challenger
    is genuinely statistically superior and meets quality floor & latency SLA.
    """
    y = synthetic_aml_holdout["y"]
    p_champ = synthetic_aml_holdout["p_champion"]
    p_chal = synthetic_aml_holdout["p_challenger_better"]

    report = compare_challenger_vs_champion(
        y_true=y,
        y_prob_challenger=p_chal,
        y_prob_champion=p_champ,
        challenger_latency_ms=2.0,
        champion_latency_ms=2.5,
        alpha=0.05,
        min_improvement=0.005,
        n_bootstraps=250,
        n_permutations=250,
    )

    assert report.promote_challenger is True
    assert report.is_statistically_superior is True
    assert report.passed_quality_floor is True
    assert report.passed_latency_sla is True
    assert report.passed_practical_lift is True
    assert report.holm_bonferroni_significant_count >= 4
    assert "APPROVED" in report.decision_summary
    assert len(report.rejection_reasons) == 0


def test_compare_challenger_vs_champion_rejects_noisy_or_inferior_challenger(synthetic_aml_holdout):
    """
    Verifies that `compare_challenger_vs_champion` rejects a Challenger whose improvement
    is merely random noise (or worse than Champion), preventing false champion overwrites.
    """
    y = synthetic_aml_holdout["y"]
    p_champ = synthetic_aml_holdout["p_champion"]
    p_noisy = synthetic_aml_holdout["p_challenger_noisy"]
    p_worse = synthetic_aml_holdout["p_challenger_worse"]

    # 1. Noisy tie: must not pass statistical significance gate
    report_noisy = compare_challenger_vs_champion(
        y_true=y,
        y_prob_challenger=p_noisy,
        y_prob_champion=p_champ,
        challenger_latency_ms=2.0,
        champion_latency_ms=2.0,
        alpha=0.05,
        min_improvement=0.005,
        n_bootstraps=200,
        n_permutations=200,
    )
    assert report_noisy.is_statistically_superior is False
    assert report_noisy.promote_challenger is False
    assert "REJECTED" in report_noisy.decision_summary

    # 2. Worse Challenger: must be rejected
    report_worse = compare_challenger_vs_champion(
        y_true=y,
        y_prob_challenger=p_worse,
        y_prob_champion=p_champ,
        challenger_latency_ms=2.0,
        champion_latency_ms=2.0,
        alpha=0.05,
        min_improvement=0.005,
        n_bootstraps=200,
        n_permutations=200,
    )
    assert report_worse.pr_auc_lift < 0.0
    assert report_worse.is_statistically_superior is False
    assert report_worse.promote_challenger is False
    assert len(report_worse.rejection_reasons) >= 2


def test_holm_bonferroni_multiple_testing_monotonicity():
    """
    Verifies that Holm-Bonferroni step-down adjusted p-values preserve order and
    properly bound family-wise error rate.
    """
    from aml.services.evaluation import HypothesisTestResult

    dummy_tests = {
        "t1": HypothesisTestResult("T1", "m", "h0", "h1", 0.1, 3.0, 0.005),
        "t2": HypothesisTestResult("T2", "m", "h0", "h1", 0.05, 2.1, 0.020),
        "t3": HypothesisTestResult("T3", "m", "h0", "h1", 0.01, 0.8, 0.200),
    }
    corrected = holm_bonferroni_correction(dummy_tests, alpha=0.05)

    # m=3: sorted p-values are 0.005 * 3 = 0.015, max(0.015, 0.020 * 2) = 0.040, max(0.040, 0.200 * 1) = 0.200
    assert corrected["t1"].adjusted_p_value == pytest.approx(0.015)
    assert corrected["t2"].adjusted_p_value == pytest.approx(0.040)
    assert corrected["t3"].adjusted_p_value == pytest.approx(0.200)
    assert corrected["t1"].details["holm_bonferroni_significant"] is True
    assert corrected["t2"].details["holm_bonferroni_significant"] is True
    assert corrected["t3"].details["holm_bonferroni_significant"] is False


def test_pipeline_does_not_overwrite_champion_when_challenger_fails_hypothesis_test(
    synthetic_aml_holdout, monkeypatch
):
    """
    Verifies that `run_full_pipeline` in `tools/pipeline.py` evaluates Challenger against
    Champion using the statistical hypothesis testing suite and REFUSES to overwrite `@champion`
    when the Challenger is not statistically superior.
    """
    y = synthetic_aml_holdout["y"]
    p_champ = synthetic_aml_holdout["p_challenger_better"]  # Strong existing Champion
    p_chal = synthetic_aml_holdout["p_champion"]  # Weaker Challenger

    X_test = np.zeros((len(y), 5), dtype=np.float64)
    X_test_df = pl.DataFrame({f"f{i}": X_test[:, i] for i in range(5)})

    chal_eval = evaluate_model_statistically(y, p_chal, latency_ms=1.5, n_bootstraps=100)

    fake_train_res: dict[str, Any] = {
        "run_id": "run_challenger_123",
        "model_uri": "runs:/run_challenger_123/xgboost-model",
        "model_version": "2",
        "pr_auc": chal_eval.pr_auc,
        "pr_auc_ci_lower": chal_eval.bootstrap_cis["pr_auc"].ci_lower,
        "pr_auc_ci_upper": chal_eval.bootstrap_cis["pr_auc"].ci_upper,
        "roc_auc": chal_eval.roc_auc,
        "roc_auc_ci_lower": chal_eval.bootstrap_cis["roc_auc"].ci_lower,
        "roc_auc_ci_upper": chal_eval.bootstrap_cis["roc_auc"].ci_upper,
        "brier_score": chal_eval.brier_score,
        "brier_skill_score": chal_eval.calibration.brier_skill_score,
        "ece": chal_eval.calibration.expected_calibration_error,
        "ks_statistic": chal_eval.ranking.ks_statistic,
        "ks_pvalue": chal_eval.ranking.ks_pvalue,
        "mcc": chal_eval.threshold_metrics.mcc,
        "f1_score": chal_eval.threshold_metrics.f1_score,
        "latency_ms": 1.5,
        "model_name": "FraudDetectionModel",
        "feature_names": [f"f{i}" for i in range(5)],
        "statistical_evaluation": chal_eval.to_dict(),
        "eval_report": chal_eval,
        "X_test": X_test,
        "X_test_df": X_test_df,
        "y_test": y,
        "y_pred_proba": p_chal,
    }

    mock_client = MagicMock()
    mock_champion_model = MagicMock(spec=["predict"])
    mock_champion_model.predict.return_value = p_champ

    monkeypatch.setattr(pipeline_mod, "get_mlflow_client", lambda: mock_client)
    monkeypatch.setattr(
        pipeline_mod, "load_champion_model", lambda client, name: (mock_champion_model, "1")
    )
    monkeypatch.setattr(pipeline_mod, "train_xgboost", lambda **kwargs: fake_train_res)
    monkeypatch.setattr(pipeline_mod.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(pipeline_mod.mlflow, "log_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline_mod.mlflow, "log_dict", lambda *a, **kw: None)

    promoted_calls = []
    monkeypatch.setattr(
        pipeline_mod,
        "promote_model_to_champion",
        lambda *a, **kw: promoted_calls.append((a, kw)),
    )

    decision = pipeline_mod.run_full_pipeline(
        model_type="xgboost",
        n_trials=2,
        min_improvement=0.005,
        alpha=0.05,
        n_bootstraps=150,
        n_permutations=150,
    )

    assert decision["promoted"] is False
    assert decision["is_statistically_superior"] is False
    assert len(promoted_calls) == 0, "Champion must NOT be overwritten when Challenger fails gate!"
    mock_client.set_model_version_tag.assert_any_call(
        "FraudDetectionModel", "2", "promotion_status", "REJECTED"
    )


def test_pipeline_promotes_challenger_when_statistically_superior(
    synthetic_aml_holdout, monkeypatch
):
    """
    Verifies that `run_full_pipeline` promotes the Challenger to `@champion` when paired
    statistical hypothesis tests confirm genuine superiority over the active Champion.
    """
    y = synthetic_aml_holdout["y"]
    p_champ = synthetic_aml_holdout["p_champion"]  # Baseline Champion
    p_chal = synthetic_aml_holdout["p_challenger_better"]  # Superior Challenger

    X_test = np.zeros((len(y), 5), dtype=np.float64)
    X_test_df = pl.DataFrame({f"f{i}": X_test[:, i] for i in range(5)})

    chal_eval = evaluate_model_statistically(y, p_chal, latency_ms=1.5, n_bootstraps=100)

    fake_train_res: dict[str, Any] = {
        "run_id": "run_challenger_456",
        "model_uri": "runs:/run_challenger_456/xgboost-model",
        "model_version": "3",
        "pr_auc": chal_eval.pr_auc,
        "pr_auc_ci_lower": chal_eval.bootstrap_cis["pr_auc"].ci_lower,
        "pr_auc_ci_upper": chal_eval.bootstrap_cis["pr_auc"].ci_upper,
        "roc_auc": chal_eval.roc_auc,
        "roc_auc_ci_lower": chal_eval.bootstrap_cis["roc_auc"].ci_lower,
        "roc_auc_ci_upper": chal_eval.bootstrap_cis["roc_auc"].ci_upper,
        "brier_score": chal_eval.brier_score,
        "brier_skill_score": chal_eval.calibration.brier_skill_score,
        "ece": chal_eval.calibration.expected_calibration_error,
        "ks_statistic": chal_eval.ranking.ks_statistic,
        "ks_pvalue": chal_eval.ranking.ks_pvalue,
        "mcc": chal_eval.threshold_metrics.mcc,
        "f1_score": chal_eval.threshold_metrics.f1_score,
        "latency_ms": 1.5,
        "model_name": "FraudDetectionModel",
        "feature_names": [f"f{i}" for i in range(5)],
        "statistical_evaluation": chal_eval.to_dict(),
        "eval_report": chal_eval,
        "X_test": X_test,
        "X_test_df": X_test_df,
        "y_test": y,
        "y_pred_proba": p_chal,
    }

    mock_client = MagicMock()
    mock_champion_model = MagicMock(spec=["predict"])
    mock_champion_model.predict.return_value = p_champ

    monkeypatch.setattr(pipeline_mod, "get_mlflow_client", lambda: mock_client)
    monkeypatch.setattr(
        pipeline_mod, "load_champion_model", lambda client, name: (mock_champion_model, "2")
    )
    monkeypatch.setattr(pipeline_mod, "train_xgboost", lambda **kwargs: fake_train_res)
    monkeypatch.setattr(pipeline_mod.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(pipeline_mod.mlflow, "log_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline_mod.mlflow, "log_dict", lambda *a, **kw: None)

    promoted_versions = []
    monkeypatch.setattr(
        pipeline_mod,
        "promote_model_to_champion",
        lambda client, model_name, new_version, **kw: promoted_versions.append(new_version),
    )
    monkeypatch.setattr(pipeline_mod, "sync_champion_to_local_models", lambda *a, **kw: None)

    decision = pipeline_mod.run_full_pipeline(
        model_type="xgboost",
        n_trials=2,
        min_improvement=0.005,
        alpha=0.05,
        n_bootstraps=150,
        n_permutations=150,
    )

    assert decision["promoted"] is True
    assert decision["is_statistically_superior"] is True
    assert promoted_versions == ["3"]
    assert decision["hypothesis_testing"]["bootstrap_pr_auc"]["significant"] is True
    assert decision["hypothesis_testing"]["delong_roc_auc"]["significant"] is True


def test_register_model_to_mlflow_gates_against_existing_champion(
    synthetic_aml_holdout, tmp_path, monkeypatch
):
    """
    Verifies that `register_model_to_mlflow` no longer blindly overwrites `@champion`
    when the candidate local model fails the paired statistical hypothesis test against
    the active `@champion`.
    """
    y = synthetic_aml_holdout["y"]
    p_strong_champ = synthetic_aml_holdout["p_challenger_better"]
    p_weak_cand = synthetic_aml_holdout["p_challenger_worse"]

    # Create a tiny valid XGBoost model file on disk
    X_dummy = np.zeros((20, 4), dtype=np.float32)
    y_dummy = np.array([0, 1] * 10, dtype=np.float32)
    dtrain = xgb.DMatrix(X_dummy, label=y_dummy)
    booster = xgb.train({"max_depth": 1, "objective": "binary:logistic"}, dtrain, num_boost_round=1)
    model_file = tmp_path / "candidate_model.json"
    booster.save_model(str(model_file))

    X_eval = np.zeros((len(y), 4), dtype=np.float64)
    X_eval_df = pl.DataFrame({f"c{i}": X_eval[:, i] for i in range(4)})

    monkeypatch.setattr(
        register_mod,
        "_prepare_evaluation_holdout",
        lambda data_path=None, preproc_path=None: (X_eval, X_eval_df, y, ["c0", "c1", "c2", "c3"]),
    )

    # First call is candidate booster -> returns p_weak_cand; second call is champion -> returns p_strong_champ
    call_preds = [p_weak_cand, p_strong_champ]
    monkeypatch.setattr(
        register_mod,
        "_predict_booster_or_pyfunc",
        lambda model, *args: (call_preds.pop(0), 1.0),
    )

    mock_client = MagicMock()
    mock_champ_meta = MagicMock()
    mock_champ_meta.version = "1"
    mock_client.get_model_version_by_alias.return_value = mock_champ_meta

    monkeypatch.setattr(register_mod, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(register_mod.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(register_mod.mlflow.xgboost, "load_model", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(register_mod.mlflow.xgboost, "log_model", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(register_mod.mlflow, "log_params", lambda *a, **kw: None)
    monkeypatch.setattr(register_mod.mlflow, "log_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(register_mod.mlflow, "log_dict", lambda *a, **kw: None)

    mock_reg = MagicMock()
    mock_reg.version = "2"
    monkeypatch.setattr(register_mod.mlflow, "register_model", lambda *a, **kw: mock_reg)

    with patch("mlflow.start_run") as mock_run:
        mock_run.return_value.__enter__.return_value.info.run_id = "run_reg_test"
        new_ver = register_mod.register_model_to_mlflow(
            model_path=str(model_file),
            model_name="FraudDetectionModel",
            alias="champion",
            n_bootstraps=100,
            n_permutations=100,
            force_promote=False,
        )

    assert new_ver == "2"
    # Candidate was registered as @challenger
    mock_client.set_registered_model_alias.assert_any_call("FraudDetectionModel", "challenger", "2")
    # Candidate was NOT promoted to @champion because it failed the hypothesis test
    champion_calls = [
        c
        for c in mock_client.set_registered_model_alias.call_args_list
        if c.args == ("FraudDetectionModel", "champion", "2")
    ]
    assert len(champion_calls) == 0, "Weak candidate must not overwrite @champion!"
    mock_client.set_model_version_tag.assert_any_call(
        "FraudDetectionModel", "2", "promotion_status", "REJECTED"
    )
