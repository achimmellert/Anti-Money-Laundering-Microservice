from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import polars as pl

import airflow.dags.aml_drift_monitoring_and_retraining_dag as drift_dag_mod
import airflow.dags.aml_model_health_and_rollback_guard_dag as rollback_dag_mod
import airflow.dags.aml_training_champion_challenger_dag as training_dag_mod
from aml.models.db_models import (
    DeadLetterQueue,
    MonitoringInference,
    TransactionsHistory,
    TransactionStatus,
    TransactionType,
)
from aml.services.monitoring import (
    audit_dead_letter_queue,
    compute_drift_monitoring_report,
    compute_psi,
    fetch_production_monitoring_window,
    reconcile_ground_truth_labels,
)


def test_psi_and_drift_monitoring_report_stationary_vs_drifted():
    """
    Verifies that `compute_psi` and `compute_drift_monitoring_report` accurately
    distinguish stationary production traffic from severe covariate, prediction,
    and concept drift.
    """
    rng = np.random.default_rng(42)
    n_ref = 600
    n_prod = 400

    ref_df = pl.DataFrame(
        {
            "amount": rng.normal(1000.0, 200.0, n_ref),
            "oldbalanceOrg": rng.normal(5000.0, 1000.0, n_ref),
            "sender_fraction_of_balance": rng.uniform(0.05, 0.50, n_ref),
        }
    )
    ref_probs = rng.beta(1.5, 8.0, n_ref)

    # 1. Stationary production window (drawn from identical distribution)
    prod_stationary_df = pl.DataFrame(
        {
            "amount": rng.normal(1000.0, 200.0, n_prod),
            "oldbalanceOrg": rng.normal(5000.0, 1000.0, n_prod),
            "sender_fraction_of_balance": rng.uniform(0.05, 0.50, n_prod),
        }
    )
    prod_stationary_probs = rng.beta(1.5, 8.0, n_prod)

    stationary_report = compute_drift_monitoring_report(
        reference_df=ref_df,
        production_df=prod_stationary_df,
        reference_probs=ref_probs,
        production_probs=prod_stationary_probs,
        feature_columns=["amount", "oldbalanceOrg", "sender_fraction_of_balance"],
        psi_threshold=0.25,
    )
    assert stationary_report.max_feature_psi < 0.15
    assert stationary_report.prediction_psi < 0.15
    assert stationary_report.should_trigger_retraining is False
    assert len(stationary_report.trigger_reasons) == 0

    # 2. Severely drifted production window (shifted feature means + degraded concept performance)
    prod_drifted_df = pl.DataFrame(
        {
            "amount": rng.normal(2500.0, 400.0, n_prod),  # Large covariate shift
            "oldbalanceOrg": rng.normal(5000.0, 1000.0, n_prod),
            "sender_fraction_of_balance": rng.uniform(0.60, 0.99, n_prod),  # Large shift
        }
    )
    prod_drifted_probs = rng.beta(5.0, 2.0, n_prod)  # Shifted prediction distribution
    prod_labels = rng.integers(0, 2, size=n_prod, dtype=np.int64)  # Near-random PR-AUC

    assert compute_psi(ref_df["amount"].to_numpy(), prod_drifted_df["amount"].to_numpy()) > 0.50

    drifted_report = compute_drift_monitoring_report(
        reference_df=ref_df,
        production_df=prod_drifted_df,
        reference_probs=ref_probs,
        production_probs=prod_drifted_probs,
        production_labels=prod_labels,
        feature_columns=["amount", "oldbalanceOrg", "sender_fraction_of_balance"],
        psi_threshold=0.25,
        baseline_pr_auc=0.85,
        max_pr_auc_drop=0.05,
    )
    assert drifted_report.should_trigger_retraining is True
    assert "amount" in drifted_report.critical_psi_features
    assert "sender_fraction_of_balance" in drifted_report.critical_psi_features
    assert drifted_report.prediction_psi > 0.25
    assert len(drifted_report.trigger_reasons) >= 2


def test_reconcile_ground_truth_and_dlq_audit_database_workflows(e2e_env):
    """
    Verifies ground-truth label reconciliation (`TransactionsHistory.isFraud` ->
    `MonitoringInference.y_true`), production window extraction, and DLQ auditing.
    """
    session_factory = e2e_env["session_factory"]
    eid_fraud = str(uuid.uuid4())
    eid_clean = str(uuid.uuid4())
    now = datetime.now(UTC).replace(tzinfo=None)

    with session_factory() as session:
        session.add_all(
            [
                TransactionsHistory(
                    event_id=eid_fraud,
                    step=10,
                    type=TransactionType.TRANSFER,
                    amount=85000.0,
                    nameOrig="C_ORIG_1",
                    nameDest="M_DEST_1",
                    oldbalanceOrg=85000.0,
                    newbalanceOrig=0.0,
                    oldbalanceDest=0.0,
                    newbalanceDest=85000.0,
                    isFraud=True,
                    timestamp=now,
                ),
                TransactionsHistory(
                    event_id=eid_clean,
                    step=11,
                    type=TransactionType.PAYMENT,
                    amount=50.0,
                    nameOrig="C_ORIG_2",
                    nameDest="M_DEST_2",
                    oldbalanceOrg=500.0,
                    newbalanceOrig=450.0,
                    oldbalanceDest=100.0,
                    newbalanceDest=150.0,
                    isFraud=False,
                    timestamp=now,
                ),
                MonitoringInference(
                    event_id=eid_fraud,
                    timestamp=now,
                    y_pred_proba=0.92,
                    y_true=None,
                    decision=TransactionStatus.BLOCKED,
                    features_json={"amount": 85000.0, "oldbalanceOrg": 85000.0},
                ),
                MonitoringInference(
                    event_id=eid_clean,
                    timestamp=now,
                    y_pred_proba=0.04,
                    y_true=None,
                    decision=TransactionStatus.SUCCESS,
                    features_json={"amount": 50.0, "oldbalanceOrg": 500.0},
                ),
                DeadLetterQueue(
                    task_name="run_inference_task",
                    event_id=str(uuid.uuid4()),
                    payload={"error": "schema mismatch"},
                    error_msg="Column mismatch",
                    created_at=now,
                ),
            ]
        )
        session.commit()

        # 1. Reconcile labels
        rec_summary = reconcile_ground_truth_labels(session)
        assert rec_summary["checked"] == 2
        assert rec_summary["updated"] == 2
        assert rec_summary["confirmed_frauds"] == 1

        # 2. Fetch reconciled monitoring window
        prod_df, prod_probs, prod_labels = fetch_production_monitoring_window(session)
        assert len(prod_df) == 2
        assert len(prod_probs) == 2
        assert prod_labels is not None
        assert set(prod_labels.tolist()) == {0, 1}

        # 3. Audit Dead Letter Queue
        dlq_healthy = audit_dead_letter_queue(session, lookback_hours=24, alert_threshold=5)
        assert dlq_healthy["total_dlq_failures"] == 1
        assert dlq_healthy["failures_by_task"]["run_inference_task"] == 1
        assert dlq_healthy["healthy"] is True

        dlq_breach = audit_dead_letter_queue(session, lookback_hours=24, alert_threshold=1)
        assert dlq_breach["healthy"] is False


def test_airflow_training_champion_challenger_dag_tasks(tmp_path, monkeypatch):
    """
    Verifies DAG 1 (`aml_champion_challenger_training_pipeline`) tasks:
    dataset validation, feature contract check, Challenger training + statistical
    hypothesis gate, and branching on promotion vs. rejection.
    """
    csv_file = tmp_path / "training.csv"
    ds_meta = training_dag_mod.prepare_and_validate_dataset(data_path=str(csv_file))
    assert ds_meta["sample_rows_checked"] == 2000
    assert ds_meta["sample_fraud_count"] > 10

    contract_meta = training_dag_mod.verify_feature_engineering_contract(ds_meta)
    assert contract_meta["contract_verified"] is True
    assert contract_meta["validated_feature_count"] == 27

    # Branching on promoted vs rejected decision
    promoted_decision = {
        "promoted": True,
        "challenger_version": "4",
        "champion_version": "3",
        "challenger_pr_auc": 0.94,
        "reason": "APPROVED by hypothesis test suite",
    }
    assert (
        training_dag_mod.branch_on_hypothesis_gate(promoted_decision)
        == "verify_promoted_champion_artifacts"
    )
    promo_out = training_dag_mod.verify_promoted_champion_artifacts(promoted_decision)
    assert promo_out["status"] == "PROMOTED"
    assert promo_out["champion_version"] == "4"

    rejected_decision = {
        "promoted": False,
        "challenger_version": "5",
        "champion_version": "4",
        "reason": "REJECTED: Bootstrap PR-AUC p=0.42 >= 0.05",
    }
    assert (
        training_dag_mod.branch_on_hypothesis_gate(rejected_decision)
        == "log_challenger_rejection_audit"
    )
    rej_out = training_dag_mod.log_challenger_rejection_audit(rejected_decision)
    assert rej_out["status"] == "REJECTED"
    assert rej_out["preserved_champion_version"] == "4"


def test_airflow_drift_monitoring_and_rollback_guard_dag_tasks(monkeypatch):
    """
    Verifies DAG 2 (`aml_drift_and_performance_monitoring`) and
    DAG 3 (`aml_champion_health_and_rollback_guard`) task routing and execution.
    """
    # DAG 2: Drift branching
    assert (
        drift_dag_mod.branch_on_retraining_trigger({"should_trigger_retraining": True})
        == "trigger_automated_retraining_pipeline"
    )
    assert (
        drift_dag_mod.branch_on_retraining_trigger({"should_trigger_retraining": False})
        == "record_healthy_stationarity_status"
    )

    retrain_payload = drift_dag_mod.trigger_automated_retraining_pipeline(
        {
            "should_trigger_retraining": True,
            "trigger_reasons": ["Critical PSI on amount"],
            "max_feature_psi": 0.42,
            "prediction_psi": 0.31,
        }
    )
    assert retrain_payload["action"] == "TRIGGER_RETRAINING"
    assert retrain_payload["target_dag_id"] == "aml_champion_challenger_training_pipeline"

    # DAG 3: Rollback guard evaluation & branching
    healthy_sla = rollback_dag_mod.evaluate_active_champion_production_sla(
        {
            "healthy": True,
            "failure_rate": 0.0,
            "failures_by_task": {},
        }
    )
    assert healthy_sla["requires_rollback"] is False
    assert (
        rollback_dag_mod.branch_on_rollback_guard(healthy_sla)
        == "confirm_champion_production_health"
    )

    breached_sla = rollback_dag_mod.evaluate_active_champion_production_sla(
        {
            "healthy": False,
            "failure_rate": 0.25,
            "failures_by_task": {"run_inference_task": 12},
        }
    )
    assert breached_sla["requires_rollback"] is True
    assert (
        rollback_dag_mod.branch_on_rollback_guard(breached_sla)
        == "execute_automated_champion_rollback"
    )

    monkeypatch.setattr(rollback_dag_mod, "get_mlflow_client", lambda: MagicMock())
    monkeypatch.setattr(rollback_dag_mod, "rollback_champion", lambda client, model_name: True)

    rb_res = rollback_dag_mod.execute_automated_champion_rollback(breached_sla)
    assert rb_res["action"] == "ROLLBACK_EXECUTED"
    assert rb_res["rollback_succeeded"] is True
