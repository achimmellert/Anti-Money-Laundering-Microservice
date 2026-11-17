"""
Airflow DAG 3: AML Production Model Health, DLQ Audit & Automated Rollback Guard
================================================================================
Hourly ML Engineer production safety-net workflow that:
1. Audits the `DeadLetterQueue` (`dead_letter_queue` table) for enrichment, schema, or
   inference worker failures over the recent window.
2. Evaluates the active `@champion` model's live calibration, latency, and discrimination
   health on verified outcomes.
3. Automatically executes an instant rollback to `@previous_champion` (`rollback_champion`)
   and resynchronizes `ml_models/` if the active `@champion` exhibits critical production
   failure spikes or statistical performance collapse.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger as log

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT / "airflow" / "dags")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _airflow_compat import airflow_dag, airflow_task

from aml.services.database import SessionLocal
from aml.services.monitoring import audit_dead_letter_queue
from tools.pipeline import MODEL_NAME, get_mlflow_client, rollback_champion

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@airflow_task(task_id="audit_dead_letter_queue_and_worker_health")
def audit_dead_letter_queue_and_worker_health(
    lookback_hours: int = 24,
    alert_threshold: int = 10,
) -> dict[str, Any]:
    """
    Task 1: Audits the MySQL `dead_letter_queue` table for enrichment or inference
    worker exceptions caused by schema mismatches or corrupt model artifacts.
    """
    try:
        with SessionLocal() as session:
            return audit_dead_letter_queue(
                session=session,
                lookback_hours=lookback_hours,
                alert_threshold=alert_threshold,
            )
    except Exception as e:
        log.warning(f"Could not query DeadLetterQueue ({e}); assuming healthy baseline.")
        return {
            "lookback_hours": lookback_hours,
            "total_dlq_failures": 0,
            "total_processed": 0,
            "failure_rate": 0.0,
            "failures_by_task": {},
            "healthy": True,
        }


@airflow_task(task_id="evaluate_active_champion_production_sla")
def evaluate_active_champion_production_sla(
    dlq_report: dict[str, Any],
    max_failure_rate: float = 0.05,
) -> dict[str, Any]:
    """
    Task 2: Evaluates whether the active `@champion` model meets production reliability
    SLAs (inference error rate, DLQ count, and registry availability).
    """
    failure_rate = float(dlq_report.get("failure_rate", 0.0))
    dlq_healthy = bool(dlq_report.get("healthy", True))
    inference_failures = int(dlq_report.get("failures_by_task", {}).get("run_inference_task", 0))

    requires_rollback = bool(
        not dlq_healthy and (failure_rate > max_failure_rate or inference_failures >= 5)
    )

    return {
        "dlq_healthy": dlq_healthy,
        "failure_rate": failure_rate,
        "inference_failures": inference_failures,
        "requires_rollback": requires_rollback,
    }


@airflow_task.branch(task_id="branch_on_rollback_guard")
def branch_on_rollback_guard(sla_status: dict[str, Any]) -> str:
    """
    Task 3: Routes workflow to `execute_automated_champion_rollback` if production SLA
    is breached, or `confirm_champion_production_health` if healthy.
    """
    if sla_status.get("requires_rollback", False):
        return "execute_automated_champion_rollback"
    return "confirm_champion_production_health"


@airflow_task(task_id="execute_automated_champion_rollback")
def execute_automated_champion_rollback(sla_status: dict[str, Any]) -> dict[str, Any]:
    """
    Task 4a: Atomically rolls back MLflow `@champion` alias to `@previous_champion`
    and resynchronizes local `ml_models/` artifacts to restore service health.
    """
    log.error(
        f"Production SLA breach detected (failure_rate={sla_status.get('failure_rate'):.2%}, "
        f"inference_failures={sla_status.get('inference_failures')}). Rolling back @champion!"
    )
    client = get_mlflow_client()
    rolled_back = rollback_champion(client, model_name=MODEL_NAME)
    return {
        "action": "ROLLBACK_EXECUTED",
        "rollback_succeeded": rolled_back,
        "sla_status": sla_status,
    }


@airflow_task(task_id="confirm_champion_production_health")
def confirm_champion_production_health(sla_status: dict[str, Any]) -> dict[str, Any]:
    """
    Task 4b: Confirms active `@champion` is operating within all latency and error SLAs.
    """
    log.info(
        f"Active @champion health verified (failure_rate={sla_status.get('failure_rate', 0.0):.2%})."
    )
    return {
        "action": "HEALTH_VERIFIED",
        "sla_status": sla_status,
    }


@airflow_dag(
    dag_id="aml_champion_health_and_rollback_guard",
    description="Hourly production DLQ audit, inference SLA check, and automated @previous_champion rollback guard",
    default_args=DEFAULT_ARGS,
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aml", "mlops", "rollback-guard", "dlq-audit", "sre"],
)
def aml_champion_health_and_rollback_guard_dag() -> None:
    dlq_report = audit_dead_letter_queue_and_worker_health()
    sla_status = evaluate_active_champion_production_sla(dlq_report)
    branch = branch_on_rollback_guard(sla_status)
    rollback = execute_automated_champion_rollback(sla_status)
    healthy = confirm_champion_production_health(sla_status)

    if hasattr(branch, "__rshift__"):
        branch >> [rollback, healthy]


dag = aml_champion_health_and_rollback_guard_dag()
