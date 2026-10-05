import threading
from loguru import logger as log
from datetime import datetime, timezone
import polars as pl
from sqlalchemy import insert
from sqlalchemy.exc import SQLAlchemyError
from celery_batches import Batches
from celery.signals import worker_init
from pydantic import ValidationError

from aml.celery_app import celery_app
from aml.services.database import SessionLocal
from aml.services.inference import FraudDetectionModel
from aml.utils.db_helpers import (
    update_transaction_state,
    save_suspicious_transaction,
    save_to_dlq
)
from aml.config.settings import settings
from aml.models.dataframes import RAW_FEATURE_COLUMNS, NUMERIC_FEATURES
from aml.models.db_models import MonitoringInference
from aml.models.schemas import TransactionInternal


# Global placeholder for the ML Model (process-wide singleton, see get_model_service)
model_service = None
_model_lock = threading.Lock()

# Only workers consuming these queues need the ML model in RAM
_MODEL_QUEUES = {"inference", "background"}

INFERENCE_MAX_RETRIES = 3
INFERENCE_RETRY_COUNTDOWN = 5


def get_model_service():
    """
    Thread-safe lazy singleton. The threads/gevent pools share one process, so
    concurrent first calls must not load the model multiple times.
    """
    global model_service
    if model_service is None:
        with _model_lock:
            if model_service is None:
                model_service = FraudDetectionModel()
    return model_service


@worker_init.connect
def init_worker_ml_model(sender=None, **kwargs):
    """
    Preload the ML model into RAM when the Celery worker boots, ensuring zero latency
    for the first transaction. Uses `worker_init` (fires for every pool type) instead of
    `worker_process_init` (prefork children only), and skips workers that don't need the model.
    """
    try:
        consumed = set(sender.app.amqp.queues.consume_from or {})
    except AttributeError:
        consumed = set()

    if consumed and not (consumed & _MODEL_QUEUES):
        return

    log.info("Worker booting: Pre-loading FraudDetectionModel into memory...")
    get_model_service()
    log.info("Worker booting complete: Model ready for inference.")


def _retry_or_dlq(entries: list[tuple], reason: str, error: Exception):
    """
    celery-batches does not support `self.retry()`; each request has to be re-sent
    individually with its original task_id. The attempt counter travels in the message headers.
    """
    for req, event_id, valid_data in entries:
        retries = int((getattr(req, "request_dict", None) or {}).get("retries") or 0)
        if retries >= INFERENCE_MAX_RETRIES:
            save_to_dlq(reason, event_id, valid_data, str(error))
            continue
        run_inference_task.apply_async(
            args=req.args,
            kwargs=req.kwargs,
            task_id=req.id,
            countdown=INFERENCE_RETRY_COUNTDOWN,
            headers={"retries": retries + 1},
        )


@celery_app.task(base=Batches, flush_every=50, flush_interval=0.5, bind=True)
def run_inference_task(self, requests):
    if not requests:
        return

    service = get_model_service()

    events = []
    rows = []

    for req in requests:
        # An upstream enrichment that ended in the DLQ returns None into the chain
        chain_data = (req.args[0] if req.args else None) or {}
        data_dict = chain_data.get("data") or req.kwargs.get("data") or {}
        event_id = chain_data.get("event_id") or req.kwargs.get("event_id")

        if not event_id or not data_dict:
            continue

        # --- 1. PARSE AT THE EDGE: Instantly convert raw dict to Pydantic model ---
        try:
            payload = TransactionInternal(**data_dict)
        except ValidationError as e:
            log.error(f"Pydantic validation failed for {event_id}: {e}")
            save_to_dlq("inference_validation_failure", event_id, data_dict, str(e))
            continue

        # --- 2. EXTRACT FEATURES CLEANLY ---
        # mode="json" ensures Enums, UUIDs, and datetimes are serialized to strings safely
        valid_data = payload.model_dump(mode="json")

        # Extract features from validated payload, falling back to data_dict
        row = {col: float(valid_data.get(col, data_dict.get(col, 0.0))) for col in NUMERIC_FEATURES}
        row["type"] = payload.type.value if hasattr(payload.type, "value") else str(payload.type)

        rows.append(row)

        # Store both the strictly-typed object and the serialized dict for downstream use
        events.append((req, event_id, payload, valid_data))

    if not rows:
        return

    # Vectorized Prediction
    X_raw = pl.DataFrame(rows).select(RAW_FEATURE_COLUMNS)
    try:
        results = service.predict_batch(X_raw)
    except Exception as e:
        log.error(f"Inference Failure: {e}")
        _retry_or_dlq([(req, eid, vd) for req, eid, _, vd in events], "inference_ml_failure", e)
        return

    feature_rows = X_raw.to_dicts()  # convert once, not per flagged event
    monitoring_entries = []
    shap_jobs = []

    with SessionLocal() as session:
        try:
            for i, (req, event_id, payload, valid_data) in enumerate(events):
                prob, is_fraud, decision = results[i]

                # Pass the strictly-typed Pydantic model
                update_transaction_state(session, event_id, decision, prob, payload)

                if decision != "SUCCESS":
                    plot_path = settings.PROJECT_ROOT / "plots" / f"blocked_{event_id}.png"

                    # Pass the Pydantic model here too!
                    save_suspicious_transaction(session, payload, event_id, str(plot_path))

                    # SHAP is dispatched only after a successful commit
                    shap_jobs.append((feature_rows[i], event_id, str(plot_path)))

                # Prepare Bulk Monitoring
                monitoring_entries.append({
                    "event_id": event_id,
                    "timestamp": datetime.now(timezone.utc),
                    "y_pred_proba": float(prob),
                    "decision": decision,
                    "features_json": {k: valid_data[k] for k in RAW_FEATURE_COLUMNS if k in valid_data}
                })

            # Final Batch Commit using SQLAlchemy 2.0 insert
            if monitoring_entries:
                session.execute(insert(MonitoringInference), monitoring_entries)

            session.commit()
            log.info(f"Batch processed: {len(events)} events.")

        except SQLAlchemyError as e:
            session.rollback()
            log.error(f"DB Error in Inference Batch: {e}")
            _retry_or_dlq([(req, eid, vd) for req, eid, _, vd in events], "inference_db_failure", e)
            return

    # Offload SHAP (Async)
    for row_dict, event_id, plot_path in shap_jobs:
        generate_shap_task.delay(row_dict, event_id, plot_path)


@celery_app.task(max_retries=2)
def generate_shap_task(row_dict: dict, event_id: str, plot_path: str):
    """Background task for SHAP visualization."""
    try:
        log.info(f"Generating SHAP visualization for Event: {event_id} -> {plot_path}")
        get_model_service().save_shap_barplot(pl.DataFrame([row_dict]), plot_path)
        log.info(f"SHAP report generated successfully for Event: {event_id}")
    except Exception as e:
        log.error(f"SHAP Task failed for {event_id}: {e}")
