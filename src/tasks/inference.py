from loguru import logger as log
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from celery_batches import Batches
from celery.signals import worker_process_init

from src.celery_app import celery_app
from src.services.database import SessionLocal
from src.services.inference import FraudDetectionModel
from src.utils.db_helpers import (
    update_transaction_state,
    save_suspicious_transaction,
    save_to_dlq
)
from src.config.settings import settings
from src.models.db_models import MonitoringInference


# Global placeholder for the ML Model
model_service = None


@worker_process_init.connect
def init_worker_ml_model(**kwargs):
    """
    Preload the ML model into RAM when the Celery
    worker process boots, ensuring zero latency for the first transaction.
    """
    global model_service
    log.info("Worker booting: Pre-loading FraudDetectionModel into memory...")
    model_service = FraudDetectionModel()
    log.info("Worker booting complete: Model ready for inference.")


@celery_app.task(base=Batches, flush_every=50, flush_interval=0.5, bind=True, max_retries=3)
def run_inference_task(self, requests):
    # Fallback just in case the task is called synchronously during testing
    global model_service
    if model_service is None:
        model_service = FraudDetectionModel()

    if not requests:
        return

    events = []
    rows = []

    for req in requests:
        chain_data = req.args[0] if req.args else {}
        data = chain_data.get("data") or req.kwargs.get("data", {})
        event_id = chain_data.get("event_id") or req.kwargs.get("event_id")

        if not event_id:
            continue

        row = {col: float(data.get(col, 0.0)) for col in settings.NUMERIC_FEATURES}
        row["type"] = str(data.get("type", ""))

        rows.append(row)
        events.append((req, event_id, data))

    if not rows:
        return

    # Vectorized Prediction
    X_raw = pd.DataFrame(rows, columns=settings.RAW_FEATURE_COLUMNS)
    try:
        results = model_service.predict_batch(X_raw)
    except Exception as e:
        log.error(f"Inference Failure: {e}")
        if self.request.retries >= self.max_retries:
            for _, eid, d in events:
                save_to_dlq("inference_ml_failure", eid, d, str(e))
            return
        raise self.retry(exc=e, countdown=5)

    monitoring_entries = []

    with SessionLocal() as session:
        try:
            for i, (req, event_id, data) in enumerate(events):
                prob, is_fraud, decision = results[i]

                # Update State (O(1)) - Ensure this helper DOES NOT call session.commit()
                update_transaction_state(session, event_id, decision, prob, data)

                if decision != "SUCCESS":
                    # Fix: Absolute Path construction via settings
                    plot_path = settings.PROJECT_ROOT / "plots" / f"blocked_{event_id}.png"

                    # Convert to string if the helper strictly expects a string
                    save_suspicious_transaction(session, data, event_id, str(plot_path))

                    # Offload SHAP (Async)
                    row_dict = X_raw.iloc[i].to_dict()
                    generate_shap_task.delay(row_dict, event_id, str(plot_path))

                # Prepare Bulk Monitoring
                monitoring_entries.append({
                    "event_id": event_id,
                    "timestamp": datetime.now(timezone.utc),
                    "y_pred_proba": float(prob),
                    "decision": decision,
                    "features_json": {k: data.get(k) for k in settings.RAW_FEATURE_COLUMNS if k in data}
                })

            # Final Batch Commit
            if monitoring_entries:
                session.bulk_insert_mappings(MonitoringInference, monitoring_entries)

            session.commit()
            log.info(f"Batch processed: {len(events)} events.")

        except SQLAlchemyError as e:
            session.rollback()
            log.error(f"DB Error in Inference Batch: {e}")
            if self.request.retries >= self.max_retries:
                for _, eid, d in events:
                    save_to_dlq("inference_db_failure", eid, d, str(e))
            else:
                raise self.retry(exc=e, countdown=5)


@celery_app.task(max_retries=2)
def generate_shap_task(row_dict: dict, event_id: str, plot_path: str):
    """Background task for SHAP visualization."""
    global model_service
    if model_service is None:
        model_service = FraudDetectionModel()

    try:
        model_service.save_shap_barplot(pd.DataFrame([row_dict]), plot_path)
    except Exception as e:
        log.error(f"SHAP Task failed for {event_id}: {e}")
