import logging
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from celery_batches import Batches

from src.celery_app import celery_app
from src.services.database import SessionLocal
from src.services.inference import FraudDetectionModel
from src.utils.db_helpers import (
    update_transaction_state,
    save_suspicious_transaction,
    save_to_dlq
)
from src.config.settings import settings
from src.models.db_models import MonitoringInference, SuspiciousTransactions

log = logging.getLogger(__name__)
model_service = None


@celery_app.task(base=Batches, flush_every=50, flush_interval=0.5, bind=True, max_retries=3)
def run_inference_task(self, requests):
    global model_service
    if model_service is None:
        model_service = FraudDetectionModel()

    if not requests:
        return

    events = []
    rows = []

    # 1. Faster Extraction Logic
    for req in requests:
        # If part of a chain, data is in args[0]. If direct call, data is in kwargs.
        chain_data = req.args[0] if req.args else {}
        data = chain_data.get("data") or req.kwargs.get("data", {})
        event_id = chain_data.get("event_id") or req.kwargs.get("event_id")

        if not event_id:
            continue

        # Dictionary comprehension is faster for mapping features
        row = {col: float(data.get(col, 0.0)) for col in settings.NUMERIC_FEATURES}
        row["type"] = str(data.get("type", ""))

        rows.append(row)
        events.append((req, event_id, data))

    if not rows:
        return

    # 2. Vectorized Prediction
    X_raw = pd.DataFrame(rows, columns=settings.RAW_FEATURE_COLUMNS)
    try:
        results = model_service.predict_batch(X_raw)
    except Exception as e:
        log.error(f"Inference Failure: {e}")
        # --- RETRY OPTIMIZATION ---
        if self.request.retries >= self.max_retries:
            for _, eid, d in events:
                save_to_dlq("inference_ml_failure", eid, d, str(e))
            return  # Stop here so we don't crash the worker
        raise self.retry(exc=e, countdown=5)

    monitoring_entries = []

    # 3. Optimized DB Session
    with SessionLocal() as session:
        try:
            for i, (req, event_id, data) in enumerate(events):
                prob, is_fraud, decision = results[i]

                # Update State (O(1) logic inside helper is key)
                update_transaction_state(session, event_id, decision, prob, data)

                if decision != "SUCCESS":
                    plot_path = f"plots/blocked_{event_id}.png"
                    save_suspicious_transaction(session, data, event_id, plot_path)

                    # Offload SHAP (Async)
                    row_dict = X_raw.iloc[[i]].to_dict('records')[0]
                    generate_shap_task.delay(row_dict, event_id)

                # Prepare Bulk Monitoring
                monitoring_entries.append({
                    "event_id": event_id,
                    "timestamp": datetime.now(timezone.utc),
                    "y_pred_proba": float(prob),
                    "decision": decision,
                    "features_json": {k: data.get(k) for k in settings.RAW_FEATURE_COLUMNS if k in data}
                })

            # 4. Final Batch Commit
            if monitoring_entries:
                session.bulk_insert_mappings(MonitoringInference, monitoring_entries)

            session.commit()
            log.info(f"Batch processed: {len(events)} events.")

        except SQLAlchemyError as e:
            session.rollback()
            log.error(f"DB Error in Inference Batch: {e}")
            # --- FINAL ATTEMPT PROTECTION ---
            if self.request.retries >= self.max_retries:
                for _, eid, d in events:
                    save_to_dlq("inference_db_failure", eid, d, str(e))
            else:
                raise self.retry(exc=e, countdown=5)


@celery_app.task(max_retries=2)
def generate_shap_task(row_dict: dict, event_id: str):
    """Background task for SHAP visualization."""
    global model_service
    if model_service is None:
        model_service = FraudDetectionModel()

    plot_path = f"plots/blocked_{event_id}.png"
    try:
        model_service.save_shap_barplot(pd.DataFrame([row_dict]), plot_path)
    except Exception as e:
        log.error(f"SHAP Task failed for {event_id}: {e}")