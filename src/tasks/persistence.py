from loguru import logger as log
from src.celery_app import celery_app
from src.services.database import SessionLocal
from src.utils.db_helpers import (
    update_transaction_state,
    save_history_transaction, # Added this
    save_to_dlq
)

@celery_app.task(bind=True, max_retries=3)
def run_persistence_task(self, data: dict, event_id: str):
    log.info(f"Starting persistence for Event: {event_id}")

    with SessionLocal() as session:
        try:
            # 1. Update the 'State' (The metadata: PENDING, PROBABILITY, etc.)
            update_transaction_state(
                session,
                event_id,
                status="PERSISTED", # Moving from PENDING to PERSISTED
                prob=0.0,
                payload=data
            )

            # 2. Save to 'History' (The actual transaction + High-speed Redis Update)
            # This triggers the Write-Through logic we discussed!
            save_history_transaction(session, data, event_id)

            # 3. Finalize both in one atomic transaction
            session.commit()
            log.success(f"Successfully persisted Event and updated Cache: {event_id}")

        except Exception as e:
            session.rollback()
            log.error(f"Persistence failed for {event_id}: {str(e)}")

            # Exponential backoff: 5s, 10s, 20s...
            retry_countdown = 5 * (2 ** self.request.retries)

            if self.request.retries >= self.max_retries:
                log.critical(f"FATAL: Max retries reached for {event_id}. Moving to DLQ.")
                save_to_dlq("persistence", event_id, data, str(e))
            else:
                log.info(f"Retrying task in {retry_countdown}s...")
                raise self.retry(exc=e, countdown=retry_countdown)