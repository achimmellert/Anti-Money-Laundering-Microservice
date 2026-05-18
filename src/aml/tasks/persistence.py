from loguru import logger as log
from aml.celery_app import celery_app
from aml.models.db_models import TransactionStatus
from aml.services.database import SessionLocal
from aml.utils.db_helpers import (
    update_transaction_state,
    save_history_transaction,
    save_to_dlq
)
from aml.models.schemas import TransactionInternal


@celery_app.task(bind=True, max_retries=3)
def run_persistence_task(self, data: dict, event_id: str):
    log.info(f"Starting persistence for Event: {event_id}")

    # 1. Re-hydrate the raw Celery dictionary into Pydantic model
    try:
        internal_tx = TransactionInternal.model_validate(data)
    except Exception as e:
        # Defensive Programming: If the data is fundamentally broken/missing fields,
        # retrying won't fix it. Send it straight to the DLQ and abort.
        log.critical(f"FATAL: Data validation failed for {event_id}. Moving to DLQ. Error: {e}")
        save_to_dlq("persistence_validation", event_id, data, str(e))
        return

    with SessionLocal() as session:
        try:
            # 2. Update the 'State'
            update_transaction_state(
                session,
                event_id,
                status=TransactionStatus.PERSISTED,
                prob=0.0,
                payload=internal_tx
            )

            # 3. Save to 'History'
            save_history_transaction(session, payload=internal_tx)

            # 4. Finalize both in one atomic transaction (Unit of Work pattern)
            session.commit()
            log.info(f"Successfully persisted Event and updated Cache: {event_id}")

        except Exception as e:
            session.rollback()
            log.error(f"Persistence DB operation failed for {event_id}: {str(e)}")

            # Exponential backoff: 5s, 10s, 20s...
            retry_countdown = 5 * (2 ** self.request.retries)

            if self.request.retries >= self.max_retries:
                log.critical(f"FATAL: Max retries reached for {event_id}. Moving to DLQ.")
                save_to_dlq("persistence", event_id, data, str(e))
            else:
                log.info(f"Retrying task in {retry_countdown}s...")
                raise self.retry(exc=e, countdown=retry_countdown)
