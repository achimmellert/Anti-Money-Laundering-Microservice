import uuid

from celery import chain
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger as log
from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session

from aml.config.settings import settings
from aml.models.db_models import DeadLetterQueue, TransactionsState
from aml.models.schemas import (
    TransactionAcceptedResponse,
    TransactionCreate,
    TransactionInternal,
    TransactionStatusResponse,
)
from aml.services.database import get_db_session
from aml.tasks.enrichment import run_enrichment_task
from aml.tasks.inference import run_inference_task
from aml.tasks.persistence import run_persistence_task
from aml.utils.db_helpers import get_user_history, redis_client

router = APIRouter()


def get_next_step() -> int:
    """Atomic counter using Redis to prevent race conditions."""
    try:
        # Seed the starting step from settings if the key is empty
        if redis_client.setnx("global_transaction_step", settings.START_STEP):
            return int(settings.START_STEP)
        return redis_client.incr("global_transaction_step")
    except Exception as e:
        log.warning(f"Redis INCR failed, falling back: {e}")
        return int(settings.START_STEP)


@router.post(
    "/transaction",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TransactionAcceptedResponse,
)
def submit_transaction(transaction: TransactionCreate):
    event_id = str(uuid.uuid4())
    step_val = get_next_step()

    internal_tx = TransactionInternal(**transaction.model_dump(), event_id=event_id, step=step_val)

    # Serialize for Celery (mode="json") since Celery use a standard JSON serializer
    celery_payload = internal_tx.model_dump(mode="json")

    # Hands it off to the two separate workers: persistence-worker and inference-worker
    try:
        # 1) persistence-worker
        run_persistence_task.delay(data=celery_payload, event_id=event_id)

        # 2) inference-worker
        inference_task_pipeline = chain(
            run_enrichment_task.s(data=celery_payload, event_id=event_id),
            run_inference_task.s(),
        )
        inference_task_pipeline.apply_async()

        log.info(f"Transaction {event_id} queued: Audit started & Intelligence Pipeline chained.")

    except Exception as e:
        log.error(f"FATAL: Failed to queue transaction {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Error queuing transaction") from e

    return {
        "event_id": event_id,
        "status": "received",
        "message": "Transaction processing initiated.",
    }


@router.get("/transaction-status/{event_id}", response_model=TransactionStatusResponse)
def get_transaction_status(event_id: str, db: Session = Depends(get_db_session)):
    stmt = select(TransactionsState).where(TransactionsState.event_id == event_id)
    row = db.execute(stmt).scalar_one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Unknown event_id")

    return row


@router.delete("/transaction-state")
def clear_transaction_state(db: Session = Depends(get_db_session)):
    try:
        stmt = delete(TransactionsState)
        db.execute(stmt)
        db.commit()
        return {"message": "TransactionState cleared successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/user-history/{user_id}")
def user_history(user_id: str, db: Session = Depends(get_db_session)):
    rows = get_user_history(db, user_id)  # last 100 entries

    return [
        {
            "step": r.step,
            "type": r.type,
            "amount": r.amount,
            "nameOrig": r.nameOrig,
            "nameDest": r.nameDest,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in rows
    ]


@router.get("/dlq")
def get_dead_letter_queue(db: Session = Depends(get_db_session)):
    try:
        stmt = select(DeadLetterQueue).order_by(desc(DeadLetterQueue.created_at)).limit(50)
        results = db.execute(stmt).scalars().all()

        return [
            {
                "id": r.id,
                "task_name": r.task_name,
                "event_id": r.event_id,
                "error_msg": r.error_msg,
                "created_at": str(r.created_at),
            }
            for r in results
        ]
    except Exception as e:
        log.error(f"DLQ Fetch Error: {e}")
        return []


@router.get("/health")
def healthcheck(db: Session = Depends(get_db_session)):
    status_dict = {"redis_ready": False, "db_ok": False}

    try:
        if redis_client.ping():
            status_dict["redis_ready"] = True
    except Exception:
        pass

    try:
        db.execute(select(1))
        status_dict["db_ok"] = True
    except Exception:
        status_dict["db_ok"] = False

    http_status = 200 if all(status_dict.values()) else 503

    return JSONResponse(content=status_dict, status_code=http_status)
