# src/api/routes/transactions.py

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from celery import chain

from src.models.schemas import TransactionCreate
from src.models.db_models import TransactionsState, DeadLetterQueue
from src.services.database import get_db_session
from src.utils.db_helpers import get_user_history, redis_client
from src.config.settings import settings
from src.tasks.persistence import run_persistence_task
from src.tasks.enrichment import run_enrichment_task
from src.tasks.inference import run_inference_task
from src.celery_app import celery_app

log = logging.getLogger(__name__)
router = APIRouter()


def get_next_step() -> int:
    """Atomic counter using Redis to prevent race conditions."""
    try:
        # Seed the starting step from settings if the key is empty
        if not redis_client.exists("global_transaction_step"):
            redis_client.set("global_transaction_step", settings.START_STEP)
        return redis_client.incr("global_transaction_step")
    except Exception as e:
        log.warning(f"Redis INCR failed, falling back: {e}")
        return int(settings.START_STEP) # Ensure it returns an integer


@router.post("/transaction", status_code=status.HTTP_202_ACCEPTED)
def submit_transaction(transaction: TransactionCreate, db: Session = Depends(get_db_session)):
    event_id = str(uuid.uuid4())
    step_val = get_next_step()

    payload = transaction.model_dump()
    payload["event_id"] = event_id
    payload["step"] = step_val
    payload["timestamp"] = transaction.timestamp.isoformat()

    try:
        run_persistence_task.delay(data=payload, event_id=event_id)

        intel_pipeline = chain(
            run_enrichment_task.s(data=payload, event_id=event_id),
            run_inference_task.s()
        )
        intel_pipeline.apply_async()

        log.info(f"Transaction {event_id} queued: Audit started & Intelligence Pipeline chained.")

    except Exception as e:
        log.error(f"FATAL: Failed to queue transaction {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Error queuing transaction")

    return {
        "event_id": event_id,
        "status": "received",
        "message": "Transaction processing initiated."
    }


@router.get("/transaction-status/{event_id}")
def get_transaction_status(event_id: str, db: Session = Depends(get_db_session)):
    row = db.query(TransactionsState).filter(TransactionsState.event_id == event_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Unknown event_id")

    return {
        "event_id": event_id,
        "status": row.status,
        "fraud_probability": row.fraud_probability,
        "decided_at": row.decided_at,
        "created_at": row.created_at,
        "type": row.type,
        "amount": row.amount,
        "nameOrig": row.nameOrig,
        "nameDest": row.nameDest,
        "step": row.step,
        "timestamp": row.timestamp,
    }


@router.delete("/transaction-state")
def clear_transaction_state(db: Session = Depends(get_db_session)):
    try:
        db.query(TransactionsState).delete()
        db.commit()
        return {"message": "TransactionState cleared successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-history/{user_id}")
def user_history(user_id: str, db: Session = Depends(get_db_session)):
    rows = get_user_history(db, user_id)
    return [{
        "step": r.step,
        "type": r.type,
        "amount": r.amount,
        "nameOrig": r.nameOrig,
        "nameDest": r.nameDest,
        "timestamp": r.timestamp.isoformat() if r.timestamp else None
    } for r in rows]


@router.get("/dlq")
def get_dead_letter_queue(db: Session = Depends(get_db_session)):
    """Modern SQLAlchemy 2.0 implementation instead of raw SQL strings."""
    try:
        stmt = select(DeadLetterQueue).order_by(desc(DeadLetterQueue.created_at)).limit(50)
        results = db.execute(stmt).scalars().all()

        return [
            {
                "id": r.id,
                "task_name": r.task_name,
                "event_id": r.event_id,
                "error_msg": r.error_msg,
                "created_at": str(r.created_at)
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
        with celery_app.connection() as conn:
            conn.connect()
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


