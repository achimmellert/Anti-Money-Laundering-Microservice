import redis
import msgspec
from loguru import logger as log
from datetime import datetime, timezone
from typing import List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_

from src.models.db_models import TransactionsHistory, TransactionsState, SuspiciousTransactions, DeadLetterQueue
from src.models.structs import HistoryStruct, HISTORY_DECODER, HISTORY_ENCODER
from src.services.database import SessionLocal
from src.config.settings import settings


# Use decode_responses=False for maximum msgspec speed
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=False)


# --- INTERNAL HELPERS ---

def _get_cached_history(session: Session, user_id: str, limit: int) -> List[HistoryStruct]:
    """Internal: Tries Redis first, falls back to SQL, then hydrates cache."""
    if not user_id: return []
    key = f"hist:{user_id}"

    # 1. High-Speed Batch Read from Redis
    try:
        raw_data = redis_client.lrange(key, 0, limit - 1)
        if raw_data:
            # The "Byte-Sandwich" trick for 80x faster decoding
            return HISTORY_DECODER.decode(b"[" + b",".join(raw_data) + b"]")
    except Exception as e:
        log.warning(f"Redis Read Error for {user_id}: {e}")

    # 2. SQL Fallback
    rows = session.query(TransactionsHistory).filter(
        or_(TransactionsHistory.nameOrig == user_id, TransactionsHistory.nameDest == user_id)
    ).order_by(desc(TransactionsHistory.step)).limit(limit).all()

    # 3. Cache Hydration (Fill the empty drawer)
    if rows:
        try:
            with redis_client.pipeline() as pipe:
                pipe.delete(key)
                for r in rows:
                    s = HistoryStruct(
                        step=r.step, type=r.type, amount=float(r.amount),
                        nameOrig=r.nameOrig, nameDest=r.nameDest,
                        oldbalanceOrg=float(r.oldbalanceOrg or 0),
                        newbalanceOrig=float(r.newbalanceOrig or 0),
                        timestamp=r.timestamp
                    )
                    pipe.rpush(key, HISTORY_ENCODER.encode(s))
                pipe.expire(key, 3600)
                pipe.execute()
        except Exception as e:
            log.warning(f"Redis Hydration Error: {e}")

    # Return the rows (SQLAlchemy handles the conversion to objects for the caller)
    return rows


# --- PUBLIC API ---

def get_account_history(session: Session, nameOrig: str, nameDest: str, limit: int = 200):
    """Combines history for both sender and receiver."""
    sender_rows = _get_cached_history(session, nameOrig, limit)
    receiver_rows = _get_cached_history(session, nameDest, limit)
    return list(sender_rows) + list(receiver_rows)


def get_user_history(session: Session, user_id: str, limit: int = 100):
    """Retrieves full history for a specific user ID."""
    return _get_cached_history(session, user_id, limit)


def update_transaction_state(session: Session, event_id: str, status: str, prob: float, payload: dict | None = None):
    """Handles the state lifecycle and prevents status race conditions."""
    ts_row = session.query(TransactionsState).filter_by(event_id=event_id).first()

    if ts_row is None and payload:
        # Initial Creation
        ts_row = TransactionsState(
            event_id=event_id,
            type=payload.get("type"),
            amount=float(payload.get("amount", 0.0)),
            nameOrig=payload.get("nameOrig"),
            nameDest=payload.get("nameDest"),
            step=int(payload.get("step", 0)),
            timestamp=datetime.fromisoformat(payload["timestamp"]) if payload.get("timestamp") else None,
        )
        session.add(ts_row)

    if ts_row:
        # Race-Condition Fix: Don't overwrite finished states with 'PENDING'
        if status == "PENDING" and ts_row.status in ["SUCCESS", "BLOCKED"]:
            return

        ts_row.status = status
        ts_row.fraud_probability = prob
        ts_row.decided_at = datetime.now(timezone.utc)


def save_history_transaction(session: Session, data: dict, event_id: str):
    """Saves to SQL and updates the sliding-window cache in Redis."""
    # 1. Clean data using modern unpacking (Filtering only columns that exist)
    hist = TransactionsHistory(
        event_id=event_id,
        **{k: v for k, v in data.items() if hasattr(TransactionsHistory, k)}
    )
    session.add(hist)

    # 2. Update Redis (Write-Through)
    try:
        struct_data = HistoryStruct(**{k: v for k, v in data.items() if k in HistoryStruct.__struct_fields__})
        encoded_obj = HISTORY_ENCODER.encode(struct_data)

        for uid in filter(None, [data.get("nameOrig"), data.get("nameDest")]):
            key = f"hist:{uid}"
            if redis_client.exists(key):
                with redis_client.pipeline() as pipe:
                    pipe.lpush(key, encoded_obj)
                    pipe.ltrim(key, 0, 199)
                    pipe.execute()
    except Exception as e:
        log.warning(f"Redis Live-Update Error: {e}")


def save_suspicious_transaction(session: Session, data: dict, event_id: str, plot_path: str):
    """Logs flagged transactions for manual review."""
    sus = SuspiciousTransactions(
        event_id=event_id,
        amount=float(data.get("amount", 0.0)),
        nameOrig=data.get("nameOrig"),
        nameDest=data.get("nameDest"),
        type=data.get("type"),
        timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
    )
    session.add(sus)
    # Note: Commit is usually handled by the calling Task's context
    log.info(f"SHAP-Plot for {event_id} stored at: {plot_path}")


def save_to_dlq(task_name: str, event_id: str, payload: dict, error_msg: str):
    """Safely moves failed tasks to a proper DB table instead of raw SQL strings."""
    try:
        with SessionLocal() as session:
            entry = DeadLetterQueue(
                task_name=task_name,
                event_id=event_id,
                payload=payload,
                error_msg=str(error_msg)
            )
            session.add(entry)
            session.commit()
            log.warning(f"Task '{task_name}' (Event {event_id}) moved to DLQ.")
    except Exception as e:
        log.error(f"DLQ CRITICAL FAILURE: {e}")