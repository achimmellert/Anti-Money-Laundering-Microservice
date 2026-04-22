# src/utils/db_helpers.py

import redis
import msgspec
import logging
from datetime import datetime, timezone
from typing import List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_

from src.models.db_models import TransactionsHistory, TransactionsState, SuspiciousTransactions, DeadLetterQueue
from src.models.structs import HistoryStruct, HISTORY_DECODER, HISTORY_ENCODER
from src.services.database import SessionLocal
from src.config.settings import settings

log = logging.getLogger(__name__)
# Change to this:
redis_client = redis.Redis.from_url(settings.REDIS_URL.unicode_string(), decode_responses=False)


# --- INTERNAL HELPERS ---

def _get_cached_history(session: Session, user_id: str, limit: int) -> List[Any]:
    """Tries Redis first, falls back to SQL, and ensures consistent return types."""
    if not user_id: return []
    key = f"hist:{user_id}"

    # 1. Redis Look-up
    try:
        raw_data = redis_client.lrange(key, 0, limit - 1)
        if raw_data:
            # Decode into list of structs
            return HISTORY_DECODER.decode(b"[" + b",".join(raw_data) + b"]")
    except Exception as e:
        log.warning(f"Redis Read Error for {user_id}: {e}")

    # 2. SQL Fallback
    rows = session.query(TransactionsHistory).filter(
        or_(TransactionsHistory.nameOrig == user_id, TransactionsHistory.nameDest == user_id)
    ).order_by(desc(TransactionsHistory.step)).limit(limit).all()

    # 3. Cache Hydration
    if rows:
        try:
            # We convert to structs for consistent caching and returning
            structs = [
                HistoryStruct(
                    step=r.step, type=r.type, amount=float(r.amount),
                    nameOrig=r.nameOrig, nameDest=r.nameDest,
                    oldbalanceOrg=float(r.oldbalanceOrg or 0),
                    newbalanceOrig=float(r.newbalanceOrig or 0),
                    timestamp=r.timestamp
                ) for r in rows
            ]
            with redis_client.pipeline() as pipe:
                pipe.delete(key)
                for s in structs:
                    pipe.rpush(key, HISTORY_ENCODER.encode(s))
                pipe.expire(key, 3600)
                pipe.execute()
            return structs
        except Exception as e:
            log.warning(f"Redis Hydration Error: {e}")

    return rows


# --- PUBLIC API ---

def get_account_history(session: Session, nameOrig: str, nameDest: str, limit: int = 200):
    sender_rows = _get_cached_history(session, nameOrig, limit)
    receiver_rows = _get_cached_history(session, nameDest, limit)
    return list(sender_rows) + list(receiver_rows)


def get_user_history(session: Session, user_id: str, limit: int = 100):
    return _get_cached_history(session, user_id, limit)


def update_transaction_state(session: Session, event_id: str, status: str, prob: float, payload: dict | None = None):
    ts_row = session.query(TransactionsState).filter_by(event_id=event_id).first()

    if ts_row is None and payload:
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
        # Race-Condition Fix: Don't let PENDING overwrite a finished state
        if status == "PENDING" and ts_row.status in ["SUCCESS", "BLOCKED"]:
            return

        ts_row.status = status
        ts_row.fraud_probability = prob
        ts_row.decided_at = datetime.now(timezone.utc)
        # Note: We don't commit here so the calling task can batch updates


def save_history_transaction(session: Session, data: dict, event_id: str):
    """Saves to SQL and updates the sliding-window cache in Redis."""
    hist = TransactionsHistory(
        event_id=event_id,
        **{k: v for k, v in data.items() if hasattr(TransactionsHistory, k)}
    )
    session.add(hist)

    # CRITICAL: Commit immediately so parallel enrichment tasks can see this record
    session.commit()

    try:
        # Prepare for Redis
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
    """Logs flagged transactions. Commit is required for the SHAP worker to find it."""
    sus = SuspiciousTransactions(
        event_id=event_id,
        amount=float(data.get("amount", 0.0)),
        nameOrig=data.get("nameOrig"),
        nameDest=data.get("nameDest"),
        type=data.get("type"),
        plot_path=plot_path,  # Now correctly assigned
        timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
    )
    session.add(sus)

    # CRITICAL: Commit here so the generate_shap_task worker finds the record instantly
    session.commit()
    log.info(f"Suspicious transaction {event_id} logged. Path: {plot_path}")


def save_to_dlq(task_name: str, event_id: str, payload: dict, error_msg: str):
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