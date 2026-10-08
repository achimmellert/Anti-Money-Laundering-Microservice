from datetime import UTC, datetime

import redis
from loguru import logger as log
from sqlalchemy import desc, or_, select
from sqlalchemy.orm import Session

from aml.config.settings import settings
from aml.models.db_models import (
    DeadLetterQueue,
    SuspiciousTransactions,
    TransactionsHistory,
    TransactionsState,
    TransactionStatus,
)
from aml.models.schemas import TransactionInternal
from aml.models.structs import HISTORY_DECODER, HISTORY_ENCODER, HistoryStruct
from aml.services.database import SessionLocal

redis_client = redis.Redis.from_url(settings.redis_uri, decode_responses=False)


# --- INTERNAL HELPERS ---


def _get_cached_history(session: Session, user_id: str, limit: int) -> list[HistoryStruct]:
    """Tries Redis first, falls back to SQL, and ensures consistent return types."""
    key = f"hist:{user_id}"

    # 1. Redis Look-up
    try:
        raw_data = redis_client.lrange(key, 0, limit - 1)
        if raw_data:
            data_bytes = [
                r if isinstance(r, (bytes, bytearray, memoryview)) else str(r).encode()
                for r in raw_data
            ]
            return HISTORY_DECODER.decode(b"[" + b",".join(data_bytes) + b"]")
    except Exception as e:
        log.warning(f"Redis Read Error for {user_id}: {e}")

    # 2. SQL Fallback
    stmt = (
        select(TransactionsHistory)
        .where(
            or_(
                TransactionsHistory.nameOrig == user_id,
                TransactionsHistory.nameDest == user_id,
            )
        )
        .order_by(desc(TransactionsHistory.step))
        .limit(limit)
    )
    rows = session.execute(stmt).scalars().all()

    if not rows:
        return []

    structs = [
        HistoryStruct(
            step=r.step,
            type=r.type.value if hasattr(r.type, "value") else str(r.type),
            amount=float(r.amount),
            nameOrig=r.nameOrig,
            nameDest=r.nameDest,
            oldbalanceOrg=float(r.oldbalanceOrg or 0),
            newbalanceOrig=float(r.newbalanceOrig or 0),
            oldbalanceDest=float(r.oldbalanceDest or 0),
            newbalanceDest=float(r.newbalanceDest or 0),
            event_id=str(r.event_id) if r.event_id else None,
            timestamp=r.timestamp,
        )
        for r in rows
    ]

    # 4. Cache Hydration
    try:
        encoded_structs = [HISTORY_ENCODER.encode(s) for s in structs]
        with redis_client.pipeline() as pipe:
            pipe.delete(key)
            if encoded_structs:
                pipe.rpush(key, *encoded_structs)
            pipe.expire(key, 3600)
            pipe.execute()
    except Exception as e:
        log.warning(f"Redis Hydration Error: {e}")

    return structs


# --- PUBLIC API ---


def get_account_history(
    session: Session, nameOrig: str, nameDest: str, limit: int = 200
) -> list[HistoryStruct]:
    sender_rows = _get_cached_history(session, nameOrig, limit)
    receiver_rows = _get_cached_history(session, nameDest, limit)
    return list(sender_rows) + list(receiver_rows)


def get_user_history(session: Session, user_id: str, limit: int = 100) -> list[HistoryStruct]:
    return _get_cached_history(session, user_id, limit)


def update_transaction_state(
    session: Session,
    event_id: str,
    status: TransactionStatus | str,
    prob: float | None,
    payload: TransactionInternal | None = None,
):
    # Safe Enum Coercion
    if isinstance(status, str):
        status = TransactionStatus(status)

    stmt = select(TransactionsState).where(TransactionsState.event_id == event_id).with_for_update()
    ts_row = session.execute(stmt).scalar_one_or_none()

    if ts_row is None and payload:
        ts_row = TransactionsState(
            event_id=event_id,
            type=payload.type,
            amount=payload.amount,
            nameOrig=payload.nameOrig,
            nameDest=payload.nameDest,
            step=payload.step,
            timestamp=payload.timestamp,
        )
        session.add(ts_row)

    if ts_row:
        # Prevent PENDING or PERSISTED from overwriting a finished decision (SUCCESS / BLOCKED)
        if status in [
            TransactionStatus.PENDING,
            TransactionStatus.PERSISTED,
        ] and ts_row.status in [TransactionStatus.SUCCESS, TransactionStatus.BLOCKED]:
            return

        ts_row.status = status
        ts_row.fraud_probability = prob

        if status in [TransactionStatus.SUCCESS, TransactionStatus.BLOCKED]:
            ts_row.decided_at = datetime.now(UTC)
        # No commit here (ensuring Unit of Work pattern)


def save_history_transaction(session: Session, payload: TransactionInternal):
    """Saves to SQL and updates the sliding-window cache in Redis."""
    hist = TransactionsHistory(
        event_id=str(payload.event_id),
        step=payload.step,
        type=payload.type,
        amount=payload.amount,
        nameOrig=payload.nameOrig,
        nameDest=payload.nameDest,
        oldbalanceOrg=payload.oldbalanceOrg,
        newbalanceOrig=payload.newbalanceOrig,
        oldbalanceDest=payload.oldbalanceDest,
        newbalanceDest=payload.newbalanceDest,
        timestamp=payload.timestamp,
    )

    session.add(hist)

    try:
        struct_data = HistoryStruct(
            step=payload.step,
            type=payload.type.value if hasattr(payload.type, "value") else str(payload.type),
            amount=payload.amount,
            nameOrig=payload.nameOrig,
            nameDest=payload.nameDest,
            oldbalanceOrg=payload.oldbalanceOrg,
            newbalanceOrig=payload.newbalanceOrig,
            oldbalanceDest=payload.oldbalanceDest,
            newbalanceDest=payload.newbalanceDest,
            event_id=str(payload.event_id),  # cast to string if needed
            timestamp=payload.timestamp,
        )
        encoded_obj = HISTORY_ENCODER.encode(struct_data)

        for uid in filter(None, [payload.nameOrig, payload.nameDest]):
            key = f"hist:{uid}"
            if redis_client.exists(key):
                with redis_client.pipeline() as pipe:
                    pipe.lpush(key, encoded_obj)
                    pipe.ltrim(key, 0, 199)
                    pipe.execute()

    except Exception as e:
        log.warning(f"Redis Live-Update Error: {e}")


def save_suspicious_transaction(
    session: Session, payload: TransactionInternal, event_id: str, plot_path: str
):
    """Logs flagged transactions. Pydantic guarantees clean data structure."""
    sus = SuspiciousTransactions(
        event_id=event_id,
        amount=payload.amount,
        nameOrig=payload.nameOrig,
        nameDest=payload.nameDest,
        type=payload.type,
        plot_path=plot_path,
        timestamp=payload.timestamp,
    )
    session.add(sus)

    # FIX: Commit removed to preserve Celery batch atomicity
    log.info(f"Suspicious transaction {event_id} logged. Path: {plot_path}")


def save_to_dlq(task_name: str, event_id: str, payload: dict, error_msg: str):
    try:
        with SessionLocal() as session:
            entry = DeadLetterQueue(
                task_name=task_name,
                event_id=event_id,
                payload=payload,
                error_msg=str(error_msg),
            )
            session.add(entry)
            session.commit()
            log.warning(f"Task '{task_name}' (Event {event_id}) moved to DLQ.")
    except Exception as e:
        log.error(f"DLQ CRITICAL FAILURE: {e}")
