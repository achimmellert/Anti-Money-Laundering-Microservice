from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    BOOLEAN,
    JSON,
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utc_now():
    """Helper to ensure all default timestamps are strictly UTC."""
    return datetime.now(UTC)


class TransactionStatus(Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    BLOCKED = "BLOCKED"
    PERSISTED = "PERSISTED"


class TransactionType(Enum):
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    PAYMENT = "PAYMENT"
    DEBIT = "DEBIT"


class TransactionsState(Base):
    __tablename__ = "transactions_state"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    event_id: Mapped[str] = mapped_column(String(36), index=True, unique=True)
    type: Mapped[TransactionType] = mapped_column(SQLAlchemyEnum(TransactionType), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    nameOrig: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    nameDest: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    step: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[TransactionStatus] = mapped_column(
        SQLAlchemyEnum(TransactionStatus), default=TransactionStatus.PENDING
    )
    fraud_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class SuspiciousTransactions(Base):
    __tablename__ = "suspicious_transactions"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    event_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    type: Mapped[TransactionType] = mapped_column(SQLAlchemyEnum(TransactionType), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    nameOrig: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    nameDest: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    plot_path: Mapped[str] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    logged_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class TransactionsHistory(Base):
    __tablename__ = "transaction_history"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    event_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    step: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[TransactionType] = mapped_column(
        SQLAlchemyEnum(TransactionType), nullable=False, index=True
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    nameOrig: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    nameDest: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    oldbalanceOrg: Mapped[float] = mapped_column(Float)
    newbalanceOrig: Mapped[float] = mapped_column(Float)
    oldbalanceDest: Mapped[float] = mapped_column(Float)
    newbalanceDest: Mapped[float] = mapped_column(Float)
    isFraud: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MonitoringInference(Base):
    __tablename__ = "monitoring_inference"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    event_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    y_pred_proba: Mapped[float] = mapped_column(Float, nullable=False)
    y_true: Mapped[bool] = mapped_column(BOOLEAN, nullable=True)
    decision: Mapped[TransactionStatus] = mapped_column(
        SQLAlchemyEnum(TransactionStatus), nullable=False
    )
    features_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    __table_args__ = (Index("ix_monitoring_ts_decision", "timestamp", "decision"),)


class DeadLetterQueue(Base):
    __tablename__ = "dead_letter_queue"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_name: Mapped[str] = mapped_column(String(255))
    event_id: Mapped[str] = mapped_column(String(255))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    error_msg: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
