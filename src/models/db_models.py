from sqlalchemy import String, Float, DateTime, BigInteger, Integer, UUID, BOOLEAN, Index
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.orm import registry, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from enum import Enum
from datetime import datetime


Base = registry().generate_base()


class TransactionStatus(Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    BLOCKED = "BLOCKED"

class TransactionType(Enum):
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    PAYMENT = "PAYMENT"
    DEBIT = "DEBIT"


class TransactionsState(Base):
    __tablename__ = "transactions_state"

    # id fürs effiziente Schreiben (autoincrement) --> schwere Last; UUID als Primärschlüssel zu verwenden wäre sehr ineffizient
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # event_id indexiert fürs schnelle Lesen (Inhaltsverzeichnis im RAM)
    event_id: Mapped[str] = mapped_column(UUID (as_uuid=False), index=True, unique=True)
    type: Mapped[TransactionType] = mapped_column(SQLAlchemyEnum(TransactionType), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    nameOrig: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    nameDest: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    step: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[TransactionStatus] = mapped_column(
        SQLAlchemyEnum(TransactionStatus), default=TransactionStatus.PENDING
    )
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SuspiciousTransactions(Base):
    __tablename__ = "suspicious_transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(UUID (as_uuid=False), unique=True, nullable=False)
    type: Mapped[TransactionType] = mapped_column(SQLAlchemyEnum(TransactionType), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    nameOrig: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    nameDest: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    logged_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TransactionsHistory(Base):
    __tablename__ = "transactions_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(UUID (as_uuid=False), unique=True, index=True)
    step: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[TransactionType] = mapped_column(SQLAlchemyEnum(TransactionType), nullable=False, index=True)
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

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(UUID (as_uuid=False), unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    y_pred_proba: Mapped[float] = mapped_column(Float, nullable=False)
    y_true: Mapped[bool] = mapped_column(BOOLEAN)
    decision: Mapped[TransactionStatus] = mapped_column(
        SQLAlchemyEnum(TransactionStatus), nullable=False
    )

    # Feature-Vektoren werden als JSON gespeichert, fürs Logging OK
    features_json: Mapped[dict] = mapped_column(MySQLJSON, nullable=False)

    # Zusammengesetzter Index aus timestamp und decision (Für späteres Dashboard)
    __table_args__ = (
        Index("ix_monitoring_ts_decision", "timestamp", "decision"),
    )
