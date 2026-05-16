# src/models/schemas.py

from pydantic import BaseModel, Field, ConfigDict, UUID4, field_validator
from datetime import datetime, timezone
from typing import Literal, Optional


# The following schemas prevent anti-pattern solution


# --- PUBLIC API Schema ---
class TransactionCreate(BaseModel):
    """The data coming directly from the API"""
    amount: float = Field(gt=0, description="Betrag in Euro", examples=["181.0"])
    type: Literal["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]
    nameOrig: str = Field(min_length=1, max_length=50, examples=["C840083671"])
    nameDest: str = Field(min_length=1, max_length=50, examples=["C38997010"])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), examples=["2026-04-12T09:30:00"])


# --- INTERNAL SYSTEM SCHEMA ---
class TransactionInternal(TransactionCreate):
    """
    The enriched transaction used by Celery, ML workers and Database helpers.
    Inherits the user input and adds system-controlled fields.
    """
    event_id: UUID4
    step: int

    oldbalanceOrg: float = 0.0
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0


class TransactionStatusResponse(BaseModel):

    event_id: UUID4
    status: str
    fraud_probability: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    decided_at: Optional[datetime] = None
    created_at: datetime
    type: str
    amount: float
    nameOrig: str
    nameDest: str
    step: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


