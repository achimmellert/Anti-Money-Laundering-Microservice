# src/models/schemas.py

from pydantic import BaseModel, Field, ConfigDict, UUID4, field_validator
from datetime import datetime, timezone
from typing import Literal, Optional
from src.models.db_models import TransactionType, TransactionStatus


# --- SHARED API LAYER ---
class TransactionBase(BaseModel):
    """Core fields shared across inputs, internal data, and responses."""
    amount: float = Field(gt=0, description="Betrag in Euro", examples=[181.0])
    type: TransactionType = Field(description="Typ der Transaktion")
    nameOrig: str = Field(min_length=1, max_length=50, examples=["C840083671"])
    nameDest: str = Field(min_length=1, max_length=50, examples=["C38997010"])


# --- PUBLIC API INPUT ---
class TransactionCreate(TransactionBase):
    """The data coming directly from the API client."""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), examples=["2026-04-12T09:30:00Z"]
    )


# --- INTERNAL SYSTEM PAYLOAD ---
class TransactionInternal(TransactionCreate):
    """
    The enriched transaction used by Celery, ML workers, and Database helpers.
    Inherits the user input and adds system-controlled fields.
    """
    event_id: UUID4
    step: int

    oldbalanceOrg: float = 0.0
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0


# --- PUBLIC API RESPONSES ---
class TransactionAcceptedResponse(BaseModel):
    """The immediate response returned to the API client when accepted."""
    event_id: UUID4
    status: Literal["received"] = "received"
    message: str = "Transaction processing initiated."


class TransactionStatusResponse(TransactionBase):
    """The complete public status of a transaction read from the database."""
    event_id: UUID4
    status: TransactionStatus = Field(description="Aktueller Verarbeitungsstatus")

    # ML probabilities range from 0.0 to 1.0 (0% to 100%)
    fraud_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    decided_at: Optional[datetime] = None
    created_at: datetime
    step: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
