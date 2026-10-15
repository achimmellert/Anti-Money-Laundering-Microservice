from datetime import UTC, datetime
from typing import Literal

from pydantic import UUID4, BaseModel, ConfigDict, Field

from aml.models.db_models import TransactionStatus, TransactionType


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
        default_factory=lambda: datetime.now(UTC),
        examples=["2026-04-12T09:30:00Z"],
    )
    oldbalanceOrg: float = Field(
        default=0.0, description="Kontostand Sender vor Transaktion", examples=[237642.64]
    )
    newbalanceOrig: float = Field(
        default=0.0, description="Kontostand Sender nach Transaktion", examples=[0.0]
    )
    oldbalanceDest: float = Field(
        default=0.0, description="Kontostand Empfänger vor Transaktion", examples=[113161.91]
    )
    newbalanceDest: float = Field(
        default=0.0, description="Kontostand Empfänger nach Transaktion", examples=[350804.55]
    )

    model_config = ConfigDict(extra="allow")


# --- INTERNAL SYSTEM PAYLOAD ---
class TransactionInternal(TransactionCreate):
    """
    The enriched transaction used by Celery, ML workers, and Database helpers.
    Inherits the user input and adds system-controlled fields.
    """

    event_id: UUID4
    step: int

    model_config = ConfigDict(extra="allow")


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
    fraud_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    decided_at: datetime | None = None
    created_at: datetime
    step: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
