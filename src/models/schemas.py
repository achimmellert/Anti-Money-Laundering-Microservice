# src/models/schemas.py

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from typing import Literal


class TransactionCreate(BaseModel):
    amount: float = Field(gt=0, description="Betrag in Euro", examples=["181.0"])
    type: Literal["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]
    nameOrig: str = Field(min_length=1, max_length=50, examples=["C840083671"])
    nameDest: str = Field(min_length=1, max_length=50, examples=["C38997010"])
    # sets the timestamp to the current time (default for constants, default_factory for dynamic values)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), examples=["2026-04-12T09:30:00"])
