# src/models/schemas.py

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from typing import Literal


class TransactionCreate(BaseModel):
    amount: float = Field(gt=0, description="Betrag in Euro", examples=["181.0"])
    type: Literal["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]
    nameOrig: str = Field(min_length=1, max_length=50, examples=["C840083671"])
    nameDest: str = Field(min_length=1, max_length=50, examples=["C38997010"])

    # --- ADD THESE 4 LINES ---
    oldbalanceOrg: float = 0.0
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    # -------------------------

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), examples=["2026-04-12T09:30:00"])
