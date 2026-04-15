import msgspec
from datetime import datetime
from typing import Optional, List

class HistoryStruct(msgspec.Struct, omit_defaults=True):
    """
    High-speed C-struct for Redis.
    Replaces Pydantic/RowObj for 80x faster parsing.
    """
    step: int
    type: str
    amount: float
    nameOrig: str
    nameDest: str
    oldbalanceOrg: float = 0.0
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    event_id: Optional[str] = None
    timestamp: Optional[datetime] = None

# Pre-compiled tools for the helper
HISTORY_DECODER = msgspec.json.Decoder(List[HistoryStruct]) # Returns a list of HistoryStruct instances
HISTORY_ENCODER = msgspec.json.Encoder()
