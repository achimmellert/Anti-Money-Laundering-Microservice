import msgspec
from datetime import datetime
from typing import Optional, List

class HistoryStruct(msgspec.Struct, omit_defaults=True):
    """
    High-speed C-struct for Redis.
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
HISTORY_DECODER = msgspec.json.Decoder(List[HistoryStruct]) # Decoder needs a precise blueprint of what to expect since the Decoder receives a raw byte string
HISTORY_ENCODER = msgspec.json.Encoder()
