from datetime import datetime

import msgspec


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
    event_id: str | None = None
    timestamp: datetime | None = None


# Pre-compiled tools for the helper
HISTORY_DECODER = msgspec.json.Decoder(
    list[HistoryStruct]
)  # Decoder needs a precise blueprint of what to expect since the Decoder receives a raw byte string
HISTORY_ENCODER = msgspec.json.Encoder()
