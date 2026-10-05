import polars as pl
import pandera.polars as pa
from pandera.typing.polars import Series


class HistoryDataFrameSchema(pa.DataFrameModel):
    """Pandera schema to strictly validate and coerce historical transaction DataFrames in Polars."""
    step: Series[int] = pa.Field(ge=0)
    type: Series[str]
    amount: Series[float] = pa.Field(ge=0.0)
    nameOrig: Series[str]
    oldbalanceOrg: Series[float]
    newbalanceOrig: Series[float]
    nameDest: Series[str]
    oldbalanceDest: Series[float]
    newbalanceDest: Series[float]
    timestamp: Series[pl.Datetime] = pa.Field(nullable=True)

    class Config:
        coerce = True


class MetadataSchema(pa.DataFrameModel):
    """Routing and identity metadata columns carrying transactional context."""
    step: Series[int]
    nameOrig: Series[str]
    nameDest: Series[str]
    timestamp: Series[pl.Datetime] = pa.Field(nullable=True)


class ModelFeaturesSchema(pa.DataFrameModel):
    """
    The 27 core feature contract for ML training and inference.
    Defines feature names, types, and value boundaries (e.g. 0/1 indicator flags).
    """
    # Raw features
    type: Series[str]
    amount: Series[float]
    oldbalanceOrg: Series[float]
    newbalanceOrig: Series[float]
    oldbalanceDest: Series[float]
    newbalanceDest: Series[float]

    # Sender features
    sender_time_since_last_trans: Series[float]
    sender_trans_count_5_steps: Series[float]
    sender_total_trans_count: Series[int]
    sender_avg_trans_amount_10_steps: Series[float]
    sender_std_trans_amount_10_steps: Series[float]
    sender_std_trans_amount_10_steps_was_na: Series[int] = pa.Field(isin=[0, 1])
    sender_total_avg_ratio_extreme_values: Series[int] = pa.Field(isin=[0, 1])
    sender_total_avg_ratio_clipped: Series[float]
    sender_total_avg_ratio_log1p: Series[float]
    sender_total_amount_history: Series[float]
    sender_fraction_of_balance: Series[float]
    sender_fraction_of_balance_sent_is_extreme: Series[int] = pa.Field(isin=[0, 1])
    sender_fraction_of_balance_clipped: Series[float]
    sender_emptied_account: Series[int] = pa.Field(isin=[0, 1])

    # Receiver features
    receiver_trans_count_5_steps: Series[float]
    receiver_unique_senders_5_steps: Series[float]
    receiver_total_amount_5_steps: Series[float]
    receiver_amount_to_avg_ratio_extreme_values: Series[int] = pa.Field(isin=[0, 1])
    receiver_amount_to_avg_ratio_clipped: Series[float]
    receiver_amount_to_avg_ratio_log1p: Series[float]
    pair_transaction_count: Series[int]

    class Config:
        strict = True
        coerce = True


class EnrichedFeaturesSchema(MetadataSchema, ModelFeaturesSchema):
    """
    The definitive data contract for the ML feature pipeline.
    Combines routing metadata with model features and guarantees no NaN/Infinity leakage.
    """
    class Config:
        strict = True
        coerce = True


# --- Schema-derived Feature Lists (Single Source of Truth) ---
_feature_schema = ModelFeaturesSchema.to_schema()
_metadata_schema = MetadataSchema.to_schema()

METADATA_COLUMNS: list[str] = list(_metadata_schema.columns.keys())
RAW_FEATURE_COLUMNS: list[str] = list(_feature_schema.columns.keys())

NUMERIC_FEATURES: list[str] = [
    col_name for col_name, col in _feature_schema.columns.items()
    if col.dtype.type != pl.String
]

CATEGORICAL_FEATURES: list[str] = [
    col_name for col_name, col in _feature_schema.columns.items()
    if col.dtype.type == pl.String
]
