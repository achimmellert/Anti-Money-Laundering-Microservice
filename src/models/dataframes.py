import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series


class HistoryDataFrameSchema(pa.DataFrameModel):
    """Pandera schema to strictly validate and coerce historical transaction DataFrames."""
    step: Series[int] = pa.Field(ge=0)
    type: Series[str]
    amount: Series[float] = pa.Field(ge=0.0)
    nameOrig: Series[str]
    oldbalanceOrg: Series[float]
    newbalanceOrig: Series[float]
    nameDest: Series[str]
    oldbalanceDest: Series[float]
    newbalanceDest: Series[float]
    timestamp: Series[pd.DatetimeTZDtype] = pa.Field(nullable=True)

    class Config:
        # Coerce=True forces Pandas to automatically cast types (e.g., string to float) if the underlying data allows it.
        coerce = True


class EnrichedFeaturesSchema(pa.DataFrameModel):
    """
    The definitive data contract for the ML Inference pipeline.
    Ensures no Infinity or NaN values leak into the XGBoost/LightGBM models.
    """

    # Base Transaction Data
    step: Series[int]
    nameOrig: Series[str]
    nameDest: Series[str]
    timestamp: Series[pd.DatetimeTZDtype] = pa.Field(nullable=True)

    # Required Raw Features
    amount: Series[float]
    type: Series[str]
    oldbalanceOrg: Series[float]
    newbalanceOrig: Series[float]
    oldbalanceDest: Series[float]
    newbalanceDest: Series[float]

    # --- Sender Features ---
    sender_time_since_last_trans: Series[float]
    sender_trans_count_5_steps: Series[float]  # Rolling counts return floats in Pandas
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

    # --- Receiver Features ---
    receiver_trans_count_5_steps: Series[float]
    receiver_unique_senders_5_steps: Series[float]
    receiver_total_amount_5_steps: Series[float]
    receiver_amount_to_avg_ratio_extreme_values: Series[int] = pa.Field(isin=[0, 1])
    receiver_amount_to_avg_ratio_clipped: Series[float]
    receiver_amount_to_avg_ratio_log1p: Series[float]
    pair_transaction_count: Series[int]

    class Config:
        strict = True # Prevents any unexpected columns from sneaking in (Crashes then)
        coerce = True
