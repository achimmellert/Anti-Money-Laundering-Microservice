# src/services/feature_engineering.py
import pandas as pd
import numpy as np

from src.config.settings import settings
from src.models.dataframes import HistoryDataFrameSchema, EnrichedFeaturesSchema

"""thresholds = {
    # Using the 99th Percentile
    "historical_avg_upper": train_df["sender_historical_avg"].quantile(0.99),
    "avg_ratio_upper": train_df["sender_total_avg_ratio"].quantile(0.99),
    "fraction_of_balance_upper": train_df["sender_fraction_of_balance"].quantile(0.99),

    "receiver_hist_avg_upper": train_df["receiver_hist_avg"].quantile(0.99),
    "receiver_ratio_upper": train_df["receiver_amount_to_avg_ratio"].quantile(0.99)
}"""


REQUIRED_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "timestamp"
]


# --- STATIC ML THRESHOLDS ---
# These should come from the Training Dataset's 99th percentile.
# So these Thresholds are considered placeholders here.
GLOBAL_THRESHOLDS = {
    "historical_avg_upper": 50000.0,
    "avg_ratio_upper": 10.0,
    "fraction_of_balance_upper": 1.0,
    "receiver_hist_avg_upper": 50000.0,
    "receiver_ratio_upper": 10.0
}


def infer_missing_balances(cur_df: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing or 0.0 balances in the current transaction using historical data.
    It acts as a synthetic time-machine.
    Takes fully validated Pandera DataFrames as inputs.
    """
    df = cur_df.copy()

    # A completely brand-new user has no history.
    if hist_df.empty:
        return df

    idx = df.index[0]

    orig_acc = df.at[idx, 'nameOrig']
    dest_acc = df.at[idx, 'nameDest']
    amount = df.at[idx, 'amount']
    tx_type = df.at[idx, 'type']

    def _get_latest_balance(target_acc: str):
        """
        Helper: Scans history to find the most recent balance for a specific account,
        regardless of whether they were the sender or receiver.
        """
        mask = (hist_df['nameOrig'] == target_acc) | (hist_df['nameDest'] == target_acc)
        matches = hist_df[mask]

        if matches.empty:
            return None

        # Because hist_df is already sorted newest-first, .iloc[0] is guaranteed to be the most recent
        latest = matches.iloc[0]

        # Return the correct balance column depending on their role in that past transaction
        if latest['nameOrig'] == target_acc:
            return latest['newbalanceOrig']
        return latest['newbalanceDest']

    # 1.) Impute Origin Balances
    if pd.isna(df.at[idx, 'oldbalanceOrg']) or df.at[idx, 'oldbalanceOrg'] == 0.0:
        last_orig_bal = _get_latest_balance(orig_acc)

        if last_orig_bal is not None:
            df.at[idx, 'oldbalanceOrg'] = last_orig_bal

            # If the new balance is also missing/0.0, it is calculated mathematically based on transaction type
            if pd.isna(df.at[idx, 'newbalanceOrig']) or df.at[idx, 'newbalanceOrig'] == 0.0:
                if tx_type in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
                    df.at[idx, 'newbalanceOrig'] = max(0.0, last_orig_bal - amount)
                elif tx_type == 'CASH_IN':
                    df.at[idx, 'newbalanceOrig'] = last_orig_bal + amount

    # 2.) Impute Destination Balances
    if pd.isna(df.at[idx, 'oldbalanceDest']) or df.at[idx, 'oldbalanceDest'] == 0.0:
        last_dest_bal = _get_latest_balance(dest_acc)

        if last_dest_bal is not None:
            df.at[idx, 'oldbalanceDest'] = last_dest_bal

            # Calculate the new destination balance if missing
            if pd.isna(df.at[idx, 'newbalanceDest']) or df.at[idx, 'newbalanceDest'] == 0.0:
                if tx_type in ['TRANSFER', 'CASH_OUT']:
                    df.at[idx, 'newbalanceDest'] = last_dest_bal + amount

    return df


# "History + 1" operation
def create_features(df: pd.DataFrame[HistoryDataFrameSchema]) -> pd.DataFrame[EnrichedFeaturesSchema]:
    df = df.copy()

    # --- SENDER FEATURES ---
    df = df.sort_values(by=["nameOrig", "step"]).reset_index(drop=True)

    sender_groups = df.groupby("nameOrig")
    df["sender_time_since_last_trans"] = sender_groups["step"].diff().fillna(0)


    rolling_amount_10 = sender_groups.rolling(window=10, on="step", min_periods=1)["amount"]
    df["sender_trans_count_5_steps"] = sender_groups.rolling(window=5, on="step", min_periods=1)[
        "amount"].count().values
    df["sender_total_trans_count"] = sender_groups.cumcount()

    df["sender_avg_trans_amount_10_steps"] = rolling_amount_10.mean().values
    df["sender_std_trans_amount_10_steps"] = rolling_amount_10.std().fillna(0.0).values
    df["sender_std_trans_amount_10_steps_was_na"] = (df["sender_std_trans_amount_10_steps"] == 0.0).astype(int)

    historical_avg = sender_groups["amount"].transform(lambda x: x.shift(1).expanding().mean())

    # Use Static Thresholds
    df["sender_total_avg_ratio_extreme_values"] = (
            (historical_avg == np.inf) | (historical_avg == -np.inf) |
            (historical_avg >= GLOBAL_THRESHOLDS["historical_avg_upper"]) | (historical_avg.isna())
    ).astype(int)

    s_clean_sender_ratio = (df["amount"] / historical_avg).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["sender_total_avg_ratio_clipped"] = s_clean_sender_ratio.clip(lower=0,
                                                                     upper=GLOBAL_THRESHOLDS["avg_ratio_upper"])

    df["sender_total_avg_ratio_log1p"] = np.log1p(df["sender_total_avg_ratio_clipped"])

    df["sender_total_amount_history"] = sender_groups["amount"].transform(lambda x: x.shift(1).cumsum().fillna(0))
    df["sender_fraction_of_balance"] = df["amount"] / df["oldbalanceOrg"]

    df["sender_fraction_of_balance_sent_is_extreme"] = (
            (df["sender_fraction_of_balance"] == np.inf) | (df["sender_fraction_of_balance"] == -np.inf) |
            (df["sender_fraction_of_balance"].isna()) |
            (df["sender_fraction_of_balance"] >= GLOBAL_THRESHOLDS["fraction_of_balance_upper"])
    ).astype(int)

    s_clean_frac = df["sender_fraction_of_balance"].replace([np.inf, -np.inf], np.nan).fillna(999).clip(lower=0, upper=GLOBAL_THRESHOLDS["fraction_of_balance_upper"])
    df["sender_fraction_of_balance_clipped"] = s_clean_frac
    df["sender_fraction_of_balance"] = np.log1p(s_clean_frac)
    df["sender_emptied_account"] = (df["newbalanceOrig"] <= 0.01).astype(int)

    # --- RECEIVER FEATURES ---
    df = df.sort_values(by=["nameDest", "step"]).reset_index(drop=True)
    receiver_groups = df.groupby("nameDest")

    df["receiver_trans_count_5_steps"] = receiver_groups.rolling(window=5, on="step", min_periods=1)[
        "amount"].count().values

    # Optimized Unique Sender calculation (Using factorize for speed)
    df["_sender_code"] = pd.factorize(df["nameOrig"])[0]
    df["receiver_unique_senders_5_steps"] = (
        df.groupby("nameDest")["_sender_code"]
        .rolling(window=5, min_periods=1)
        .apply(lambda x: len(np.unique(x)), raw=True).values
    )

    df["receiver_total_amount_5_steps"] = receiver_groups.rolling(window=5, on="step", min_periods=1)[
        "amount"].sum().values

    receiver_hist_avg = receiver_groups["amount"].transform(lambda x: x.shift(1).expanding().mean())
    df["receiver_amount_to_avg_ratio_extreme_values"] = (
            (receiver_hist_avg == np.inf) | (receiver_hist_avg == -np.inf) |
            (receiver_hist_avg >= GLOBAL_THRESHOLDS["receiver_hist_avg_upper"]) | (receiver_hist_avg.isna())
    ).astype(int)

    s_clean_receiver = (df["amount"] / receiver_hist_avg).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["receiver_amount_to_avg_ratio_clipped"] = s_clean_receiver.clip(lower=0,
                                                                       upper=GLOBAL_THRESHOLDS["receiver_ratio_upper"])
    df["receiver_amount_to_avg_ratio_log1p"] = np.log1p(df["receiver_amount_to_avg_ratio_clipped"])

    df["pair_transaction_count"] = df.groupby(["nameOrig", "nameDest"]).cumcount()

    return df[settings.RAW_FEATURE_COLUMNS + ["step", "nameOrig", "nameDest", "timestamp"]].copy()
