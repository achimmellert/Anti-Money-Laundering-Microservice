# src/services/feature_engineering.py
import pandas as pd
import numpy as np
from src.config.settings import settings

"""thresholds = {
    # Method 1: Using the 99th Percentile (Standard approach for ratios)
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
# These should come from your Training Dataset's 99th percentile, NOT calculated on the fly.
GLOBAL_THRESHOLDS = {
    "historical_avg_upper": 50000.0,  # Replace with training IQR upper bound
    "avg_ratio_upper": 10.0,  # e.g., anything 10x their normal amount is clipped
    "fraction_of_balance_upper": 1.0,  # Can't send more than 100% of balance (usually)
    "receiver_hist_avg_upper": 50000.0,
    "receiver_ratio_upper": 10.0
}


def infer_missing_balances(cur: dict, hist_df: pd.DataFrame) -> dict:
    tx_type = (cur.get("type") or "").upper()
    amt = float(cur.get("amount", 0.0))

    # Fast lookups using boolean masking instead of sorting the whole dataframe
    last_new_sender = None
    last_new_receiver = None
    if not hist_df.empty and "step" in hist_df.columns:
        sender_mask = hist_df["nameOrig"] == cur["nameOrig"]
        if sender_mask.any():
            last_new_sender = hist_df.loc[sender_mask, "newbalanceOrig"].dropna().tail(1)
            last_new_sender = last_new_sender.iloc[0] if not last_new_sender.empty else None

        receiver_mask = hist_df["nameDest"] == cur["nameDest"]
        if receiver_mask.any():
            last_new_receiver = hist_df.loc[receiver_mask, "newbalanceDest"].dropna().tail(1)
            last_new_receiver = last_new_receiver.iloc[0] if not last_new_receiver.empty else None

    old_org = float(cur.get("oldbalanceOrg", 0.0))
    old_dest = float(cur.get("oldbalanceDest", 0.0))

    if (old_org <= 0) and (last_new_sender is not None):
        old_org = float(last_new_sender)
    if (old_dest <= 0) and (last_new_receiver is not None):
        old_dest = float(last_new_receiver)

    if tx_type in {"TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"}:
        new_org = max(old_org - amt, 0.0)
    else:  # CASH_IN
        new_org = max(old_org + amt, 0.0)

    if tx_type in {"TRANSFER", "CASH_IN"}:
        new_dest = max(old_dest + amt, 0.0)
    else:
        new_dest = old_dest

    cur["oldbalanceOrg"] = old_org
    cur["newbalanceOrig"] = new_org
    cur["oldbalanceDest"] = old_dest
    cur["newbalanceDest"] = new_dest
    return cur


# "History + 1" operation
def create_features(df: pd.DataFrame) -> pd.DataFrame:
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

    s_clean_frac = df["sender_fraction_of_balance"].replace([np.inf, -np.inf], np.nan).fillna(999).clip(lower=0, upper=
    GLOBAL_THRESHOLDS["fraction_of_balance_upper"])
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
        # using a fast custom function or leaving raw=False if unavoidable, but .values aligns it correctly
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


