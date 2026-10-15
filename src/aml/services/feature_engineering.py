# src/aml/services/feature_engineering.py
import polars as pl

from aml.models.dataframes import (
    METADATA_COLUMNS,
    RAW_FEATURE_COLUMNS,
    EnrichedFeaturesSchema,
)

REQUIRED_COLS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "timestamp",
]

# --- STATIC ML THRESHOLDS ---
GLOBAL_THRESHOLDS = {
    "historical_avg_upper": 50000.0,
    "avg_ratio_upper": 10.0,
    "fraction_of_balance_upper": 1.0,
    "receiver_hist_avg_upper": 50000.0,
    "receiver_ratio_upper": 10.0,
}


def infer_missing_balances(cur_df: pl.DataFrame, hist_df: pl.DataFrame) -> pl.DataFrame:
    """
    Imputes missing or 0.0 balances in the current transaction using historical data.
    Acts as a synthetic time-machine to protect model feature integrity.
    """
    if hist_df.is_empty():
        return cur_df

    cur_dict = cur_df.to_dicts()[0]
    orig_acc = cur_dict["nameOrig"]
    dest_acc = cur_dict["nameDest"]
    amount = float(cur_dict["amount"])
    tx_type = str(cur_dict["type"])

    def _get_latest_balance(target_acc: str):
        matches = hist_df.filter(
            (pl.col("nameOrig") == target_acc) | (pl.col("nameDest") == target_acc)
        )
        if matches.is_empty():
            return None
        # hist_df is sorted descending by step
        latest = matches.to_dicts()[0]
        if latest["nameOrig"] == target_acc:
            return float(latest["newbalanceOrig"])
        return float(latest["newbalanceDest"])

    # 1.) Impute Origin Balances
    cur_old_orig = cur_dict.get("oldbalanceOrg")
    if cur_old_orig is None or cur_old_orig == 0.0:
        last_orig_bal = _get_latest_balance(orig_acc)
        if last_orig_bal is not None:
            cur_dict["oldbalanceOrg"] = last_orig_bal
            cur_new_orig = cur_dict.get("newbalanceOrig")
            if cur_new_orig is None or cur_new_orig == 0.0:
                if tx_type in ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]:
                    cur_dict["newbalanceOrig"] = max(0.0, last_orig_bal - amount)
                elif tx_type == "CASH_IN":
                    cur_dict["newbalanceOrig"] = last_orig_bal + amount

    # 2.) Impute Destination Balances
    cur_old_dest = cur_dict.get("oldbalanceDest")
    if cur_old_dest is None or cur_old_dest == 0.0:
        last_dest_bal = _get_latest_balance(dest_acc)
        if last_dest_bal is not None:
            cur_dict["oldbalanceDest"] = last_dest_bal
            cur_new_dest = cur_dict.get("newbalanceDest")
            if (cur_new_dest is None or cur_new_dest == 0.0) and tx_type in [
                "TRANSFER",
                "CASH_OUT",
            ]:
                cur_dict["newbalanceDest"] = last_dest_bal + amount

    return pl.DataFrame([cur_dict], schema=cur_df.schema)


def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates rolling and expanding features over the historical transaction window using Polars.
    Enforces strict data contracts via Pandera runtime validation before returning.
    """
    has_orig_idx = "_orig_idx" in df.columns
    if not has_orig_idx:
        df = df.with_row_index("_orig_idx")

    if "timestamp" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Datetime).alias("timestamp"))

    # --- SENDER FEATURES ---
    df = df.sort(["nameOrig", "step"])

    hist_avg_expr = pl.col("amount").cum_sum().shift(1) / pl.int_range(0, pl.len()).cast(pl.Float64)
    ratio_sender_expr = (
        pl.when(hist_avg_expr.is_null() | hist_avg_expr.is_infinite() | (hist_avg_expr == 0))
        .then(0.0)
        .otherwise(pl.col("amount") / hist_avg_expr)
    )

    ratio_sender_clipped = (
        pl.when(ratio_sender_expr.is_null() | ratio_sender_expr.is_infinite())
        .then(0.0)
        .otherwise(ratio_sender_expr.clip(0.0, GLOBAL_THRESHOLDS["avg_ratio_upper"]))
    )

    frac_raw = (
        pl.when(pl.col("oldbalanceOrg") > 0)
        .then(pl.col("amount") / pl.col("oldbalanceOrg"))
        .otherwise(None)
    )

    frac_is_extreme = (
        pl.when(
            frac_raw.is_null()
            | frac_raw.is_infinite()
            | (frac_raw >= GLOBAL_THRESHOLDS["fraction_of_balance_upper"])
        )
        .then(1)
        .otherwise(0)
    )

    frac_clean = (
        pl.when(frac_raw.is_null() | frac_raw.is_infinite())
        .then(GLOBAL_THRESHOLDS["fraction_of_balance_upper"])
        .otherwise(frac_raw.clip(0.0, GLOBAL_THRESHOLDS["fraction_of_balance_upper"]))
    )

    df = (
        df.with_columns(
            sender_time_since_last_trans=pl.col("step")
            .diff()
            .fill_null(0.0)
            .cast(pl.Float64)
            .over("nameOrig"),
            sender_trans_count_5_steps=pl.col("amount")
            .is_not_null()
            .cast(pl.Float64)
            .rolling_sum(window_size=5, min_samples=1)
            .over("nameOrig"),
            sender_total_trans_count=(pl.int_range(0, pl.len())).cast(pl.Int64).over("nameOrig"),
            sender_avg_trans_amount_10_steps=pl.col("amount")
            .rolling_mean(window_size=10, min_samples=1)
            .over("nameOrig"),
            sender_std_trans_amount_10_steps=pl.col("amount")
            .rolling_std(window_size=10, min_samples=1)
            .fill_null(0.0)
            .over("nameOrig"),
            _hist_avg=hist_avg_expr.over("nameOrig"),
            sender_total_amount_history=pl.col("amount")
            .cum_sum()
            .shift(1)
            .fill_null(0.0)
            .over("nameOrig"),
            sender_fraction_of_balance_sent_is_extreme=frac_is_extreme,
            sender_fraction_of_balance_clipped=frac_clean,
            sender_fraction_of_balance=frac_clean.log1p(),
            sender_emptied_account=pl.when(pl.col("newbalanceOrig") <= 0.01).then(1).otherwise(0),
        )
        .with_columns(
            sender_std_trans_amount_10_steps_was_na=pl.when(
                pl.col("sender_std_trans_amount_10_steps") == 0.0
            )
            .then(1)
            .otherwise(0),
            sender_total_avg_ratio_extreme_values=pl.when(
                pl.col("_hist_avg").is_null()
                | pl.col("_hist_avg").is_infinite()
                | (pl.col("_hist_avg") >= GLOBAL_THRESHOLDS["historical_avg_upper"])
            )
            .then(1)
            .otherwise(0),
            sender_total_avg_ratio_clipped=ratio_sender_clipped.over("nameOrig"),
            sender_total_avg_ratio_log1p=ratio_sender_clipped.log1p().over("nameOrig"),
        )
        .drop("_hist_avg")
    )

    # --- RECEIVER FEATURES ---
    df = df.sort(["nameDest", "step"])

    rec_hist_avg_expr = pl.col("amount").cum_sum().shift(1) / pl.int_range(0, pl.len()).cast(
        pl.Float64
    )
    ratio_rec_expr = (
        pl.when(
            rec_hist_avg_expr.is_null() | rec_hist_avg_expr.is_infinite() | (rec_hist_avg_expr == 0)
        )
        .then(0.0)
        .otherwise(pl.col("amount") / rec_hist_avg_expr)
    )

    ratio_rec_clipped = (
        pl.when(ratio_rec_expr.is_null() | ratio_rec_expr.is_infinite())
        .then(0.0)
        .otherwise(ratio_rec_expr.clip(0.0, GLOBAL_THRESHOLDS["receiver_ratio_upper"]))
    )

    df = (
        df.with_columns(
            receiver_trans_count_5_steps=pl.col("amount")
            .is_not_null()
            .cast(pl.Float64)
            .rolling_sum(window_size=5, min_samples=1)
            .over("nameDest"),
            receiver_unique_senders_5_steps=pl.col("nameOrig")
            .cast(pl.Categorical)
            .to_physical()
            .rolling_map(lambda s: float(len(set(s.to_list()))), window_size=5, min_samples=1)
            .over("nameDest")
            .cast(pl.Float64),
            receiver_total_amount_5_steps=pl.col("amount")
            .rolling_sum(window_size=5, min_samples=1)
            .over("nameDest"),
            _rec_hist_avg=rec_hist_avg_expr.over("nameDest"),
            pair_transaction_count=(pl.int_range(0, pl.len()))
            .cast(pl.Int64)
            .over(["nameOrig", "nameDest"]),
        )
        .with_columns(
            receiver_amount_to_avg_ratio_extreme_values=pl.when(
                pl.col("_rec_hist_avg").is_null()
                | pl.col("_rec_hist_avg").is_infinite()
                | (pl.col("_rec_hist_avg") >= GLOBAL_THRESHOLDS["receiver_hist_avg_upper"])
            )
            .then(1)
            .otherwise(0),
            receiver_amount_to_avg_ratio_clipped=ratio_rec_clipped.over("nameDest"),
            receiver_amount_to_avg_ratio_log1p=ratio_rec_clipped.log1p().over("nameDest"),
        )
        .drop("_rec_hist_avg")
    )

    # Restore exact input row order to prevent feature-label misalignment
    df = df.sort("_orig_idx")
    if not has_orig_idx:
        df = df.drop("_orig_idx")

    # Extract final features specified by the schema contracts
    final_cols = RAW_FEATURE_COLUMNS + METADATA_COLUMNS
    feat_df = df.select(final_cols)

    # --- PANDERA RUNTIME VALIDATION ---
    return EnrichedFeaturesSchema.validate(feat_df)
