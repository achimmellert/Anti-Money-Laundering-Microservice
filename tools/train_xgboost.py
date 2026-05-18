import os
import numpy as np
import polars as pl
from loguru import logger as log
import warnings
import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from optuna.integration import XGBoostPruningCallback


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training.csv")

# Set MLflow to use a local folder instead of a Docker container
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns").replace("\\", "/")  # Ensure forward slashes for URI
MLFLOW_URI = f"file:///{MLRUNS_DIR}"


# === 1. OPTIMIZED FEATURE ENGINEERING ===
def _get_threshold(ser: pl.Series, factor: float = 1.5) -> float:
    """Calculates upper threshold based on IQR."""
    clean = ser.drop_nulls().drop_nans()
    if clean.len() == 0:
        return 0.0
    q1 = clean.quantile(0.25, interpolation="linear")
    q3 = clean.quantile(0.75, interpolation="linear")
    if q1 is None or q3 is None:
        return 0.0
    iqr = q3 - q1
    return float(q3 + factor * iqr)


def create_features(df: pl.DataFrame) -> pl.DataFrame:
    log.info("Starting optimized Feature Engineering with Polars...")

    # --- SENDER FEATURES ---
    df = df.sort(["nameOrig", "step"])

    hist_avg_expr = pl.col("amount").cum_sum().shift(1) / pl.int_range(0, pl.len()).cast(pl.Float64)

    df = df.with_columns(
        sender_time_since_last_trans=pl.col("step").diff().fill_null(0).cast(pl.Float64).over("nameOrig"),
        sender_total_trans_count=pl.int_range(0, pl.len()).cast(pl.Int64).over("nameOrig"),
        sender_avg_trans_amount_10_steps=pl.col("amount").rolling_mean(window_size=10, min_samples=1).over("nameOrig"),
        sender_std_trans_amount_10_steps=pl.col("amount").rolling_std(window_size=10, min_samples=1).fill_null(0.0).over("nameOrig"),
        _hist_avg=hist_avg_expr.over("nameOrig"),
        sender_fraction_of_balance=pl.col("amount") / (pl.col("oldbalanceOrg") + 1e-5),
        sender_emptied_account=pl.when(pl.col("newbalanceOrig") <= 0.01).then(1).otherwise(0),
    )

    df = df.with_columns(
        sender_std_trans_amount_10_steps_was_na=pl.when(pl.col("sender_std_trans_amount_10_steps") == 0.0).then(1).otherwise(0)
    )

    hist_avg_upper = _get_threshold(df["_hist_avg"])

    s_clean_ratio_expr = pl.when(
        pl.col("_hist_avg").is_null() | pl.col("_hist_avg").is_infinite() | pl.col("_hist_avg").is_nan() | (pl.col("_hist_avg") == 0)
    ).then(0.0).otherwise(pl.col("amount") / pl.col("_hist_avg"))

    df = df.with_columns(
        sender_total_avg_ratio_extreme_values=pl.when(
            pl.col("_hist_avg").is_null() | pl.col("_hist_avg").is_infinite() | pl.col("_hist_avg").is_nan() | (pl.col("_hist_avg") >= hist_avg_upper)
        ).then(1).otherwise(0),
        _s_clean_ratio=s_clean_ratio_expr,
    )

    ratio_upper = _get_threshold(df["_s_clean_ratio"])
    frac_upper = _get_threshold(df["sender_fraction_of_balance"])

    s_clean_frac_expr = pl.when(
        pl.col("sender_fraction_of_balance").is_null() | pl.col("sender_fraction_of_balance").is_infinite() | pl.col("sender_fraction_of_balance").is_nan()
    ).then(999.0).otherwise(pl.col("sender_fraction_of_balance")).clip(0.0, frac_upper)

    df = df.with_columns(
        sender_total_avg_ratio_log1p=pl.col("_s_clean_ratio").clip(0.0, ratio_upper).log1p(),
        sender_fraction_of_balance=s_clean_frac_expr.log1p(),
    ).drop(["_hist_avg", "_s_clean_ratio"])

    # --- RECEIVER FEATURES ---
    df = df.sort(["nameDest", "step"])
    rec_hist_avg_expr = pl.col("amount").cum_sum().shift(1) / pl.int_range(0, pl.len()).cast(pl.Float64)

    df = df.with_columns(
        receiver_trans_count_5_steps=pl.col("amount").is_not_null().cast(pl.Float64).rolling_sum(window_size=5, min_samples=1).over("nameDest"),
        receiver_total_amount_5_steps=pl.col("amount").rolling_sum(window_size=5, min_samples=1).over("nameDest"),
        _rec_hist_avg=rec_hist_avg_expr.over("nameDest"),
    )

    s_clean_rec_expr = pl.when(
        pl.col("_rec_hist_avg").is_null() | pl.col("_rec_hist_avg").is_infinite() | pl.col("_rec_hist_avg").is_nan()
    ).then(0.0).otherwise(pl.col("amount") / (pl.col("_rec_hist_avg") + 1e-5))

    df = df.with_columns(_s_clean_rec=s_clean_rec_expr)
    rec_ratio_upper = _get_threshold(df["_s_clean_rec"])

    df = df.with_columns(
        receiver_amount_to_avg_ratio_log1p=pl.col("_s_clean_rec").clip(0.0, rec_ratio_upper).log1p(),
        pair_transaction_count=pl.int_range(0, pl.len()).cast(pl.Int64).over(["nameOrig", "nameDest"]),
    ).drop(["_rec_hist_avg", "_s_clean_rec"])

    log.info("Feature Engineering completed.")
    return df


# === 2. MAIN TRAINING PIPELINE ===
def main():
    # Setup Local MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("AML_Fraud_Detection_XGBoost")

    if not os.path.exists(DATA_PATH):
        log.error(f"Data not found at: {DATA_PATH}")
        return

    # Load and process data
    log.info("Loading raw data...")
    df_raw = pl.read_csv(DATA_PATH)
    df = create_features(df_raw)

    # Cleanup unused columns
    drop_cols = ["step", "nameOrig", "nameDest", "timestamp", "isFlaggedFraud"]
    df = df.drop([c for c in drop_cols if c in df.columns])

    # Categorical handling: Encode 'type' as integer.
    df = df.with_columns(
        type=pl.Series(OrdinalEncoder().fit_transform(df.select("type").to_numpy()).ravel()).cast(pl.Int64)
    )

    X = df.drop("isFraud")
    y = df["isFraud"]

    # Datasets Splits: Train (70%), Val (15%), Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

    log.info(f"Data ready. Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # === OPTUNA HYPERPARAMETER SEARCH ===
    log.info("--- Starting Optuna Optimization ---")

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            # Important for imbalanced datasets:
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0)
        }

        # Using the Scikit-Learn API: xgb.XGBClassifier
        clf = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",  # Required for CUDA
            device="cuda",  # Run on GPU
            early_stopping_rounds=20,  # Stop if validation doesn't improve for 20 trees
            random_state=42
        )

        pruning_callback = XGBoostPruningCallback(trial, "validation_0-aucpr")

        # Fit the model
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[pruning_callback],
            verbose=False
        )

        preds = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)

    # Run the study
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective_xgb, n_trials=25)

    log.info(f"Best Trial PR-AUC: {study.best_value:.4f}")

    # === TRAIN FINAL MODEL & LOG TO MLFLOW ===
    log.info("--- Training Final Model ---")

    # Combine Train and Val for the final push
    X_train_full = pl.concat([X_train, X_val])
    y_train_full = pl.concat([y_train, y_val])

    best_params = study.best_params

    final_clf = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=42
    )

    final_clf.fit(X_train_full, y_train_full, verbose=False)

    with mlflow.start_run(run_name="Best_XGBoost_Model"):
        mlflow.log_params(best_params)
        mlflow.log_param("device", "cuda")

        test_preds = final_clf.predict_proba(X_test)[:, 1]
        test_pr_auc = average_precision_score(y_test, test_preds)
        test_roc_auc = roc_auc_score(y_test, test_preds)

        mlflow.log_metric("test_pr_auc", test_pr_auc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)

        mlflow.xgboost.log_model(
            xgb_model=final_clf,
            artifact_path="xgboost-model",
            registered_model_name="Production_Fraud_XGBoost"
        )

        log.info(f"Final Model Test PR-AUC: {test_pr_auc:.4f}")
        log.info(f"Final Model Test ROC-AUC: {test_roc_auc:.4f}")
        log.info(f"Success! Model and metrics saved to local directory: {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
