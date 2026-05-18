import os
import numpy as np
import pandas as pd
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
def _get_threshold(ser: pd.Series, factor: float = 1.5):
    """Calculates upper threshold based on IQR."""
    Q1 = ser.quantile(0.25)
    Q3 = ser.quantile(0.75)
    IQR = Q3 - Q1
    return Q3 + factor * IQR


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting optimized Feature Engineering...")
    df = df.copy()

    # --- SENDER FEATURES ---
    df = df.sort_values(by=["nameOrig", "step"]).reset_index(drop=True)
    sender_groups = df.groupby("nameOrig")

    df["sender_time_since_last_trans"] = sender_groups["step"].diff().fillna(0)
    df["sender_total_trans_count"] = sender_groups.cumcount()

    rolling_amount_10 = sender_groups["amount"].rolling(window=10, min_periods=1)
    df["sender_avg_trans_amount_10_steps"] = rolling_amount_10.mean().reset_index(level=0, drop=True)
    df["sender_std_trans_amount_10_steps"] = rolling_amount_10.std().reset_index(level=0, drop=True).fillna(0.0)
    df["sender_std_trans_amount_10_steps_was_na"] = (df["sender_std_trans_amount_10_steps"] == 0.0).astype(int)

    df["sender_cum_sum"] = sender_groups["amount"].cumsum()
    historical_avg = (df["sender_cum_sum"] - df["amount"]) / df["sender_total_trans_count"].replace(0, np.nan)
    historical_avg_upper = _get_threshold(historical_avg)

    df["sender_total_avg_ratio_extreme_values"] = (
            (historical_avg == np.inf) | (historical_avg == -np.inf) |
            (historical_avg >= historical_avg_upper) | (historical_avg.isna())
    ).astype(int)

    s_clean_sender_ratio = (df["amount"] / historical_avg).replace([np.inf, -np.inf], np.nan).fillna(0)
    ratio_upper = _get_threshold(s_clean_sender_ratio)
    df["sender_total_avg_ratio_log1p"] = np.log1p(s_clean_sender_ratio.clip(lower=0, upper=ratio_upper))

    df["sender_fraction_of_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1e-5)
    frac_upper = _get_threshold(df["sender_fraction_of_balance"])

    s_clean_frac = df["sender_fraction_of_balance"].replace([np.inf, -np.inf], np.nan).fillna(999).clip(lower=0,
                                                                                                        upper=frac_upper)
    df["sender_fraction_of_balance"] = np.log1p(s_clean_frac)
    df["sender_emptied_account"] = (df["newbalanceOrig"] <= 0.01).astype(int)

    # --- RECEIVER FEATURES ---
    df = df.sort_values(by=["nameDest", "step"]).reset_index(drop=True)
    receiver_groups = df.groupby("nameDest")

    df["receiver_trans_count_5_steps"] = receiver_groups["amount"].rolling(window=5, min_periods=1).count().reset_index(
        level=0, drop=True)
    df["receiver_total_amount_5_steps"] = receiver_groups["amount"].rolling(window=5, min_periods=1).sum().reset_index(
        level=0, drop=True)

    df["receiver_cum_count"] = receiver_groups.cumcount()
    df["receiver_cum_sum"] = receiver_groups["amount"].cumsum()
    receiver_hist_avg = (df["receiver_cum_sum"] - df["amount"]) / df["receiver_cum_count"].replace(0, np.nan)

    s_clean_receiver = (df["amount"] / (receiver_hist_avg + 1e-5)).replace([np.inf, -np.inf], np.nan).fillna(0)
    rec_ratio_upper = _get_threshold(s_clean_receiver)
    df["receiver_amount_to_avg_ratio_log1p"] = np.log1p(s_clean_receiver.clip(lower=0, upper=rec_ratio_upper))
    df["pair_transaction_count"] = df.groupby(["nameOrig", "nameDest"]).cumcount()

    df = df.drop(columns=["sender_cum_sum", "receiver_cum_count", "receiver_cum_sum"])

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
    df_raw = pd.read_csv(DATA_PATH)
    df = create_features(df_raw)

    # Cleanup unused columns
    drop_cols = ["step", "nameOrig", "nameDest", "timestamp", "isFlaggedFraud"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Categorical handling: Encode 'type' as integer.
    df["type"] = OrdinalEncoder().fit_transform(df[["type"]]).astype(int)

    X = df.drop(columns=["isFraud"])
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
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

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
