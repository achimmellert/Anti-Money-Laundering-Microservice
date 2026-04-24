import os
import numpy as np
import pandas as pd
import logging
import warnings
import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from optuna.integration import XGBoostPruningCallback

# --- Import your custom wrapper here ---
from src.services.model_wrappers import SklearnPyTorchWrapper

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training.csv")
MLFLOW_URI = "http://localhost:5000"


# === 1. OPTIMIZED FEATURE ENGINEERING ===
def _get_threshold(ser: pd.Series, factor: float = 1.5):
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
    mlflow.set_tracking_uri(MLFLOW_URI)

    if not os.path.exists(DATA_PATH):
        log.error(f"Data not found at: {DATA_PATH}")
        return

    # Load and process features
    df_raw = pd.read_csv(DATA_PATH)
    df = create_features(df_raw)

    drop_cols = ["step", "nameOrig", "nameDest", "timestamp", "isFlaggedFraud"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["type"] = OrdinalEncoder().fit_transform(df[["type"]]).astype(int)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    categorical_cols = ["type"]
    continuous_cols = [col for col in X.columns if col not in categorical_cols]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

    # === EXPERIMENT 1: XGBoost Optimization ===
    log.info("--- Starting Experiment 1: XGBoost Optimization ---")
    mlflow.set_experiment("AML_XGBoost_Standalone")

    def objective_xgb(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
            "device": "cuda",
            "tree_method": "hist",
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[XGBoostPruningCallback(trial, "validation_0-aucpr")], verbose=False)
        return average_precision_score(y_val, clf.predict_proba(X_val)[:, 1])

    pruner = optuna.pruners.HyperbandPruner()
    study_xgb = optuna.create_study(direction="maximize", pruner=pruner)
    study_xgb.optimize(objective_xgb, n_trials=20)
    best_xgb = xgb.XGBClassifier(**study_xgb.best_params, random_state=42, device="cuda", tree_method="hist")
    best_xgb.fit(X_train, y_train)

    # === EXPERIMENT 2: PyTorch Tabular Optimization ===
    log.info("--- Starting Experiment 2: PyTorch Tabular Optimization ---")

    def objective_tabular(trial):
        layers = trial.suggest_categorical("layers", ["32-16", "64-32", "128-64"])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Using the imported wrapper
        model = SklearnPyTorchWrapper(
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            layers=layers, dropout=dropout, learning_rate=lr, epochs=15, trial=trial
        )
        model.fit(X_train, y_train, X_val, y_val)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study_tabular = optuna.create_study(direction="maximize", pruner=pruner)
    study_tabular.optimize(objective_tabular, n_trials=15)

    best_tabular = SklearnPyTorchWrapper(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        **study_tabular.best_params,
        epochs=20
    )
    best_tabular.fit(X_train, y_train)

    # === EXPERIMENT 3: Stacking Ensemble ===
    log.info("--- Starting Experiment 3: Stacking Ensemble ---")
    mlflow.set_experiment("AML_Stacking_Ensemble")

    estimators = [
        ('xgb', best_xgb),
        ('pytorch_tabular', best_tabular)
    ]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=3,
        passthrough=False
    )

    stacking_clf.fit(X_train, y_train)

    with mlflow.start_run(run_name="Stacking_XGB_PyTorchTabular"):
        mlflow.log_params({f"xgb_{k}": v for k, v in study_xgb.best_params.items()})
        mlflow.log_params({f"tabular_{k}": v for k, v in study_tabular.best_params.items()})

        test_preds_stack = stacking_clf.predict_proba(X_test)[:, 1]
        test_pr_auc = average_precision_score(y_test, test_preds_stack)

        mlflow.log_metric("test_pr_auc", test_pr_auc)
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, test_preds_stack))

        mlflow.sklearn.log_model(stacking_clf, "stacking-model", registered_model_name="Modern_FraudDetection_Stacking")
        log.info(f"Stacking Ensemble Test PR-AUC: {test_pr_auc:.4f}")
        log.info("Finished: Models successfully trained and logged to MLflow.")


if __name__ == "__main__":
    main()