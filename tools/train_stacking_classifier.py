import os
import numpy as np
import polars as pl
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

from aml.services.model_wrappers import SklearnPyTorchWrapper


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training.csv")
MLFLOW_URI = "http://localhost:5000"


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
    mlflow.set_tracking_uri(MLFLOW_URI)

    if not os.path.exists(DATA_PATH):
        log.error(f"Data not found at: {DATA_PATH}")
        return

    # Load and process features
    log.info("Loading raw data...")
    df_raw = pl.read_csv(DATA_PATH)
    df = create_features(df_raw)

    drop_cols = ["step", "nameOrig", "nameDest", "timestamp", "isFlaggedFraud"]
    df = df.drop([c for c in drop_cols if c in df.columns])

    df = df.with_columns(
        type=pl.Series(OrdinalEncoder().fit_transform(df.select("type").to_numpy()).ravel()).cast(pl.Int64)
    )

    X = df.drop("isFraud")
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
