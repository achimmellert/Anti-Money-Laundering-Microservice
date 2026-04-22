# src/config/settings.py

from typing import List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import MySQLDsn, RedisDsn


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
env_path = project_root / ".env"

MODEL_PATH: str = "models/model.pkl"
FEATNAMES_PATH: str = "models/feature_names.npy"
PREPROC_PATH: str = "models/preprocessor.joblib"

START_STEP: str = "743"


class Settings(BaseSettings):
    REDIS_URL: RedisDsn
    DATABASE_URL: MySQLDsn
    MODEL_PATH: str
    FEATNAMES_PATH: str
    START_STEP: str

    RAW_FEATURE_COLUMNS: List[str] = [
        "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "sender_time_since_last_trans", "sender_trans_count_5_steps", "sender_total_trans_count",
        "sender_avg_trans_amount_10_steps", "sender_std_trans_amount_10_steps",
        "sender_std_trans_amount_10_steps_was_na",
        "sender_total_avg_ratio_extreme_values", "sender_total_avg_ratio_clipped", "sender_total_avg_ratio_log1p",
        "sender_total_amount_history", "sender_fraction_of_balance", "sender_fraction_of_balance_sent_is_extreme",
        "sender_fraction_of_balance_clipped", "sender_emptied_account",
        "receiver_trans_count_5_steps", "receiver_unique_senders_5_steps", "receiver_total_amount_5_steps",
        "receiver_amount_to_avg_ratio_extreme_values", "receiver_amount_to_avg_ratio_clipped",
        "receiver_amount_to_avg_ratio_log1p",
        "pair_transaction_count"
    ]

    NUMERIC_FEATURES: List[str] = [c for c in RAW_FEATURE_COLUMNS if c != "type"]


    # nur für Local Development
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
