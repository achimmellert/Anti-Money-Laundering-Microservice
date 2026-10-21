from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import xgboost as xgb
from loguru import logger as log
from mlflow.tracking import MlflowClient

# Ensure project root is in sys.path when invoked directly as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from aml.config.settings import settings


def resolve_mlflow_uri(uri: str | None = None) -> str:
    candidate = uri or settings.MLFLOW_TRACKING_URI
    if candidate.startswith("http://") or candidate.startswith("https://"):
        try:
            import urllib.request

            urllib.request.urlopen(candidate, timeout=0.6)
            return candidate
        except Exception:
            db_path = (settings.PROJECT_ROOT / "mlflow.db").as_posix()
            fallback = f"sqlite:///{db_path}"
            log.info(
                f"MLflow server at '{candidate}' unreachable. Using SQLite registry store: '{fallback}'"
            )
            return fallback
    return candidate


MLFLOW_URI = resolve_mlflow_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_URI)

BASE_DIR = str(settings.PROJECT_ROOT)
MODEL_PATH = str(settings.MODEL_PATH)


def register_model_to_mlflow(
    model_path: str = MODEL_PATH,
    model_name: str = "FraudDetectionModel",
    alias: str = "champion",
) -> str | None:
    """
    Registers a local XGBoost model to MLflow and assigns modern model aliases.

    Follows modern MLOps best practice:
    1. Registers model version.
    2. Demotes existing '@champion' to '@previous_champion' (for instant rollback).
    3. Sets new model version to '@champion'.
    4. Backwards compatibility: sets stage='Production' gracefully if supported.
    """
    if not os.path.exists(model_path):
        log.error(f"Model file not found at {model_path}.")
        return None

    booster = xgb.Booster()
    booster.load_model(model_path)
    log.info(f"Loaded local model from {model_path}.")

    experiment_name = "AML_Fraud_Detection"
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    # Track existing champion version for graceful demotion
    prev_champion_version: str | None = None
    try:
        model_meta = client.get_model_version_by_alias(model_name, alias)
        prev_champion_version = model_meta.version
        log.info(f"Current '@{alias}' is version {prev_champion_version}.")
    except Exception:
        log.info(f"No existing '@{alias}' found for model '{model_name}'.")

    with mlflow.start_run(run_name="Model_Import") as run:
        mlflow.xgboost.log_model(xgb_model=booster, artifact_path="xgboost-model")
        model_uri = f"runs:/{run.info.run_id}/xgboost-model"
        reg_result = mlflow.register_model(model_uri=model_uri, name=model_name)
        new_version = str(reg_result.version)

    # Reassign previous champion if existing
    if prev_champion_version and prev_champion_version != new_version:
        try:
            client.set_registered_model_alias(
                model_name, "previous_champion", prev_champion_version
            )
            log.info(f"Assigned '@previous_champion' to version {prev_champion_version}.")
        except Exception as e:
            log.warning(f"Could not assign '@previous_champion': {e}")

    # Set new model as champion
    client.set_registered_model_alias(model_name, alias, new_version)
    log.info(f"Model version {new_version} successfully assigned alias '@{alias}'!")

    # Backward compatibility with legacy systems still checking stage="Production"
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        log.debug(f"Stage transition skipped (modern MLflow prefers aliases): {e}")

    return new_version


def main():
    register_model_to_mlflow()


if __name__ == "__main__":
    main()
