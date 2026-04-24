import os
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# 1. MLflow Tracking URI definieren (Da du das Skript lokal ausführst: localhost)
MLFLOW_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_URI)

# 2. Pfad zum lokalen Modell
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "model.json")

def main():
    """
    Lädt ein lokales Modell, registriert es in MLflow, setzt es auf 'Production'
    und archiviert automatisch alle älteren Versionen.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Fehler: Modell unter {MODEL_PATH} nicht gefunden.")
        return

    # Lokales XGBoost Modell laden
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    print("Lokales Modell geladen.")

    # 3. Modell in MLflow einloggen und registrieren
    experiment_name = "AML_Fraud_Detection"
    mlflow.set_experiment(experiment_name)

    model_name = "FraudDetectionModel"
    with mlflow.start_run(run_name="Initial_Import") as run:
        # 3.1 Modell-Dateien loggen (ohne direkte Registrierung)
        mlflow.xgboost.log_model(
            xgb_model=booster,
            artifact_path="xgboost-model"
        )
        
        # 3.2 Modell explizit in die Registry eintragen
        model_uri = f"runs:/{run.info.run_id}/xgboost-model"
        mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # 4. Die hochgeladene Version als "Production" markieren
    client = MlflowClient()
    latest_version_info = client.get_latest_versions(model_name, stages=["None"])[0]
    client.transition_model_version_stage(name=model_name, version=latest_version_info.version, stage="Production")
    print(f"✅ Modell-Version {latest_version_info.version} auf 'Production' gesetzt!")
    
    # 5. (BEST PRACTICE) Alle anderen Versionen, die nicht 'Production' sind, archivieren
    all_versions = client.search_model_versions(f"name='{model_name}'")
    for v in all_versions:
        if v.current_stage not in ["Production", "Archived"]:
            print(f"Archiviere alte Version {v.version} (Stage: {v.current_stage})...")
            client.transition_model_version_stage(name=model_name, version=v.version, stage="Archived")

    print("✅ Aufräumen abgeschlossen.")

if __name__ == "__main__":
    main()
