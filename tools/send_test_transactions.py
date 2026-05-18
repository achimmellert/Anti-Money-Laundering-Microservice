import csv
import json
import time
import os
import requests
from datetime import datetime


# === Konfiguration ===
FASTAPI_URL = "http://localhost:8000/api/v1/transaction"

# Baut den absoluten Pfad dynamisch auf Basis des Skript-Standorts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "test.csv")
BATCH_SIZE = 10                     # optional: wie viele pro Pause
SLEEP_BETWEEN_BATCHES = 1.0         # Sekunden warten nach jedem Batch
MAX_RECORDS = 100                   # Anzahl der Datensätze, die gesendet werden sollen


# === Hilfsfunktion ===
def send_transaction(row):
    """
    Baut aus einer CSV-Zeile den Request-Body und sendet ihn an FastAPI.
    """
    try:
        payload = {
            "amount": float(row["amount"]),
            "nameOrig": row["nameOrig"],
            "nameDest": row["nameDest"],
            "type": row["type"],
            # Add these lines to pull the balances from the CSV!
            "oldbalanceOrg": float(row.get("oldbalanceOrg", 0.0)),
            "newbalanceOrig": float(row.get("newbalanceOrig", 0.0)),
            "oldbalanceDest": float(row.get("oldbalanceDest", 0.0)),
            "newbalanceDest": float(row.get("newbalanceDest", 0.0)),
            "timestamp": datetime.now().isoformat()
        }

        resp = requests.post(FASTAPI_URL, json=payload, timeout=5)
        if resp.status_code in (200, 202):
            print(f"[OK] {row['nameOrig']} → {row['nameDest']} ({payload['type']} {payload['amount']})")
        else:
            print(f"[ERR {resp.status_code}] {resp.text}")
    except Exception as e:
        print(f"[EXC] {e}")

# === Hauptlogik ===
def main():
    with open(CSV_PATH, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        batch = 0
        for row in reader:
            if count >= MAX_RECORDS:
                break

            send_transaction(row)
            count += 1

            # kleine Pause nach jedem Batch, um FastAPI & Redis nicht zu überlasten
            if count % BATCH_SIZE == 0:
                batch += 1
                print(f"--- {count} Transaktionen gesendet (Batch {batch}) ---")
                time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\nFertig! Insgesamt {count} Transaktionen gesendet.")

if __name__ == "__main__":
    main()
