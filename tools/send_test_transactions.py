"""
AML Live Test Transaction Streaming Client
===========================================
Streams real transaction records from `data/test.csv` to the FastAPI microservice
endpoint (`/api/v1/transaction`).

Configurable to precise throughput rates (e.g. 100 transactions/second for 1000 total).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Default Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FASTAPI_URL = "http://localhost:8000/api/v1/transaction"
DEFAULT_CSV_PATH = str(BASE_DIR / "data" / "test.csv")
DEFAULT_MAX_RECORDS = 1000
DEFAULT_RATE = 100.0  # transactions per second


def send_transaction(
    row: dict[str, Any],
    session: requests.Session | None = None,
    url: str = DEFAULT_FASTAPI_URL,
    quiet: bool = False,
) -> bool:
    """
    Constructs a validated request payload from a CSV row and sends it to FastAPI.
    """
    http = session or requests
    try:
        payload = {
            "amount": float(row["amount"]),
            "nameOrig": row["nameOrig"],
            "nameDest": row["nameDest"],
            "type": row["type"],
            "oldbalanceOrg": float(row.get("oldbalanceOrg", 0.0)),
            "newbalanceOrig": float(row.get("newbalanceOrig", 0.0)),
            "oldbalanceDest": float(row.get("oldbalanceDest", 0.0)),
            "newbalanceDest": float(row.get("newbalanceDest", 0.0)),
            "timestamp": datetime.now().isoformat(),
        }

        # Inspect ground truth label if present
        is_fraud = row.get("isFraud")
        fraud_label = ""
        if is_fraud is not None:
            fraud_label = (
                " [Ground Truth: FRAUD]"
                if str(is_fraud).strip() in ("1", "1.0", "True")
                else " [Ground Truth: CLEAN]"
            )

        resp = http.post(url, json=payload, timeout=5)
        if resp.status_code in (200, 202):
            if not quiet:
                data = (
                    resp.json()
                    if resp.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                event_id = data.get("event_id", "")
                id_str = f" | Event: {event_id[:8]}..." if event_id else ""
                print(
                    f"[OK] {row['nameOrig']} -> {row['nameDest']} ({payload['type']} {payload['amount']:.2f}){fraud_label}{id_str}"
                )
            return True
        else:
            print(f"[ERR {resp.status_code}] {resp.text}")
            return False
    except Exception as e:
        print(f"[EXC] Connection failed: {e}")
        return False


def stream_test_transactions(
    csv_path: str = DEFAULT_CSV_PATH,
    url: str = DEFAULT_FASTAPI_URL,
    max_records: int = DEFAULT_MAX_RECORDS,
    rate: float = DEFAULT_RATE,
    quiet: bool = False,
) -> int:
    """
    Reads rows from the CSV file and streams them at the target rate (transactions/second).
    Uses a drift-free token bucket scheduler and an HTTP connection pool.
    """
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Test dataset not found at '{csv_path}'!")
        print("Please run 'uv run python tools/split_data.py' first to generate 'data/test.csv'.\n")
        return 0

    print(f"Reading test data from: {csv_path}")
    print(f"Target API Endpoint:    {url}")
    print(f"Streaming {max_records} transactions at {rate:.0f} tx/sec...\n")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=100)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    sent_count = 0
    t_start = time.perf_counter()
    report_interval = max(10, int(rate))  # Report roughly once every second

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if sent_count >= max_records:
                break

            send_transaction(row, session=session, url=url, quiet=quiet)
            sent_count += 1

            # Report rate telemetry every interval
            if sent_count % report_interval == 0 or sent_count == max_records:
                elapsed = max(0.001, time.perf_counter() - t_start)
                actual_rate = sent_count / elapsed
                print(
                    f"--- Progress: {sent_count}/{max_records} sent (Elapsed: {elapsed:.2f}s | Throughput: {actual_rate:.1f} tx/s) ---"
                )

            # Precise drift-free pacing
            target_next = t_start + (sent_count / rate)
            sleep_time = target_next - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    total_elapsed = max(0.001, time.perf_counter() - t_start)
    final_rate = sent_count / total_elapsed
    print(
        f"\nComplete! Successfully dispatched {sent_count} transactions in {total_elapsed:.2f}s (Average: {final_rate:.1f} tx/s)."
    )
    return sent_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream real test transactions to the AML FastAPI service at a controlled rate."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to test CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_FASTAPI_URL,
        help=f"FastAPI transaction endpoint (default: {DEFAULT_FASTAPI_URL})",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help=f"Total transactions to send (default: {DEFAULT_MAX_RECORDS})",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help=f"Throughput rate in transactions/second (default: {DEFAULT_RATE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-transaction logs and show only summary progress",
    )
    args = parser.parse_args()

    stream_test_transactions(
        csv_path=args.csv_path,
        url=args.url,
        max_records=args.max_records,
        rate=args.rate,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
