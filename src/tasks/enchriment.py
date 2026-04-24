import pandas as pd
from loguru import logger as log
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

from src.celery_app import celery_app
from src.services.database import SessionLocal
from src.services.feature_engineering import create_features, infer_missing_balances, REQUIRED_COLS
from src.utils.db_helpers import get_account_history, save_to_dlq
from src.config.settings import settings

@celery_app.task(bind=True, max_retries=3)
def run_enrichment_task(self, data: dict, event_id: str):
    """
    Enriches raw transaction data with historical features.
    Uses a context manager for session handling and returns data for the Celery Chain.
    """
    if not event_id:
        log.warning(f"Aborting: No event_id provided for payload: {data}")
        return None

    # 1. High-Speed History Retrieval
    # We use the context manager 'with' to ensure the session is always closed.
    with SessionLocal() as session:
        try:
            name_orig = data.get("nameOrig")
            name_dest = data.get("nameDest")

            # Fetch rows (Helper uses Redis-first logic)
            rows = get_account_history(session, name_orig, name_dest, limit=200) or []

            # Convert to list of dicts for Pandas efficiency
            rows_data = [{
                "step": r.step, "type": r.type, "amount": r.amount,
                "nameOrig": r.nameOrig, "oldbalanceOrg": r.oldbalanceOrg, "newbalanceOrig": r.newbalanceOrig,
                "nameDest": r.nameDest, "oldbalanceDest": r.oldbalanceDest, "newbalanceDest": r.newbalanceDest,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None
            } for r in rows]

            # 2. Historical DataFrame Construction
            if rows_data:
                hist_df = (
                    pd.DataFrame(rows_data)
                    .drop_duplicates(keep="last")
                    .sort_values("step", ascending=False)
                    .reset_index(drop=True)
                )
                # Ensure all features exist for the Engineering function
                for col in REQUIRED_COLS:
                    if col not in hist_df.columns:
                        dtype = "float64" if col not in ["type", "nameOrig", "nameDest", "timestamp"] else "object"
                        hist_df[col] = pd.Series(dtype=dtype)
            else:
                hist_df = pd.DataFrame(columns=REQUIRED_COLS)
                log.info(f"New User context: No history for {name_orig} or {name_dest}.")

            # 3. Process Current Transaction
            ts_raw = data.get("timestamp")
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else None
            except (ValueError, TypeError):
                ts = None

            cur = {
                "step": int(data.get("step", 0)),
                "type": data.get("type"),
                "amount": float(data.get("amount", 0.0)),
                "nameOrig": name_orig,
                "oldbalanceOrg": float(data.get("oldbalanceOrg", 0.0)),
                "newbalanceOrig": float(data.get("newbalanceOrig", 0.0)),
                "nameDest": name_dest,
                "oldbalanceDest": float(data.get("oldbalanceDest", 0.0)),
                "newbalanceDest": float(data.get("newbalanceDest", 0.0)),
                "timestamp": ts.isoformat() if ts else None
            }

            # 4. Feature Engineering Pipeline
            # Impute missing balances based on historical trends
            cur = infer_missing_balances(cur, hist_df)

            # Concatenate and generate rolling/aggregate features
            full_df = pd.concat([hist_df, pd.DataFrame([cur])], ignore_index=True)
            feat_df = create_features(full_df)

            # Extract the newly enriched row
            enriched_row = feat_df.iloc[[-1]].copy()
            enriched_data = enriched_row[settings.RAW_FEATURE_COLUMNS + ["step", "nameOrig", "nameDest", "timestamp"]].iloc[0].to_dict()
            enriched_data["event_id"] = event_id

            log.info(f"Enrichment successful for {event_id}. Flowing to Chain...")

            # RETURN for the next link in the Celery Chain (Inference)
            return {"data": enriched_data, "event_id": event_id}

        except SQLAlchemyError as e:
            log.error(f"Database Error during enrichment [{event_id}]: {e}")
            if self.request.retries >= self.max_retries:
                save_to_dlq("enrichment_db_failure", event_id, data, str(e))
            else:
                raise self.retry(exc=e, countdown=5)

        except Exception as e:
            log.error(f"Unexpected Engineering Error [{event_id}]: {e}")
            if self.request.retries >= self.max_retries:
                save_to_dlq("enrichment_logic_failure", event_id, data, str(e))
            else:
                raise self.retry(exc=e, countdown=5)