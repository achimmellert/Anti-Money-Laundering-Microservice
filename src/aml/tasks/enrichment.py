import polars as pl
import pandera.polars as pa
from loguru import logger as log
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

from aml.celery_app import celery_app
from aml.services.database import SessionLocal
from aml.models.schemas import TransactionInternal
from aml.models.dataframes import HistoryDataFrameSchema, EnrichedFeaturesSchema
from aml.services.feature_engineering import create_features, infer_missing_balances
from aml.utils.db_helpers import get_account_history, save_to_dlq
from aml.config.settings import settings


HIST_SCHEMA = {
    "step": pl.Int64,
    "type": pl.String,
    "amount": pl.Float64,
    "nameOrig": pl.String,
    "oldbalanceOrg": pl.Float64,
    "newbalanceOrig": pl.Float64,
    "nameDest": pl.String,
    "oldbalanceDest": pl.Float64,
    "newbalanceDest": pl.Float64,
    "timestamp": pl.Datetime,
}


@celery_app.task(bind=True, max_retries=3)
def run_enrichment_task(self, data: dict, event_id: str):
    """
    Enriches raw transaction data with historical features.
    Uses a context manager for session handling and returns data for the Celery Chain.
    """
    log.info(f"Starting enrichment for Event: {event_id}")

    # 1. Pydantic Gateway: Validate incoming Celery payload instantly
    try:
        current_tx = TransactionInternal.model_validate(data)
    except Exception as e:
        log.critical(f"FATAL: Input validation failed for {event_id}. Moving to DLQ.")
        save_to_dlq("enchrichment_validation", event_id, data, str(e))
        return

    with SessionLocal() as session:
        try:
            # 2. High-Speed History Retrieval (List of msgspec Structs)
            rows = get_account_history(
                session,
                current_tx.nameOrig,
                current_tx.nameDest,
                limit=200
            ) or []

            # 3. Construct Historical DataFrame
            if rows:
                rows_data = [
                    {
                        "step": r.step, "type": r.type, "amount": float(r.amount),
                        "nameOrig": r.nameOrig, "oldbalanceOrg": float(r.oldbalanceOrg), "newbalanceOrig": float(r.newbalanceOrig),
                        "nameDest": r.nameDest, "oldbalanceDest": float(r.oldbalanceDest), "newbalanceDest": float(r.newbalanceDest),
                        "timestamp": r.timestamp
                    } for r in rows]

                hist_df = pl.DataFrame(rows_data, schema=HIST_SCHEMA)
                hist_df = (
                    hist_df.unique(maintain_order=True)
                    .sort("step", descending=True)
                )

            else:
                log.info(f"New User context: No history for {current_tx.nameOrig} or {current_tx.nameDest}.")
                # Generate empty DataFrame with exact required columns
                hist_df = pl.DataFrame(schema=HIST_SCHEMA)

            # 4. Construct Current Transaction DataFrame (Pydantic already guaranteed types)
            cur_dict = current_tx.model_dump(mode="python", exclude={"event_id"})
            cur_dict["type"] = cur_dict["type"].value if hasattr(cur_dict["type"], "value") else str(cur_dict["type"])
            cur_df = pl.DataFrame([cur_dict], schema=HIST_SCHEMA)

            # 5. Pandera Validation
            cur_df = HistoryDataFrameSchema.validate(cur_df)
            hist_df = HistoryDataFrameSchema.validate(hist_df)

            # 6. Feature Engineering Pipeline
            cur_df_imputed = infer_missing_balances(cur_df, hist_df)
            full_df = pl.concat([hist_df, cur_df_imputed])
            feat_df = create_features(full_df)

            # 7. Extract the newly enriched row for the Chain
            enriched_data = feat_df.to_dicts()[-1]

            # Serialize datetimes back to strings for JSON/Celery transmission
            if enriched_data.get("timestamp") is not None:
                if hasattr(enriched_data["timestamp"], "isoformat"):
                    enriched_data["timestamp"] = enriched_data["timestamp"].isoformat()
                else:
                    enriched_data["timestamp"] = str(enriched_data["timestamp"])

            enriched_data["event_id"] = event_id

            log.info(f"Enrichment successful for {event_id}. Flowing to Chain...")

            return {"data": enriched_data, "event_id": event_id}

        except pa.errors.SchemaError as e:
            log.critical(f"FATAL: Pandera Schema validation failed for {event_id}. Data corrupted.")
            save_to_dlq("enrichment_pandera_failure", event_id, data, str(e))
            return

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
