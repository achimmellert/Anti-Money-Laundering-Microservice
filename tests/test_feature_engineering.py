from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from aml.models.dataframes import HistoryDataFrameSchema
from aml.services.feature_engineering import create_features, infer_missing_balances


@pytest.fixture
def sample_history_df() -> pl.DataFrame:
    """Fixture providing valid historical transactions for feature testing in Polars."""
    return pl.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": "ACC_A",
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 400.0,
                "nameDest": "ACC_B",
                "oldbalanceDest": 50.0,
                "newbalanceDest": 150.0,
                "timestamp": datetime(2026, 4, 12, 9, 0, tzinfo=UTC),
            },
            {
                "step": 2,
                "type": "TRANSFER",
                "amount": 200.0,
                "nameOrig": "ACC_A",
                "oldbalanceOrg": 400.0,
                "newbalanceOrig": 200.0,
                "nameDest": "ACC_B",
                "oldbalanceDest": 150.0,
                "newbalanceDest": 350.0,
                "timestamp": datetime(2026, 4, 12, 10, 0, tzinfo=UTC),
            },
            {
                "step": 3,
                "type": "CASH_OUT",
                "amount": 50.0,
                "nameOrig": "ACC_C",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 950.0,
                "nameDest": "ACC_B",
                "oldbalanceDest": 350.0,
                "newbalanceDest": 400.0,
                "timestamp": datetime(2026, 4, 12, 11, 0, tzinfo=UTC),
            },
        ]
    )


@pytest.fixture
def sample_current_df() -> pl.DataFrame:
    """Fixture providing a valid incoming transaction in Polars."""
    return pl.DataFrame(
        [
            {
                "step": 4,
                "type": "TRANSFER",
                "amount": 150.0,
                "nameOrig": "ACC_A",
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "nameDest": "ACC_B",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "timestamp": datetime(2026, 4, 12, 12, 0, tzinfo=UTC),
            }
        ]
    )


def test_history_dataframe_schema_validation(sample_history_df):
    """TEST: HistoryDataFrameSchema accepts valid timestamps and types without crashing."""
    validated = HistoryDataFrameSchema.validate(sample_history_df)
    assert len(validated) == 3
    assert "timestamp" in validated.columns


def test_infer_missing_balances(sample_history_df, sample_current_df):
    """TEST: infer_missing_balances reconstructs 0.0 balances using latest history."""
    # sample_history_df is sorted descending by step as enrichment_task expects
    hist_sorted = sample_history_df.sort("step", descending=True)
    imputed = infer_missing_balances(sample_current_df, hist_sorted)

    # ACC_A had 200.0 after step 2, so oldbalanceOrg should be 200.0, newbalanceOrig 200 - 150 = 50.0
    assert imputed["oldbalanceOrg"][0] == 200.0
    assert imputed["newbalanceOrig"][0] == 50.0

    # ACC_B had 400.0 after step 3, so oldbalanceDest should be 400.0, newbalanceDest 400 + 150 = 550.0
    assert imputed["oldbalanceDest"][0] == 400.0
    assert imputed["newbalanceDest"][0] == 550.0


def test_create_features_end_to_end(sample_history_df, sample_current_df):
    """TEST: create_features runs without errors and produces valid EnrichedFeaturesSchema."""
    full_df = pl.concat([sample_history_df, sample_current_df])
    full_df = HistoryDataFrameSchema.validate(full_df)

    feat_df = create_features(full_df)

    # Validate against contract schema
    assert isinstance(feat_df, pl.DataFrame)
    assert len(feat_df) == 4

    # The current transaction should be at the last row
    last_row = feat_df.to_dicts()[-1]
    assert last_row["nameOrig"] == "ACC_A"
    assert last_row["nameDest"] == "ACC_B"
    assert last_row["step"] == 4

    # Engineered features for ACC_A must reflect historical transactions
    assert last_row["sender_total_trans_count"] == 2  # ACC_A had 2 previous transactions
    assert last_row["pair_transaction_count"] == 2  # Pair ACC_A -> ACC_B appeared twice before
    assert last_row["sender_avg_trans_amount_10_steps"] is not None
    assert not np.isnan(last_row["sender_avg_trans_amount_10_steps"])
    assert not np.isinf(last_row["sender_total_avg_ratio_log1p"])
