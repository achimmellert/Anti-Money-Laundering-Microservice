import os
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("DATABASE_URL", "mysql+pymysql://root:root@localhost:3306/aml")

import pytest
import pandera.polars as pa
import polars as pl
from aml.models.dataframes import NUMERIC_FEATURES


# 1. Define the statistical rules for your AML data in Polars
aml_schema = pa.DataFrameSchema({
    "type": pa.Column(str, pa.Check.isin(["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"])),
    "amount": pa.Column(float, pa.Check.ge(0)),
    "oldbalanceOrg": pa.Column(float, pa.Check.ge(0)),
    "newbalanceOrig": pa.Column(float, pa.Check.ge(0)),
    "oldbalanceDest": pa.Column(float, pa.Check.ge(0)),
    "newbalanceDest": pa.Column(float, pa.Check.ge(0)),
    **{col: pa.Column(float) for col in NUMERIC_FEATURES if col not in [
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    ]}
})


@pytest.fixture
def mock_input_df() -> pl.DataFrame:
    """Generates 5 rows of valid synthetic AML data in Polars on the fly."""
    types = ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]
    rows = []
    for i in range(5):
        row = {
            "type": types[i % len(types)],
            "amount": float(100.0 * (i + 1)),
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 400.0,
            "oldbalanceDest": 50.0,
            "newbalanceDest": 150.0,
        }
        for col in NUMERIC_FEATURES:
            if col not in row:
                row[col] = 1.0
        rows.append(row)
    df = pl.DataFrame(rows)
    return aml_schema.validate(df)


@pytest.fixture
def mock_invalid_df(mock_input_df) -> pl.DataFrame:
    """Generates data missing a critical column to test error handling."""
    return mock_input_df.drop("amount")
