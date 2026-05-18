import pytest
import pandera.pandas as pa
import pandas as pd
from src.config.settings import settings


# 1. Define the statistical rules for your AML data
aml_schema = pa.DataFrameSchema({
    "type": pa.Column(str, pa.Check.isin(["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"])),
    "amount": pa.Column(float, pa.Check.ge(0)),
    "oldbalanceOrg": pa.Column(float, pa.Check.ge(0)),
    "newbalanceOrig": pa.Column(float, pa.Check.ge(0)),
    "oldbalanceDest": pa.Column(float, pa.Check.ge(0)),
    "newbalanceDest": pa.Column(float, pa.Check.ge(0)),
    # Assuming other numeric columns default to float for the ML model
    **{col: pa.Column(float) for col in settings.NUMERIC_FEATURES if col not in [
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    ]}
})

@pytest.fixture
def mock_input_df() -> pd.DataFrame:
    """Generates 5 rows of perfectly valid synthetic AML data on the fly."""
    # pandera's example() creates synthetic data matching the rules above
    return aml_schema.example(size=5)

@pytest.fixture
def mock_invalid_df(mock_input_df) -> pd.DataFrame:
    """Generates data missing a critical column to test error handling."""
    return mock_input_df.drop(columns=["amount"])
