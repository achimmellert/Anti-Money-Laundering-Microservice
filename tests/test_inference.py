import pytest
import numpy as np
import polars as pl
from unittest.mock import patch, MagicMock

from aml.services.inference import FraudDetectionModel
from aml.config.settings import settings


@pytest.fixture
def mock_fraud_model():
    """
    Creates an instance of FraudDetectionModel but intercepts its __init__
    so it doesn't try to download models from MLflow during tests.
    """
    with patch.object(FraudDetectionModel, '_load_model') as mock_load:
        model_service = FraudDetectionModel()

        # 1. Mock the Preprocessor
        model_service.preproc = MagicMock()

        model_service.preproc.transform.return_value = np.zeros((5, len(settings.RAW_FEATURE_COLUMNS)))

        # 2. Mock the XGBoost Model
        model_service.model = MagicMock()
        # Mock it to return a 90% fraud probability for all 5 synthetic rows
        model_service.model.predict.return_value = np.array([0.90] * 5)

        # 3. Set required properties
        model_service.feature_names = settings.RAW_FEATURE_COLUMNS
        model_service.threshold = 0.5

        yield model_service


def test_predict_batch_success(mock_fraud_model, mock_input_df):
    """TEST: Does the model process a valid Pandera DataFrame correctly?"""

    # Act: Run the synthetic data through our model service
    results = mock_fraud_model.predict_batch(mock_input_df)

    # Assert: Verify the output format and logic
    assert len(results) == 5, "Should return exactly 5 predictions"

    for prob, is_fraud, decision in results:
        assert isinstance(prob, float)
        assert prob == 0.90
        assert is_fraud == 1
        assert decision == "BLOCKED"

    # Verify preprocessor was actually called with the dataframe
    mock_fraud_model.preproc.transform.assert_called_once_with(mock_input_df)


def test_prepare_dmatrix_column_mismatch_raises_error(mock_fraud_model, mock_input_df):
    """TEST: Does the system catch schema mismatches?"""

    # Simulate a scenario where the preprocessor output doesn't match the feature names
    mock_fraud_model.preproc.transform.return_value = np.zeros((5, 3))  # Only 3 columns returned

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        mock_fraud_model._prepare_dmatrix(mock_input_df)

    assert "Column mismatch" in str(exc_info.value)
