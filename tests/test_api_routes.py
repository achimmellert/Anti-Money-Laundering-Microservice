import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from aml.api.main import app
from aml.services.database import get_db_session


@pytest.fixture
def client():
    """FastAPI test client with mocked DB initialization."""
    with patch("aml.api.lifespan.init_db"):
        with TestClient(app) as test_client:
            yield test_client


def test_root_endpoint(client):
    """TEST: Root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AML Fraud Detection API"}


def test_submit_transaction_validation_error(client):
    """TEST: Submitting invalid payload returns 422 Unprocessable Entity."""
    # Negative amount is forbidden
    payload = {
        "amount": -50.0,
        "type": "TRANSFER",
        "nameOrig": "C123",
        "nameDest": "C456"
    }
    response = client.post("/api/v1/transaction", json=payload)
    assert response.status_code == 422


def test_submit_transaction_success(client):
    """TEST: Valid transaction submission returns 202 Accepted and queues tasks."""
    payload = {
        "amount": 250.0,
        "type": "TRANSFER",
        "nameOrig": "C840083671",
        "nameDest": "C38997010"
    }

    with patch("aml.api.routes.transactions.run_persistence_task.delay") as mock_persist, \
         patch("aml.api.routes.transactions.chain") as mock_chain:

        mock_pipeline = MagicMock()
        mock_chain.return_value = mock_pipeline

        response = client.post("/api/v1/transaction", json=payload)

        assert response.status_code == 202
        data = response.json()
        assert "event_id" in data
        assert data["status"] == "received"
        assert data["message"] == "Transaction processing initiated."

        mock_persist.assert_called_once()
        mock_pipeline.apply_async.assert_called_once()


def test_get_unknown_transaction_status_returns_404(client):
    """TEST: Requesting status for non-existent event_id returns 404."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    app.dependency_overrides[get_db_session] = lambda: mock_db
    try:
        response = client.get("/api/v1/transaction-status/non-existent-uuid")
        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown event_id"
    finally:
        app.dependency_overrides.clear()
