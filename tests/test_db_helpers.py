import pytest
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aml.models.db_models import Base, TransactionsState, TransactionStatus, TransactionType
from aml.models.schemas import TransactionInternal
from aml.utils.db_helpers import update_transaction_state, save_history_transaction


@pytest.fixture
def sqlite_session():
    """Provides an isolated SQLite in-memory database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_update_transaction_state_prevents_persisted_overwriting_blocked(sqlite_session):
    """
    CRITICAL TEST: Ensures that persistence-worker setting status=PERSISTED
    cannot overwrite an ML decision of BLOCKED or reset the fraud_probability to 0.0.
    """
    event_id = str(uuid.uuid4())
    payload = TransactionInternal(
        amount=500.0,
        type=TransactionType.TRANSFER,
        nameOrig="C111",
        nameDest="C222",
        event_id=event_id,
        step=743,
        timestamp=datetime.now(timezone.utc)
    )

    # 1. Inference worker flags transaction as BLOCKED with high fraud score
    update_transaction_state(sqlite_session, event_id, status=TransactionStatus.BLOCKED, prob=0.92, payload=payload)
    sqlite_session.commit()

    # Verify initial state
    row = sqlite_session.query(TransactionsState).filter_by(event_id=event_id).first()
    assert row is not None
    assert row.status == TransactionStatus.BLOCKED
    assert row.fraud_probability == 0.92

    # 2. Persistence worker attempts to set PERSISTED with prob=0.0
    update_transaction_state(sqlite_session, event_id, status=TransactionStatus.PERSISTED, prob=0.0, payload=payload)
    sqlite_session.commit()

    # 3. Assert that BLOCKED decision and 0.92 score were NOT overwritten!
    row_after = sqlite_session.query(TransactionsState).filter_by(event_id=event_id).first()
    assert row_after.status == TransactionStatus.BLOCKED, "PERSISTED must not overwrite BLOCKED!"
    assert row_after.fraud_probability == 0.92, "Fraud probability must not be reset to 0.0!"


def test_update_transaction_state_prevents_persisted_overwriting_success(sqlite_session):
    """TEST: Ensures that status=PERSISTED cannot overwrite a SUCCESS decision."""
    event_id = str(uuid.uuid4())
    payload = TransactionInternal(
        amount=50.0,
        type=TransactionType.PAYMENT,
        nameOrig="C333",
        nameDest="C444",
        event_id=event_id,
        step=743,
        timestamp=datetime.now(timezone.utc)
    )

    # 1. Inference worker marks transaction as SUCCESS
    update_transaction_state(sqlite_session, event_id, status=TransactionStatus.SUCCESS, prob=0.03, payload=payload)
    sqlite_session.commit()

    # 2. Persistence worker attempts to set PERSISTED
    update_transaction_state(sqlite_session, event_id, status=TransactionStatus.PERSISTED, prob=0.0, payload=payload)
    sqlite_session.commit()

    # 3. Assert SUCCESS is preserved
    row = sqlite_session.query(TransactionsState).filter_by(event_id=event_id).first()
    assert row.status == TransactionStatus.SUCCESS


def test_update_transaction_state_allows_persisted_initially(sqlite_session):
    """TEST: A new transaction can be initialized as PERSISTED if persistence runs first."""
    event_id = str(uuid.uuid4())
    payload = TransactionInternal(
        amount=100.0,
        type=TransactionType.TRANSFER,
        nameOrig="C555",
        nameDest="C666",
        event_id=event_id,
        step=743,
        timestamp=datetime.now(timezone.utc)
    )

    # Persistence runs first
    update_transaction_state(sqlite_session, event_id, status=TransactionStatus.PERSISTED, prob=0.0, payload=payload)
    sqlite_session.commit()

    row = sqlite_session.query(TransactionsState).filter_by(event_id=event_id).first()
    assert row.status == TransactionStatus.PERSISTED

    # Then inference worker updates it to BLOCKED
    update_transaction_state(sqlite_session, event_id, status=TransactionStatus.BLOCKED, prob=0.88, payload=payload)
    sqlite_session.commit()

    row_after = sqlite_session.query(TransactionsState).filter_by(event_id=event_id).first()
    assert row_after.status == TransactionStatus.BLOCKED
    assert row_after.fraud_probability == 0.88
