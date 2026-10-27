import uuid
from datetime import UTC, datetime

import aml.tasks.enrichment as enrichment_mod
import aml.tasks.inference as inference_mod
import aml.tasks.persistence as persistence_mod
from tests.test_e2e_pipeline import CeleryBatchRequest, RealPipelineMLModel


def _create_tx_payload(
    step: int, name_orig: str = "C_USER_RACE_1", name_dest: str = "M_MERCHANT_RACE_1"
):
    event_id = str(uuid.uuid4())
    return event_id, {
        "amount": 250.0,
        "type": "TRANSFER",
        "nameOrig": name_orig,
        "nameDest": name_dest,
        "step": step,
        "timestamp": datetime.now(UTC).isoformat(),
        "event_id": event_id,
    }


def test_persistence_before_enrichment_does_not_contaminate_history(e2e_env):
    """
    REGRESSION TEST (Bug A):
    When persistence finishes before enrichment, the transaction must NOT be counted
    as its own previous history (point-in-time correctness).
    """
    event_id, payload = _create_tx_payload(step=1000, name_orig="C_USER_RACE_A")

    # Persistence runs first
    persistence_mod.run_persistence_task.apply(args=[payload, event_id])

    # Enrichment runs second
    result = enrichment_mod.run_enrichment_task.apply(args=[payload, event_id]).result
    assert result is not None
    data = result["data"]

    # Since this is the user's very first transaction, history counts must be 0
    assert data["sender_total_trans_count"] == 0
    assert data["pair_transaction_count"] == 0
    assert data["sender_total_amount_history"] == 0.0


def test_later_transaction_does_not_leak_into_earlier_enrichment(e2e_env):
    """
    REGRESSION TEST (Bug B):
    If an out-of-order transaction with higher step is persisted first,
    an earlier transaction being enriched must not return the later transaction's row.
    """
    user_id = "C_USER_RACE_B"
    eid_early, payload_early = _create_tx_payload(step=2000, name_orig=user_id)
    eid_late, payload_late = _create_tx_payload(step=2001, name_orig=user_id)

    # Later transaction persisted first
    persistence_mod.run_persistence_task.apply(args=[payload_late, eid_late])

    # Earlier transaction enriched
    result_early = enrichment_mod.run_enrichment_task.apply(args=[payload_early, eid_early]).result
    assert result_early is not None
    assert result_early["event_id"] == eid_early
    assert result_early["data"]["step"] == 2000


def test_none_chain_input_does_not_crash_batch_inference(e2e_env, monkeypatch):
    """
    REGRESSION TEST (Bug C):
    When an upstream enrichment task fails and sends to DLQ, it returns None.
    A None input in the Celery Batch must be skipped without crashing the rest of the batch.
    """
    fake_model = RealPipelineMLModel(forced_proba=0.10)
    monkeypatch.setattr(inference_mod, "model_service", fake_model)

    _, valid_payload = _create_tx_payload(step=3000, name_orig="C_USER_RACE_C")
    valid_enrichment = enrichment_mod.run_enrichment_task.apply(
        args=[valid_payload, valid_payload["event_id"]]
    ).result

    # Batch contains one failed (None) request and one valid request
    batch_requests = [CeleryBatchRequest(None), CeleryBatchRequest(valid_enrichment)]

    # Must process without raising AttributeError: 'NoneType' object has no attribute 'get'
    inference_mod.run_inference_task.run(batch_requests)
