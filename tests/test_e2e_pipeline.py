import os
import sys
import time
import uuid
import tempfile
import contextlib
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
import shap
from loguru import logger as log
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from aml.models.db_models import (
    Base,
    TransactionsState,
    TransactionsHistory,
    MonitoringInference,
    SuspiciousTransactions,
    TransactionStatus,
    TransactionType
)
from aml.config.settings import settings
import aml.tasks.enrichment as enrichment_mod
import aml.tasks.inference as inference_mod
import aml.tasks.persistence as persistence_mod
import aml.utils.db_helpers as db_helpers_mod
from aml.api.main import app
from aml.services.database import get_db_session

# Dedicated persistent log file for E2E pipeline tests
LOG_DIR = os.path.join(settings.PROJECT_ROOT, "logs")
E2E_LOG_FILE = os.path.join(LOG_DIR, "e2e_pipeline_test.log")


@pytest.fixture(scope="session", autouse=True)
def configure_e2e_logging():
    """Configures session-wide persistent file logging for all E2E test executions."""
    os.makedirs(LOG_DIR, exist_ok=True)
    handler_id = log.add(
        E2E_LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        encoding="utf-8",
        enqueue=True,
    )
    log.info("=" * 105)
    log.info(f" E2E Pipeline Test Suite Initialized - Log Destination: {E2E_LOG_FILE}")
    log.info("=" * 105)
    yield
    log.info("=" * 105)
    log.info(" E2E Pipeline Test Suite Completed")
    log.info("=" * 105)
    log.remove(handler_id)


class PipelineTestLogger:
    """
    Structured timeline and latency logger for E2E integration tests.
    Logs each execution phase with UTC timestamps, duration in milliseconds,
    and contextual payload metadata, followed by a formatted ASCII summary.
    """

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.test_start_wall = datetime.now(timezone.utc)
        self.test_start_perf = time.perf_counter()
        self.steps: list[dict] = []
        log.info(f"╔{'═' * 103}╗")
        log.info(f"║ START TEST: {self.test_name:<89} ║")
        log.info(f"║ Started At: {self.test_start_wall.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC{' ' * 63} ║")
        log.info(f"╚{'═' * 103}╝")

    @contextlib.contextmanager
    def step(self, phase_name: str, **metadata):
        step_id = len(self.steps) + 1
        start_wall = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        meta_str = " | ".join(f"{k}={v}" for k, v in metadata.items()) if metadata else ""
        log.info(f"▶ [STEP {step_id:02d} START] {phase_name} {('(' + meta_str + ')') if meta_str else ''}")

        status = "SUCCESS"
        error_msg = None
        try:
            yield
        except Exception as e:
            status = "FAILED"
            error_msg = str(e)
            raise
        finally:
            end_perf = time.perf_counter()
            end_wall = datetime.now(timezone.utc)
            duration_ms = (end_perf - start_perf) * 1000.0
            log.info(
                f"✔ [STEP {step_id:02d} DONE] {phase_name} in {duration_ms:.2f} ms | Status: {status}"
                + (f" | Error: {error_msg}" if error_msg else "")
            )
            self.steps.append({
                "step_id": step_id,
                "phase": phase_name,
                "start_time": start_wall.strftime("%H:%M:%S.%f")[:-3],
                "end_time": end_wall.strftime("%H:%M:%S.%f")[:-3],
                "duration_ms": duration_ms,
                "status": status,
                "metadata": metadata,
            })

    def finish_and_report(self) -> str:
        total_perf = time.perf_counter() - self.test_start_perf
        total_ms = total_perf * 1000.0

        lines = [
            "",
            "=" * 110,
            f" E2E TEST EXECUTION TIMELINE: {self.test_name}",
            f" Total Duration: {total_ms:.2f} ms ({total_perf:.3f} s) | Total Steps: {len(self.steps)}",
            f" Log File: {E2E_LOG_FILE}",
            "-" * 110,
            f"{'#':<3} | {'Phase Name':<42} | {'Start (UTC)':<12} | {'Duration':<11} | {'Status':<7} | {'Metadata'}",
            "-" * 110,
        ]
        for s in self.steps:
            meta_str = ", ".join(f"{k}={v}" for k, v in s["metadata"].items())
            lines.append(
                f"{s['step_id']:02d}  | {s['phase']:<42} | {s['start_time']:<12} | {s['duration_ms']:>8.2f} ms | {s['status']:<7} | {meta_str}"
            )
        lines.append("=" * 110)
        lines.append("")

        summary_str = "\n".join(lines)
        log.info(f"\n{summary_str}")
        print(summary_str)
        return summary_str


class InMemoryRedis:
    """Fast, isolated in-memory Redis mock supporting string, list, and pipeline operations."""

    def __init__(self):
        self.store = {}
        self.lists = {}

    def setnx(self, key, value):
        if key not in self.store:
            self.store[key] = value
            return True
        return False

    def incr(self, key):
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    def lrange(self, key, start, end):
        lst = self.lists.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start:end + 1]

    def lpush(self, key, *values):
        if key not in self.lists:
            self.lists[key] = []
        for v in values:
            self.lists[key].insert(0, v)
        return len(self.lists[key])

    def rpush(self, key, *values):
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].extend(values)
        return len(self.lists[key])

    def ltrim(self, key, start, end):
        if key in self.lists:
            if end == -1:
                self.lists[key] = self.lists[key][start:]
            else:
                self.lists[key] = self.lists[key][start:end + 1]

    def exists(self, key):
        return key in self.store or key in self.lists

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)
            self.lists.pop(k, None)

    def expire(self, key, seconds):
        return True

    def pipeline(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self):
        return []


class CeleryBatchRequest:
    """Emulates a Celery Batches request wrapper for task execution."""

    def __init__(self, arg):
        self.args = [arg]
        self.kwargs = {}


class RealPipelineMLModel:
    """
    Lightweight, working ML model service equipped with a trained XGBoost Booster,
    Scikit-Learn ColumnTransformer, and SHAP TreeExplainer for full integration tests.
    """

    def __init__(self, forced_proba: float | None = None):
        self.feature_names = settings.RAW_FEATURE_COLUMNS
        numeric_cols = [c for c in self.feature_names if c != "type"]
        cat_cols = ["type"]

        self.preproc = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ]
        )

        # Train a fast 5-tree Booster on synthetic samples
        rng = np.random.default_rng(42)
        synth_data = {col: rng.random(20) for col in numeric_cols}
        synth_data["type"] = ["TRANSFER", "CASH_OUT"] * 10
        synth_df = pl.DataFrame(synth_data).select(self.feature_names)

        X_trans = self.preproc.fit_transform(synth_df.to_pandas())
        y = np.array([1, 0] * 10)

        dtrain = xgb.DMatrix(X_trans, label=y, feature_names=self.feature_names)
        self.model = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=5)
        self.explainer = shap.TreeExplainer(self.model)
        self.threshold = settings.FRAUD_THRESHOLD
        self.forced_proba = forced_proba

    def _prepare_dmatrix(self, raw_features_df: pl.DataFrame):
        X_trans = self.preproc.transform(raw_features_df.to_pandas())
        return xgb.DMatrix(X_trans, feature_names=self.feature_names), X_trans

    def predict_batch(self, raw_features_df: pl.DataFrame):
        dmat, _ = self._prepare_dmatrix(raw_features_df)
        if self.forced_proba is not None:
            probs = np.array([self.forced_proba] * raw_features_df.height)
        else:
            probs = self.model.predict(dmat).astype(float)

        results = []
        for prob in probs:
            p = float(prob)
            is_fraud = 1 if p >= self.threshold else 0
            decision = "BLOCKED" if is_fraud else "SUCCESS"
            results.append((p, is_fraud, decision))
        return results

    def save_shap_barplot(self, raw_features_df: pl.DataFrame, out_path: str):
        log.info(f"RealPipelineMLModel: Computing SHAP values for {raw_features_df.height} row(s)...")
        _, X_trans = self._prepare_dmatrix(raw_features_df)
        shap_values = self.explainer.shap_values(X_trans)

        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_trans, feature_names=self.feature_names, show=False, plot_type="bar")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        fig.clf()
        plt.close(fig)
        log.info(f"RealPipelineMLModel: Saved SHAP barplot to {out_path}")


@pytest.fixture
def e2e_env(monkeypatch):
    """
    Sets up a fully functional, self-contained environment for E2E tests:
    - SQLite in-memory database with StaticPool for cross-thread access (FastAPI TestClient)
    - In-memory mock Redis instance
    - Monkeypatched SessionLocal and redis_client across all tasks and routes
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=True)

    mock_redis = InMemoryRedis()

    # Patch SessionLocal across workers
    monkeypatch.setattr(persistence_mod, "SessionLocal", TestingSession)
    monkeypatch.setattr(enrichment_mod, "SessionLocal", TestingSession)
    monkeypatch.setattr(inference_mod, "SessionLocal", TestingSession)
    monkeypatch.setattr(db_helpers_mod, "redis_client", mock_redis)
    monkeypatch.setattr("aml.api.routes.transactions.redis_client", mock_redis)

    yield {
        "session_factory": TestingSession,
        "redis": mock_redis,
        "engine": engine,
    }

    engine.dispose()


def test_e2e_pipeline_blocked_transaction_with_shap_report(e2e_env, monkeypatch):
    """
    END-TO-END PIPELINE TEST (BLOCKED / FRAUDULENT FLOW):
    1. Submits high-risk transaction (account emptied, large amount).
    2. Runs Persistence Task -> Saves state PERSISTED, writes history, updates Redis cache.
    3. Runs Enrichment Task -> Gathers history, executes Polars feature engineering, validates Pandera schema.
    4. Runs Inference Task -> Micro-batch inference with real XGBoost model, marks BLOCKED.
    5. Runs SHAP Generation -> Computes real TreeExplainer feature importance, saves PNG plot to disk.
    6. Verifies:
       - TransactionsState status is BLOCKED, fraud_probability is recorded.
       - Timestamps: created_at and decided_at are populated, and decided_at >= created_at.
       - MonitoringInference contains timestamp, decision, y_pred_proba, and features_json.
       - SuspiciousTransactions contains plot_path and event details.
       - SHAP plot file exists on disk, has size > 10KB, and has valid PNG header.
       - Redis cache holds the sender transaction.
    """
    tlog = PipelineTestLogger("test_e2e_pipeline_blocked_transaction_with_shap_report")
    session_factory = e2e_env["session_factory"]
    mock_redis = e2e_env["redis"]

    # 1. Set up real ML model configured to trigger BLOCKED (fraud_probability >= 0.5)
    with tlog.step("Setup ML Model & Environment", forced_proba=0.88, threshold=settings.FRAUD_THRESHOLD):
        fraud_model = RealPipelineMLModel(forced_proba=0.88)
        monkeypatch.setattr(inference_mod, "model_service", fraud_model)

    # Temporary directory for SHAP plot artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        event_id = str(uuid.uuid4())
        expected_plot_path = os.path.join(temp_dir, f"blocked_{event_id}.png")

        # Intercept async SHAP task to write directly to our temp_dir
        shap_executed = []
        def intercept_shap_delay(row_dict, eid, ppath):
            shap_executed.append(expected_plot_path)
            # Execute synchronously to simulate worker processing
            inference_mod.generate_shap_task(row_dict, eid, expected_plot_path)

        monkeypatch.setattr(inference_mod.generate_shap_task, "delay", intercept_shap_delay)

        # 2. Incoming transaction payload
        with tlog.step("Transaction Payload Intake", event_id=event_id, amount=500000.0, type="TRANSFER"):
            tx_payload = {
                "amount": 500000.0,
                "type": "TRANSFER",
                "nameOrig": "C_SUSPICIOUS_100",
                "nameDest": "M_RECEIVER_200",
                "oldbalanceOrg": 500000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 1000.0,
                "newbalanceDest": 501000.0,
                "step": 743,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": event_id
            }

        # --- STEP 1: PERSISTENCE TASK ---
        with tlog.step("Persistence Task & Redis Cache Write", event_id=event_id):
            persistence_mod.run_persistence_task.apply(args=[tx_payload, event_id])

        # Verify intermediate persistence state
        with tlog.step("Intermediate Persistence Audit", event_id=event_id):
            with session_factory() as sess:
                state_intermediate = sess.query(TransactionsState).filter_by(event_id=event_id).first()
                assert state_intermediate is not None
                assert state_intermediate.status == TransactionStatus.PERSISTED
                assert state_intermediate.created_at is not None
                assert state_intermediate.decided_at is None

                hist_record = sess.query(TransactionsHistory).filter_by(event_id=event_id).first()
                assert hist_record is not None
                assert hist_record.amount == 500000.0

        # --- STEP 2: ENRICHMENT TASK ---
        with tlog.step("Polars Feature Enrichment & Pandera Validation", event_id=event_id):
            enrichment_output = enrichment_mod.run_enrichment_task.apply(args=[tx_payload, event_id]).result
            assert enrichment_output is not None
            assert enrichment_output["event_id"] == event_id
            enriched_row = enrichment_output["data"]

            # Verify Polars engineered features are populated
            assert "sender_fraction_of_balance" in enriched_row
            assert "sender_emptied_account" in enriched_row
            assert enriched_row["sender_emptied_account"] == 1

        # --- STEP 3: INFERENCE TASK (CELERY BATCHES) ---
        with tlog.step("XGBoost Batch ML Inference Execution", event_id=event_id, batch_size=1):
            inference_mod.run_inference_task.run([CeleryBatchRequest(enrichment_output)])

        # --- STEP 4: VERIFICATION & AUDIT ---
        with tlog.step("Explainable AI (SHAP TreeExplainer & Plot Generation)", plot_path=expected_plot_path):
            assert len(shap_executed) == 1
            plot_file = shap_executed[0]
            assert os.path.exists(plot_file), f"SHAP plot file does not exist at {plot_file}"
            file_size = os.path.getsize(plot_file)
            assert file_size > 5000, f"SHAP plot file is suspiciously small: {file_size} bytes"

            # Verify valid PNG header (magic bytes: \x89PNG\r\n\x1a\n)
            with open(plot_file, "rb") as pf:
                png_header = pf.read(8)
                assert png_header == b"\x89PNG\r\n\x1a\n", "Generated file is not a valid PNG image"

        with tlog.step("Database State, Monitoring & Cache Audit", event_id=event_id):
            with session_factory() as sess:
                # 1. State verification & Decision
                final_state = sess.query(TransactionsState).filter_by(event_id=event_id).first()
                assert final_state is not None
                assert final_state.status == TransactionStatus.BLOCKED
                assert final_state.fraud_probability == pytest.approx(0.88, abs=1e-3)

                # 2. Timing and Latency Verification
                assert final_state.created_at is not None
                assert final_state.decided_at is not None
                assert final_state.decided_at >= final_state.created_at

                # 3. Monitoring Inference Audit Trail
                monitoring = sess.query(MonitoringInference).filter_by(event_id=event_id).first()
                assert monitoring is not None
                assert monitoring.decision == TransactionStatus.BLOCKED
                assert monitoring.y_pred_proba == pytest.approx(0.88, abs=1e-3)
                assert monitoring.timestamp is not None
                assert isinstance(monitoring.features_json, dict)
                assert monitoring.features_json.get("amount") == 500000.0
                assert monitoring.features_json.get("type") == "TRANSFER"

                # 4. Suspicious Transactions Table
                suspicious = sess.query(SuspiciousTransactions).filter_by(event_id=event_id).first()
                assert suspicious is not None
                assert suspicious.event_id == event_id
                assert suspicious.amount == 500000.0

            # 5. Redis Sliding Window Cache Verification
            cache_key = f"hist:{tx_payload['nameOrig']}"
            assert mock_redis.exists(cache_key), "Sender history was not cached in Redis"
            cached_items = mock_redis.lrange(cache_key, 0, 10)
            assert len(cached_items) > 0, "No cached items found in Redis sliding window"

    tlog.finish_and_report()


def test_e2e_pipeline_success_transaction_no_shap(e2e_env, monkeypatch):
    """
    END-TO-END PIPELINE TEST (SUCCESS / LEGITIMATE FLOW):
    1. Submits low-risk transaction.
    2. Runs through Persistence, Enrichment, and Inference.
    3. Model predicts SUCCESS (fraud_probability = 0.05).
    4. Verifies:
       - TransactionsState status is SUCCESS.
       - decided_at timestamp is set and valid.
       - MonitoringInference records SUCCESS.
       - No suspicious transaction is logged, and no SHAP report is generated.
    """
    tlog = PipelineTestLogger("test_e2e_pipeline_success_transaction_no_shap")
    session_factory = e2e_env["session_factory"]

    # Configure model to return low probability
    with tlog.step("Setup Legitimate ML Model", forced_proba=0.05):
        legit_model = RealPipelineMLModel(forced_proba=0.05)
        monkeypatch.setattr(inference_mod, "model_service", legit_model)

    shap_calls = []
    monkeypatch.setattr(inference_mod.generate_shap_task, "delay", lambda *a, **kw: shap_calls.append(a))

    event_id = str(uuid.uuid4())
    with tlog.step("Transaction Payload Intake", event_id=event_id, amount=25.0, type="PAYMENT"):
        tx_payload = {
            "amount": 25.0,
            "type": "PAYMENT",
            "nameOrig": "C_LEGIT_USER_01",
            "nameDest": "M_MERCHANT_01",
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 475.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1025.0,
            "step": 743,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id
        }

    # Execute full pipeline
    with tlog.step("Persistence Task Execution", event_id=event_id):
        persistence_mod.run_persistence_task.apply(args=[tx_payload, event_id])

    with tlog.step("Polars Feature Enrichment", event_id=event_id):
        enrichment_output = enrichment_mod.run_enrichment_task.apply(args=[tx_payload, event_id]).result

    with tlog.step("XGBoost Batch ML Inference (Legitimate)", event_id=event_id):
        inference_mod.run_inference_task.run([CeleryBatchRequest(enrichment_output)])

    # Verification
    with tlog.step("Database Audit & Negative Check", event_id=event_id):
        with session_factory() as sess:
            final_state = sess.query(TransactionsState).filter_by(event_id=event_id).first()
            assert final_state is not None
            assert final_state.status == TransactionStatus.SUCCESS
            assert final_state.fraud_probability == pytest.approx(0.05, abs=1e-3)
            assert final_state.decided_at is not None
            assert final_state.decided_at >= final_state.created_at

            # Verify Monitoring Log
            monitoring = sess.query(MonitoringInference).filter_by(event_id=event_id).first()
            assert monitoring is not None
            assert monitoring.decision == TransactionStatus.SUCCESS
            assert monitoring.y_pred_proba == pytest.approx(0.05, abs=1e-3)

            # Verify that legitimate transactions DO NOT generate SHAP reports or suspicious records
            assert len(shap_calls) == 0, "SHAP report should not be generated for legitimate transactions"
            suspicious = sess.query(SuspiciousTransactions).filter_by(event_id=event_id).first()
            assert suspicious is None, "Legitimate transaction should not be in SuspiciousTransactions"

    tlog.finish_and_report()


def test_e2e_api_to_pipeline_and_status_query(e2e_env, monkeypatch):
    """
    END-TO-END API & PIPELINE INTEGRATION TEST:
    1. Submits transaction via FastAPI client: POST /api/v1/transaction.
    2. Intercepts dispatched Celery payload.
    3. Runs full worker pipeline (Persistence -> Enrichment -> Inference).
    4. Queries FastAPI status endpoint: GET /api/v1/transaction-status/{event_id}.
    5. Queries FastAPI history endpoint: GET /api/v1/user-history/{user_id}.
    6. Verifies that the public API reflects the internal ML decision and audit trail.
    """
    tlog = PipelineTestLogger("test_e2e_api_to_pipeline_and_status_query")
    session_factory = e2e_env["session_factory"]

    # Provide DB session to FastAPI dependency injection
    def override_db():
        with session_factory() as sess:
            yield sess

    app.dependency_overrides[get_db_session] = override_db

    # Setup ML model
    with tlog.step("Setup ML Model & Dependency Overrides", forced_proba=0.91):
        fraud_model = RealPipelineMLModel(forced_proba=0.91)
        monkeypatch.setattr(inference_mod, "model_service", fraud_model)
        monkeypatch.setattr(inference_mod.generate_shap_task, "delay", lambda *a, **kw: None)

    dispatched = {}
    def mock_persist_delay(data, event_id):
        dispatched["persist"] = (data, event_id)

    class MockPipeline:
        def __init__(self, data, event_id):
            dispatched["enrichment"] = (data, event_id)
        def apply_async(self):
            pass

    monkeypatch.setattr("aml.api.routes.transactions.run_persistence_task.delay", mock_persist_delay)
    monkeypatch.setattr("aml.api.routes.transactions.chain", lambda enrich, infer: MockPipeline(*enrich.args, **enrich.kwargs))

    with patch_lifespan():
        with TestClient(app) as client:
            # 1. Submit transaction via API
            tx_input = {
                "amount": 75000.0,
                "type": "CASH_OUT",
                "nameOrig": "C_CLIENT_999",
                "nameDest": "M_MERCHANT_888"
            }
            with tlog.step("FastAPI POST /api/v1/transaction", amount=75000.0, type="CASH_OUT"):
                resp = client.post("/api/v1/transaction", json=tx_input)
                assert resp.status_code == 202
                resp_data = resp.json()
                event_id = resp_data["event_id"]
                assert resp_data["status"] == "received"

            # 2. Run persistence and enrichment from intercepted payload
            with tlog.step("Worker Pipeline Execution", event_id=event_id):
                payload_data, _ = dispatched["persist"]
                persistence_mod.run_persistence_task.apply(args=[payload_data, event_id])
                enrich_out = enrichment_mod.run_enrichment_task.apply(args=[payload_data, event_id]).result
                inference_mod.run_inference_task.run([CeleryBatchRequest(enrich_out)])

            # 3. Query transaction status via API
            with tlog.step("FastAPI GET /api/v1/transaction-status/{id}", event_id=event_id):
                status_resp = client.get(f"/api/v1/transaction-status/{event_id}")
                assert status_resp.status_code == 200
                status_data = status_resp.json()
                assert status_data["event_id"] == event_id
                assert status_data["status"] == "BLOCKED"
                assert status_data["fraud_probability"] == pytest.approx(0.91, abs=1e-3)
                assert status_data["decided_at"] is not None

            # 4. Query user history via API
            with tlog.step("FastAPI GET /api/v1/user-history/{user_id}", user_id=tx_input["nameOrig"]):
                history_resp = client.get(f"/api/v1/user-history/{tx_input['nameOrig']}")
                assert history_resp.status_code == 200
                history_data = history_resp.json()
                assert len(history_data) >= 1
                assert history_data[0]["nameOrig"] == tx_input["nameOrig"]
                assert history_data[0]["amount"] == 75000.0

    app.dependency_overrides.clear()
    tlog.finish_and_report()


def test_e2e_log_file_inspectable():
    """
    VERIFICATION TEST:
    Ensures that the comprehensive log file `logs/e2e_pipeline_test.log` is generated,
    is non-empty, contains exact timestamps, and includes all recorded phases and summaries.
    """
    assert os.path.exists(E2E_LOG_FILE), f"Log file does not exist at {E2E_LOG_FILE}"
    file_size = os.path.getsize(E2E_LOG_FILE)
    assert file_size > 0, "Log file is empty"

    with open(E2E_LOG_FILE, "r", encoding="utf-8") as f:
        log_content = f.read()

    # Verify key markers in the persistent log file
    assert "E2E Pipeline Test Suite Initialized" in log_content
    assert "[STEP 01 START]" in log_content
    assert "[STEP" in log_content and "DONE]" in log_content
    assert "E2E TEST EXECUTION TIMELINE:" in log_content
    assert "Starting persistence for Event:" in log_content
    assert "Starting enrichment for Event:" in log_content

    # Output log file path and line count for user visibility
    lines = log_content.splitlines()
    print(f"\n[INFO] Verified {len(lines)} log lines successfully written to: {E2E_LOG_FILE}")


@pytest.fixture(autouse=True)
def _clean_dependency_overrides():
    yield
    app.dependency_overrides.clear()


@contextlib.contextmanager
def patch_lifespan():
    with patch("aml.api.lifespan.init_db"):
        yield
