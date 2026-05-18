# Enterprise Anti-Money Laundering (AML) Engine

An asynchronous, high-throughput machine learning microservice designed to detect fraudulent financial transactions in real-time. 

Built with **FastAPI, Celery, XGBoost, and Redis**, this system is engineered for scale, featuring vectorized micro-batch inference, absolute data type safety, and real-time model explainability (SHAP).

---

## Tech Stack

* **API Layer:** FastAPI, Pydantic 2.0
* **Async Task Queue:** Celery (with `celery-batches`), Gevent (I/O bounds), threading (CPU bounds)
* **Caching & Broker:** Redis, `msgspec` (High-performance JSON serialization)
* **Database:** MySQL 8.0, SQLAlchemy 2.0 (Async-ready, strictly typed)
* **Data Processing & Validation:** Polars (Rust-based columnar DataFrame engine), Pandera (Polars schema contracts)
* **Machine Learning:** XGBoost, Scikit-Learn, MLflow (Model Registry), SHAP (Explainability)
* **Testing:** Pytest, Pandera
* **Deployment:** Docker, Docker Compose

---

## Key Engineering Patterns Implemented

This project goes beyond standard configurations by implementing robust, enterprise-ready architectural patterns:

### 1. Vectorized Micro-Batching
Instead of evaluating transactions one by one, the inference worker utilizes `celery-batches` to queue incoming requests for a fraction of a second, combining up to 50 transactions into a single Polars DataFrame. This allows XGBoost to process them in vectorized batches, drastically reducing CPU overhead and increasing throughput.

### 2. "Parse, Don't Validate" (Edge Protection)
Untyped dictionaries are caught at the very edge of the application (inside the Celery workers) and immediately parsed into strictly validated Pydantic models (`TransactionInternal`). This guarantees absolute type safety (including Enums, UUIDs, and datetimes) before data ever touches the core business logic or database helpers.

### 3. Zero-Latency Worker Initialization
Using Celery's `@worker_process_init` signals, the ML model (XGBoost) and the preprocessor are downloaded from MLflow and loaded directly into RAM *before* the worker starts accepting traffic. This eliminates the "cold start" latency spike on the first prediction. Connection pools (`engine.dispose()`) are also reset on fork to prevent multi-process socket pollution.

### 4. Dedicated Background Queues
Heavy, CPU-bound processes—like generating Matplotlib SHAP plots for flagged transactions—are routed to a dedicated `background` queue with a limited thread pool. This ensures that slow explainability tasks never block the real-time inference queue.

### 5. Distributed Race-Condition Prevention
Database updates utilize SQLAlchemy's `.with_for_update()` to enforce exclusive row-level locks. If multiple asynchronous workers attempt to process or update the same transaction simultaneously, the database safely sequences them, preventing dirty writes.

### 6. Centralized Configuration Truth
All infrastructure secrets, ML thresholds, and dynamic file paths are controlled via a strict Pydantic `BaseSettings` class. File paths are resolved via absolute `pathlib` traversal, immunizing the application against "Current Working Directory" bugs common in Docker containerization.

---

## System Services (Docker Compose)

The application is fully containerized and orchestrated via Docker Compose, spinning up the following isolated services:

1. **`fastapi-app`**: The public-facing REST API. Accepts transaction payloads, validates them, and pushes them to Redis queues.
2. **`enrichment-worker`**: Celery gevent pool that gathers historical data and calculates rolling features.
3. **`inference-worker`**: Celery thread pool that executes micro-batched XGBoost predictions.
4. **`background-worker`**: Dedicated Celery worker for heavy, non-blocking tasks (like SHAP visualizations).
5. **`persistence-worker`**: Celery worker responsible for writing clean data payloads to MySQL.
6. **`redis`**: In-memory data store acting as both the Celery message broker and the sliding-window cache for transaction histories.
7. **`mysql`**: Relational database storing transaction states, tracking metrics, and the Dead Letter Queue (DLQ).
8. **`mlflow`**: Local model registry serving the production-ready ML artifacts.

---

## Testing

The testing suite utilizes **Property-Based Testing**. Instead of relying on brittle, hardcoded `.csv` files, the test suite uses `pandera.example()` to dynamically generate fresh, statistically valid mock DataFrames on the fly. 

External services (like MLflow and Redis) are abstracted using Python's native `unittest.mock` to ensure unit tests run in milliseconds without requiring active infrastructure.

**Run the test suite locally:**
```bash
# Sync virtual environment from uv.lock
uv sync

# Run the test suite
uv run pytest
```

---

## Quick Start

**1. Clone the repository and configure environments**
Ensure your `.env` file is populated with the correct database credentials.

**2. Boot the infrastructure**
```bash
docker compose up -d
```

**3. Monitor the system**
You can view the real-time API documentation (Swagger) by navigating to:
* `http://localhost:8000/docs`

MLflow Model Registry UI is available at:
* `http://localhost:5000`