# src/api/main.py

import os
import logging
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.lifespan import lifespan
from src.api.routes.transactions import router as transaction_router

# Centralized Logging Configuration
LOG_PATH = os.getenv("API_LOG_PATH", "logs/api_events.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(lifespan=lifespan)
app.include_router(transaction_router, prefix="/api/v1", tags=["transactions"])

Instrumentator().instrument(app).expose(app)

@app.get("/")
def read_root():
    return {"message": "AML Fraud Detection API"}
