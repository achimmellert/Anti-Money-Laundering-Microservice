import os
from pathlib import Path
from fastapi import FastAPI

from aml.api.lifespan import lifespan
from aml.api.routes.transactions import router as transaction_router


LOG_PATH = os.getenv("API_LOG_PATH", "logs/api_events.log")
Path(LOG_PATH).parent.mkdir(exist_ok=True, parents=True)

app = FastAPI(lifespan=lifespan)
app.include_router(transaction_router, prefix="/api/v1", tags=["transactions"])

@app.get("/")
def read_root():
    return {"message": "AML Fraud Detection API"}
