# src/api/lifespan.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from src.services.database import init_db

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FastAPI starting up...")

    try:
        # Prevents synchronous blocking of the async event loop
        await run_in_threadpool(init_db)
        log.info("Datenbank-Tabellen initialisiert.")
    except Exception as e:
        log.error(f"FEHLER: Konnte DB nicht initialisieren: {e}")

    yield

    log.info("FastAPI shutting down...")
