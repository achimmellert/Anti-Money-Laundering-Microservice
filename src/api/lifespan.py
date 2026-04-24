# src/api/lifespan.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from src.services.database import init_db

log = logging.getLogger(__name__)


# src/api/lifespan.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FastAPI starting up...")

    # Let this crash the app if it fails, rather than suppressing it
    await run_in_threadpool(init_db)
    log.info("Datenbank-Tabellen initialisiert.")

    yield
    log.info("FastAPI shutting down...")
