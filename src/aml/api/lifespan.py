from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from loguru import logger as log

from aml.services.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FastAPI starting up...")

    await run_in_threadpool(init_db)
    log.info("Database tables initialized.")

    yield
    log.info("FastAPI shutting down...")
