# src/services/database.py

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config.settings import settings


engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        # session is loaned to the fastapi router
        yield db
    finally:
        # session is returned to the pool when fastapi router has finished
        db.close()


def init_db() -> None:
    # Models importieren, damit sie bei Base registriert sind
    from src.models import db_models  # noqa: F401
    from src.models.db_models import Base
    Base.metadata.create_all(bind=engine)