from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote_plus
from pydantic import MySQLDsn, RedisDsn, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Verhindert UserWarning auf Systemen ohne gemountetes /run/secrets Verzeichnis (reine pathlib-Prüfung)
SECRETS_DIR: Optional[Path] = Path("/run/secrets") if Path("/run/secrets").is_dir() else None


class Settings(BaseSettings):
    # --- ENVIRONMENT ---
    environment: Literal["local", "staging", "production"] = "local"
    PROJECT_ROOT: Path = PROJECT_ROOT

    # --- MYSQL CONFIGURATION ---
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "aml_user"
    mysql_password: SecretStr = SecretStr("aml_secure_password")
    mysql_db: str = "aml_db"
    # Optionale direkte Übergabe (z.B. aus Docker Compose oder Cloud-Umgebungen)
    DATABASE_URL: Optional[MySQLDsn] = None

    # --- REDIS CONFIGURATION ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[SecretStr] = None
    REDIS_URL: Optional[RedisDsn] = None

    # --- BUSINESS & ML CONFIGURATIONS ---
    START_STEP: int = 743
    FRAUD_THRESHOLD: float = 0.5
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # --- ABSOLUTE ARTIFACT FILE PATHS ---
    MODEL_PATH: Path = PROJECT_ROOT / "ml_models" / "model.json"
    FEATNAMES_PATH: Path = PROJECT_ROOT / "ml_models" / "feature_names.npy"
    PREPROC_PATH: Path = PROJECT_ROOT / "ml_models" / "preprocessor.pkl"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        secrets_dir=SECRETS_DIR,
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def assemble_connection_urls(self):
        """
        Dual-Mode: Verwendet DATABASE_URL / REDIS_URL falls direkt gesetzt,
        oder baut sie sicher aus den Einzelparametern zusammen (inkl. URL-Encoding).
        """
        if self.DATABASE_URL is None:
            encoded_password = quote_plus(self.mysql_password.get_secret_value())
            self.DATABASE_URL = MySQLDsn(
                f"mysql+pymysql://{self.mysql_user}:{encoded_password}"
                f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            )

        if self.REDIS_URL is None:
            if self.redis_password and self.redis_password.get_secret_value():
                encoded_password = quote_plus(self.redis_password.get_secret_value())
                self.REDIS_URL = RedisDsn(f"redis://:{encoded_password}@{self.redis_host}:{self.redis_port}/0")
            else:
                self.REDIS_URL = RedisDsn(f"redis://{self.redis_host}:{self.redis_port}/0")

        return self

    @computed_field
    @property
    def sqlalchemy_database_uri(self) -> str:
        """Alias-String für SQLAlchemy (kompatibel mit modernen Konventionen)."""
        return self.DATABASE_URL.unicode_string() if self.DATABASE_URL else ""

    @computed_field
    @property
    def redis_uri(self) -> str:
        """Alias-String für Redis (kompatibel mit modernen Konventionen)."""
        return self.REDIS_URL.unicode_string() if self.REDIS_URL else ""


@lru_cache
def get_settings() -> Settings:
    """Singleton-Pattern via Cache: Lädt und validiert die Config genau einmal beim Start."""
    return Settings()


# Singleton-Instanz (Fail-Fast beim Modul-Import):
settings = get_settings()
