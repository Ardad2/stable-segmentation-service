"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class LogLevel(str, Enum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"


class Backend(str, Enum):
    mock = "mock"
    sam2 = "sam2"
    custom = "custom"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ---- service ----
    app_env: AppEnv = AppEnv.development
    log_level: LogLevel = LogLevel.info
    api_prefix: str = "/api"
    service_name: str = "stable-segmentation-service"

    # ---- backend ----
    segmentation_backend: Backend = Backend.mock
    model_device: str = "cpu"

    # ---- SAM-2 (ignored when backend != sam2) ----
    sam2_checkpoint: str = ""
    sam2_config: str = ""

    @field_validator("model_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"model_device must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
