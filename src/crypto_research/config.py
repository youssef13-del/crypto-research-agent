from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from platformdirs import user_data_path
from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_database_url() -> str:
    database_path = user_data_path("ChainScope", appauthor=False) / "chainscope.db"
    return f"sqlite+pysqlite:///{database_path.as_posix()}"


class LLMProvider(StrEnum):
    GROQ = "groq"
    DISABLED = "disabled"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: LLMProvider = LLMProvider.DISABLED
    groq_api_key: SecretStr | None = None
    groq_model: str = Field(default="openai/gpt-oss-120b", min_length=1, max_length=200)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_seconds: float = Field(default=30, gt=0)
    groq_max_retries: int = Field(default=0, ge=0, le=0)
    groq_max_tokens: int = Field(default=2500, ge=100)
    groq_min_request_interval_seconds: float = Field(default=2.1, ge=0, le=10)
    groq_rate_limit_max_wait_seconds: float = Field(default=10, ge=0, le=30)

    coingecko_api_key: SecretStr | None = None
    defillama_base_url: str = "https://api.llama.fi"
    coinmetrics_base_url: str = "https://community-api.coinmetrics.io/v4"
    binance_futures_base_url: str = "https://fapi.binance.com"

    database_url: str = Field(default_factory=_default_database_url, min_length=1)
    research_retention_days: int = Field(default=365, ge=30, le=3650)

    minimum_training_samples: int = Field(default=200, ge=50)
    minimum_validation_samples: int = Field(default=50, ge=20)
    time_series_folds: int = Field(default=5, ge=2, le=10)
    minimum_mae_improvement: float = Field(default=0.02, ge=0)
    minimum_directional_accuracy: float = Field(default=0.52, ge=0, le=1)
    maximum_absolute_forecast_return: float = Field(default=0.20, gt=0, le=1)
    maximum_interval_width: float = Field(default=0.40, gt=0)

    @classmethod
    def load_application(cls) -> Settings:
        """Load process settings and the project ``.env`` from any launch directory.

        The packaged Streamlit entry point can be started from a shortcut or from the
        virtual-environment directory. Pydantic's relative ``env_file`` lookup would then
        miss the project's credentials and silently select the safe disabled provider.
        Environment variables still take precedence over the discovered dotenv file.
        """

        env_file = _discover_env_file()
        return cls(_env_file=env_file if env_file is not None else None)

    @field_validator(
        "groq_api_key",
        "coingecko_api_key",
        mode="before",
    )
    @classmethod
    def empty_secret_is_unset(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("groq_model", mode="before")
    @classmethod
    def normalize_required_model(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            return _default_database_url()
        if not normalized.startswith("sqlite"):
            raise ValueError("DATABASE_URL must use SQLite.")
        return normalized

    @field_validator(
        "groq_base_url",
        "defillama_base_url",
        "coinmetrics_base_url",
        "binance_futures_base_url",
    )
    @classmethod
    def validate_provider_base_url(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        label = (info.field_name or "provider_base_url").upper()
        candidate = value.strip().rstrip("/")
        if any(character.isspace() or ord(character) < 32 for character in candidate) or (
            "\\" in candidate
        ):
            raise ValueError(f"{label} is not a valid HTTP(S) URL.")
        try:
            parsed = urlsplit(candidate)
            _ = parsed.port
        except ValueError as exc:
            raise ValueError(f"{label} is not a valid HTTP(S) URL.") from exc
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{label} is not a valid HTTP(S) URL.")
        return candidate

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None = None) -> Settings:
        """Load known settings from a Streamlit-style mapping without forwarding extra secrets."""

        known_fields = {name.upper(): name for name in cls.model_fields}
        overrides: dict[str, Any] = {
            known_fields[key.upper()]: value
            for key, value in (values or {}).items()
            if isinstance(key, str) and key.upper() in known_fields
        }
        return cls(**overrides)

    @model_validator(mode="after")
    def validate_provider(self) -> Settings:
        if self.groq_rate_limit_max_wait_seconds < self.groq_min_request_interval_seconds:
            raise ValueError(
                "GROQ_RATE_LIMIT_MAX_WAIT_SECONDS must be at least "
                "GROQ_MIN_REQUEST_INTERVAL_SECONDS."
            )
        if self.llm_provider is LLMProvider.GROQ:
            key = (
                self.groq_api_key.get_secret_value().strip()
                if self.groq_api_key is not None
                else ""
            )
            if not key:
                raise ValueError("Groq mode requires GROQ_API_KEY.")
            if not key.casefold().startswith("gsk_"):
                raise ValueError("GROQ_API_KEY must contain a Groq key beginning with gsk_.")
        return self


def _discover_env_file() -> Path | None:
    candidates = [Path.cwd() / ".env"]
    candidates.extend(parent / ".env" for parent in Path(__file__).resolve().parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
