from pathlib import Path

import pytest
from pydantic import ValidationError

import crypto_research.config as settings_module
from crypto_research.config import LLMProvider, Settings


def test_settings_default_to_safe_provider_configuration() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider is LLMProvider.DISABLED
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.groq_base_url == "https://api.groq.com/openai/v1"
    assert settings.groq_timeout_seconds == 30
    assert settings.groq_min_request_interval_seconds == 2.1
    assert settings.groq_rate_limit_max_wait_seconds == 10
    assert settings.groq_max_retries == 0
    assert settings.groq_max_tokens == 2500
    assert settings.database_url.startswith("sqlite+pysqlite:///")
    assert settings.research_retention_days == 365
    assert settings.binance_futures_base_url == "https://fapi.binance.com"


@pytest.mark.parametrize(
    "database_url",
    ("mysql://localhost/chainscope", "postgresql+psycopg://localhost/chainscope"),
)
def test_settings_reject_unsupported_database_schemes(database_url: str) -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(database_url=database_url, _env_file=None)


def test_settings_require_a_groq_key_only_in_groq_mode() -> None:
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        Settings(llm_provider=LLMProvider.GROQ, groq_api_key="", _env_file=None)

    settings = Settings(llm_provider=LLMProvider.DISABLED, _env_file=None)

    assert settings.groq_api_key is None


@pytest.mark.parametrize("key", ["not-a-groq-key", "sk_other-provider-key"])
def test_settings_reject_a_non_groq_key_in_groq_mode(key: str) -> None:
    with pytest.raises(ValidationError, match="beginning with gsk_"):
        Settings(
            llm_provider=LLMProvider.GROQ,
            groq_api_key=key,
            _env_file=None,
        )


def test_settings_normalize_the_groq_model() -> None:
    settings = Settings(
        llm_provider=LLMProvider.DISABLED,
        groq_model=" openai/gpt-oss-120b ",
        _env_file=None,
    )

    assert settings.groq_model == "openai/gpt-oss-120b"

    with pytest.raises(ValidationError, match="groq_model"):
        Settings(
            llm_provider=LLMProvider.DISABLED,
            groq_model=" ",
            _env_file=None,
        )


def test_settings_trim_provider_keys() -> None:
    settings = Settings(
        llm_provider=LLMProvider.DISABLED,
        groq_api_key="  gsk_test-key  ",
        _env_file=None,
    )

    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "gsk_test-key"


def test_settings_from_mapping_accepts_only_known_streamlit_values() -> None:
    settings = Settings.from_mapping(
        {
            "LLM_PROVIDER": "disabled",
            "COINGECKO_API_KEY": "coin-key",
            "DEFILLAMA_BASE_URL": "https://defi.example.test/",
            "BINANCE_FUTURES_BASE_URL": "https://fapi.example.test/",
            "TIME_SERIES_FOLDS": 4,
            "UNRELATED_SECRET": "must-not-be-forwarded",
        }
    )

    assert settings.llm_provider is LLMProvider.DISABLED
    assert settings.coingecko_api_key is not None
    assert settings.coingecko_api_key.get_secret_value() == "coin-key"
    assert settings.defillama_base_url == "https://defi.example.test"
    assert settings.binance_futures_base_url == "https://fapi.example.test"
    assert settings.time_series_folds == 4
    assert "unrelated_secret" not in type(settings).model_fields


def test_settings_reject_invalid_binance_futures_url() -> None:
    with pytest.raises(ValidationError, match="BINANCE_FUTURES_BASE_URL"):
        Settings(binance_futures_base_url="file:///tmp/binance", _env_file=None)


@pytest.mark.parametrize(
    "value",
    ["https://example.com/a path", "https://example.com\\unexpected"],
)
def test_provider_base_urls_reject_ambiguous_path_characters(value: str) -> None:
    with pytest.raises(ValidationError, match=r"valid HTTP\(S\) URL"):
        Settings(
            llm_provider=LLMProvider.DISABLED,
            defillama_base_url=value,
            _env_file=None,
        )


def test_removed_legacy_environment_fields_are_ignored() -> None:
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        groq_fallback_model="retired-model",
        embeddings_base_url="https://embeddings.example.test",
        news_semantic_rerank=True,
        auth_provider="retired-provider",
        groq_router_model="retired-router",
        _env_file=None,
    )

    assert settings.llm_provider is LLMProvider.DISABLED
    assert not {
        "redis_url",
        "groq_fallback_model",
        "embeddings_base_url",
        "news_semantic_rerank",
        "auth_provider",
        "groq_router_model",
    }.intersection(type(settings).model_fields)


def test_dotenv_loading_remains_available_through_pydantic_settings(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=disabled\nMINIMUM_TRAINING_SAMPLES=250\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.llm_provider is LLMProvider.DISABLED
    assert settings.minimum_training_samples == 250
    assert "default_forecast_horizon" not in type(settings).model_fields


def test_application_settings_use_the_discovered_dotenv_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=disabled\nMINIMUM_TRAINING_SAMPLES=275\n", encoding="utf-8")
    monkeypatch.setattr(settings_module, "_discover_env_file", lambda: env_file)

    settings = Settings.load_application()

    assert settings.llm_provider is LLMProvider.DISABLED
    assert settings.minimum_training_samples == 275
