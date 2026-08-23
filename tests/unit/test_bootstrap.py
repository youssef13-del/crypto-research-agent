from typing import Any

import pytest
from tests.support.fakes import FakeLLM

import crypto_research.bootstrap as bootstrap
from crypto_research.bootstrap import create_forecast_service, create_research_runtime
from crypto_research.config import LLMProvider, Settings
from crypto_research.forecasting.service import ForecastPolicy
from crypto_research.orchestration.runtime import ResearchRuntime


def test_bootstrap_composes_the_typed_research_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class CaptureRuntime:
        def __init__(self, **dependencies: object) -> None:
            captured.update(dependencies)

    monkeypatch.setattr(bootstrap, "ResearchRuntime", CaptureRuntime)
    runtime = create_research_runtime(
        Settings(llm_provider=LLMProvider.DISABLED, _env_file=None),
        owner_id="test-owner",
        llm=FakeLLM(),
    )

    assert isinstance(runtime, CaptureRuntime)
    assert set(captured) == {
        "market_agent",
        "news_agent",
        "fundamentals_agent",
        "onchain_agent",
        "forecast_agent",
        "specialist_analysis",
        "history",
    }


def test_bootstrap_returns_the_runtime_contract() -> None:
    runtime = create_research_runtime(
        Settings(llm_provider=LLMProvider.DISABLED, _env_file=None),
        owner_id="test-owner",
        llm=FakeLLM(),
    )

    assert isinstance(runtime, ResearchRuntime)


def test_bootstrap_composes_forecast_policy_from_settings() -> None:
    settings = Settings(
        llm_provider=LLMProvider.DISABLED,
        minimum_training_samples=220,
        minimum_validation_samples=55,
        time_series_folds=4,
        minimum_mae_improvement=0.03,
        minimum_directional_accuracy=0.6,
        maximum_absolute_forecast_return=0.15,
        maximum_interval_width=0.3,
        _env_file=None,
    )

    service = create_forecast_service(settings)

    assert service.policy == ForecastPolicy(
        minimum_training_samples=220,
        minimum_validation_samples=55,
        time_series_folds=4,
        minimum_mae_improvement=0.03,
        minimum_directional_accuracy=0.6,
        maximum_absolute_forecast_return=0.15,
        maximum_interval_width=0.3,
    )
