import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from crypto_research.agents.market.market_analyzer import MarketLiveOutput
from crypto_research.config import LLMProvider, Settings
from crypto_research.domain.research import AgentAnswer
from crypto_research.llm.client import (
    DisabledLLMAdapter,
    LLMResponseError,
    LLMRole,
    ResilientLLMAdapter,
)
from crypto_research.llm.groq import GroqAdapter


class _StrictStructuredClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0
        self.schemas: list[object] = []
        self.options: dict[str, object] = {}
        self.kwargs: list[dict[str, object]] = []

    def model_copy(self, *, update: dict[str, object]) -> _StrictStructuredClient:
        self.options = update
        return self

    def with_structured_output(self, schema: object, **kwargs: object) -> _StrictStructuredClient:
        self.schemas.append(schema)
        self.kwargs.append(dict(kwargs))
        return self

    def invoke(self, _messages: object) -> dict[str, object]:
        self.calls += 1
        return {
            "raw": SimpleNamespace(usage_metadata={}, response_metadata={"headers": {}}),
            "parsed": self.payload,
            "parsing_error": None,
        }


def _groq_adapter(client: _StrictStructuredClient) -> GroqAdapter:
    return GroqAdapter(
        Settings(
            llm_provider=LLMProvider.DISABLED,
            groq_model="openai/gpt-oss-120b",
            _env_file=None,
        ),
        client=cast(Any, client),
    )


def test_market_specialist_uses_one_strict_closed_schema_request() -> None:
    client = _StrictStructuredClient(
        {
            "verdict": "Market posture is mixed.",
            "assets": [
                {
                    "symbol": "BTC/USD",
                    "market_analysis": "Momentum is constructive.",
                    "risk_analysis": "Observed risk is evidence-bound.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )

    result = _groq_adapter(client).generate_structured(
        role=LLMRole.MARKET,
        system_prompt="Return the strict market object.",
        user_prompt='{"assets":["BTC/USD"]}',
        output_schema=MarketLiveOutput,
    )

    assert result.assets[0].symbol == "BTC/USD"
    assert client.calls == 1
    assert client.kwargs == [{"method": "json_schema", "strict": True, "include_raw": True}]
    provider_schema = cast(dict[str, Any], client.schemas[0])
    parameters = cast(dict[str, Any], provider_schema["parameters"])
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(parameters["properties"])
    assert client.options["max_tokens"] == 600


def test_invalid_specialist_shape_is_not_retried_in_loose_mode() -> None:
    client = _StrictStructuredClient({"verdict": "Missing required fields."})

    with pytest.raises(LLMResponseError, match="strict specialist output"):
        _groq_adapter(client).generate_structured(
            role=LLMRole.MARKET,
            system_prompt="Return the strict market object.",
            user_prompt='{"assets":["BTC/USD"]}',
            output_schema=MarketLiveOutput,
        )

    assert client.calls == 1


def test_disabled_provider_returns_evidence_only_agent_fallback() -> None:
    result = DisabledLLMAdapter().generate_structured(
        role=LLMRole.MARKET,
        system_prompt="system",
        user_prompt='{"agent":"market_agent","question":"Analyze BTC"}',
        output_schema=AgentAnswer,
    )

    assert result.agent == "market_agent"
    assert result.status == "unavailable"


def test_resilient_adapter_falls_back_for_controlled_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = ResilientLLMAdapter(cast(Any, _FailureAdapter(LLMResponseError("secret"))))

    with caplog.at_level(logging.WARNING):
        result = adapter.generate_structured(
            role=LLMRole.MARKET,
            system_prompt="secret-system",
            user_prompt='{"agent":"market_agent","question":"secret-question"}',
            output_schema=AgentAnswer,
        )

    assert result.agent == "market_agent"
    assert adapter.last_call_used_fallback is True
    assert "secret-system" not in caplog.text
    assert "secret-question" not in caplog.text


def test_resilient_adapter_can_disable_deterministic_fallback() -> None:
    adapter = ResilientLLMAdapter(
        cast(Any, _FailureAdapter(LLMResponseError("provider failed"))),
        deterministic_fallback=False,
    )

    with pytest.raises(LLMResponseError):
        adapter.generate_structured(
            role=LLMRole.FORECAST,
            system_prompt="system",
            user_prompt='{"agent":"forecast_agent","question":"Forecast BTC"}',
            output_schema=AgentAnswer,
        )

    assert adapter.last_call_used_fallback is False


def test_resilient_adapter_propagates_programming_errors() -> None:
    adapter = ResilientLLMAdapter(cast(Any, _FailureAdapter(RuntimeError("boom"))))

    with pytest.raises(RuntimeError, match="boom"):
        adapter.generate_structured(
            role=LLMRole.RESEARCH,
            system_prompt="system",
            user_prompt='{"agent":"news_agent","question":"News"}',
            output_schema=AgentAnswer,
        )


class _FailureAdapter:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate_structured(self, **_kwargs: object) -> object:
        raise self.error
