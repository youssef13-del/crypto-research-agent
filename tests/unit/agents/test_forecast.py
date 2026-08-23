import pytest
from tests.support.forecasting import market_evidence, permissive_policy, synthetic_candles

from crypto_research.agents.forecast.forecast_agent import ForecastAgent
from crypto_research.domain.forecast import ForecastRequest, ForecastSettings
from crypto_research.domain.research import AnalysisAsset, AnalysisRequest
from crypto_research.forecasting.service import ForecastPolicy, _run_forecast
from crypto_research.llm.client import LLMCallTelemetry, LLMResponseError, LLMRole


class _Service:
    def run(self, request: ForecastRequest):  # type: ignore[no-untyped-def]
        return _run_forecast(
            market=market_evidence(synthetic_candles()),
            request=request,
            policy=permissive_policy(),
        )


class _StrictService:
    def run(self, request: ForecastRequest):  # type: ignore[no-untyped-def]
        return _run_forecast(
            market=market_evidence(synthetic_candles()),
            request=request,
            policy=ForecastPolicy(
                minimum_training_samples=100,
                minimum_validation_samples=20,
                time_series_folds=3,
                minimum_mae_improvement=100.0,
                minimum_directional_accuracy=1.0,
                maximum_interval_width=0.01,
            ),
        )


class _LLM:
    def __init__(self, payload: object | None = None) -> None:
        self.calls = 0
        self.payload = (
            payload
            if payload is not None
            else {
                "summary": "The validated batch shows a bounded directional estimate.",
                "limitations": ["Model relationships can change after the training window."],
                "confidence": 0.7,
            }
        )
        self.last_call_telemetry = (
            LLMCallTelemetry(
                model="test-model",
                status_category="success",
                request_id="forecast-request",
                usage_tokens=120,
                estimated_tokens=180,
            ),
        )

    def generate_structured(self, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Forecast summaries must not use provider-side strict schemas.")

    def generate_specialist_json(self, *, role, system_prompt, user_prompt):  # type: ignore[no-untyped-def]
        self.calls += 1
        assert role is LLMRole.FORECAST
        assert '"forecasts"' in user_prompt
        assert "headline" not in user_prompt.casefold()
        assert "market & risk" not in user_prompt.casefold()
        assert "Return JSON only" in system_prompt
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_forecast_llm_interpretation_cannot_change_model_output() -> None:
    llm = _LLM()
    agent = ForecastAgent(service=_Service(), llm=llm)
    request = AnalysisRequest(
        user_intent="Forecast BTC",
        assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")],
        forecast_settings=ForecastSettings(),
    )
    result = agent.run(request)
    before = result.model_dump(mode="json")

    answer = agent.analyze("Forecast BTC", result)

    assert answer.analysis_state == "live"
    assert "Predicted prices" in answer.answer
    assert "BTC/USD" in answer.answer
    assert "→" in answer.answer
    assert "[validation passed]" in answer.answer
    assert "**Model view**" in answer.answer
    assert answer.structured_analysis is not None
    assert (
        answer.structured_analysis.verdict
        == "The validated batch shows a bounded directional estimate."
    )
    assert answer.structured_analysis.sections == []
    assert result.model_dump(mode="json") == before
    assert result.asset_results[0].model_output.predicted_price > 0  # type: ignore[union-attr]
    assert llm.calls == 1


@pytest.mark.parametrize("asset_count", [1, 2, 3, 4])
def test_forecast_uses_one_batch_summary_call_for_one_to_four_assets(asset_count: int) -> None:
    llm = _LLM()
    agent = ForecastAgent(service=_Service(), llm=llm)
    request = AnalysisRequest(
        user_intent="Forecast four assets",
        assets=[
            AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin"),
            AnalysisAsset(requested_name="BTC", symbol="BTC/USDT", coin_id="bitcoin"),
            AnalysisAsset(requested_name="BTC", symbol="BTC/EUR", coin_id="bitcoin"),
            AnalysisAsset(requested_name="BTC", symbol="BTC/GBP", coin_id="bitcoin"),
        ][:asset_count],
        forecast_settings=ForecastSettings(),
    )

    # Reuse distinct market symbols while keeping deterministic synthetic histories.
    class Service:
        def run(self, item: ForecastRequest):  # type: ignore[no-untyped-def]
            market = market_evidence(synthetic_candles()).model_copy(
                update={"symbol": item.symbol, "coin_id": item.coin_id}
            )
            return _run_forecast(market=market, request=item, policy=permissive_policy())

    agent = ForecastAgent(service=Service(), llm=llm)
    result = agent.run(request)
    agent.analyze("Forecast four assets", result)

    assert llm.calls == 1
    assert len(result.asset_results) == asset_count


@pytest.mark.parametrize("alias", ["summary", "answer", "verdict"])
def test_forecast_summary_accepts_safe_text_aliases(alias: str) -> None:
    llm = _LLM({alias: "The batch has mixed validation strength.", "confidence": "medium"})
    agent = ForecastAgent(service=_Service(), llm=llm)
    result = agent.run(
        AnalysisRequest(
            user_intent="Forecast BTC",
            assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")],
        )
    )

    answer = agent.analyze("Forecast BTC", result)

    assert "Predicted prices" in answer.answer
    assert answer.answer.endswith("The batch has mixed validation strength.")
    assert answer.confidence == 0.6
    assert answer.analysis_state == "live"
    assert llm.calls == 1


def test_forecast_summary_removes_generated_heading_and_repetition() -> None:
    sentence = "Validation supports a bounded directional estimate."
    llm = _LLM(
        {
            "summary": f"### **Model view:** {sentence} {sentence}",
            "limitations": ["- Coverage: Regime changes remain uncertain."],
            "confidence": "medium",
        }
    )
    agent = ForecastAgent(service=_Service(), llm=llm)
    result = agent.run(
        AnalysisRequest(
            user_intent="Forecast BTC",
            assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")],
        )
    )

    answer = agent.analyze("Forecast BTC", result)

    assert answer.structured_analysis is not None
    assert answer.structured_analysis.verdict == sentence
    assert answer.uncertainty == ["Coverage: Regime changes remain uncertain."]


def test_forecast_summary_includes_numbers_when_quality_gates_fail() -> None:
    llm = _LLM()
    agent = ForecastAgent(service=_StrictService(), llm=llm)
    result = agent.run(
        AnalysisRequest(
            user_intent="Forecast BTC",
            assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")],
        )
    )

    answer = agent.analyze("Forecast BTC", result)

    assert result.asset_results[0].status == "suppressed"
    assert result.asset_results[0].prediction is None
    assert "Predicted prices" in answer.answer
    assert "not trusted" in answer.answer
    assert "$" in answer.answer
    assert "%" in answer.answer
    assert answer.structured_analysis is not None
    assert (
        answer.structured_analysis.verdict
        == "The validated batch shows a bounded directional estimate."
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"summary": "   ", "confidence": 0.8},
        ["not", "an", "object"],
        LLMResponseError("rate limited"),
    ],
)
def test_invalid_or_failed_forecast_summary_keeps_computed_results(payload: object) -> None:
    llm = _LLM(payload)
    agent = ForecastAgent(service=_Service(), llm=llm)
    result = agent.run(
        AnalysisRequest(
            user_intent="Forecast BTC",
            assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")],
        )
    )
    before = result.model_dump(mode="json")

    answer = agent.analyze("Forecast BTC", result)

    assert answer.analysis_state == "evidence_only"
    assert "Predicted prices" in answer.answer
    assert answer.answer.count("Live model context is currently unavailable.") == 1
    assert result.model_dump(mode="json") == before
    assert llm.calls == 1


def test_failed_quality_forecast_fallback_keeps_organized_numeric_summary() -> None:
    llm = _LLM(LLMResponseError("rate limited"))
    agent = ForecastAgent(service=_StrictService(), llm=llm)
    result = agent.run(
        AnalysisRequest(
            user_intent="Forecast BTC",
            assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")],
        )
    )

    answer = agent.analyze("Forecast BTC", result)

    assert answer.analysis_state == "evidence_only"
    assert "Predicted prices" in answer.answer
    assert "not trusted" in answer.answer
    assert "$" in answer.answer
    assert "%" in answer.answer
