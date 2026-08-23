from datetime import timedelta

import pytest
from tests.support.forecasting import market_evidence, permissive_policy, synthetic_candles

from crypto_research.domain.forecast import ForecastFailure, ForecastRequest
from crypto_research.domain.market import MarketEvidence
from crypto_research.forecasting.models import MODEL_REGISTRY
from crypto_research.forecasting.service import ForecastPolicy, ForecastService, _run_forecast


def test_registry_exposes_two_experimental_models() -> None:
    assert set(MODEL_REGISTRY) == {"gradient_boosting_huber", "ridge"}


@pytest.mark.parametrize("model_id", tuple(MODEL_REGISTRY))
def test_each_model_returns_a_validated_run(model_id: str) -> None:
    candles = synthetic_candles()
    market = market_evidence(candles)
    result = _run_forecast(
        market=market,
        request=ForecastRequest(
            asset="Bitcoin",
            coin_id="bitcoin",
            symbol="BTC/USD",
            model_id=model_id,
            horizon_hours=24,
        ),
        policy=permissive_policy(),
    )

    assert result.request.model_id == model_id
    assert result.metrics.validation_samples > 0
    assert result.model.feature_columns
    assert result.model.random_state == (42 if model_id == "gradient_boosting_huber" else None)
    assert result.quality.passed
    assert result.prediction is not None
    assert result.model_output == result.prediction
    assert result.prediction.timestamp > market.last_time
    assert result.prediction.predicted_return == pytest.approx(
        result.prediction.predicted_price / market.current_price - 1
    )


def test_failed_quality_gates_withhold_the_point_prediction() -> None:
    candles = synthetic_candles()
    result = _run_forecast(
        market=market_evidence(candles),
        request=ForecastRequest(asset="Bitcoin", coin_id="bitcoin"),
        policy=ForecastPolicy(
            minimum_training_samples=100,
            minimum_validation_samples=20,
            time_series_folds=3,
            minimum_mae_improvement=100.0,
            minimum_directional_accuracy=1.0,
            maximum_interval_width=0.01,
        ),
    )

    assert result.status == "suppressed"
    assert result.prediction is None
    assert result.model_output.predicted_price > 0
    assert result.quality.prediction_suppressed is True


def test_service_returns_typed_data_failure() -> None:
    request = ForecastRequest(asset="Bitcoin", coin_id="bitcoin")

    def unavailable(**_: object) -> MarketEvidence:
        raise TimeoutError("provider unavailable")

    result = ForecastService(market_fetcher=unavailable).run(request)

    assert isinstance(result, ForecastFailure)
    assert result.code == "DATA_UNAVAILABLE"
    assert result.message


def test_service_preserves_stale_data_failure_code() -> None:
    request = ForecastRequest(asset="Bitcoin", coin_id="bitcoin")

    def stale(**_: object) -> MarketEvidence:
        raise ValueError("Market provider returned stale candles.")

    result = ForecastService(market_fetcher=stale).run(request)

    assert isinstance(result, ForecastFailure)
    assert result.code == "STALE_DATA"
    assert "stale" in result.message


def test_service_rejects_non_contiguous_provider_history() -> None:
    candles = synthetic_candles()
    candles = [
        candle.model_copy(
            update={
                "timestamp": candle.timestamp
                + (timedelta(hours=1) if index >= 100 else timedelta())
            }
        )
        for index, candle in enumerate(candles)
    ]
    market = market_evidence(candles)

    def gapped(**_: object) -> MarketEvidence:
        return market

    result = ForecastService(market_fetcher=gapped).run(
        ForecastRequest(asset="Bitcoin", coin_id="bitcoin")
    )

    assert isinstance(result, ForecastFailure)
    assert result.code == "INVALID_DATA"
    assert "contiguous" in result.message


def test_service_rejects_market_evidence_for_a_different_request() -> None:
    market = market_evidence(synthetic_candles())

    def mismatched(**_: object) -> MarketEvidence:
        return market

    result = ForecastService(market_fetcher=mismatched).run(
        ForecastRequest(asset="Ethereum", coin_id="ethereum", symbol="ETH/USD")
    )

    assert isinstance(result, ForecastFailure)
    assert result.code == "INVALID_DATA"
    assert "does not match" in result.message


def test_forecast_rejects_gapped_candles() -> None:
    candles = synthetic_candles()
    del candles[100]
    with pytest.raises(ValueError, match="contiguous"):
        _run_forecast(
            market=market_evidence(candles),
            request=ForecastRequest(asset="Bitcoin"),
            policy=permissive_policy(),
        )
