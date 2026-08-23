import pytest
from tests.support.fakes import synthetic_candles
from tests.support.services import candles_for_prices

from crypto_research.tools.market import calculate_indicators


def test_technical_indicators_return_valid_snapshot() -> None:
    snapshot = calculate_indicators(synthetic_candles(80))

    assert snapshot.trend in {"bullish", "bearish", "neutral"}
    assert snapshot.rsi is not None
    assert 0 <= snapshot.rsi <= 100
    assert snapshot.support is not None
    assert snapshot.resistance is not None
    assert snapshot.support <= snapshot.resistance


def test_rsi_handles_zero_losses_and_flat_prices() -> None:
    rising = calculate_indicators(candles_for_prices([100.0 + index for index in range(50)]))
    flat = calculate_indicators(candles_for_prices([100.0] * 50))

    assert rising.rsi == 100.0
    assert flat.rsi == 50.0


def test_technical_indicators_require_candles() -> None:
    with pytest.raises(ValueError, match="At least one candle"):
        calculate_indicators([])
