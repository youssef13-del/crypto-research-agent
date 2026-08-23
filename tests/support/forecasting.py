"""Forecasting-specific test fixtures and synthetic market data."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from crypto_research.domain.market import Candle, MarketEvidence
from crypto_research.forecasting.service import ForecastPolicy


def permissive_policy() -> ForecastPolicy:
    return ForecastPolicy(
        minimum_training_samples=100,
        minimum_validation_samples=20,
        time_series_folds=3,
        minimum_mae_improvement=-1.0,
        minimum_directional_accuracy=0.0,
        maximum_absolute_forecast_return=10.0,
        maximum_interval_width=10.0,
    )


def synthetic_candles(count: int = 520) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values: list[Candle] = []
    for index in range(count):
        price = 100 + index * 0.08 + math.sin(index / 7) * 2
        values.append(
            Candle(
                timestamp=start + timedelta(hours=index),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000 + index,
            )
        )
    return values


def market_evidence(candles: list[Candle]) -> MarketEvidence:
    return MarketEvidence(
        exchange="kraken",
        symbol="BTC/USD",
        timeframe="1h",
        candles=candles,
        first_time=candles[0].timestamp,
        last_time=candles[-1].timestamp,
        current_price=candles[-1].close,
        collected_at=datetime.now(UTC),
        coin_id="bitcoin",
    )
