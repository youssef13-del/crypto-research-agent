from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from crypto_research.domain.analytics import build_market_posture, calculate_market_features
from crypto_research.domain.evidence import TechnicalSnapshot
from crypto_research.domain.market import Candle, MarketEvidence


def test_market_features_calculate_exact_24_hour_return() -> None:
    market = _market(
        [float(price) for price in range(100, 125)],
        timeframe="1h",
        step=timedelta(hours=1),
    )

    features = calculate_market_features(market)
    change = next(item for item in features.returns if item.label == "24h")

    assert change.status == "available"
    assert change.reference_price == 100
    assert change.latest_price == 124
    assert change.change_absolute == 24
    assert change.return_decimal == pytest.approx(0.24)
    assert change.return_percent == pytest.approx(24)


@pytest.mark.parametrize(
    ("timeframe", "step", "count"),
    [
        ("5m", timedelta(minutes=5), 289),
        ("15m", timedelta(minutes=15), 97),
        ("30m", timedelta(minutes=30), 49),
        ("1h", timedelta(hours=1), 25),
        ("4h", timedelta(hours=4), 7),
        ("1d", timedelta(days=1), 2),
    ],
)
def test_market_features_support_24_hour_return_across_timeframes(
    timeframe: str,
    step: timedelta,
    count: int,
) -> None:
    prices = [100.0 + index for index in range(count)]
    features = calculate_market_features(_market(prices, timeframe=timeframe, step=step))

    change = next(item for item in features.returns if item.label == "24h")

    assert change.status == "available"
    assert change.period_start is not None
    assert change.period_end - change.period_start == timedelta(hours=24)


def test_market_features_label_insufficient_24_hour_history() -> None:
    market = _market(
        [100.0 + index for index in range(24)],
        timeframe="1h",
        step=timedelta(hours=1),
    )

    change = next(item for item in calculate_market_features(market).returns if item.label == "24h")

    assert change.status == "unavailable"
    assert change.reason is not None
    assert "does not cover" in change.reason


def test_market_features_reject_gapped_24_hour_window() -> None:
    market = _market(
        [100.0 + index for index in range(26)],
        timeframe="1h",
        step=timedelta(hours=1),
    )
    candles = list(market.candles)
    del candles[-10]
    market = market.model_copy(
        update={
            "candles": candles,
            "first_time": candles[0].timestamp,
            "last_time": candles[-1].timestamp,
            "current_price": candles[-1].close,
        }
    )

    features = calculate_market_features(market)
    change = next(item for item in features.returns if item.label == "24h")

    assert features.contiguous is False
    assert change.status == "unavailable"
    assert change.reason is not None
    assert "not contiguous" in change.reason


def test_market_features_include_compact_window_statistics() -> None:
    market = _market(
        [100.0, 120.0, 90.0, 110.0],
        timeframe="1h",
        step=timedelta(hours=1),
    )

    features = calculate_market_features(market)

    assert features.high == 121
    assert features.low == 89
    assert features.base_volume == 4_000
    assert features.quote_volume == 420_000
    assert features.maximum_drawdown == pytest.approx(0.25)
    assert features.fresh_at_collection is True


def test_build_market_posture_includes_full_snapshot() -> None:
    market = _market(
        [float(price) for price in range(100, 125)],
        timeframe="1h",
        step=timedelta(hours=1),
    )
    technical = TechnicalSnapshot(
        trend="bullish",
        rsi=61.3,
        macd=4.0,
        atr=9.0,
        volatility=0.012,
        support=90_000,
        resistance=110_000,
    )

    contextual_timeframes = cast(
        "list[tuple[str, str]]",
        [("1d", "bullish"), ("", "bearish"), ("1w", None)],
    )
    posture = build_market_posture(
        market,
        technical,
        contextual_timeframes=contextual_timeframes,
    )

    assert posture.symbol == "BTC/USD"
    assert posture.exchange == "kraken"
    assert posture.timeframe == "1h"
    assert posture.as_of == market.last_time
    assert posture.collected_at == market.collected_at
    assert posture.price == 124
    assert posture.change_24h_percent == pytest.approx(24)
    assert posture.change_24h_absolute == pytest.approx(24)
    assert posture.high == 125
    assert posture.low == 99
    assert posture.range_percent == pytest.approx(20.97, rel=1e-2)
    assert posture.quote_volume == 2_800_000
    assert posture.maximum_drawdown == 0
    assert posture.trend == "bullish"
    assert posture.rsi == 61.3
    assert posture.rsi_band == "strong"
    assert posture.macd == 4.0
    assert posture.atr == 9.0
    assert posture.volatility == 0.012
    assert posture.support == 90_000
    assert posture.resistance == 110_000
    assert posture.fresh is True
    assert posture.data_delay_seconds == 0
    assert posture.contextual_confirmation == ["1d:bullish"]


def test_build_market_posture_degrades_without_24h_or_technical_value() -> None:
    market = _market(
        [100.0 + index for index in range(24)],
        timeframe="1h",
        step=timedelta(hours=1),
    )
    technical = TechnicalSnapshot(
        status="unavailable",
        limitation="Too few candles.",
        trend="bearish",
    )

    posture = build_market_posture(market, technical)

    assert posture.change_24h_percent is None
    assert posture.change_24h_absolute is None
    assert posture.rsi is None
    assert posture.rsi_band == "neutral"
    assert posture.support is None
    assert posture.resistance is None
    assert posture.contextual_confirmation == []
    assert posture.trend == "bearish"


@pytest.mark.parametrize(
    ("rsi", "band"),
    [
        (None, "neutral"),
        (29.9, "oversold"),
        (30, "weak"),
        (44.9, "weak"),
        (45, "neutral"),
        (55, "neutral"),
        (55.1, "strong"),
        (70, "strong"),
        (70.1, "overbought"),
    ],
)
def test_build_market_posture_labels_rsi_bands_at_boundaries(rsi: float | None, band: str) -> None:
    market = _market([100.0, 110.0, 120.0], timeframe="1h", step=timedelta(hours=1))

    posture = build_market_posture(market, TechnicalSnapshot(trend="neutral", rsi=rsi))

    assert posture.rsi_band == band


def _market(
    prices: list[float],
    *,
    timeframe: str,
    step: timedelta,
) -> MarketEvidence:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        Candle(
            timestamp=start + index * step,
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1_000,
        )
        for index, price in enumerate(prices)
    ]
    return MarketEvidence(
        exchange="kraken",
        symbol="BTC/USD",
        timeframe=timeframe,
        candles=candles,
        first_time=candles[0].timestamp,
        last_time=candles[-1].timestamp,
        current_price=candles[-1].close,
        collected_at=candles[-1].timestamp + step,
        coin_id="bitcoin",
    )
