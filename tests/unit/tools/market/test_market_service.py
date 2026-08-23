from datetime import UTC, datetime, timedelta

import pytest
from tests.support.services import candles_for_prices, ohlcv_row

from crypto_research.domain.market import MarketEvidence
from crypto_research.domain.research import AnalysisRequest
from crypto_research.tools.market import (
    FutureMarketDataError,
    _latest_contiguous_candles,
    _parse_candles,
    fetch_market_comparison,
    fetch_market_evidence,
    fetch_market_snapshots,
)


def test_market_snapshots_keep_available_assets_when_one_fetch_fails() -> None:
    def market_service(**kwargs: object) -> MarketEvidence:
        symbol = str(kwargs["symbol"])
        if symbol == "ETH/USD":
            raise OSError("provider unavailable")
        candles = candles_for_prices([100.0 + index for index in range(72)])
        return MarketEvidence(
            exchange="kraken",
            symbol=symbol,
            timeframe="1h",
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=candles[-1].timestamp,
        )

    snapshots, warnings = fetch_market_snapshots(
        exchange_name="kraken",
        symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
        timeframe="1h",
        limit=72,
        market_service=market_service,
    )

    assert [market.symbol for market, _ in snapshots] == ["BTC/USD", "SOL/USD"]
    assert warnings == ["ETH/USD market data was unavailable (OSError)."]


def test_market_comparison_aligns_available_assets_to_shared_timestamps() -> None:
    starts = {
        "BTC/USD": datetime(2026, 1, 1, tzinfo=UTC),
        "ETH/USD": datetime(2026, 1, 1, 1, tzinfo=UTC),
    }

    def market_service(**kwargs: object) -> MarketEvidence:
        symbol = str(kwargs["symbol"])
        candles = candles_for_prices(
            [100.0 + index for index in range(72)],
            start=starts[symbol],
        )
        return MarketEvidence(
            exchange="kraken",
            symbol=symbol,
            timeframe="1h",
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=datetime.now(UTC),
        )

    snapshots, warnings = fetch_market_comparison(
        request=AnalysisRequest(
            user_intent="Compare BTC and ETH",
            symbol="BTC/USD",
            comparison_symbols=["BTC/USD", "ETH/USD"],
        ),
        market_service=market_service,
    )

    assert not warnings
    assert len(snapshots) == 2
    assert [c.timestamp for c in snapshots[0][0].candles] == [
        c.timestamp for c in snapshots[1][0].candles
    ]


def test_single_market_snapshot_does_not_emit_comparison_warning() -> None:
    candles = candles_for_prices([100.0 + index for index in range(72)])

    def market_service(**kwargs: object) -> MarketEvidence:
        return MarketEvidence(
            exchange="kraken",
            symbol=str(kwargs["symbol"]),
            timeframe="1h",
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=datetime.now(UTC),
        )

    snapshots, warnings = fetch_market_comparison(
        request=AnalysisRequest(user_intent="Review Bitcoin"),
        market_service=market_service,
    )

    assert len(snapshots) == 1
    assert warnings == []


def test_market_comparison_labels_unaligned_partial_histories() -> None:
    starts = {
        "BTC/USD": datetime(2026, 1, 1, tzinfo=UTC),
        "ETH/USD": datetime(2026, 2, 1, tzinfo=UTC),
    }

    def market_service(**kwargs: object) -> MarketEvidence:
        symbol = str(kwargs["symbol"])
        candles = candles_for_prices(
            [100.0 + index for index in range(72)],
            start=starts[symbol],
        )
        return MarketEvidence(
            exchange="kraken",
            symbol=symbol,
            timeframe="1h",
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=datetime.now(UTC),
        )

    snapshots, warnings = fetch_market_comparison(
        request=AnalysisRequest(
            user_intent="Compare BTC and ETH",
            comparison_symbols=["BTC/USD", "ETH/USD"],
        ),
        market_service=market_service,
    )

    assert len(snapshots) == 2
    assert any("not directly comparable" in warning for warning in warnings)
    assert snapshots[0][0].first_time != snapshots[1][0].first_time


def test_market_rows_exclude_still_open_candle() -> None:
    now = datetime(2026, 1, 1, 10, 30, tzinfo=UTC)
    complete_time = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    open_time = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    rows = [
        ohlcv_row(complete_time, 100.0),
        ohlcv_row(open_time, 101.0),
    ]

    candles = _parse_candles(rows, timeframe="1h", now=now).candles

    assert [candle.timestamp for candle in candles] == [complete_time]


def test_market_rows_quarantine_future_dated_candles() -> None:
    now = datetime(2026, 1, 1, 10, 30, tzinfo=UTC)
    rows = [ohlcv_row(datetime(2026, 1, 1, 12, tzinfo=UTC), 100.0)]

    with pytest.raises(FutureMarketDataError, match="future-dated candles"):
        _parse_candles(rows, timeframe="1h", now=now)


def test_market_data_records_quality_exclusions_without_leaking_invalid_candles() -> None:
    now = datetime(2026, 1, 2, 12, 30, tzinfo=UTC)
    complete_start = datetime(2026, 1, 1, 15, tzinfo=UTC)
    rows = [
        ohlcv_row(complete_start + timedelta(hours=index), 100.0 + index) for index in range(21)
    ]
    rows.extend(
        [
            ohlcv_row(datetime(2026, 1, 2, 13, tzinfo=UTC), 200.0),
            ohlcv_row(datetime(2026, 1, 2, 12, tzinfo=UTC), 201.0),
            ohlcv_row(complete_start, 99.0),
            [1, 2, 3],
        ]
    )

    result = _parse_candles(rows, timeframe="1h", now=now)

    quality = result.quality
    assert len(result.candles) == 21
    assert quality.excluded_future == quality.excluded_incomplete == 1
    assert quality.excluded_duplicates == quality.excluded_malformed == 1
    assert all(candle.timestamp <= now - timedelta(hours=1) for candle in result.candles)


def test_market_rows_reject_malformed_provider_payload() -> None:
    with pytest.raises(ValueError, match="No valid complete candles"):
        _parse_candles([[1, 2, 3]], timeframe="1h")


def test_market_data_paginates_and_resamples_supported_source_timeframe() -> None:
    now = datetime.now(UTC)
    current_bucket = datetime.fromtimestamp(
        int(now.timestamp()) // (4 * 60 * 60) * (4 * 60 * 60),
        tz=UTC,
    )
    start = current_bucket - timedelta(hours=1_200)
    rows = [ohlcv_row(start + timedelta(hours=2 * index), 100.0 + index) for index in range(600)]
    calls: list[tuple[str, int | None, int]] = []

    class CoinbaseLikeExchange:
        timeframes = {"1h": "ONE_HOUR", "2h": "TWO_HOUR"}

        def load_markets(self) -> dict[str, object]:
            return {"BTC/USD": {}}

        def fetch_ohlcv(
            self,
            symbol: str,
            *,
            timeframe: str,
            since: int | None,
            limit: int,
        ) -> list[list[float]]:
            del symbol
            calls.append((timeframe, since, limit))
            available = [row for row in rows if since is None or row[0] >= since]
            return available[: min(limit, 300)]

        def close(self) -> None:
            return None

    result = fetch_market_evidence(
        exchange_name="coinbase",
        symbol="BTC/USD",
        timeframe="4h",
        limit=300,
        exchange_factory=lambda _: CoinbaseLikeExchange(),
    )

    assert len(result.candles) == 300
    assert result.timeframe == "4h"
    assert result.candles[0].open == 100.0
    assert result.candles[0].close == 101.0
    assert result.candles[0].volume == 2_000
    assert len(calls) >= 3
    assert {timeframe for timeframe, _, _ in calls} == {"2h"}
    assert all(limit <= 300 for _, _, limit in calls)


def test_market_data_keeps_newest_contiguous_candle_window() -> None:
    candles = candles_for_prices([100.0, 101.0, 102.0, 103.0])
    candles[2] = candles[2].model_copy(
        update={"timestamp": candles[2].timestamp + timedelta(hours=2)}
    )
    candles[3] = candles[3].model_copy(
        update={"timestamp": candles[3].timestamp + timedelta(hours=2)}
    )

    contiguous = _latest_contiguous_candles(candles, timeframe="1h")

    assert contiguous == candles[2:]


def test_market_data_keeps_requested_complete_rows_before_dropping_open_candle() -> None:
    now = datetime.now(UTC)
    current_bucket = datetime.fromtimestamp(
        int(now.timestamp()) // 3600 * 3600,
        tz=UTC,
    )
    rows = [
        ohlcv_row(current_bucket - timedelta(hours=21 - index), 100.0 + index)
        for index in range(22)
    ]

    class Exchange:
        timeframes = {"1h": "1h"}

        def load_markets(self) -> dict[str, object]:
            return {"BTC/USD": {}}

        def fetch_ohlcv(self, *args: object, **kwargs: object) -> list[list[float]]:
            return rows

        def close(self) -> None:
            return None

    result = fetch_market_evidence(
        exchange_name="kraken",
        symbol="BTC/USD",
        timeframe="1h",
        limit=20,
        exchange_factory=lambda _: Exchange(),
    )

    assert len(result.candles) == 20
    assert result.last_time == current_bucket - timedelta(hours=1)


def test_market_data_rejects_stale_candles() -> None:
    start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(days=30)
    rows = [ohlcv_row(start + timedelta(hours=index), 100.0 + index) for index in range(20)]

    class Exchange:
        timeframes = {"1h": "1h"}

        def load_markets(self) -> dict[str, object]:
            return {"BTC/USD": {}}

        def fetch_ohlcv(self, *args: object, **kwargs: object) -> list[list[float]]:
            return rows

        def close(self) -> None:
            return None

    with pytest.raises(ValueError, match="stale candles"):
        fetch_market_evidence(
            exchange_name="kraken",
            symbol="BTC/USD",
            timeframe="1h",
            limit=20,
            exchange_factory=lambda _: Exchange(),
        )


def test_market_data_rejects_undersized_current_window() -> None:
    completed = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    rows = [ohlcv_row(completed, 100.0)]

    class Exchange:
        timeframes = {"1h": "1h"}

        def load_markets(self) -> dict[str, object]:
            return {"BTC/USD": {}}

        def fetch_ohlcv(self, *args: object, **kwargs: object) -> list[list[float]]:
            return rows

        def close(self) -> None:
            return None

    with pytest.raises(ValueError, match="at least 20 recent contiguous candles"):
        fetch_market_evidence(
            exchange_name="kraken",
            symbol="BTC/USD",
            timeframe="1h",
            limit=250,
            exchange_factory=lambda _: Exchange(),
        )
