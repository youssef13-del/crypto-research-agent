from typing import cast

from tests.support.services import candles_for_prices

from crypto_research.domain.market import MarketEvidence
from crypto_research.domain.research import TechnicalSnapshot
from crypto_research.tools.market import (
    DEFAULT_OPPORTUNITY_WATCHLIST,
    scan_crypto_opportunities,
)


def test_opportunity_scan_ranks_live_watchlist_candidates() -> None:
    def market_service(**kwargs: object) -> MarketEvidence:
        symbol = str(kwargs["symbol"])
        slope = {"BTC/USD": 0.1, "ETH/USD": 0.2, "SOL/USD": 1.0}[symbol]
        candles = candles_for_prices([100 + index * slope for index in range(72)])
        return MarketEvidence(
            exchange="kraken",
            symbol=symbol,
            timeframe="1h",
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=candles[-1].timestamp,
            coin_id=str(kwargs.get("coin_id") or "").lower() or None,
        )

    result = scan_crypto_opportunities(
        watchlist=(("BTC", "bitcoin"), ("ETH", "ethereum"), ("SOL", "solana")),
        market_service=market_service,
    )

    assert result.candidates[0].asset == "SOL"
    assert result.candidates[0].score >= result.candidates[1].score
    assert "highest-ranked asset" in result.summary


def test_opportunity_watchlist_excludes_stablecoin_cash_proxies() -> None:
    assert "USDC" not in {asset for asset, _ in DEFAULT_OPPORTUNITY_WATCHLIST}


def test_opportunity_ranking_uses_technical_trend_and_explains_score() -> None:
    candles = candles_for_prices([100 + index * 0.2 for index in range(72)])

    def snapshot_fetcher(
        **_: object,
    ) -> tuple[list[tuple[MarketEvidence, TechnicalSnapshot]], list[str]]:
        def market(symbol: str) -> MarketEvidence:
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

        return [
            (market("BTC/USD"), TechnicalSnapshot(trend="bearish")),
            (market("ETH/USD"), TechnicalSnapshot(trend="bullish")),
        ], []

    result = scan_crypto_opportunities(
        watchlist=(("BTC", "bitcoin"), ("ETH", "ethereum")),
        snapshot_fetcher=snapshot_fetcher,
    )

    assert [candidate.asset for candidate in result.candidates] == ["ETH", "BTC"]
    assert "realized volatility" in result.candidates[0].reason
    assert "relative liquidity" in result.candidates[0].reason


def test_opportunity_uses_24_intervals_for_momentum_and_volume() -> None:
    requested_limits: list[int] = []

    def market_service(**kwargs: object) -> MarketEvidence:
        requested_limits.append(cast(int, kwargs["limit"]))
        candles = candles_for_prices([float(price) for price in range(100, 125)])
        return MarketEvidence(
            exchange="kraken",
            symbol=str(kwargs["symbol"]),
            timeframe="1h",
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=candles[-1].timestamp,
            coin_id="bitcoin",
        )

    result = scan_crypto_opportunities(
        limit=1,
        watchlist=(("BTC", "bitcoin"),),
        market_service=market_service,
    )

    candidate = result.candidates[0]
    assert requested_limits == [25]
    assert candidate.momentum_24h == 24.0


def test_opportunity_scan_keeps_successes_without_refetching_failures() -> None:
    calls: dict[str, int] = {}

    def market_service(**kwargs: object) -> MarketEvidence:
        symbol = str(kwargs["symbol"])
        calls[symbol] = calls.get(symbol, 0) + 1
        if symbol == "ETH/USD":
            raise OSError("provider unavailable")
        candles = candles_for_prices([100 + index for index in range(72)])
        return MarketEvidence(
            exchange="kraken",
            symbol=symbol,
            timeframe="1h",
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=candles[-1].timestamp,
            coin_id=str(kwargs.get("coin_id") or "").lower() or None,
        )

    result = scan_crypto_opportunities(
        watchlist=(("BTC", "bitcoin"), ("ETH", "ethereum")),
        market_service=market_service,
    )

    assert [candidate.asset for candidate in result.candidates] == ["BTC"]
    assert calls == {"BTC/USD": 1, "ETH/USD": 1}
    assert result.warnings == ["ETH/USD market data was unavailable (OSError)."]


def test_opportunity_scan_skips_incomplete_24_hour_windows() -> None:
    def market_service(**kwargs: object) -> MarketEvidence:
        symbol = str(kwargs["symbol"])
        candle_count = 24 if symbol == "BTC/USD" else 25
        candles = candles_for_prices([100.0 + index for index in range(candle_count)])
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

    result = scan_crypto_opportunities(
        watchlist=(("BTC", "bitcoin"), ("ETH", "ethereum")),
        market_service=market_service,
    )

    assert [candidate.asset for candidate in result.candidates] == ["ETH"]
    assert result.warnings == ["BTC/USD could not be scanned (ValueError)."]
