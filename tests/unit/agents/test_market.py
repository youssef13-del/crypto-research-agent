from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from tests.support.fakes import fake_market_service, fake_market_services, mock_request

from crypto_research.agents.market.market_agent import MarketAgent
from crypto_research.domain.evidence import DerivativesEvidence
from crypto_research.domain.market import Candle, MarketEvidence
from crypto_research.domain.research import (
    CollectionContext,
    MarketAgentResult,
    MarketComparisonResult,
    OpportunityCandidate,
    OpportunityScanResult,
    ResearchCapability,
    TechnicalSnapshot,
)
from crypto_research.tools.market import FutureMarketDataError


def test_market_agent_is_deterministic() -> None:
    agent = MarketAgent(services=fake_market_services())
    assert agent is not None


def test_market_agent_uses_opportunity_service_for_discovery() -> None:
    expected = OpportunityScanResult(
        exchange="kraken",
        timeframe="1h",
        collected_at=datetime(2026, 1, 1, tzinfo=UTC),
        summary="SOL leads the deterministic scan.",
        candidates=[
            OpportunityCandidate(
                rank=1,
                asset="SOL",
                symbol="SOL/USD",
                current_price=150,
                score=80,
                momentum_24h=5,
                volatility_24h=2,
                trend="bullish",
                reason="Strong momentum.",
            )
        ],
    )
    scanner = MagicMock(return_value=expected)
    result = MarketAgent(
        services=fake_market_services(discovery=scanner),
    ).run(mock_request(), requested_capabilities=[ResearchCapability.DISCOVERY])

    scanner.assert_called_once()
    assert scanner.call_args.args == (mock_request(),)
    assert "snapshot_fetcher" in scanner.call_args.kwargs
    assert result == expected


def test_market_agent_passes_collection_controls_to_discovery_injection() -> None:
    expected = OpportunityScanResult(
        exchange="kraken",
        timeframe="1h",
        collected_at=datetime(2026, 1, 1, tzinfo=UTC),
        summary="SOL leads the deterministic scan.",
        candidates=[
            OpportunityCandidate(
                rank=1,
                asset="SOL",
                symbol="SOL/USD",
                current_price=150,
                score=80,
                momentum_24h=5,
                volatility_24h=2,
                trend="bullish",
                reason="Strong momentum.",
            )
        ],
    )
    calls: list[object] = []

    def scanner(request: object, **_: object) -> OpportunityScanResult:
        calls.append(request)
        return expected

    request = mock_request()
    result = MarketAgent(services=fake_market_services(discovery=scanner)).run(
        request,
        requested_capabilities=[ResearchCapability.DISCOVERY],
        collection_context=CollectionContext(collected_at=datetime(2026, 1, 1, tzinfo=UTC)),
    )

    assert result == expected
    assert calls == [request]


def test_market_agent_keeps_collection_controls_for_the_default_tool() -> None:
    expected = OpportunityScanResult(
        exchange="kraken",
        timeframe="1h",
        collected_at=datetime(2026, 1, 1, tzinfo=UTC),
        summary="SOL leads the deterministic scan.",
        candidates=[
            OpportunityCandidate(
                rank=1,
                asset="SOL",
                symbol="SOL/USD",
                current_price=150,
                score=80,
                momentum_24h=5,
                volatility_24h=2,
                trend="bullish",
                reason="Strong momentum.",
            )
        ],
    )
    received: dict[str, object] = {}

    def scanner(
        request: object,
        *,
        snapshot_fetcher: object,
        collected_at: datetime,
    ) -> OpportunityScanResult:
        received.update(
            request=request,
            snapshot_fetcher=snapshot_fetcher,
            collected_at=collected_at,
        )
        return expected

    def snapshot_fetcher(**_: object) -> tuple[list[object], list[str]]:
        return [], []

    request = mock_request()
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    result = MarketAgent(
        services=fake_market_services(discovery=scanner, snapshots=snapshot_fetcher),
    ).run(
        request,
        requested_capabilities=[ResearchCapability.DISCOVERY],
        collection_context=CollectionContext(collected_at=cutoff),
    )

    assert result == expected
    assert received == {
        "request": request,
        "snapshot_fetcher": snapshot_fetcher,
        "collected_at": cutoff,
    }


def test_market_agent_passes_collection_controls_to_market_injection() -> None:
    calls: list[tuple[str, str, str, int]] = []

    def market_service(
        *,
        exchange_name: str,
        symbol: str,
        timeframe: str,
        limit: int,
        **_: object,
    ) -> MarketEvidence:
        calls.append((exchange_name, symbol, timeframe, limit))
        return fake_market_service()

    result = MarketAgent(services=fake_market_services(market_evidence=market_service)).run(
        mock_request(),
        requested_capabilities=[ResearchCapability.MARKET],
        collection_context=CollectionContext(collected_at=datetime(2026, 1, 20, tzinfo=UTC)),
    )

    assert isinstance(result, MarketAgentResult)
    assert calls[0] == ("kraken", "BTC/USD", "1h", 320)


def test_market_agent_returns_a_limitation_for_future_provider_candles() -> None:
    result = MarketAgent(
        services=fake_market_services(
            market_evidence=lambda **_: (_ for _ in ()).throw(FutureMarketDataError("future"))
        )
    ).run(mock_request(), requested_capabilities=[ResearchCapability.MARKET])

    assert isinstance(result, MarketComparisonResult)
    assert result.assets == []
    assert "future-dated market data were excluded" in result.warnings[0]


def test_market_agent_keeps_a_typed_comparison_when_the_provider_fails() -> None:
    request = mock_request().model_copy(update={"comparison_symbols": ["BTC/USD", "ETH/USD"]})
    result = MarketAgent(
        services=fake_market_services(
            comparison=lambda **_: (_ for _ in ()).throw(OSError("provider unavailable"))
        )
    ).run(request, requested_capabilities=[ResearchCapability.MARKET])

    assert isinstance(result, MarketComparisonResult)
    assert result.assets == []
    assert "unavailable" in result.warnings[0]


def test_market_agent_isolates_derivatives_failures_across_comparison_assets() -> None:
    request = mock_request().model_copy(update={"comparison_symbols": ["BTC/USD", "ETH/USD"]})
    btc = fake_market_service()
    eth = btc.model_copy(update={"symbol": "ETH/USD"})
    calls: list[tuple[str, str]] = []

    def derivatives_fetcher(*, asset: str, base_url: str, **_: object) -> DerivativesEvidence:
        calls.append((asset, base_url))
        if asset == "ETH":
            raise OSError("isolated provider failure")
        return DerivativesEvidence(
            asset=asset,
            status="not_applicable",
            collected_at=btc.collected_at,
            warnings=["No active contract."],
        )

    result = MarketAgent(
        services=fake_market_services(
            comparison=lambda **_: (
                [
                    (btc, TechnicalSnapshot(trend="neutral")),
                    (eth, TechnicalSnapshot(trend="neutral")),
                ],
                [],
            ),
            derivatives=derivatives_fetcher,
        ),
        derivatives_base_url="https://fapi.example.test",
    ).run(
        request,
        requested_capabilities=[ResearchCapability.MARKET, ResearchCapability.DERIVATIVES],
        collection_context=CollectionContext(collected_at=btc.collected_at),
    )

    assert isinstance(result, MarketComparisonResult)
    assert len(result.assets) == 2
    assert result.assets[0].derivatives is not None
    assert result.assets[0].derivatives.status == "not_applicable"
    assert result.assets[1].derivatives is not None
    assert result.assets[1].derivatives.status == "unavailable"
    assert calls == [
        ("BTC", "https://fapi.example.test"),
        ("ETH", "https://fapi.example.test"),
    ]


def test_market_agent_quarantines_future_injected_market_evidence() -> None:
    future = datetime(2026, 1, 2, tzinfo=UTC)
    candle = Candle(
        timestamp=future,
        open=1,
        high=1,
        low=1,
        close=1,
        volume=1,
    )
    market = MarketEvidence(
        exchange="kraken",
        symbol="BTC/USD",
        timeframe="1h",
        candles=[candle],
        first_time=future,
        last_time=future,
        current_price=1,
        collected_at=future,
    )

    result = MarketAgent(services=fake_market_services(market_evidence=lambda **_: market)).run(
        mock_request(),
        requested_capabilities=[ResearchCapability.MARKET],
        collection_context=CollectionContext(collected_at=datetime(2026, 1, 1, tzinfo=UTC)),
    )

    assert isinstance(result, MarketComparisonResult)
    assert result.assets == []
    assert "future-dated" in result.warnings[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exchange", "coinbase"),
        ("symbol", "ETH/USD"),
        ("timeframe", "4h"),
    ],
)
def test_market_agent_quarantines_injected_data_for_the_wrong_market_identity(
    field: str,
    value: str,
) -> None:
    market = fake_market_service().model_copy(update={field: value})

    result = MarketAgent(services=fake_market_services(market_evidence=lambda **_: market)).run(
        mock_request(),
        requested_capabilities=[ResearchCapability.MARKET],
    )

    assert isinstance(result, MarketComparisonResult)
    assert result.assets == []
    assert "did not match the selected exchange" in result.warnings[0]


def test_market_agent_quarantines_a_discovery_scan_for_another_exchange() -> None:
    scan = OpportunityScanResult(
        exchange="binance",
        timeframe="1h",
        collected_at=datetime(2026, 1, 1, tzinfo=UTC),
        summary="A mismatched scan must not be rendered.",
        candidates=[
            OpportunityCandidate(
                rank=1,
                asset="SOL",
                symbol="SOL/USD",
                current_price=150,
                score=80,
                momentum_24h=5,
                volatility_24h=2,
                trend="bullish",
                reason="Strong momentum.",
            )
        ],
    )

    result = MarketAgent(services=fake_market_services(discovery=lambda _, **__: scan)).run(
        mock_request(),
        requested_capabilities=[ResearchCapability.DISCOVERY],
    )

    assert isinstance(result, MarketComparisonResult)
    assert result.assets == []
    assert "did not match the selected exchange" in result.warnings[0]


def test_market_agent_adds_higher_timeframe_confirmation_for_hourly_research() -> None:
    calls: list[str] = []

    def market_service(**kwargs: object) -> MarketEvidence:
        timeframe = str(kwargs["timeframe"])
        interval_hours = {"1h": 1, "4h": 4, "1d": 24}[timeframe]
        candles = [
            Candle(
                timestamp=datetime(2026, 1, 1, tzinfo=UTC)
                + index * timedelta(hours=interval_hours),
                open=100 + index,
                high=102 + index,
                low=99 + index,
                close=101 + index,
                volume=1_000,
            )
            for index in range(60)
        ]
        calls.append(timeframe)
        return MarketEvidence(
            exchange="kraken",
            symbol="BTC/USD",
            timeframe=timeframe,
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=candles[-1].timestamp,
        )

    result = MarketAgent(services=fake_market_services(market_evidence=market_service)).run(
        mock_request(), requested_capabilities=[ResearchCapability.MARKET]
    )

    assert isinstance(result, MarketAgentResult)
    assert calls == ["1h", "4h", "1d"]
    assert [(item.timeframe, item.status) for item in result.contextual_timeframes] == [
        ("4h", "complete"),
        ("1d", "complete"),
    ]
