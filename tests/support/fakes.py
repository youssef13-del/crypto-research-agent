import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from crypto_research.domain.market import Candle, MarketEvidence
from crypto_research.domain.research import (
    AgentAnswer,
    AnalysisRequest,
    FundamentalEvidence,
    NewsEvidence,
    NewsItem,
)
from crypto_research.llm.client import LLMResponseError, LLMRole
from crypto_research.tools.types import MarketServices


def mock_request() -> AnalysisRequest:
    return AnalysisRequest(
        user_intent="Full Bitcoin analysis please",
        exchange="kraken",
        symbol="BTC/USD",
        timeframe="1h",
        candle_limit=320,
        coin_id="bitcoin",
    )


class FakeLLM:
    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[Any],
    ) -> Any:
        del system_prompt
        if output_schema is AgentAnswer:
            payload = json.loads(user_prompt)
            agent = payload.get("agent", "market_agent")
            return AgentAnswer(
                agent=agent,
                answer=f"{agent} reviewed the question using the supplied evidence.",
                confidence=0.8,
                status="complete",
            )
        raise AssertionError(f"Unexpected schema for {role.value}: {output_schema.__name__}")


def fake_market_service(**_: object) -> MarketEvidence:
    candles = synthetic_candles()
    return MarketEvidence(
        exchange="kraken",
        symbol="BTC/USD",
        timeframe="1h",
        candles=candles,
        first_time=candles[0].timestamp,
        last_time=candles[-1].timestamp,
        current_price=candles[-1].close,
        collected_at=candles[-1].timestamp,
        coin_id="bitcoin",
    )


def fake_news_service(**_: object) -> NewsEvidence:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return NewsEvidence(
        items=[
            NewsItem(
                publisher="Example",
                title="Bitcoin market update",
                excerpt="BTC market news.",
                url="https://example.test/btc",
                published_at=timestamp,
            )
        ],
        query="Bitcoin BTC",
        collected_at=timestamp,
        warnings=[],
    )


def fake_fundamental_service(**_: object) -> FundamentalEvidence:
    return FundamentalEvidence(
        name="Bitcoin",
        symbol="btc",
        market_cap=1_000_000.0,
        rank=1,
        circulating_supply=19_000_000.0,
        total_supply=21_000_000.0,
        max_supply=21_000_000.0,
        categories=["Cryptocurrency"],
        homepage="https://bitcoin.org",
        genesis_date="2009-01-03",
        status="available",
        collected_at=datetime(2026, 1, 1, tzinfo=UTC),
        warnings=[],
    )


def fake_market_services(**overrides: Any) -> MarketServices:
    """Build a focused market dependency bundle with optional test replacements."""

    from crypto_research.tools.derivatives import fetch_derivatives_evidence
    from crypto_research.tools.market import (
        calculate_comparison_metrics,
        calculate_indicators,
        fetch_market_comparison,
        fetch_market_snapshots,
        scan_crypto_opportunities,
    )

    dependencies: dict[str, Any] = {
        "market_evidence": fake_market_service,
        "indicators": calculate_indicators,
        "snapshots": fetch_market_snapshots,
        "comparison": fetch_market_comparison,
        "comparison_metrics": calculate_comparison_metrics,
        "discovery": scan_crypto_opportunities,
        "derivatives": fetch_derivatives_evidence,
    }
    dependencies.update(overrides)
    return MarketServices(**dependencies)


def synthetic_candles(count: int = 320, *, interval_hours: int = 1) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    for index in range(count):
        price = 100 + index * 0.2 + math.sin(index / 7) * 2
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=index * interval_hours),
                open=price,
                high=price + 2,
                low=price - 2,
                close=price + 0.5,
                volume=1_000 + index,
            )
        )
    return candles


def provider_response_error(status_code: int) -> LLMResponseError:
    provider_error = RuntimeError("provider failed")
    provider_error.status_code = status_code  # type: ignore[attr-defined]
    error = LLMResponseError("live provider failed")
    error.__cause__ = provider_error
    return error
