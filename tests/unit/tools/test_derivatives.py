from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from crypto_research.domain.evidence import DerivativesEvidence
from crypto_research.tools import derivatives
from crypto_research.tools.cache import TTLCache

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)


def _millis(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _exchange_info(*, include_contract: bool = True) -> dict[str, object]:
    return {
        "symbols": (
            [
                {
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "symbol": "BTCUSDT",
                }
            ]
            if include_contract
            else [
                {
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "contractType": "CURRENT_QUARTER",
                    "status": "TRADING",
                    "symbol": "BTCUSDT_260925",
                }
            ]
        )
    }


def _funding_rows(*, symbol: str = "BTCUSDT") -> list[dict[str, object]]:
    return [
        {
            "symbol": symbol,
            "fundingTime": _millis(NOW - timedelta(hours=8 * index)),
            "fundingRate": str(0.0001 * index),
        }
        for index in (3, 2, 1)
    ]


def _open_interest_rows(*, symbol: str = "BTCUSDT") -> list[dict[str, object]]:
    return [
        {
            "symbol": symbol,
            "timestamp": _millis(NOW - timedelta(hours=hours)),
            "sumOpenInterestValue": value,
        }
        for hours, value in ((25, "1000000"), (12, "1100000"), (1, "1250000"))
    ]


def _client(
    *,
    exchange: object | None = None,
    funding: object | None = None,
    open_interest: object | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/exchangeInfo"):
            return httpx.Response(200, json=exchange if exchange is not None else _exchange_info())
        if request.url.path.endswith("/fundingRate"):
            return httpx.Response(200, json=funding if funding is not None else _funding_rows())
        if request.url.path.endswith("/openInterestHist"):
            return httpx.Response(
                200,
                json=open_interest if open_interest is not None else _open_interest_rows(),
            )
        raise AssertionError(f"Unexpected Binance request: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_binance_resolves_active_perpetual_and_orders_observations() -> None:
    with _client() as client:
        result = derivatives.fetch_derivatives_evidence(
            asset="BTC",
            timeframe="1h",
            client=client,
            collected_at=NOW,
        )

    assert result.status == "complete"
    assert result.contract_symbol == "BTCUSDT"
    assert result.venue == "Binance USD-M Futures"
    assert [item.observed_at for item in result.funding_history] == sorted(
        item.observed_at for item in result.funding_history
    )
    assert result.latest_funding_rate == pytest.approx(0.0001)
    assert result.latest_open_interest_usd == 1_250_000
    assert result.open_interest_change_24h_pct == pytest.approx(25)


def test_binance_returns_not_applicable_without_an_active_contract() -> None:
    with _client(exchange=_exchange_info(include_contract=False)) as client:
        result = derivatives.fetch_derivatives_evidence(
            asset="BTC", timeframe="1h", client=client, collected_at=NOW
        )

    assert result.status == "not_applicable"
    assert result.contract_symbol is None
    assert "no active" in result.warnings[0]


@pytest.mark.parametrize(
    ("funding", "open_interest", "warning"),
    [
        (_funding_rows(symbol="ETHUSDT"), _open_interest_rows(), "Funding-rate"),
        (_funding_rows(), _open_interest_rows(symbol="ETHUSDT"), "Open-interest"),
        (
            [*_funding_rows(), _funding_rows()[0]],
            _open_interest_rows(),
            "Funding-rate",
        ),
        (
            [{"symbol": "BTCUSDT", "fundingTime": "bad", "fundingRate": "x"}],
            _open_interest_rows(),
            "Funding-rate",
        ),
    ],
)
def test_binance_rejects_mismatched_duplicate_or_malformed_series(
    funding: object,
    open_interest: object,
    warning: str,
) -> None:
    with _client(funding=funding, open_interest=open_interest) as client:
        result = derivatives.fetch_derivatives_evidence(
            asset="BTC", timeframe="1h", client=client, collected_at=NOW
        )

    assert result.status == "partial"
    assert warning in " ".join(result.warnings)


def test_binance_excludes_future_observations_and_rejects_stale_series() -> None:
    funding = [
        {
            "symbol": "BTCUSDT",
            "fundingTime": _millis(NOW + timedelta(hours=1)),
            "fundingRate": "0.001",
        },
        {
            "symbol": "BTCUSDT",
            "fundingTime": _millis(NOW - timedelta(days=3)),
            "fundingRate": "0.001",
        },
    ]
    with _client(funding=funding) as client:
        result = derivatives.fetch_derivatives_evidence(
            asset="BTC", timeframe="1h", client=client, collected_at=NOW
        )

    assert result.status == "partial"
    assert result.funding_history == []
    assert "future-dated" in " ".join(result.warnings)
    assert "stale" in " ".join(result.warnings)


def test_binance_retries_one_transient_request() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path.endswith("/exchangeInfo"):
            return httpx.Response(200, json=_exchange_info())
        if request.url.path.endswith("/fundingRate"):
            attempts += 1
            if attempts == 1:
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json=_funding_rows())
        return httpx.Response(200, json=_open_interest_rows())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = derivatives.fetch_derivatives_evidence(
            asset="BTC", timeframe="1h", client=client, collected_at=NOW
        )

    assert attempts == 2
    assert result.status == "complete"


def test_binance_uses_stale_cache_when_live_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = TTLCache[tuple[str, str, str], DerivativesEvidence](
        300,
        clone=lambda value: value.model_copy(deep=True),
        max_entries=4,
    )
    monkeypatch.setattr(derivatives, "_DERIVATIVES_CACHE", cache)
    clients = [
        _client(),
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(503, json={"error": "unavailable"})
            )
        ),
    ]
    monkeypatch.setattr(derivatives, "make_http_client", lambda _timeout: clients.pop(0))

    live = derivatives.fetch_derivatives_evidence(asset="BTC", timeframe="1h", collected_at=NOW)
    cached = derivatives.fetch_derivatives_evidence(
        asset="BTC", timeframe="1h", collected_at=NOW + timedelta(minutes=10)
    )

    assert live.source_state == "live"
    assert cached.source_state == "cached"
    assert cached.latest_open_interest_usd == live.latest_open_interest_usd
    assert "cached snapshot" in " ".join(cached.warnings)
