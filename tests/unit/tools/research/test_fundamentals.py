from typing import Any, cast

from tests.support.services import (
    CapturingFundamentalClient,
    FundamentalClient,
    FundamentalResponse,
)

from crypto_research.tools.fundamentals import fetch_fundamental_evidence


def test_fundamentals_discard_mismatched_coingecko_asset() -> None:
    result = fetch_fundamental_evidence(
        symbol="BTC/USD",
        coin_id="ethereum",
        client=cast(Any, FundamentalClient({"symbol": "eth"})),
    )

    assert result.status == "unavailable"
    assert result.warnings == [
        "CoinGecko asset did not match the requested market symbol; fundamentals were discarded."
    ]


def test_fundamentals_map_ticker_symbol_to_coingecko_coin_id() -> None:
    payload: dict[str, object] = {"symbol": "btc", "market_data": {}}
    client = CapturingFundamentalClient(payload)

    result = fetch_fundamental_evidence(
        symbol="BTC/USD",
        coin_id=None,
        client=cast(Any, client),
    )

    assert result.status == "available"
    assert client.urls == ["https://api.coingecko.com/api/v3/coins/bitcoin"]


def test_fundamentals_retry_known_coin_id_after_bad_slug_404() -> None:
    client = CapturingFundamentalClient(
        {
            "https://api.coingecko.com/api/v3/coins/btc": FundamentalResponse({}, status_code=404),
            "https://api.coingecko.com/api/v3/coins/bitcoin": FundamentalResponse(
                {"symbol": "btc", "market_data": {}},
                status_code=200,
            ),
        }
    )

    result = fetch_fundamental_evidence(
        symbol="BTC/USD",
        coin_id="btc",
        client=cast(Any, client),
    )

    assert result.status == "available"
    assert client.urls == ["https://api.coingecko.com/api/v3/coins/bitcoin"]


def test_fundamentals_parse_optional_developer_activity() -> None:
    payload: dict[str, object] = {
        "symbol": "btc",
        "market_data": {},
        "last_updated": "2026-08-14T11:30:00Z",
        "developer_data": {
            "forks": 42,
            "stars": 1_234,
            "pull_request_contributors": 18,
            "pull_requests_merged": 77,
            "commit_count_4_weeks": 56,
        },
    }

    result = fetch_fundamental_evidence(
        symbol="BTC/USD",
        coin_id="bitcoin",
        client=cast(Any, FundamentalClient(payload)),
    )

    assert result.developer_activity is not None
    assert result.developer_activity.commits_4_weeks == 56
    assert result.developer_activity.stars == 1_234
    assert result.developer_activity.forks == 42
    assert result.developer_activity.contributors == 18
    assert result.developer_activity.merged_pull_requests == 77
    assert result.developer_activity.provider_updated_at is not None


def test_fundamentals_ignore_missing_or_malformed_optional_developer_fields() -> None:
    payload: dict[str, object] = {
        "symbol": "btc",
        "market_data": {},
        "last_updated": "not-a-timestamp",
        "developer_data": {
            "forks": -1,
            "stars": "many",
            "pull_request_contributors": True,
            "pull_requests_merged": 1.5,
        },
    }

    result = fetch_fundamental_evidence(
        symbol="BTC/USD",
        coin_id="bitcoin",
        client=cast(Any, FundamentalClient(payload)),
    )

    assert result.status == "available"
    assert result.developer_activity is None
