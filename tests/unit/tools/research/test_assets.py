from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

import crypto_research.tools.assets as resolution_service
from crypto_research.domain.research import AnalysisRequest, AssetResolution
from crypto_research.tools.assets import (
    _cacheable_resolution,
    resolve_analysis_request,
)
from crypto_research.tools.assets import (
    _resolve_crypto_asset as resolve_crypto_asset,
)


def test_asset_resolver_confirms_exact_project_name() -> None:
    client = _client(
        {
            "coins": [
                {
                    "id": "uniswap",
                    "symbol": "uni",
                    "name": "Uniswap",
                    "market_cap_rank": 25,
                },
                {
                    "id": "uniswap-wormhole",
                    "symbol": "uni",
                    "name": "Uniswap (Wormhole)",
                    "market_cap_rank": None,
                },
            ]
        }
    )

    result = resolve_crypto_asset(query="Uniswap", client=client)

    assert result.status == "confirmed"
    assert result.selected is not None
    assert result.selected.coin_id == "uniswap"
    assert result.selected.symbol == "UNI"
    assert result.resolved_at.tzinfo is UTC


def test_asset_resolver_refuses_ambiguous_ticker() -> None:
    client = _client(
        {
            "coins": [
                {"id": "alpha-one", "symbol": "abc", "name": "Alpha One"},
                {"id": "alpha-two", "symbol": "abc", "name": "Alpha Two"},
            ]
        }
    )

    result = resolve_crypto_asset(query="ABC", client=client)

    assert result.status == "ambiguous"
    assert result.selected is None
    assert len(result.candidates) == 2


def test_known_asset_request_resolves_without_network() -> None:
    request = AnalysisRequest(user_intent="Review ETH", symbol="ETH/USD", coin_id="ethereum")

    result = resolve_analysis_request(request)

    assert result.asset_resolution is not None
    assert result.asset_resolution.status == "confirmed"
    assert result.coin_id == "ethereum"


def test_confirmed_asset_resolution_requires_a_selected_candidate() -> None:
    with pytest.raises(ValidationError, match="requires a selected candidate"):
        AssetResolution(
            query="Bitcoin",
            status="confirmed",
            resolved_at=datetime.now(UTC),
        )


def test_semantic_asset_query_is_used_for_dynamic_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def resolver(**values: object) -> AssetResolution:
        captured.append(str(values["query"]))
        return AssetResolution(
            query=str(values["query"]),
            status="not_found",
            resolved_at=datetime.now(UTC),
        )

    monkeypatch.setattr(resolution_service, "_resolve_crypto_asset", resolver)
    request = AnalysisRequest(
        user_intent="Research PayPal USD",
        asset_query="PayPal USD",
        symbol="TOKEN/USD",
        coin_id=None,
    )

    resolve_analysis_request(request)

    assert captured == ["PayPal USD"]


def test_transient_asset_resolution_failures_are_not_cacheable() -> None:
    transient_failure = AssetResolution(
        query="new token",
        status="not_found",
        warnings=["Asset provider was unavailable (TimeoutError)."],
        resolved_at=datetime.now(UTC),
    )
    verified_miss = AssetResolution(
        query="not-a-real-token",
        status="not_found",
        resolved_at=datetime.now(UTC),
    )

    assert _cacheable_resolution(transient_failure) is False
    assert _cacheable_resolution(verified_miss) is True


@pytest.mark.parametrize("payload", [[], {}, {"coins": {}}, {"coins": [{"id": "broken"}]}])
def test_malformed_search_payload_is_unavailable_and_not_cacheable(payload: object) -> None:
    result = resolve_crypto_asset(query="new token", client=_client(payload))

    assert result.status == "unavailable"
    assert result.warnings == ["Asset provider returned an unexpected payload."]
    assert _cacheable_resolution(result) is False


def test_verified_search_miss_remains_distinct_from_provider_failure() -> None:
    result = resolve_crypto_asset(query="not-a-real-token", client=_client({"coins": []}))

    assert result.status == "not_found"
    assert result.warnings == []
    assert _cacheable_resolution(result) is True


def _client(payload: object) -> httpx.Client:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))
