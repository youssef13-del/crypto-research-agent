from __future__ import annotations

import httpx

from crypto_research.tools.assets import (
    fetch_asset_search_payload,
    request_contract_lookup,
)
from crypto_research.tools.fundamentals import request_coin_detail


def test_asset_search_requests_the_provider_search_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"coins": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = fetch_asset_search_payload(
            http=client,
            query="PayPal USD",
            base_url="https://provider.test/api/v3/",
        )

    assert payload == {"coins": []}
    assert str(requests[0].url) == "https://provider.test/api/v3/search?query=PayPal+USD"


def test_contract_lookup_preserves_not_found_response_for_domain_handling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = request_contract_lookup(
            http=client,
            network="ethereum",
            contract_address="0xabc",
            base_url="https://provider.test/api/v3/",
        )

    assert response.status_code == 404
    assert str(response.request.url) == (
        "https://provider.test/api/v3/coins/ethereum/contract/0xabc"
    )


def test_coin_detail_applies_provider_parameters_and_demo_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"symbol": "btc"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = request_coin_detail(
            http=client,
            coin_id="bitcoin",
            api_key="demo-key",
            base_url="https://provider.test/api/v3/",
        )

    assert response.status_code == 200
    assert requests[0].headers["x-cg-demo-api-key"] == "demo-key"
    assert dict(requests[0].url.params) == {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "true",
    }
