import httpx
import pytest

from crypto_research.tools.fundamentals import fetch_defi_evidence


def test_defillama_protocol_payload_is_normalized() -> None:
    client = _client(
        {
            "name": "Aave",
            "slug": "aave",
            "category": "Lending",
            "chains": ["Ethereum", "Arbitrum"],
            "currentChainTvls": {"Ethereum": 100.0, "Arbitrum": 50.0},
            "change_1d": 1.5,
            "change_7d": -2.0,
            "url": "https://aave.com",
        }
    )

    result = fetch_defi_evidence(protocol_slug="aave", client=client)

    assert result.status == "available"
    assert result.protocol == "Aave"
    assert result.tvl_usd == 150.0
    assert result.chains == ["Ethereum", "Arbitrum"]


def test_defillama_failure_becomes_unavailable_evidence() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = fetch_defi_evidence(protocol_slug="missing", client=client)

    assert result.status == "unavailable"
    assert result.warnings


@pytest.mark.parametrize("payload", [{}, {"name": "Aave", "chains": "Ethereum"}])
def test_defillama_malformed_success_payload_is_unavailable(payload: object) -> None:
    result = fetch_defi_evidence(protocol_slug="aave", client=_client(payload))

    assert result.status == "unavailable"
    assert result.warnings


def _client(payload: object) -> httpx.Client:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))
