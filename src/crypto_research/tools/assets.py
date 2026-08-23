from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime

import httpx

from crypto_research.domain.research import (
    COIN_ID_BY_ASSET,
    AnalysisRequest,
    AssetCandidate,
    AssetResolution,
)
from crypto_research.shared.text import normalize_text as _text
from crypto_research.tools.cache import TTLCache
from crypto_research.tools.http import make_http_client

COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"


def fetch_asset_search_payload(
    *,
    http: httpx.Client,
    query: str,
    base_url: str = COINGECKO_API_BASE_URL,
) -> object:
    response = request_asset_search(http=http, query=query, base_url=base_url)
    response.raise_for_status()
    return response.json()


def request_asset_search(
    *,
    http: httpx.Client,
    query: str,
    base_url: str = COINGECKO_API_BASE_URL,
) -> httpx.Response:
    return http.get(f"{base_url.rstrip('/')}/search", params={"query": query})


def request_contract_lookup(
    *,
    http: httpx.Client,
    network: str,
    contract_address: str,
    base_url: str = COINGECKO_API_BASE_URL,
) -> httpx.Response:
    return http.get(f"{base_url.rstrip('/')}/coins/{network}/contract/{contract_address}")


_RESOLUTION_CACHE = TTLCache[tuple[str, str | None, str | None], AssetResolution](
    namespace="asset_resolution",
    clone=lambda result: result.model_copy(deep=True),
    serialize=lambda result: result.model_dump(mode="json"),
    deserialize=AssetResolution.model_validate,
)


def _resolve_crypto_asset(
    *,
    query: str,
    api_key: str | None = None,
    network: str | None = None,
    contract_address: str | None = None,
    timeout_seconds: float = 12,
    client: httpx.Client | None = None,
) -> AssetResolution:
    """Resolve a symbol, name, CoinGecko ID, or network contract without guessing."""

    normalized_query = query.strip()
    normalized_network = network.strip().lower() if network else None
    normalized_contract = contract_address.strip() if contract_address else None
    cache_key = (normalized_query.casefold(), normalized_network, normalized_contract)
    headers = {"x-cg-demo-api-key": api_key} if api_key else None
    if client is None:

        def load() -> AssetResolution:
            http = make_http_client(timeout_seconds, headers=headers)
            try:
                return _resolve_asset(
                    http=http,
                    query=normalized_query,
                    network=normalized_network,
                    contract_address=normalized_contract,
                )
            finally:
                with suppress(Exception):
                    http.close()

        return _RESOLUTION_CACHE.get_or_load(
            cache_key,
            load,
            cache_if=_cacheable_resolution,
        )
    return _resolve_asset(
        http=client,
        query=normalized_query,
        network=normalized_network,
        contract_address=normalized_contract,
    )


def _resolve_asset(
    *,
    http: httpx.Client,
    query: str,
    network: str | None,
    contract_address: str | None,
) -> AssetResolution:
    if network and contract_address:
        return _resolve_contract(
            http=http,
            query=query,
            network=network,
            contract_address=contract_address,
        )
    return _resolve_search(http=http, query=query)


def resolve_analysis_request(
    request: AnalysisRequest,
    *,
    api_key: str | None = None,
) -> AnalysisRequest:
    """Attach a verified provider identity to a request when one can be established."""

    if request.asset_resolution is not None:
        return request
    base, quote = request.symbol.split("/", maxsplit=1)
    known_id = COIN_ID_BY_ASSET.get(base)
    if known_id is not None and request.contract_address is None:
        candidate = AssetCandidate(coin_id=known_id, symbol=base, name=known_id.replace("-", " "))
        resolution = AssetResolution(
            query=base,
            status="confirmed",
            selected=candidate,
            candidates=[candidate],
            source="local verified catalog",
            resolved_at=datetime.now(UTC),
        )
    else:
        query = request.contract_address or request.asset_query or request.coin_id or base
        resolution = _resolve_crypto_asset(
            query=query,
            api_key=api_key,
            network=request.network,
            contract_address=request.contract_address,
        )

    selected = resolution.selected
    updates: dict[str, object] = {"asset_resolution": resolution}
    if selected is not None:
        updates.update(
            {
                "coin_id": selected.coin_id,
                "symbol": f"{selected.symbol}/{quote}",
                "network": selected.network or request.network,
                "contract_address": selected.contract_address or request.contract_address,
            }
        )
    return request.model_copy(update=updates)


def _resolve_contract(
    *,
    http: httpx.Client,
    query: str,
    network: str,
    contract_address: str,
) -> AssetResolution:
    resolved_at = datetime.now(UTC)
    try:
        response = request_contract_lookup(
            http=http,
            network=network,
            contract_address=contract_address,
            base_url=COINGECKO_API_BASE_URL,
        )
        if response.status_code == 404:
            return AssetResolution(
                query=query,
                status="not_found",
                source="CoinGecko contract lookup",
                resolved_at=resolved_at,
            )
        response.raise_for_status()
        payload = response.json()
        candidate = _candidate_from_coin_payload(
            payload,
            network=network,
            contract_address=contract_address,
        )
    except Exception as exc:
        return AssetResolution(
            query=query,
            status="unavailable",
            source="CoinGecko contract lookup",
            resolved_at=resolved_at,
            warnings=[f"Contract identity could not be confirmed ({type(exc).__name__})."],
        )
    return AssetResolution(
        query=query,
        status="confirmed",
        selected=candidate,
        candidates=[candidate],
        source="CoinGecko contract lookup",
        resolved_at=resolved_at,
    )


def _resolve_search(*, http: httpx.Client, query: str) -> AssetResolution:
    resolved_at = datetime.now(UTC)
    try:
        payload = fetch_asset_search_payload(
            http=http,
            query=query,
            base_url=COINGECKO_API_BASE_URL,
        )
    except Exception as exc:
        return AssetResolution(
            query=query,
            status="unavailable",
            resolved_at=resolved_at,
            warnings=[f"Asset provider was unavailable ({type(exc).__name__})."],
        )

    if not isinstance(payload, Mapping) or "coins" not in payload:
        return _invalid_search_payload(query, resolved_at=resolved_at)
    root = payload
    rows = root.get("coins")
    if not isinstance(rows, list):
        return _invalid_search_payload(query, resolved_at=resolved_at)
    search_rows = rows
    candidates = [
        candidate
        for row in search_rows[:20]
        if (candidate := _candidate_from_search_row(row)) is not None
    ][:5]
    if not candidates:
        if search_rows:
            return _invalid_search_payload(query, resolved_at=resolved_at)
        return AssetResolution(query=query, status="not_found", resolved_at=resolved_at)

    folded = query.casefold().strip()
    exact_names = [
        item for item in candidates if folded in {item.name.casefold(), item.coin_id.casefold()}
    ]
    exact_symbols = [item for item in candidates if folded == item.symbol.casefold()]
    exact = exact_names or exact_symbols
    if len(exact) == 1:
        return AssetResolution(
            query=query,
            status="confirmed",
            selected=exact[0],
            candidates=candidates,
            resolved_at=resolved_at,
        )
    return AssetResolution(
        query=query,
        status="ambiguous",
        candidates=exact or candidates,
        resolved_at=resolved_at,
        warnings=["Multiple assets matched; a name, network, or contract address is required."],
    )


def _candidate_from_search_row(value: object) -> AssetCandidate | None:
    if not isinstance(value, Mapping):
        return None
    coin_id = _text(value.get("id"))
    symbol = _text(value.get("symbol"))
    name = _text(value.get("name"))
    if not coin_id or not symbol or not name:
        return None
    rank_value = value.get("market_cap_rank")
    rank = rank_value if isinstance(rank_value, int) and not isinstance(rank_value, bool) else None
    return AssetCandidate(coin_id=coin_id, symbol=symbol, name=name, market_cap_rank=rank)


def _candidate_from_coin_payload(
    value: object,
    *,
    network: str,
    contract_address: str,
) -> AssetCandidate:
    if not isinstance(value, Mapping):
        raise ValueError("Coin provider returned an invalid payload.")
    coin_id = _text(value.get("id"))
    symbol = _text(value.get("symbol"))
    name = _text(value.get("name"))
    if not coin_id or not symbol or not name:
        raise ValueError("Coin provider omitted identity fields.")
    rank_value = value.get("market_cap_rank")
    rank = rank_value if isinstance(rank_value, int) and not isinstance(rank_value, bool) else None
    return AssetCandidate(
        coin_id=coin_id,
        symbol=symbol,
        name=name,
        market_cap_rank=rank,
        network=network,
        contract_address=contract_address,
    )


def _cacheable_resolution(result: AssetResolution) -> bool:
    """Do not preserve transient provider failures as long-lived lookup results."""
    return result.status != "unavailable" and (result.status != "not_found" or not result.warnings)


def _invalid_search_payload(query: str, *, resolved_at: datetime) -> AssetResolution:
    return AssetResolution(
        query=query,
        status="unavailable",
        resolved_at=resolved_at,
        warnings=["Asset provider returned an unexpected payload."],
    )
