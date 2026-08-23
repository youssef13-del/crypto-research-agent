"""CoinGecko fundamentals and DefiLlama protocol evidence."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from math import isfinite

import httpx
from pydantic import ValidationError

from crypto_research.domain.evidence import DeveloperActivity
from crypto_research.domain.research import COIN_ID_BY_ASSET, DefiEvidence, FundamentalEvidence
from crypto_research.shared.text import normalize_text as optional_clean_text
from crypto_research.tools.assets import COINGECKO_API_BASE_URL
from crypto_research.tools.cache import TTLCache
from crypto_research.tools.http import make_http_client, normalize_collection_time


def request_coin_detail(
    *,
    http: httpx.Client,
    coin_id: str,
    api_key: str | None,
    base_url: str = COINGECKO_API_BASE_URL,
) -> httpx.Response:
    headers = {"x-cg-demo-api-key": api_key} if api_key else None
    return http.get(
        f"{base_url.rstrip('/')}/coins/{coin_id}",
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "true",
        },
        headers=headers,
    )


DEFAULT_DEFILLAMA_BASE_URL = "https://api.llama.fi"

# DefiLlama protocol slugs for the supported assets that are themselves DeFi protocols.
# Assets without an entry are not applicable: their coin id is not a DefiLlama protocol slug.
DEFILLAMA_PROTOCOL_SLUGS: dict[str, str] = {
    "aave": "aave",
    "uniswap": "uniswap",
    "curve-dao-token": "curve",
    "lido-dao": "lido",
    "maker": "makerdao",
    "compound-governance-token": "compound",
    "sushi": "sushiswap",
    "yearn-finance": "yearn",
    "balancer": "balancer",
    "pancakeswap-token": "pancakeswap",
    "the-graph": "the-graph",
    "wrapped-bitcoin": "wrapped-bitcoin",
}


def defillama_slug_for(coin_id: str | None) -> str | None:
    """Return the DefiLlama protocol slug for a coin id, if one is registered."""
    if coin_id is None:
        return None
    return DEFILLAMA_PROTOCOL_SLUGS.get(coin_id.strip().lower())


_DEFI_CACHE = TTLCache[tuple[str, str], DefiEvidence](
    namespace="defi",
    clone=lambda result: result.model_copy(deep=True),
    serialize=lambda result: result.model_dump(mode="json"),
    deserialize=DefiEvidence.model_validate,
)


def fetch_defi_evidence(
    *,
    protocol_slug: str,
    base_url: str = DEFAULT_DEFILLAMA_BASE_URL,
    timeout_seconds: float = 12,
    client: httpx.Client | None = None,
    collected_at: datetime | None = None,
) -> DefiEvidence:
    slug = protocol_slug.strip().lower()
    normalized_base_url = base_url.rstrip("/")
    observation_time = normalize_collection_time(collected_at)
    cache_key = (normalized_base_url.casefold(), slug)
    cached = _DEFI_CACHE.get(cache_key, allow_stale=True) if client is None else None
    owns_client = client is None
    http = client or make_http_client(timeout_seconds)
    try:
        result: DefiEvidence | None = None
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = http.get(f"{normalized_base_url}/protocol/{slug}")
                response.raise_for_status()
                result = _parse_protocol(response.json(), slug=slug, collected_at=observation_time)
            except Exception as exc:
                last_error = exc
                continue
            break
        if result is None:
            result = DefiEvidence(
                slug=slug,
                status="unavailable",
                collected_at=observation_time,
                warnings=[
                    "DefiLlama protocol data was unavailable "
                    f"({type(last_error).__name__ if last_error else 'unknown'})."
                ],
            )
        return _finalize_defi_result(
            result,
            cached=cached,
            cache_key=cache_key,
            use_cache=client is None,
            collected_at=observation_time,
        )
    finally:
        if owns_client:
            with suppress(Exception):
                http.close()


def _parse_protocol(
    value: object,
    *,
    slug: str,
    collected_at: datetime | None = None,
) -> DefiEvidence:
    if not isinstance(value, Mapping):
        raise ValueError("DefiLlama returned an invalid payload.")
    name = optional_clean_text(value.get("name"))
    provider_slug = optional_clean_text(value.get("slug"))
    if name is None and provider_slug is None:
        raise ValueError("DefiLlama omitted the protocol identity.")
    raw_chains = value.get("chains", [])
    if not isinstance(raw_chains, list):
        raise ValueError("DefiLlama returned an invalid chains field.")
    chains = [item for item in raw_chains if isinstance(item, str) and item.strip()]
    return DefiEvidence(
        protocol=name or provider_slug,
        slug=provider_slug or slug,
        category=optional_clean_text(value.get("category")),
        chains=chains,
        tvl_usd=_current_tvl(value),
        change_1d=_number(value.get("change_1d")),
        change_7d=_number(value.get("change_7d")),
        homepage=optional_clean_text(value.get("url")),
        collected_at=collected_at or datetime.now(UTC),
    )


def _finalize_defi_result(
    result: DefiEvidence,
    *,
    cached: DefiEvidence | None,
    cache_key: tuple[str, str],
    use_cache: bool,
    collected_at: datetime,
) -> DefiEvidence:
    """Persist validated live protocol data and mark only fallback cache as cached."""

    if result.status == "available":
        live = result.model_copy(update={"source_state": "live"})
        if use_cache:
            _DEFI_CACHE.set(cache_key, live)
        return live
    if cached is not None and cached.collected_at <= collected_at:
        return cached.model_copy(
            update={
                "source_state": "cached",
                "warnings": list(
                    dict.fromkeys(
                        (
                            *cached.warnings,
                            "Live DeFi verification failed; a fresh cached snapshot is shown.",
                        )
                    )
                ),
            }
        )
    return result.model_copy(update={"source_state": "live"})


def _current_tvl(value: Mapping[object, object]) -> float | None:
    direct = _number(value.get("tvl"))
    if direct is not None:
        return max(0, direct)
    history = value.get("tvl")
    if isinstance(history, list) and history and isinstance(history[-1], Mapping):
        historical = _number(history[-1].get("totalLiquidityUSD"))
        if historical is not None:
            return max(0, historical)
    chain_tvls = value.get("currentChainTvls")
    if isinstance(chain_tvls, Mapping):
        numbers = [_number(item) for item in chain_tvls.values()]
        usable = [item for item in numbers if item is not None and item >= 0]
        return sum(usable) if usable else None
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if isfinite(result) else None


_FUNDAMENTALS_CACHE_TTL = 1_800.0
_fundamentals_cache = TTLCache[str, FundamentalEvidence](
    _FUNDAMENTALS_CACHE_TTL,
    namespace="fundamentals",
    clone=lambda result: result.model_copy(deep=True),
    serialize=lambda result: result.model_dump(mode="json"),
    deserialize=FundamentalEvidence.model_validate,
)


class _ProviderPayloadError(ValueError):
    """Raised when CoinGecko returns a structurally invalid success payload."""


def fetch_fundamental_evidence(
    *,
    symbol: str,
    coin_id: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 20,
    client: httpx.Client | None = None,
    collected_at: datetime | None = None,
) -> FundamentalEvidence:
    cache_key = f"{symbol}:{coin_id}"
    observation_time = normalize_collection_time(collected_at)
    cached = _fundamentals_cache.get(cache_key, allow_stale=True) if client is None else None

    resolved_ids = _coin_id_candidates(symbol=symbol, coin_id=coin_id)
    owns_client = client is None
    http = client or make_http_client(timeout_seconds)
    try:
        response = None
        missing_ids: list[str] = []
        for resolved_id in resolved_ids:
            response, request_error = _request_coin_detail_with_retry(
                http=http,
                coin_id=resolved_id,
                api_key=api_key,
            )
            if response is None:
                return _finalize_fundamental_result(
                    _unavailable(
                        "CoinGecko request failed "
                        f"({type(request_error).__name__ if request_error else 'unknown'}); "
                        "fundamentals are unavailable.",
                        collected_at=observation_time,
                    ),
                    cached=cached,
                    cache_key=cache_key,
                    use_cache=client is None,
                    collected_at=observation_time,
                )
            if response.status_code == 404:
                missing_ids.append(resolved_id)
                continue
            if response.status_code >= 400:
                return _finalize_fundamental_result(
                    _unavailable(
                        f"CoinGecko HTTP {response.status_code}", collected_at=observation_time
                    ),
                    cached=cached,
                    cache_key=cache_key,
                    use_cache=client is None,
                    collected_at=observation_time,
                )
            break
        if response is None or response.status_code == 404:
            missing = ", ".join(missing_ids or resolved_ids)
            return _finalize_fundamental_result(
                _unavailable(
                    f"CoinGecko could not find fundamentals for {missing}; "
                    "fundamentals are unavailable.",
                    collected_at=observation_time,
                ),
                cached=cached,
                cache_key=cache_key,
                use_cache=client is None,
                collected_at=observation_time,
            )
        try:
            payload = response.json()
        except (httpx.HTTPError, TypeError, ValueError):  # fmt: skip
            return _finalize_fundamental_result(
                _unavailable(
                    "CoinGecko returned invalid JSON; fundamentals are unavailable.",
                    collected_at=observation_time,
                ),
                cached=cached,
                cache_key=cache_key,
                use_cache=client is None,
                collected_at=observation_time,
            )
        try:
            result = _parse_payload(payload, symbol, collected_at=observation_time)
            return _finalize_fundamental_result(
                result,
                cached=cached,
                cache_key=cache_key,
                use_cache=client is None,
                collected_at=observation_time,
            )
        except (_ProviderPayloadError, ValidationError):  # fmt: skip
            return _finalize_fundamental_result(
                _unavailable(
                    "CoinGecko returned an unexpected provider payload; "
                    "fundamentals are unavailable.",
                    collected_at=observation_time,
                ),
                cached=cached,
                cache_key=cache_key,
                use_cache=client is None,
                collected_at=observation_time,
            )
    finally:
        if owns_client:
            with suppress(Exception):
                http.close()


def _finalize_fundamental_result(
    result: FundamentalEvidence,
    *,
    cached: FundamentalEvidence | None,
    cache_key: str,
    use_cache: bool,
    collected_at: datetime,
) -> FundamentalEvidence:
    """Prefer a validated live result; use cache only after its verification fails."""

    if result.status == "available":
        live = result.model_copy(update={"source_state": "live"})
        if use_cache:
            _fundamentals_cache.set(cache_key, live)
        return live
    if cached is not None and cached.collected_at <= collected_at:
        return cached.model_copy(
            update={
                "source_state": "cached",
                "warnings": list(
                    dict.fromkeys(
                        (
                            *cached.warnings,
                            "Live fundamentals verification failed; "
                            "a fresh cached snapshot is shown.",
                        )
                    )
                ),
            }
        )
    return result.model_copy(update={"source_state": "live"})


def _request_coin_detail_with_retry(
    *,
    http: httpx.Client,
    coin_id: str,
    api_key: str | None,
) -> tuple[httpx.Response | None, Exception | None]:
    """Retry one transient CoinGecko failure without changing the selected asset.

    When a demo/rate-limited key is rejected (401/403/429), retry the same
    request once without the key header against CoinGecko's keyless public
    endpoint so fundamentals remain available.
    """

    for attempt in range(2):
        try:
            response = request_coin_detail(http=http, coin_id=coin_id, api_key=api_key)
        except (httpx.HTTPError, OSError) as exc:
            if attempt == 1:
                return None, exc
            continue
        if attempt == 0 and response.status_code in {401, 403, 429} and api_key:
            # A demo/rate-limited key can be rejected; retry once keyless.
            try:
                keyless = request_coin_detail(http=http, coin_id=coin_id, api_key=None)
            except (httpx.HTTPError, OSError) as exc:
                return None, exc
            if keyless.status_code < 400:
                return keyless, None
            continue
        if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
            continue
        return response, None
    return None, None


def _coin_id_candidates(*, symbol: str, coin_id: str | None) -> tuple[str, ...]:
    asset = symbol.split("/", maxsplit=1)[0].strip().upper()
    mapped = COIN_ID_BY_ASSET.get(asset, asset.lower())
    if coin_id is None:
        return (mapped,)
    normalized = coin_id.strip().lower()
    if not normalized or normalized.upper() == asset:
        return (mapped,)
    return tuple(dict.fromkeys((normalized, mapped)))


def _parse_payload(
    payload: object,
    requested_market: str,
    *,
    collected_at: datetime | None = None,
) -> FundamentalEvidence:
    root = _mapping(payload)
    provider_symbol = root.get("symbol")
    if not isinstance(provider_symbol, str) or not provider_symbol.strip():
        raise _ProviderPayloadError
    requested_symbol = requested_market.split("/", maxsplit=1)[0].lower()
    if provider_symbol.lower() != requested_symbol:
        return _unavailable(
            "CoinGecko asset did not match the requested market symbol; "
            "fundamentals were discarded.",
            collected_at=collected_at,
        )

    market = _mapping(root.get("market_data"))
    links_value = root.get("links")
    links = {} if links_value is None else _mapping(links_value)
    homepages = _string_list(links.get("homepage"))
    homepage = next((item for item in homepages if item), None)

    return FundamentalEvidence(
        name=_optional_string(root.get("name")),
        symbol=provider_symbol,
        market_cap=_usd_market_value(market, "market_cap"),
        rank=_optional_integer(root.get("market_cap_rank")),
        circulating_supply=_optional_number(market.get("circulating_supply")),
        total_supply=_optional_number(market.get("total_supply")),
        max_supply=_optional_number(market.get("max_supply")),
        categories=_string_list(root.get("categories")),
        homepage=homepage,
        genesis_date=_optional_string(root.get("genesis_date")),
        developer_activity=_developer_activity(root),
        status="available",
        warnings=[],
        collected_at=collected_at or datetime.now(UTC),
    )


def _developer_activity(root: Mapping[str, object]) -> DeveloperActivity | None:
    raw = root.get("developer_data")
    if not isinstance(raw, Mapping):
        return None
    values = {
        "forks": _optional_nonnegative_count(raw.get("forks")),
        "stars": _optional_nonnegative_count(raw.get("stars")),
        "contributors": _optional_nonnegative_count(raw.get("pull_request_contributors")),
        "merged_pull_requests": _optional_nonnegative_count(raw.get("pull_requests_merged")),
        "commits_4_weeks": _optional_nonnegative_count(raw.get("commit_count_4_weeks")),
        "provider_updated_at": _optional_provider_time(root.get("last_updated")),
    }
    if all(value is None for value in values.values()):
        return None
    return DeveloperActivity.model_validate(values)


def _optional_nonnegative_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not numeric.is_integer() or numeric < 0:
        return None
    return int(numeric)


def _optional_provider_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return None
    return parsed.astimezone(UTC)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise _ProviderPayloadError
    return value


def _usd_market_value(market: Mapping[str, object], field: str) -> int | float | None:
    values = market.get(field)
    if values is None:
        return None
    return _optional_number(_mapping(values).get("usd"))


def _optional_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _ProviderPayloadError
    return value


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ProviderPayloadError
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _ProviderPayloadError
    return value


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _ProviderPayloadError
    return value


def _unavailable(
    warning: str,
    *,
    collected_at: datetime | None = None,
) -> FundamentalEvidence:
    return FundamentalEvidence(
        status="unavailable",
        collected_at=collected_at or datetime.now(UTC),
        warnings=[warning],
    )
