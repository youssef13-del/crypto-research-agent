"""Validated, keyless Binance USD-M perpetual-futures evidence."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

import httpx

from crypto_research.domain.evidence import (
    DerivativesEvidence,
    FundingRateObservation,
    OpenInterestObservation,
)
from crypto_research.shared.time import timeframe_delta
from crypto_research.tools.cache import TTLCache
from crypto_research.tools.http import make_http_client, normalize_collection_time

DEFAULT_BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"

_DERIVATIVES_CACHE = TTLCache[tuple[str, str, str], DerivativesEvidence](
    namespace="derivatives",
    clone=lambda result: result.model_copy(deep=True),
    serialize=lambda result: result.model_dump(mode="json"),
    deserialize=DerivativesEvidence.model_validate,
)
_CONTRACT_CACHE = TTLCache[str, dict[str, str]](
    900,
    clone=dict,
)


def fetch_derivatives_evidence(
    *,
    asset: str,
    timeframe: str,
    base_url: str = DEFAULT_BINANCE_FUTURES_BASE_URL,
    timeout_seconds: float = 12,
    client: httpx.Client | None = None,
    collected_at: datetime | None = None,
) -> DerivativesEvidence:
    """Fetch public funding and open-interest observations for one asset."""

    normalized_asset = asset.split("/", maxsplit=1)[0].strip().upper()
    if not normalized_asset or not normalized_asset.isalnum():
        raise ValueError("Derivatives asset must be an alphanumeric ticker.")
    normalized_timeframe = timeframe.strip().casefold()
    timeframe_delta(normalized_timeframe)
    normalized_base = base_url.strip().rstrip("/")
    observation_time = normalize_collection_time(collected_at)
    cache_key = (normalized_base.casefold(), normalized_asset, normalized_timeframe)
    cached = _DERIVATIVES_CACHE.get(cache_key, allow_stale=True) if client is None else None

    owns_client = client is None
    http = client or make_http_client(timeout_seconds)
    try:
        try:
            contracts = _contracts(http, normalized_base, use_cache=client is None)
            contract_symbol = contracts.get(normalized_asset)
            if contract_symbol is None:
                result = DerivativesEvidence(
                    asset=normalized_asset,
                    status="not_applicable",
                    collected_at=observation_time,
                    warnings=[
                        "Binance USD-M Futures has no active USDT perpetual contract "
                        "for this asset."
                    ],
                )
            else:
                result = _fetch_contract_evidence(
                    http,
                    normalized_base,
                    asset=normalized_asset,
                    contract_symbol=contract_symbol,
                    timeframe=normalized_timeframe,
                    collected_at=observation_time,
                )
        except Exception as exc:
            result = DerivativesEvidence(
                asset=normalized_asset,
                status="unavailable",
                collected_at=observation_time,
                warnings=[f"Binance derivatives data were unavailable ({type(exc).__name__})."],
            )
        return _finalize_result(
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


def _contracts(
    http: httpx.Client,
    base_url: str,
    *,
    use_cache: bool,
) -> dict[str, str]:
    def load() -> dict[str, str]:
        payload = _request_json_with_retry(http, f"{base_url}/fapi/v1/exchangeInfo")
        if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
            raise ValueError("Binance returned invalid contract metadata.")
        contracts: dict[str, str] = {}
        valid_rows = 0
        for value in payload["symbols"]:
            if not isinstance(value, dict):
                continue
            base_asset = _text(value.get("baseAsset"))
            quote_asset = _text(value.get("quoteAsset"))
            contract_type = _text(value.get("contractType"))
            status = _text(value.get("status"))
            symbol = _text(value.get("symbol"))
            if not all((base_asset, quote_asset, contract_type, status, symbol)):
                continue
            assert base_asset is not None and symbol is not None
            valid_rows += 1
            if quote_asset == "USDT" and contract_type == "PERPETUAL" and status == "TRADING":
                if base_asset in contracts and contracts[base_asset] != symbol:
                    raise ValueError("Binance returned duplicate active contract identities.")
                contracts[base_asset] = symbol
        if not valid_rows:
            raise ValueError("Binance returned malformed contract metadata.")
        return contracts

    return _CONTRACT_CACHE.get_or_load(base_url.casefold(), load) if use_cache else load()


def _fetch_contract_evidence(
    http: httpx.Client,
    base_url: str,
    *,
    asset: str,
    contract_symbol: str,
    timeframe: str,
    collected_at: datetime,
) -> DerivativesEvidence:
    warnings: list[str] = []
    funding: list[FundingRateObservation] = []
    open_interest: list[OpenInterestObservation] = []
    try:
        payload = _request_json_with_retry(
            http,
            f"{base_url}/fapi/v1/fundingRate",
            params={"symbol": contract_symbol, "limit": 24},
        )
        funding, excluded = _parse_funding(payload, contract_symbol, cutoff=collected_at)
        if excluded:
            warnings.append(f"{excluded} future-dated funding observation(s) were excluded.")
        if funding and funding[-1].observed_at < collected_at - timedelta(hours=36):
            funding = []
            warnings.append("Binance funding observations were stale and were excluded.")
    except Exception as exc:
        warnings.append(f"Funding-rate history was unavailable ({type(exc).__name__}).")
    try:
        period = _open_interest_period(timeframe)
        payload = _request_json_with_retry(
            http,
            f"{base_url}/futures/data/openInterestHist",
            params={"symbol": contract_symbol, "period": period, "limit": 48},
        )
        open_interest, excluded = _parse_open_interest(
            payload,
            contract_symbol,
            cutoff=collected_at,
        )
        if excluded:
            warnings.append(f"{excluded} future-dated open-interest observation(s) were excluded.")
        maximum_age = max(timeframe_delta(period) * 2, timedelta(hours=3))
        if open_interest and open_interest[-1].observed_at < collected_at - maximum_age:
            open_interest = []
            warnings.append("Binance open-interest observations were stale and were excluded.")
    except Exception as exc:
        warnings.append(f"Open-interest history was unavailable ({type(exc).__name__}).")

    status = (
        "complete"
        if funding and open_interest
        else "partial"
        if funding or open_interest
        else "unavailable"
    )
    latest_funding = funding[-1].rate if funding else None
    recent_funding = [
        item.rate for item in funding if item.observed_at >= collected_at - timedelta(hours=24)
    ]
    latest_open_interest = open_interest[-1].value_usd if open_interest else None
    return DerivativesEvidence(
        asset=asset,
        contract_symbol=contract_symbol,
        status=status,
        funding_history=funding,
        open_interest_history=open_interest,
        latest_funding_rate=latest_funding,
        average_funding_rate_24h=(
            sum(recent_funding) / len(recent_funding) if recent_funding else None
        ),
        latest_open_interest_usd=latest_open_interest,
        open_interest_change_24h_pct=_open_interest_change_24h(open_interest),
        source_url=f"https://www.binance.com/en/futures/{contract_symbol}",
        collected_at=collected_at,
        warnings=list(dict.fromkeys(warnings)),
    )


def _parse_funding(
    payload: object,
    contract_symbol: str,
    *,
    cutoff: datetime,
) -> tuple[list[FundingRateObservation], int]:
    if not isinstance(payload, list):
        raise ValueError("Binance returned invalid funding history.")
    observations: list[FundingRateObservation] = []
    seen: set[datetime] = set()
    excluded = 0
    for row in payload:
        if not isinstance(row, dict) or _text(row.get("symbol")) != contract_symbol:
            raise ValueError("Binance funding history did not match the requested contract.")
        observed_at = _timestamp(row.get("fundingTime"))
        if observed_at in seen:
            raise ValueError("Binance funding history contained duplicate timestamps.")
        seen.add(observed_at)
        if observed_at > cutoff:
            excluded += 1
            continue
        observations.append(
            FundingRateObservation(observed_at=observed_at, rate=_number(row.get("fundingRate")))
        )
    observations.sort(key=lambda item: item.observed_at)
    return observations[-24:], excluded


def _parse_open_interest(
    payload: object,
    contract_symbol: str,
    *,
    cutoff: datetime,
) -> tuple[list[OpenInterestObservation], int]:
    if not isinstance(payload, list):
        raise ValueError("Binance returned invalid open-interest history.")
    observations: list[OpenInterestObservation] = []
    seen: set[datetime] = set()
    excluded = 0
    for row in payload:
        if not isinstance(row, dict) or _text(row.get("symbol")) != contract_symbol:
            raise ValueError("Binance open-interest history did not match the requested contract.")
        observed_at = _timestamp(row.get("timestamp"))
        if observed_at in seen:
            raise ValueError("Binance open-interest history contained duplicate timestamps.")
        seen.add(observed_at)
        if observed_at > cutoff:
            excluded += 1
            continue
        observations.append(
            OpenInterestObservation(
                observed_at=observed_at,
                value_usd=_nonnegative_number(row.get("sumOpenInterestValue")),
            )
        )
    observations.sort(key=lambda item: item.observed_at)
    return observations[-48:], excluded


def _open_interest_change_24h(observations: list[OpenInterestObservation]) -> float | None:
    if len(observations) < 2:
        return None
    latest = observations[-1]
    target = latest.observed_at - timedelta(hours=24)
    previous = next(
        (item for item in reversed(observations[:-1]) if item.observed_at <= target), None
    )
    if previous is None or previous.value_usd <= 0:
        return None
    return (latest.value_usd / previous.value_usd - 1) * 100


def _open_interest_period(timeframe: str) -> str:
    if timeframe in {"4h", "1d"}:
        return timeframe
    if timeframe in {"1h", "2h"}:
        return timeframe
    return "1h"


def _request_json_with_retry(
    http: httpx.Client,
    url: str,
    *,
    params: Mapping[str, str | int | float | bool | None] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = http.get(url, params=params)
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            if attempt == 0:
                continue
            break
        if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
            last_error = httpx.HTTPStatusError(
                f"Transient Binance HTTP {response.status_code}",
                request=response.request,
                response=response,
            )
            continue
        try:
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            last_error = exc
            break
    assert last_error is not None
    raise last_error


def _finalize_result(
    result: DerivativesEvidence,
    *,
    cached: DerivativesEvidence | None,
    cache_key: tuple[str, str, str],
    use_cache: bool,
    collected_at: datetime,
) -> DerivativesEvidence:
    if result.status in {"complete", "partial", "not_applicable"}:
        live = result.model_copy(update={"source_state": "live"})
        if use_cache:
            _DERIVATIVES_CACHE.set(cache_key, live)
        return live
    if cached is not None and cached.collected_at <= collected_at:
        return cached.model_copy(
            update={
                "source_state": "cached",
                "warnings": list(
                    dict.fromkeys(
                        (
                            *cached.warnings,
                            "Live Binance derivatives verification failed; a cached "
                            "snapshot is shown.",
                        )
                    )
                ),
            }
        )
    return result.model_copy(update={"source_state": "live"})


def _timestamp(value: object) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Binance returned an invalid timestamp.")
    return datetime.fromtimestamp(float(value) / 1_000, tz=UTC)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError("Binance returned an invalid numeric value.")
    result = float(value)
    if not isfinite(result):
        raise ValueError("Binance returned a non-finite numeric value.")
    return result


def _nonnegative_number(value: object) -> float:
    result = _number(value)
    if result < 0:
        raise ValueError("Binance returned a negative open-interest value.")
    return result


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


__all__ = ["DEFAULT_BINANCE_FUTURES_BASE_URL", "fetch_derivatives_evidence"]
