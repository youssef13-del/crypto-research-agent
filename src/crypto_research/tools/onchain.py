"""Coin Metrics Community network-activity evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from statistics import fmean
from typing import Literal

import httpx

from crypto_research.domain.evidence import (
    OnChainEvidence,
    OnChainMetricId,
    OnChainMetricSeries,
    OnChainObservation,
)
from crypto_research.domain.research import AnalysisAsset
from crypto_research.tools.cache import TTLCache
from crypto_research.tools.http import make_http_client

DEFAULT_COINMETRICS_BASE_URL = "https://community-api.coinmetrics.io/v4"
_METRICS: tuple[OnChainMetricId, ...] = (
    "AdrActCnt",
    "AdrNewCnt",
    "TxCnt",
    "TxTfrValAdjUSD",
    "FeeTotUSD",
)
_PROVIDER_METRICS = (*_METRICS, "FeeTotNtv", "PriceUSD")
_METRIC_DETAILS: dict[OnChainMetricId, tuple[str, Literal["count", "usd"]]] = {
    "AdrActCnt": ("Active addresses", "count"),
    "AdrNewCnt": ("New addresses", "count"),
    "TxCnt": ("Transactions", "count"),
    "TxTfrValAdjUSD": ("Adjusted transfer value", "usd"),
    "FeeTotUSD": ("Network fees", "usd"),
}
_ASSET_IDS = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "solana": "sol",
    "ripple": "xrp",
    "cardano": "ada",
    "dogecoin": "doge",
    "polkadot": "dot",
    "avalanche-2": "avax",
    "chainlink": "link",
    "litecoin": "ltc",
    "aave": "aave",
    "usd-coin": "usdc",
}
_CACHE = TTLCache[tuple[str, str], OnChainEvidence](
    namespace="onchain",
    clone=lambda item: item.model_copy(deep=True),
    max_entries=64,
    serialize=lambda item: item.model_dump(mode="json"),
    deserialize=OnChainEvidence.model_validate,
)
_FRACTION = re.compile(r"(\.\d{6})\d+(?=Z$)")


def fetch_onchain_evidence(
    *,
    asset: AnalysisAsset,
    base_url: str = DEFAULT_COINMETRICS_BASE_URL,
    collected_at: datetime | None = None,
    client: httpx.Client | None = None,
) -> OnChainEvidence:
    """Fetch and normalize 30 daily network-activity observations for one verified asset."""

    collected = _utc(collected_at or datetime.now(UTC))
    provider_asset = _ASSET_IDS.get(asset.coin_id or "")
    if provider_asset is None:
        return OnChainEvidence(
            asset=asset.symbol,
            status="not_applicable",
            collected_at=collected,
            warnings=["Coin Metrics Community has no verified mapping for this asset."],
        )

    normalized_base = base_url.strip().rstrip("/")
    cache_key = (normalized_base.casefold(), provider_asset)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached.model_copy(update={"source_state": "cached"}, deep=True)
    stale = _CACHE.get(cache_key, allow_stale=True)
    if _CACHE.failure_cached(cache_key):
        if stale is not None:
            return _cached_after_failure(stale)
        return OnChainEvidence(
            asset=asset.symbol,
            provider_asset=provider_asset,
            status="unavailable",
            source_url=normalized_base,
            collected_at=collected,
            warnings=["Coin Metrics is temporarily paused after a recent provider failure."],
        )

    owns_client = client is None
    http = client or make_http_client(10)
    try:
        available_metrics = _available_daily_metrics(
            http,
            base_url=normalized_base,
            provider_asset=provider_asset,
            collected_at=collected,
        )
        if not available_metrics:
            return OnChainEvidence(
                asset=asset.symbol,
                provider_asset=provider_asset,
                status="unavailable",
                source_url=normalized_base,
                collected_at=collected,
                warnings=[
                    "Coin Metrics Community does not provide daily network-activity "
                    "metrics for this asset."
                ],
            )
        response = http.get(
            f"{normalized_base}/timeseries/asset-metrics",
            params={
                "assets": provider_asset,
                "metrics": ",".join(available_metrics),
                "frequency": "1d",
                "start_time": (collected - timedelta(days=35)).date().isoformat(),
                "end_time": collected.date().isoformat(),
                "page_size": 100,
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError("Coin Metrics returned an invalid data payload.")
        series = _metric_series(_derive_usd_fees(rows), collected_at=collected)
        warnings: list[str] = []
        missing = [metric for metric in _METRICS if metric not in {item.metric for item in series}]
        if missing:
            warnings.append("Unavailable community metrics: " + ", ".join(missing) + ".")
        evidence = OnChainEvidence(
            asset=asset.symbol,
            provider_asset=provider_asset,
            status="complete" if len(series) >= 3 else "partial" if series else "unavailable",
            metrics=series,
            source_url=normalized_base,
            collected_at=collected,
            warnings=warnings
            or (["Coin Metrics returned no usable daily metrics."] if not series else []),
        )
        if evidence.metrics:
            _CACHE.set(cache_key, evidence)
        return evidence
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        _CACHE.remember_failure(cache_key)
        if stale is not None:
            return _cached_after_failure(stale)
        return OnChainEvidence(
            asset=asset.symbol,
            provider_asset=provider_asset,
            status="unavailable",
            source_url=normalized_base,
            collected_at=collected,
            warnings=[_provider_failure_message(exc)],
        )
    finally:
        if owns_client:
            http.close()


def _cached_after_failure(evidence: OnChainEvidence) -> OnChainEvidence:
    return evidence.model_copy(
        update={
            "source_state": "cached",
            "warnings": list(
                dict.fromkeys(
                    (
                        *evidence.warnings,
                        "Live Coin Metrics verification failed; a recent cached snapshot is shown.",
                    )
                )
            ),
        },
        deep=True,
    )


def _available_daily_metrics(
    http: httpx.Client,
    *,
    base_url: str,
    provider_asset: str,
    collected_at: datetime,
) -> tuple[str, ...]:
    response = http.get(
        f"{base_url}/catalog-v2/asset-metrics",
        params={
            "assets": provider_asset,
            "metrics": ",".join(_PROVIDER_METRICS),
        },
    )
    response.raise_for_status()
    payload = response.json()
    assets = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(assets, list):
        raise ValueError("Coin Metrics returned an invalid catalog payload.")

    available: set[str] = set()
    for asset in assets:
        if not isinstance(asset, Mapping) or asset.get("asset") != provider_asset:
            continue
        metrics = asset.get("metrics")
        if not isinstance(metrics, list):
            continue
        for item in metrics:
            if not isinstance(item, Mapping) or item.get("metric") not in _PROVIDER_METRICS:
                continue
            frequencies = item.get("frequencies")
            if isinstance(frequencies, list) and any(
                isinstance(frequency, Mapping)
                and frequency.get("frequency") == "1d"
                and frequency.get("community", True) is True
                and _catalog_frequency_is_current(frequency, collected_at=collected_at)
                for frequency in frequencies
            ):
                available.add(str(item["metric"]))
    network_metrics = set(_METRICS) | {"FeeTotNtv"}
    if not available.intersection(network_metrics):
        return ()
    return tuple(metric for metric in _PROVIDER_METRICS if metric in available)


def _catalog_frequency_is_current(
    frequency: Mapping[object, object],
    *,
    collected_at: datetime,
) -> bool:
    max_time = frequency.get("max_time")
    if max_time is None:
        return True
    timestamp = _timestamp(max_time)
    return timestamp is not None and collected_at - timestamp <= timedelta(days=3)


def _derive_usd_fees(rows: list[object]) -> list[object]:
    normalized: list[object] = []
    for row in rows:
        if not isinstance(row, Mapping) or _number(row.get("FeeTotUSD")) is not None:
            normalized.append(row)
            continue
        native_fees = _number(row.get("FeeTotNtv"))
        price = _number(row.get("PriceUSD"))
        if native_fees is None or price is None:
            normalized.append(row)
            continue
        derived = dict(row)
        derived["FeeTotUSD"] = native_fees * price
        normalized.append(derived)
    return normalized


def _provider_failure_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 429:
            return "Coin Metrics Community rate limit was reached; try again shortly."
        return f"Coin Metrics Community is temporarily unavailable (HTTP {status_code})."
    return "Coin Metrics Community returned unusable data; on-chain metrics are unavailable."


def _metric_series(
    rows: list[object],
    *,
    collected_at: datetime,
) -> list[OnChainMetricSeries]:
    observations: dict[OnChainMetricId, dict[datetime, float]] = {metric: {} for metric in _METRICS}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        observed_at = _timestamp(row.get("time"))
        if (
            observed_at is None
            or observed_at > collected_at
            or observed_at < collected_at - timedelta(days=72)
        ):
            continue
        for metric in _METRICS:
            value = _number(row.get(metric))
            if value is not None:
                observations[metric][observed_at] = value

    result: list[OnChainMetricSeries] = []
    for metric in _METRICS:
        points = [
            OnChainObservation(observed_at=timestamp, value=value)
            for timestamp, value in sorted(observations[metric].items())[-30:]
        ]
        if not points or collected_at - points[-1].observed_at > timedelta(days=3):
            continue
        current = points[-7:]
        previous = points[-14:-7]
        current_average = fmean(item.value for item in current) if len(current) == 7 else None
        previous_average = fmean(item.value for item in previous) if len(previous) == 7 else None
        change = (
            ((current_average / previous_average) - 1) * 100
            if current_average is not None and previous_average not in {None, 0.0}
            else None
        )
        label, unit = _METRIC_DETAILS[metric]
        result.append(
            OnChainMetricSeries(
                metric=metric,
                label=label,
                unit=unit,
                observations=points,
                latest_value=points[-1].value,
                latest_at=points[-1].observed_at,
                seven_day_average=current_average,
                previous_seven_day_average=previous_average,
                seven_day_change_pct=change,
            )
        )
    return result


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        normalized = _FRACTION.sub(r"\1", value.strip()).replace("Z", "+00:00")
        return _utc(datetime.fromisoformat(normalized))
    except ValueError:
        return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value))
    except TypeError, ValueError:
        return None
    return number if number >= 0 and number < float("inf") else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("On-chain timestamps must be timezone-aware.")
    return value.astimezone(UTC)


__all__ = ["DEFAULT_COINMETRICS_BASE_URL", "fetch_onchain_evidence"]
