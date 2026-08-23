from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from math import ceil, log10
from statistics import pstdev
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

from crypto_research.domain.analytics import MAX_DATA_DELAY_INTERVALS
from crypto_research.domain.market import (
    Candle,
    ComparisonMetrics,
    MarketDataQuality,
    MarketEvidence,
)
from crypto_research.domain.market import FutureMarketDataError as FutureMarketDataError
from crypto_research.domain.research import (
    COIN_ID_BY_ASSET,
    AnalysisRequest,
    OpportunityCandidate,
    OpportunityScanResult,
    TechnicalSnapshot,
)
from crypto_research.shared.time import timeframe_delta
from crypto_research.tools.cache import TTLCache

_OHLCV_PAGE_LIMIT = 300


def fetch_ohlcv_rows(
    *,
    exchange_name: str,
    symbol: str,
    timeframe: str,
    limit: int,
    exchange_factory: Any | None = None,
) -> list[list[float]]:
    """Fetch recent raw OHLCV rows and normalize provider timeframe differences."""

    if exchange_factory is None:
        import ccxt  # type: ignore[import-untyped]

        exchange_type = getattr(ccxt, exchange_name, None)
        if exchange_type is None:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        create_exchange = partial(exchange_type, {"enableRateLimit": True})
    else:
        create_exchange = partial(exchange_factory, exchange_name)
    return _fetch_rows_with_retry(
        create_exchange=create_exchange,
        exchange_name=exchange_name,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )


def _fetch_rows_with_retry(
    *,
    create_exchange: Callable[[], Any],
    exchange_name: str,
    symbol: str,
    timeframe: str,
    limit: int,
) -> list[list[float]]:
    last_error: Exception | None = None
    for _ in range(2):
        exchange = create_exchange()
        try:
            markets = exchange.load_markets()
            if symbol not in markets:
                raise ValueError(f"Unsupported symbol for {exchange_name}: {symbol}")
            provider_timeframe = _provider_timeframe(
                timeframe,
                getattr(exchange, "timeframes", {timeframe: timeframe}),
            )
            if provider_timeframe is None:
                raise ValueError(f"Unsupported timeframe for {exchange_name}: {timeframe}")
            ratio = _timeframe_ratio(provider_timeframe, timeframe)
            provider_limit = limit * ratio + ratio
            rows = _fetch_recent_rows(
                exchange=exchange,
                symbol=symbol,
                timeframe=provider_timeframe,
                limit=provider_limit,
            )
            if provider_timeframe != timeframe:
                rows = _resample_ohlcv_rows(
                    rows,
                    source_timeframe=provider_timeframe,
                    target_timeframe=timeframe,
                )
            return rows[-(limit + 1) :]
        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
        finally:
            close = getattr(exchange, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
    if last_error is not None:
        raise last_error
    raise ValueError("Market data could not be fetched.")


def _provider_timeframe(requested: str, available: object) -> str | None:
    supported = set(available) if isinstance(available, dict | list | tuple | set) else {requested}
    if requested in supported:
        return requested

    target_seconds = timeframe_delta(requested).total_seconds()
    candidates: list[tuple[float, str]] = []
    for candidate in supported:
        try:
            candidate_seconds = timeframe_delta(str(candidate)).total_seconds()
        except TypeError, ValueError:
            continue
        if candidate_seconds < target_seconds and target_seconds % candidate_seconds == 0:
            candidates.append((candidate_seconds, str(candidate)))
    return max(candidates, default=(0.0, ""))[1] or None


def _timeframe_ratio(source: str, target: str) -> int:
    source_seconds = timeframe_delta(source).total_seconds()
    target_seconds = timeframe_delta(target).total_seconds()
    ratio = target_seconds / source_seconds
    if ratio < 1 or not ratio.is_integer():
        raise ValueError(f"Cannot aggregate {source} candles into {target} candles.")
    return int(ratio)


def _fetch_recent_rows(
    *,
    exchange: Any,
    symbol: str,
    timeframe: str,
    limit: int,
) -> list[list[float]]:
    period_ms = int(timeframe_delta(timeframe).total_seconds() * 1000)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    since = now_ms - period_ms * (limit + 2)
    max_requests = ceil((limit + 2) / _OHLCV_PAGE_LIMIT) + 2
    rows_by_time: dict[int, list[float]] = {}

    for _ in range(max_requests):
        page = cast(
            list[list[float]],
            exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=min(_OHLCV_PAGE_LIMIT, limit + 2),
            ),
        )
        if not page:
            break
        try:
            for row in page:
                rows_by_time[int(coerce_ohlcv_number(row[0]))] = row
            last_timestamp = max(int(coerce_ohlcv_number(row[0])) for row in page)
        except IndexError, TypeError, ValueError:
            return page
        next_since = last_timestamp + period_ms
        if next_since <= since:
            break
        since = next_since
        if since >= now_ms:
            break

    return [rows_by_time[key] for key in sorted(rows_by_time)][-limit:]


def _resample_ohlcv_rows(
    rows: Sequence[Sequence[object]],
    *,
    source_timeframe: str,
    target_timeframe: str,
) -> list[list[float]]:
    ratio = _timeframe_ratio(source_timeframe, target_timeframe)
    source_ms = int(timeframe_delta(source_timeframe).total_seconds() * 1000)
    target_ms = int(timeframe_delta(target_timeframe).total_seconds() * 1000)
    by_time: dict[int, list[float]] = {}
    for index, row in enumerate(rows):
        try:
            if isinstance(row, str | bytes) or len(row) < 6:
                raise ValueError
            timestamp = int(coerce_ohlcv_number(row[0]))
            by_time[timestamp] = [coerce_ohlcv_number(value) for value in row[:6]]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Market provider returned an invalid OHLCV row at index {index}."
            ) from exc

    buckets: dict[int, list[list[float]]] = {}
    for timestamp in sorted(by_time):
        bucket = timestamp - timestamp % target_ms
        buckets.setdefault(bucket, []).append(by_time[timestamp])

    aggregated: list[list[float]] = []
    for bucket, bucket_rows in sorted(buckets.items()):
        expected_timestamps = [bucket + offset * source_ms for offset in range(ratio)]
        if [int(row[0]) for row in bucket_rows] != expected_timestamps:
            continue
        aggregated.append(
            [
                float(bucket),
                bucket_rows[0][1],
                max(row[2] for row in bucket_rows),
                min(row[3] for row in bucket_rows),
                bucket_rows[-1][4],
                sum(row[5] for row in bucket_rows),
            ]
        )
    return aggregated


def coerce_ohlcv_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError("OHLCV values must be numeric.")
    return float(value)


_CACHE_TTL_SECONDS = 15.0
_MINIMUM_TECHNICAL_CANDLES = 20
_MATURE_TECHNICAL_CANDLES = 50
_MARKET_CACHE = TTLCache[tuple[str, str, str, int, str | None, bool], MarketEvidence](
    _CACHE_TTL_SECONDS,
    clone=lambda result: result.model_copy(deep=True),
    namespace="market",
    serialize=lambda result: result.model_dump(mode="json"),
    deserialize=MarketEvidence.model_validate,
)


def fetch_market_evidence(
    *,
    exchange_name: str,
    symbol: str,
    timeframe: str,
    limit: int,
    coin_id: str | None = None,
    strict_contiguity: bool = False,
    exchange_factory: Any | None = None,
    cache_ttl_seconds: float = _CACHE_TTL_SECONDS,
    collected_at: datetime | None = None,
) -> MarketEvidence:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    observation_time = collected_at or datetime.now(UTC)
    if (
        observation_time.tzinfo is None
        or observation_time.tzinfo.utcoffset(observation_time) is None
    ):
        raise ValueError("collected_at must be timezone-aware.")
    observation_time = observation_time.astimezone(UTC)
    cache_key = (exchange_name, symbol, timeframe, limit, coin_id, strict_contiguity)
    cached = (
        _MARKET_CACHE.get(cache_key, ttl_seconds=cache_ttl_seconds, allow_stale=True)
        if exchange_factory is None
        else None
    )
    try:
        # The provider adapter performs one same-exchange retry.  A cached
        # observation is strictly a recovery path, never a way to skip a live
        # verification attempt.
        rows = fetch_ohlcv_rows(
            exchange_name=exchange_name,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            exchange_factory=exchange_factory,
        )

        parsed = _parse_candles(rows, timeframe=timeframe, now=observation_time)
        parsed_candles = parsed.candles
        contiguous_candles = (
            parsed_candles
            if strict_contiguity
            else _latest_contiguous_candles(parsed_candles, timeframe=timeframe)
        )
        candles = contiguous_candles[-limit:]
        excluded_prefix = len(parsed_candles) - len(contiguous_candles)
        quality = parsed.quality.model_copy(
            update={
                "accepted_candles": len(candles),
                "excluded_noncontiguous_prefix": excluded_prefix,
                "warnings": _quality_warnings(parsed.quality, excluded_prefix),
            }
        )
        _validate_market_window(
            candles,
            timeframe=timeframe,
            requested_limit=limit,
            collected_at=observation_time,
            require_contiguous=strict_contiguity,
        )
        result = MarketEvidence(
            exchange=exchange_name,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            first_time=candles[0].timestamp,
            last_time=candles[-1].timestamp,
            current_price=candles[-1].close,
            collected_at=observation_time,
            coin_id=coin_id,
            data_source=f"{exchange_name.title()} OHLCV adapter",
            source_state="live",
            data_quality=quality,
        )
    except Exception:
        if cached is not None and _market_cache_is_current(cached, collected_at=observation_time):
            return _cached_market_evidence(cached)
        raise
    if exchange_factory is None:
        _MARKET_CACHE.set(cache_key, result)
    return result


def _market_cache_is_current(
    evidence: MarketEvidence,
    *,
    collected_at: datetime,
) -> bool:
    age = collected_at - evidence.collected_at
    return (
        evidence.collected_at <= collected_at
        and evidence.last_time <= collected_at
        and age <= timeframe_delta(evidence.timeframe)
    )


def _cached_market_evidence(evidence: MarketEvidence) -> MarketEvidence:
    quality = evidence.data_quality.model_copy(
        update={
            "warnings": list(
                dict.fromkeys(
                    (
                        *evidence.data_quality.warnings,
                        "Live market verification failed; a fresh cached snapshot is shown.",
                    )
                )
            )[:12]
        }
    )
    return evidence.model_copy(update={"source_state": "cached", "data_quality": quality})


@dataclass(frozen=True, slots=True)
class _ParsedCandles:
    candles: list[Candle]
    quality: MarketDataQuality


def _parse_candles(
    rows: Sequence[Sequence[object]],
    *,
    timeframe: str,
    now: datetime | None = None,
) -> _ParsedCandles:
    by_time: dict[datetime, Candle] = {}
    current_time = now or datetime.now(UTC)
    candle_duration = timeframe_delta(timeframe)
    timeframe_seconds = int(candle_duration.total_seconds())
    quality = MarketDataQuality()
    for row in rows:
        try:
            if isinstance(row, str | bytes) or len(row) < 6:
                raise ValueError
            timestamp = datetime.fromtimestamp(coerce_ohlcv_number(row[0]) / 1000, tz=UTC)
            if int(timestamp.timestamp()) % timeframe_seconds:
                quality = quality.model_copy(
                    update={"excluded_misaligned": quality.excluded_misaligned + 1}
                )
                continue
            if timestamp > current_time:
                quality = quality.model_copy(
                    update={"excluded_future": quality.excluded_future + 1}
                )
                continue
            if timestamp + candle_duration > current_time:
                quality = quality.model_copy(
                    update={"excluded_incomplete": quality.excluded_incomplete + 1}
                )
                continue
            if timestamp in by_time:
                quality = quality.model_copy(
                    update={"excluded_duplicates": quality.excluded_duplicates + 1}
                )
                continue
            candle = Candle(
                timestamp=timestamp,
                open=coerce_ohlcv_number(row[1]),
                high=coerce_ohlcv_number(row[2]),
                low=coerce_ohlcv_number(row[3]),
                close=coerce_ohlcv_number(row[4]),
                volume=coerce_ohlcv_number(row[5]),
            )
        except IndexError, OSError, OverflowError, TypeError, ValueError:
            quality = quality.model_copy(
                update={"excluded_malformed": quality.excluded_malformed + 1}
            )
            continue
        by_time[timestamp] = candle
    candles = [by_time[key] for key in sorted(by_time)]
    if not candles:
        if quality.excluded_future:
            raise FutureMarketDataError(
                "Market provider returned only future-dated candles; they were excluded."
            )
        raise ValueError("No valid complete candles returned.")
    return _ParsedCandles(candles=candles, quality=quality)


def _quality_warnings(quality: MarketDataQuality, excluded_prefix: int) -> list[str]:
    values = {
        "future-dated": quality.excluded_future,
        "unfinished": quality.excluded_incomplete,
        "malformed": quality.excluded_malformed,
        "misaligned": quality.excluded_misaligned,
        "duplicate": quality.excluded_duplicates,
        "non-contiguous historical": excluded_prefix,
    }
    return [f"{count} {label} candle(s) excluded." for label, count in values.items() if count]


def _latest_contiguous_candles(candles: list[Candle], *, timeframe: str) -> list[Candle]:
    expected_delta = timeframe_delta(timeframe)
    start = 0
    for index, (previous, current) in enumerate(
        zip(candles, candles[1:], strict=False),
        start=1,
    ):
        if current.timestamp - previous.timestamp != expected_delta:
            start = index
    return candles[start:]


def _validate_market_window(
    candles: list[Candle],
    *,
    timeframe: str,
    requested_limit: int,
    collected_at: datetime,
    require_contiguous: bool = False,
) -> None:
    minimum = min(requested_limit, _MINIMUM_TECHNICAL_CANDLES)
    if len(candles) < minimum:
        raise ValueError(
            f"Market data requires at least {minimum} recent contiguous candles; "
            f"received {len(candles)}."
        )
    candle_delta = timeframe_delta(timeframe)
    if require_contiguous and any(
        current.timestamp - previous.timestamp != candle_delta
        for previous, current in zip(candles, candles[1:], strict=False)
    ):
        raise ValueError("Market provider returned non-contiguous candles.")
    latest_close_time = candles[-1].timestamp + candle_delta
    if collected_at - latest_close_time > candle_delta * MAX_DATA_DELAY_INTERVALS:
        raise ValueError("Market provider returned stale candles.")


def fetch_market_snapshots(
    *,
    exchange_name: str,
    symbols: Sequence[str],
    timeframe: str,
    limit: int,
    coin_ids: Sequence[str | None] | None = None,
    market_service: Callable[..., MarketEvidence] = fetch_market_evidence,
    max_workers: int = 5,
    collected_at: datetime | None = None,
) -> tuple[list[tuple[MarketEvidence, TechnicalSnapshot]], list[str]]:
    if not symbols:
        raise ValueError("At least one market symbol is required.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    ordered_symbols = list(symbols)
    resolved_ids = list(coin_ids) if coin_ids is not None else [None] * len(ordered_symbols)
    if len(resolved_ids) != len(ordered_symbols):
        raise ValueError("coin_ids must align with symbols.")

    with ThreadPoolExecutor(max_workers=min(max_workers, len(ordered_symbols))) as pool:
        requests = [
            (
                symbol,
                pool.submit(
                    market_service,
                    exchange_name=exchange_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    coin_id=coin_id,
                    collected_at=collected_at,
                ),
            )
            for symbol, coin_id in zip(ordered_symbols, resolved_ids, strict=True)
        ]
        snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]] = []
        warnings: list[str] = []
        for symbol, future in requests:
            try:
                market = future.result()
            except Exception as exc:
                warnings.append(f"{symbol} market data was unavailable ({type(exc).__name__}).")
                continue
            try:
                snapshot = calculate_indicators(market.candles)
            except Exception as exc:
                warnings.append(
                    f"{symbol} technical indicators could not be calculated ({type(exc).__name__})."
                )
                continue
            snapshots.append((market, snapshot))
    return snapshots, warnings


def fetch_market_comparison(
    *,
    request: AnalysisRequest,
    market_service: Callable[..., MarketEvidence] = fetch_market_evidence,
    collected_at: datetime | None = None,
) -> tuple[list[tuple[MarketEvidence, TechnicalSnapshot]], list[str]]:
    symbols = request.comparison_symbols or [request.symbol]
    coin_ids = [COIN_ID_BY_ASSET.get(symbol.split("/", maxsplit=1)[0]) for symbol in symbols]
    snapshots, warnings = fetch_market_snapshots(
        exchange_name=request.exchange,
        symbols=symbols,
        timeframe=request.timeframe,
        limit=request.candle_limit,
        coin_ids=coin_ids,
        market_service=market_service,
        collected_at=collected_at,
    )
    if len(symbols) < 2:
        return snapshots, warnings
    if len(snapshots) < 2:
        warnings.append("Fewer than two requested assets were available for comparison.")
        return snapshots, warnings
    aligned, alignment_warning = _align_comparison_windows(snapshots)
    if alignment_warning is not None:
        warnings.append(alignment_warning)
    return aligned, warnings


def _align_comparison_windows(
    snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]],
) -> tuple[list[tuple[MarketEvidence, TechnicalSnapshot]], str | None]:
    """Use the exact shared candle timestamps for every comparison asset."""

    shared = set.intersection(
        *({candle.timestamp for candle in market.candles} for market, _ in snapshots)
    )
    timestamps = sorted(shared)
    if len(timestamps) < _MINIMUM_TECHNICAL_CANDLES:
        return (
            snapshots,
            "The available assets did not share enough candle timestamps for a standardized "
            "comparison. Individual histories remain available but are not directly comparable.",
        )

    aligned: list[tuple[MarketEvidence, TechnicalSnapshot]] = []
    for market, _ in snapshots:
        by_time = {candle.timestamp: candle for candle in market.candles}
        candles = [by_time[timestamp] for timestamp in timestamps]
        aligned_market = market.model_copy(
            update={
                "candles": candles,
                "first_time": candles[0].timestamp,
                "last_time": candles[-1].timestamp,
                "current_price": candles[-1].close,
            }
        )
        aligned.append((aligned_market, calculate_indicators(candles)))
    return aligned, None


def calculate_comparison_metrics(market: MarketEvidence) -> ComparisonMetrics:
    """Calculate comparable return, range, volatility, and volume metrics."""

    closes = np.array([candle.close for candle in market.candles], dtype=float)
    returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([], dtype=float)
    return ComparisonMetrics(
        period_start=market.first_time,
        period_end=market.last_time,
        start_price=float(closes[0]),
        end_price=float(closes[-1]),
        price_return=float(closes[-1] / closes[0] - 1),
        volatility=float(np.std(returns)) if len(returns) else 0.0,
        high=max(candle.high for candle in market.candles),
        low=min(candle.low for candle in market.candles),
        total_volume=sum(candle.volume for candle in market.candles),
        observation_count=len(market.candles),
    )


def calculate_indicators(candles: list[Candle]) -> TechnicalSnapshot:
    if not candles:
        raise ValueError("At least one candle is required for technical analysis.")
    if len(candles) < _MATURE_TECHNICAL_CANDLES:
        return TechnicalSnapshot(
            status="unavailable",
            limitation=(
                f"Technical indicators require {_MATURE_TECHNICAL_CANDLES} contiguous completed "
                f"candles; received {len(candles)}."
            ),
            trend="neutral",
        )
    frame = pd.DataFrame(
        ((candle.close, candle.high, candle.low) for candle in candles),
        columns=("close", "high", "low"),
        dtype=float,
    )
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    returns = close.pct_change(fill_method=None).dropna()
    sma = float(close.rolling(20, min_periods=1).mean().iloc[-1])
    ema = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    latest_gain = float(gain.iloc[-1])
    latest_loss = float(loss.iloc[-1])
    if np.isnan(latest_gain) or np.isnan(latest_loss):
        rsi = 50.0
    elif latest_loss == 0:
        rsi = 100.0 if latest_gain > 0 else 50.0
    else:
        rsi = 100 - (100 / (1 + latest_gain / latest_loss))
    fast_ema = close.ewm(span=12, adjust=False).mean()
    slow_ema = close.ewm(span=26, adjust=False).mean()
    macd = float(fast_ema.iloc[-1] - slow_ema.iloc[-1])
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(14, min_periods=1).mean().iloc[-1])
    volatility = (
        float(returns.rolling(20, min_periods=1).std().fillna(0).iloc[-1]) if len(returns) else 0.0
    )
    support = float(low.tail(20).min())
    resistance = float(high.tail(20).max())
    trend = (
        "bullish"
        if close.iloc[-1] > sma and macd > 0
        else "bearish"
        if close.iloc[-1] < sma and macd < 0
        else "neutral"
    )
    return TechnicalSnapshot(
        sma=sma,
        ema=ema,
        rsi=rsi,
        macd=macd,
        atr=atr,
        volatility=volatility,
        support=support,
        resistance=resistance,
        trend=trend,
    )


WatchlistItem = tuple[str, str]


class OpportunityScore(TypedDict):
    asset: str
    symbol: str
    current_price: float
    momentum_24h: float
    volatility_24h: float
    quote_volume_24h: float
    trend: str


DEFAULT_OPPORTUNITY_WATCHLIST = tuple(x for x in COIN_ID_BY_ASSET.items() if x[0] != "USDC")


def scan_crypto_opportunities(
    *,
    exchange_name: str = "kraken",
    quote: str = "USD",
    timeframe: str = "1h",
    limit: int = 72,
    watchlist: Sequence[WatchlistItem] = DEFAULT_OPPORTUNITY_WATCHLIST,
    market_service: Callable[..., MarketEvidence] = fetch_market_evidence,
    snapshot_fetcher: Callable[
        ...,
        tuple[list[tuple[MarketEvidence, TechnicalSnapshot]], list[str]],
    ] = fetch_market_snapshots,
    collected_at: datetime | None = None,
) -> OpportunityScanResult:
    warnings: list[str] = []
    evidence: list[tuple[MarketEvidence, TechnicalSnapshot]] = []
    requested_limit = max(limit, _intervals_per_day(timeframe) + 1)
    symbols = [f"{asset}/{quote}" for asset, _ in watchlist]
    coin_ids = [coin_id for _, coin_id in watchlist]
    snapshots, snapshot_warnings = snapshot_fetcher(
        exchange_name=exchange_name,
        symbols=symbols,
        timeframe=timeframe,
        limit=requested_limit,
        coin_ids=coin_ids,
        market_service=market_service,
        max_workers=min(6, max(1, len(watchlist))),
        collected_at=collected_at,
    )
    warnings.extend(snapshot_warnings)
    for market, technical in snapshots:
        try:
            _require_complete_day(market)
            evidence.append((market, technical))
        except Exception as exc:
            warnings.append(f"{market.symbol} could not be scanned ({type(exc).__name__}).")

    if not evidence:
        raise ValueError("No cryptocurrencies could be scanned.")

    raw = [_score_market(market, technical) for market, technical in evidence]
    max_volume = max(item["quote_volume_24h"] for item in raw) or 1.0
    candidates = [_candidate_from_score(item, max_volume=max_volume) for item in raw]
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    candidates = [
        OpportunityCandidate(**{**candidate.model_dump(), "rank": index})
        for index, candidate in enumerate(ranked, start=1)
    ]
    summary = _build_summary(candidates, exchange_name=exchange_name, warnings=warnings)
    collected_at = max(market.collected_at for market, _ in evidence)
    return OpportunityScanResult(
        exchange=exchange_name,
        timeframe=timeframe,
        candidates=candidates,
        collected_at=collected_at,
        summary=summary,
        warnings=warnings,
    )


def _score_market(market: MarketEvidence, technical: TechnicalSnapshot) -> OpportunityScore:
    candles = market.candles
    interval_count = _intervals_per_day(market.timeframe)
    price_window = candles[-(interval_count + 1) :]
    volume_window = candles[-interval_count:]
    first = price_window[0].close
    last = price_window[-1].close
    momentum = (last - first) / first * 100
    returns = _returns(price_window)
    volatility = pstdev(returns) * 100 if len(returns) > 1 else 0.0
    quote_volume = sum(candle.close * candle.volume for candle in volume_window)
    sma = sum(candle.close for candle in price_window) / len(price_window)
    trend = (
        technical.trend
        if technical.status == "available"
        else ("bullish" if last > sma else "bearish" if last < sma else "neutral")
    )
    return {
        "asset": market.symbol.split("/")[0],
        "symbol": market.symbol,
        "current_price": market.current_price,
        "momentum_24h": momentum,
        "volatility_24h": volatility,
        "quote_volume_24h": quote_volume,
        "trend": trend,
    }


def _require_complete_day(market: MarketEvidence) -> None:
    interval_count = _intervals_per_day(market.timeframe)
    required = interval_count + 1
    if len(market.candles) < required:
        raise ValueError("Opportunity scoring requires a complete 24-hour candle window.")
    window = market.candles[-required:]
    expected_delta = timeframe_delta(market.timeframe)
    if any(
        current.timestamp - previous.timestamp != expected_delta
        for previous, current in zip(window, window[1:], strict=False)
    ):
        raise ValueError("Opportunity scoring requires contiguous 24-hour candles.")


def _candidate_from_score(
    item: OpportunityScore,
    *,
    max_volume: float,
) -> OpportunityCandidate:
    momentum = item["momentum_24h"]
    volatility = item["volatility_24h"]
    quote_volume = item["quote_volume_24h"]
    trend_bonus = 12.0 if item["trend"] == "bullish" else 5.0 if item["trend"] == "neutral" else 0.0
    liquidity_ratio = log10(max(quote_volume, 1.0)) / log10(max(max_volume, 10.0))
    liquidity = min(20.0, max(0.0, liquidity_ratio * 20))
    momentum_score = max(0.0, min(45.0, 25.0 + momentum * 2.0))
    volatility_penalty = min(30.0, volatility * 4.0)
    score = max(0.0, min(100.0, momentum_score + liquidity + trend_bonus - volatility_penalty))
    reason = (
        f"{momentum:+.2f}% 24h momentum with a {item['trend']} trend; "
        f"{volatility:.2f}% realized volatility and {liquidity:.1f}/20 relative liquidity."
    )
    return OpportunityCandidate(
        rank=1,
        asset=item["asset"],
        symbol=item["symbol"],
        current_price=item["current_price"],
        score=round(score, 2),
        momentum_24h=round(momentum, 2),
        volatility_24h=round(volatility, 2),
        trend=str(item["trend"]),
        reason=reason,
    )


def _returns(candles: Sequence[Candle]) -> list[float]:
    return [
        (current.close - previous.close) / previous.close
        for previous, current in zip(candles, candles[1:], strict=False)
        if previous.close > 0
    ]


def _intervals_per_day(timeframe: str) -> int:
    seconds = timeframe_delta(timeframe).total_seconds()
    return max(1, round(86_400 / seconds))


def _build_summary(
    candidates: list[OpportunityCandidate],
    *,
    exchange_name: str,
    warnings: list[str],
) -> str:
    top = candidates[0]
    runner_up = candidates[1:3]
    lines = [
        (
            f"The highest-ranked asset in this {exchange_name} watchlist scan is "
            f"{top.asset} ({top.symbol}). It scored {top.score:.1f}/100, "
            f"with {top.momentum_24h:+.2f}% 24h momentum, {top.trend} trend, and "
            f"{top.volatility_24h:.2f}% short-term volatility."
        ),
    ]
    if runner_up:
        lines.append(
            "Closest alternatives: "
            + ", ".join(
                f"{item.asset} ({item.score:.1f}/100, {item.momentum_24h:+.2f}% momentum)"
                for item in runner_up
            )
            + "."
        )
    lines.append(
        "This is a relative market screen, not a forecast or trading recommendation. Scores can "
        "change quickly and only compare assets with validated data in this run."
    )
    if warnings:
        lines.append(f"Some assets were unavailable during the scan: {len(warnings)} skipped.")
    return "\n\n".join(lines)
