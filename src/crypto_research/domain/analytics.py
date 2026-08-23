"""Deterministic market features and observed-risk calculations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import timedelta
from math import sqrt

from crypto_research.domain.evidence import FundamentalEvidence, NewsEvidence, TechnicalSnapshot
from crypto_research.domain.market import (
    MarketEvidence,
    MarketFeatureSummary,
    MarketPostureSummary,
    MarketWindowLabel,
    MarketWindowReturn,
)
from crypto_research.domain.research import (
    AnalysisAsset,
    AssetResearchBundle,
    AssetRiskResult,
    FundamentalsAgentResult,
    MarketAgentResult,
    MarketComparisonResult,
    ResearchAgentResult,
    RiskAssessment,
    RiskResult,
    risk_band,
)
from crypto_research.shared.time import timeframe_delta

MAX_DATA_DELAY_INTERVALS = 2
_RETURN_WINDOWS: tuple[tuple[MarketWindowLabel, int], ...] = (
    ("1h", 1),
    ("4h", 4),
    ("24h", 24),
    ("7d", 24 * 7),
)


def calculate_market_features(market: MarketEvidence) -> MarketFeatureSummary:
    """Reduce validated OHLCV into deterministic features for language-model evidence."""

    candles = market.candles
    candle_delta = timeframe_delta(market.timeframe)
    contiguous = all(
        current.timestamp - previous.timestamp == candle_delta
        for previous, current in zip(candles, candles[1:], strict=False)
    )
    latest_close_time = market.last_time + candle_delta
    data_delay = max(timedelta(0), market.collected_at - latest_close_time)
    high = max(candle.high for candle in candles)
    low = min(candle.low for candle in candles)
    closes = [candle.close for candle in candles]
    peak = closes[0]
    maximum_drawdown = 0.0
    for close in closes:
        peak = max(peak, close)
        maximum_drawdown = max(maximum_drawdown, (peak - close) / peak)

    return MarketFeatureSummary(
        latest_completed_close=market.current_price,
        observed_at=market.last_time,
        window_start=market.first_time,
        window_end=market.last_time,
        candle_count=len(candles),
        high=high,
        low=low,
        range_absolute=high - low,
        range_percent=(high - low) / market.current_price * 100,
        base_volume=sum(candle.volume for candle in candles),
        quote_volume=sum(candle.close * candle.volume for candle in candles),
        maximum_drawdown=maximum_drawdown,
        contiguous=contiguous,
        fresh_at_collection=(data_delay <= candle_delta * MAX_DATA_DELAY_INTERVALS),
        data_delay_seconds=data_delay.total_seconds(),
        returns=[
            _calculate_window_return(
                market,
                label=label,
                hours=hours,
                candle_delta=candle_delta,
            )
            for label, hours in _RETURN_WINDOWS
        ],
    )


def _calculate_window_return(
    market: MarketEvidence,
    *,
    label: MarketWindowLabel,
    hours: int,
    candle_delta: timedelta,
) -> MarketWindowReturn:
    horizon = timedelta(hours=hours)
    target_time = market.last_time - horizon
    by_time = {candle.timestamp: index for index, candle in enumerate(market.candles)}
    target_index = by_time.get(target_time)
    unavailable_reason: str | None = None
    status = "unavailable"
    if horizon < candle_delta or horizon.total_seconds() % candle_delta.total_seconds() != 0:
        unavailable_reason = f"The {market.timeframe} timeframe is too coarse for {label}."
        status = "not_applicable"
    elif target_index is None:
        unavailable_reason = f"The available history does not cover an exact {label} window."
    else:
        window = market.candles[target_index:]
        expected_count = int(horizon / candle_delta) + 1
        if len(window) != expected_count or any(
            current.timestamp - previous.timestamp != candle_delta
            for previous, current in zip(window, window[1:], strict=False)
        ):
            unavailable_reason = f"The {label} candle window is not contiguous."

    if unavailable_reason is not None or target_index is None:
        return MarketWindowReturn(
            label=label,
            hours=hours,
            status=status,
            latest_price=market.current_price,
            period_end=market.last_time,
            reason=unavailable_reason or f"The {label} return is unavailable.",
        )

    reference = market.candles[target_index]
    change = market.current_price - reference.close
    price_return = change / reference.close
    return MarketWindowReturn(
        label=label,
        hours=hours,
        status="available",
        reference_price=reference.close,
        latest_price=market.current_price,
        change_absolute=change,
        return_decimal=price_return,
        return_percent=price_return * 100,
        period_start=reference.timestamp,
        period_end=market.last_time,
    )


def build_market_posture(
    market: MarketEvidence,
    technical: TechnicalSnapshot,
    contextual_timeframes: Sequence[tuple[str, str]] = (),
) -> MarketPostureSummary:
    """Build the deterministic market posture shared by summary and digest.

    ``contextual_timeframes`` carries ``(timeframe, trend)`` pairs for the
    higher-timeframe confirmation line; callers derive them from validated
    ``MarketTimeframeEvidence`` so the posture never touches raw candles.
    """

    features = calculate_market_features(market)
    change_24h = next(
        (item for item in features.returns if item.label == "24h" and item.status == "available"),
        None,
    )
    return MarketPostureSummary(
        symbol=market.symbol,
        exchange=market.exchange,
        timeframe=market.timeframe,
        as_of=market.last_time,
        collected_at=market.collected_at,
        price=market.current_price,
        change_24h_percent=change_24h.return_percent if change_24h is not None else None,
        change_24h_absolute=change_24h.change_absolute if change_24h is not None else None,
        window_returns=features.returns,
        high=features.high,
        low=features.low,
        range_percent=features.range_percent,
        quote_volume=features.quote_volume,
        maximum_drawdown=features.maximum_drawdown,
        trend=technical.trend,
        rsi=technical.rsi,
        rsi_band=_rsi_band(technical.rsi),
        macd=technical.macd,
        atr=technical.atr,
        volatility=technical.volatility,
        support=technical.support,
        resistance=technical.resistance,
        fresh=features.fresh_at_collection,
        data_delay_seconds=features.data_delay_seconds,
        contextual_confirmation=[
            f"{timeframe}:{trend}"
            for timeframe, trend in contextual_timeframes
            if timeframe and trend
        ],
    )


def _rsi_band(rsi: float | None) -> str:
    if rsi is None:
        return "neutral"
    if rsi < 30:
        return "oversold"
    if rsi < 45:
        return "weak"
    if rsi <= 55:
        return "neutral"
    if rsi <= 70:
        return "strong"
    return "overbought"


_ADVERSE_NEWS_HIGH = re.compile(
    r"(?i)\b(?:exploit(?:ed)?|hack(?:ed)?|insolven(?:t|cy)|funds?\s+(?:stolen|drained)|"
    r"bridge\s+attack|chain\s+halt)\b"
)
_ADVERSE_NEWS_MEDIUM = re.compile(
    r"(?i)\b(?:lawsuit|enforcement|investigation|outage|security incident|vulnerability|"
    r"delist(?:ed|ing)?)\b"
)


def evaluate_research_risk(
    *,
    market_result: MarketAgentResult | None = None,
    market_comparison_result: MarketComparisonResult | None = None,
    research_result: ResearchAgentResult | None = None,
    fundamentals_result: FundamentalsAgentResult | None = None,
    assets: list[AnalysisAsset] | None = None,
) -> RiskResult:
    """Score every requested asset from independently matched validated evidence."""

    ordered_assets = list(assets or [])
    if not ordered_assets and research_result is not None:
        ordered_assets = [bundle.asset for bundle in research_result.asset_results]
    if not ordered_assets and fundamentals_result is not None:
        ordered_assets = [bundle.asset for bundle in fundamentals_result.asset_results]
    if not ordered_assets:
        return RiskResult(
            assessment=_calculate_observed_risk(
                market=market_result.market if market_result is not None else None,
                technical=market_result.technical if market_result is not None else None,
                fundamentals=(
                    fundamentals_result.fundamentals
                    if fundamentals_result is not None
                    else research_result.fundamentals
                    if research_result is not None
                    else None
                ),
                news=research_result.news if research_result is not None else None,
            )
        )

    fundamentals_source = fundamentals_result or research_result
    news_bundles = {
        bundle.asset.key: bundle
        for bundle in (research_result.asset_results if research_result is not None else [])
    }
    fundamentals_bundles = {
        bundle.asset.key: bundle
        for bundle in (fundamentals_source.asset_results if fundamentals_source is not None else [])
    }
    market_by_symbol = {
        item.market.symbol: (item.market, item.technical)
        for item in (
            market_comparison_result.assets if market_comparison_result is not None else []
        )
    }

    results: list[AssetRiskResult] = []
    for index, asset in enumerate(ordered_assets):
        news_bundle = news_bundles.get(asset.key)
        fundamentals_bundle = fundamentals_bundles.get(asset.key)
        market, technical = market_by_symbol.get(asset.symbol, (None, None))
        if market is None and index == 0 and market_result is not None:
            market, technical = market_result.market, market_result.technical
        fundamentals = _fundamentals_for_asset(
            fundamentals_bundle,
            fundamentals_source if index == 0 else None,
        )
        news = _news_for_asset(news_bundle, research_result if index == 0 else None)
        results.append(
            AssetRiskResult(
                asset=asset,
                assessment=_calculate_observed_risk(
                    market=market,
                    technical=technical,
                    fundamentals=fundamentals,
                    news=news,
                ),
            )
        )
    return RiskResult(
        assessment=(
            results[0].assessment if len(results) == 1 else _aggregate_assessments(results)
        ),
        asset_results=results,
    )


def _calculate_observed_risk(
    *,
    market: MarketEvidence | None,
    technical: TechnicalSnapshot | None,
    fundamentals: FundamentalEvidence | None,
    news: NewsEvidence | None,
    fundamentals_available: bool | None = None,
    news_available: bool | None = None,
) -> RiskAssessment:
    components: dict[str, float] = {}
    available_weight = 0.0
    coverage_gaps: list[str] = []

    if technical is not None and technical.status == "available":
        if technical.volatility is not None:
            hours = _timeframe_hours(market.timeframe) if market else 1.0
            volatility_24h = technical.volatility * sqrt(max(1.0, 24.0 / hours))
            components["realized_volatility"] = _scaled_points(
                volatility_24h,
                floor=0.02,
                ceiling=0.12,
                weight=25.0,
            )
            available_weight += 25.0
        else:
            coverage_gaps.append("Realized volatility is unavailable.")

        trend_stress = 5.0 if technical.trend == "bearish" else 0.0
        if technical.rsi is not None:
            trend_stress += 5.0 if technical.rsi < 25 or technical.rsi > 75 else 0.0
            available_weight += 10.0
            components["trend_rsi_stress"] = trend_stress
        else:
            coverage_gaps.append("RSI evidence is unavailable.")
    else:
        coverage_gaps.extend(
            [
                "Current market and technical evidence is unavailable.",
                "Realized volatility is unavailable.",
                "RSI evidence is unavailable.",
            ]
        )

    if market is not None and technical is not None and technical.status == "available":
        drawdown, change_24h = _market_observations(market)
        components["maximum_drawdown"] = min(20.0, drawdown / 0.30 * 20.0)
        available_weight += 20.0
        if change_24h is not None:
            components["absolute_24h_move"] = _scaled_points(
                abs(change_24h),
                floor=2.0,
                ceiling=20.0,
                weight=15.0,
            )
            available_weight += 15.0
        else:
            coverage_gaps.append("An exact 24-hour return is unavailable.")
    elif technical is not None and technical.status == "available":
        coverage_gaps.extend(
            [
                "Maximum drawdown is unavailable.",
                "An exact 24-hour return is unavailable.",
            ]
        )

    fundamental_is_available = (
        fundamentals is not None and fundamentals.status == "available"
        if fundamentals_available is None
        else fundamentals_available
    )
    if fundamental_is_available:
        if fundamentals is not None and fundamentals.rank is not None:
            components["market_size"] = (
                0.0 if fundamentals.rank <= 20 else 5.0 if fundamentals.rank <= 100 else 10.0
            )
            available_weight += 10.0
        elif fundamentals is None:
            available_weight += 10.0
        else:
            coverage_gaps.append("Market-cap rank is unavailable.")

        if (
            fundamentals is not None
            and fundamentals.circulating_supply is not None
            and fundamentals.max_supply is not None
            and fundamentals.max_supply > 0
        ):
            ratio = fundamentals.circulating_supply / fundamentals.max_supply
            components["supply_dilution"] = 10.0 if ratio < 0.5 else 5.0 if ratio < 0.8 else 0.0
            available_weight += 10.0
        elif fundamentals is None:
            available_weight += 15.0
        else:
            coverage_gaps.append("Circulating-versus-maximum-supply coverage is unavailable.")
    else:
        coverage_gaps.append("Fundamental data is unavailable.")

    news_is_available = (
        bool(news.items) if news_available is None and news is not None else bool(news_available)
    )
    if news_is_available:
        components["adverse_news"] = _adverse_news_points(news)
        available_weight += 10.0 if news is not None else 25.0
    else:
        coverage_gaps.append("Recent news evidence is unavailable.")

    observed_points = sum(components.values())
    score = min(100.0, observed_points / available_weight * 100.0) if available_weight else 0.0
    factors = _risk_factors(components, score=score, confidence=available_weight)
    return RiskAssessment(
        score=round(score, 1),
        band=risk_band(score),
        factors=factors,
        components={name: round(value, 2) for name, value in components.items()},
        evidence_confidence=min(100.0, available_weight),
        coverage_gaps=list(dict.fromkeys(coverage_gaps)),
    )


def _scaled_points(value: float, *, floor: float, ceiling: float, weight: float) -> float:
    if value <= floor:
        return 0.0
    return min(weight, (value - floor) / (ceiling - floor) * weight)


def _timeframe_hours(timeframe: str) -> float:
    value = timeframe.strip().casefold()
    if value.endswith("m"):
        return float(value[:-1]) / 60.0
    if value.endswith("h"):
        return float(value[:-1])
    if value.endswith("d"):
        return float(value[:-1]) * 24.0
    return 1.0


def _market_observations(market: MarketEvidence) -> tuple[float, float | None]:
    closes = [candle.close for candle in market.candles]
    peak = closes[0]
    drawdown = 0.0
    for close in closes:
        peak = max(peak, close)
        drawdown = max(drawdown, (peak - close) / peak)
    target = market.last_time - timedelta(hours=24)
    reference = next(
        (candle.close for candle in market.candles if candle.timestamp == target),
        None,
    )
    change = None if reference is None else (market.current_price / reference - 1.0) * 100.0
    return drawdown, change


def _adverse_news_points(news: NewsEvidence | None) -> float:
    if news is None:
        return 0.0
    text = " ".join(f"{item.title} {item.excerpt}" for item in news.items)
    if _ADVERSE_NEWS_HIGH.search(text):
        return 10.0
    if _ADVERSE_NEWS_MEDIUM.search(text):
        return 5.0
    return 0.0


def _risk_factors(
    components: dict[str, float],
    *,
    score: float,
    confidence: float,
) -> list[str]:
    labels = {
        "realized_volatility": "Elevated realized volatility.",
        "maximum_drawdown": "Meaningful drawdown within the observed window.",
        "absolute_24h_move": "Large absolute 24-hour price movement.",
        "trend_rsi_stress": "Bearish trend or extreme RSI conditions.",
        "market_size": "Smaller market-cap rank increases observed structural risk.",
        "supply_dilution": "A large share of maximum supply remains outside circulation.",
        "adverse_news": "Recent verified coverage contains an adverse risk event.",
    }
    ranked = sorted(components.items(), key=lambda item: item[1], reverse=True)
    factors = [labels[name] for name, value in ranked if value > 0][:3]
    if not factors:
        factors.append("No elevated signal was detected in the available risk components.")
    if confidence < 60:
        factors.append(
            f"Observed risk is low-confidence because evidence coverage is {confidence:.0f}/100."
        )
    elif score >= 50:
        factors.append("Multiple available components contribute to the elevated score.")
    return factors


def _fundamentals_for_asset(
    bundle: AssetResearchBundle | None,
    aggregate: FundamentalsAgentResult | ResearchAgentResult | None,
) -> FundamentalEvidence | None:
    if bundle is not None and bundle.fundamentals is not None:
        return bundle.fundamentals
    return aggregate.fundamentals if aggregate is not None else None


def _news_for_asset(
    bundle: AssetResearchBundle | None,
    aggregate: ResearchAgentResult | None,
) -> NewsEvidence | None:
    if bundle is not None and bundle.news is not None:
        return bundle.news
    return aggregate.news if aggregate is not None else None


def _aggregate_assessments(results: list[AssetRiskResult]) -> RiskAssessment:
    score = max(result.assessment.score for result in results)
    return RiskAssessment(
        score=score,
        band=risk_band(score),
        factors=[
            f"{result.asset.symbol}: {factor}"
            for result in results
            for factor in result.assessment.factors
        ],
        components={
            f"{result.asset.evidence_key}.{name}": value
            for result in results
            for name, value in result.assessment.components.items()
        },
        evidence_confidence=min(result.assessment.evidence_confidence for result in results),
        coverage_gaps=[
            f"{result.asset.symbol}: {gap}"
            for result in results
            for gap in result.assessment.coverage_gaps
        ],
    )


__all__ = ["evaluate_research_risk"]
