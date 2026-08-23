from __future__ import annotations

from collections.abc import Sequence

from crypto_research.agents.base import AgentManifest, collection_kwargs
from crypto_research.domain.analytics import build_market_posture
from crypto_research.domain.core import SupportedTimeframe
from crypto_research.domain.evidence import DerivativesEvidence
from crypto_research.domain.market import (
    FutureMarketDataError,
    MarketEvidence,
    MarketPostureSummary,
)
from crypto_research.domain.research import (
    AnalysisRequest,
    CollectionContext,
    MarketAgentResult,
    MarketComparisonAsset,
    MarketComparisonResult,
    MarketTimeframeEvidence,
    OpportunityCandidate,
    OpportunityScanResult,
    ResearchCapability,
    TechnicalSnapshot,
)
from crypto_research.shared.formatting import format_money
from crypto_research.tools.types import MarketServices

from .market_collector import selected_capabilities

MarketAgentOutput = OpportunityScanResult | MarketAgentResult | MarketComparisonResult
MARKET_MANIFEST = AgentManifest(
    id="market_agent",
    label="Market & Risk Agent",
    capabilities=frozenset(
        {
            ResearchCapability.MARKET,
            ResearchCapability.RISK,
            ResearchCapability.DISCOVERY,
            ResearchCapability.DERIVATIVES,
        }
    ),
)


class MarketDataIdentityError(ValueError):
    """A provider response belongs to a different selected market request."""


_CONTEXTUAL_TIMEFRAMES: dict[str, tuple[SupportedTimeframe, ...]] = {
    "1m": ("4h", "1d"),
    "5m": ("4h", "1d"),
    "15m": ("4h", "1d"),
    "30m": ("4h", "1d"),
    "1h": ("4h", "1d"),
    "4h": ("1d",),
    "1d": (),
}


class MarketAgent:
    def __init__(
        self,
        *,
        services: MarketServices,
        derivatives_base_url: str = "https://fapi.binance.com",
    ) -> None:
        self._services = services
        self._derivatives_base_url = derivatives_base_url

    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> MarketAgentOutput:
        selected = selected_capabilities(requested_capabilities)
        if ResearchCapability.DISCOVERY in selected:
            try:
                scan_kwargs: dict[str, object] = collection_kwargs(collection_context)
                scan_kwargs["snapshot_fetcher"] = self._services.snapshots
                result = self._services.discovery(request, **scan_kwargs)
                _require_current_scan(result, collection_context, request)
                return result
            except Exception as exc:
                return MarketComparisonResult(
                    warnings=[_market_unavailable_warning("Market discovery", exc)]
                )
        is_multi = len(request.comparison_symbols) > 1

        if is_multi:
            try:
                snapshots, warnings = self._services.comparison(
                    request=request,
                    **collection_kwargs(collection_context),
                )
            except Exception as exc:
                return MarketComparisonResult(
                    warnings=[_market_unavailable_warning("Market comparison", exc)]
                )
            assets_by_symbol: dict[str, MarketComparisonAsset] = {}
            asset_by_symbol = {asset.symbol: asset for asset in request.ordered_assets()}
            expected_symbols = tuple(request.comparison_symbols or asset_by_symbol)
            for market, technical in snapshots:
                try:
                    _require_current_market(market, collection_context)
                    _require_market_identity(
                        market,
                        exchange=request.exchange,
                        symbol=market.symbol,
                        timeframe=request.timeframe,
                    )
                    if market.symbol not in expected_symbols:
                        raise MarketDataIdentityError(
                            "Market provider returned an unrequested comparison symbol."
                        )
                    if market.symbol in assets_by_symbol:
                        raise MarketDataIdentityError(
                            "Market provider returned a duplicate comparison symbol."
                        )
                    asset = asset_by_symbol.get(market.symbol)
                    assets_by_symbol[market.symbol] = MarketComparisonAsset(
                        market=market,
                        technical=technical,
                        metrics=self._services.comparison_metrics(market),
                        derivatives=(
                            self._collect_derivatives(
                                symbol=market.symbol,
                                timeframe=request.timeframe,
                                collection_context=collection_context,
                            )
                            if ResearchCapability.DERIVATIVES in selected
                            else None
                        ),
                        contextual_timeframes=self._collect_contextual_timeframes(
                            request,
                            symbol=market.symbol,
                            coin_id=asset.coin_id if asset is not None else None,
                            collection_context=collection_context,
                        ),
                    )
                except Exception as exc:
                    warnings.append(_market_unavailable_warning(market.symbol, exc))
            assets: list[MarketComparisonAsset] = []
            for symbol in expected_symbols:
                item = assets_by_symbol.get(symbol)
                if item is None:
                    if not any(symbol in warning for warning in warnings):
                        warnings.append(
                            f"{symbol} market data were unavailable (no validated provider result)."
                        )
                    continue
                assets.append(item)
            return MarketComparisonResult(
                assets=assets,
                warnings=warnings,
            )

        try:
            market = self._services.market_evidence(
                exchange_name=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
                limit=request.candle_limit,
                coin_id=request.coin_id,
                **collection_kwargs(collection_context),
            )
            _require_current_market(market, collection_context)
            _require_market_identity(
                market,
                exchange=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
            )
        except Exception as exc:
            return MarketComparisonResult(
                warnings=[_market_unavailable_warning(request.symbol, exc)]
            )
        try:
            technical = self._services.indicators(market.candles)
        except Exception as exc:
            return MarketComparisonResult(
                warnings=[_market_unavailable_warning(request.symbol, exc)]
            )
        contextual_timeframes = self._collect_contextual_timeframes(
            request,
            symbol=request.symbol,
            coin_id=request.coin_id,
            collection_context=collection_context,
        )
        return MarketAgentResult(
            market=market,
            technical=technical,
            derivatives=(
                self._collect_derivatives(
                    symbol=market.symbol,
                    timeframe=request.timeframe,
                    collection_context=collection_context,
                )
                if ResearchCapability.DERIVATIVES in selected
                else None
            ),
            summary=_build_market_summary(
                market,
                technical,
                contextual_timeframes=contextual_timeframes,
            ),
            contextual_timeframes=contextual_timeframes,
        )

    def _collect_derivatives(
        self,
        *,
        symbol: str,
        timeframe: str,
        collection_context: CollectionContext | None,
    ) -> DerivativesEvidence:
        asset = symbol.split("/", maxsplit=1)[0].upper()
        try:
            evidence = self._services.derivatives(
                asset=asset,
                timeframe=timeframe,
                base_url=self._derivatives_base_url,
                **collection_kwargs(collection_context),
            )
            return _sanitize_derivatives_evidence(
                evidence,
                symbol=symbol,
                context=collection_context,
            )
        except Exception as exc:
            return DerivativesEvidence(
                asset=asset,
                status="unavailable",
                **collection_kwargs(collection_context),
                warnings=[f"Binance derivatives data were unavailable ({type(exc).__name__})."],
            )

    def _collect_contextual_timeframes(
        self,
        request: AnalysisRequest,
        *,
        symbol: str,
        coin_id: str | None,
        collection_context: CollectionContext | None,
    ) -> list[MarketTimeframeEvidence]:
        evidence: list[MarketTimeframeEvidence] = []
        for timeframe in contextual_timeframes(request.timeframe):
            try:
                market = self._services.market_evidence(
                    exchange_name=request.exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=max(request.candle_limit, 50),
                    coin_id=coin_id,
                    **collection_kwargs(collection_context),
                )
                _require_current_market(market, collection_context)
                _require_market_identity(
                    market,
                    exchange=request.exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                technical = self._services.indicators(market.candles)
                if technical.status == "unavailable":
                    evidence.append(
                        MarketTimeframeEvidence(
                            timeframe=timeframe,
                            status="partial",
                            market=market,
                            technical=technical,
                            limitation=technical.limitation,
                        )
                    )
                else:
                    evidence.append(
                        MarketTimeframeEvidence(
                            timeframe=timeframe,
                            status="complete",
                            market=market,
                            technical=technical,
                        )
                    )
            except Exception as exc:
                evidence.append(
                    MarketTimeframeEvidence(
                        timeframe=timeframe,
                        status="unavailable",
                        limitation=_market_unavailable_warning(f"{symbol} {timeframe}", exc),
                    )
                )
        return evidence


def _market_unavailable_warning(symbol: str, exc: Exception) -> str:
    if type(exc).__name__ == "FutureMarketDataError":
        return (
            f"{symbol} future-dated market data were excluded; current market data are unavailable."
        )
    if isinstance(exc, MarketDataIdentityError):
        return f"{symbol} market data did not match the selected exchange, asset, or timeframe."
    return f"{symbol} market data were unavailable ({type(exc).__name__})."


def _require_current_market(
    market: MarketEvidence,
    context: CollectionContext | None,
) -> None:
    if context is None:
        return
    cutoff = context.collected_at
    if (
        market.collected_at > cutoff
        or market.last_time > cutoff
        or any(candle.timestamp > cutoff for candle in market.candles)
    ):
        raise FutureMarketDataError("Market provider returned future-dated observations.")


def _require_market_identity(
    market: MarketEvidence,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> None:
    """Reject a provider/cache result that does not match this exact request."""

    expected = (exchange.casefold(), symbol.upper(), timeframe.casefold())
    received = (market.exchange.casefold(), market.symbol.upper(), market.timeframe.casefold())
    if received != expected:
        raise MarketDataIdentityError(
            "Market provider returned data for "
            f"{market.exchange} {market.symbol} {market.timeframe}, expected "
            f"{exchange} {symbol} {timeframe}."
        )


def _require_current_scan(
    result: OpportunityScanResult,
    context: CollectionContext | None,
    request: AnalysisRequest,
) -> None:
    _require_scan_identity(result, context, request)
    for candidate in result.candidates:
        _require_discovery_candidate_identity(candidate, request)


def _require_scan_identity(
    result: OpportunityScanResult,
    context: CollectionContext | None,
    request: AnalysisRequest,
) -> None:
    if context is not None and result.collected_at > context.collected_at:
        raise FutureMarketDataError("Market provider returned a future-dated opportunity scan.")
    if result.exchange != request.exchange or result.timeframe != request.timeframe:
        raise MarketDataIdentityError(
            "Market provider returned a discovery scan for a different exchange or timeframe."
        )


def _require_discovery_candidate_identity(
    candidate: OpportunityCandidate,
    request: AnalysisRequest,
) -> None:
    try:
        base, quote = candidate.symbol.split("/", maxsplit=1)
    except ValueError as exc:
        raise MarketDataIdentityError(
            "Market provider returned a discovery candidate without a market pair."
        ) from exc
    expected_quote = request.symbol.split("/", maxsplit=1)[1]
    if base != candidate.asset or quote != expected_quote:
        raise MarketDataIdentityError(
            "Market provider returned a discovery candidate for a different asset or quote."
        )


def _sanitize_derivatives_evidence(
    evidence: DerivativesEvidence,
    *,
    symbol: str,
    context: CollectionContext | None,
) -> DerivativesEvidence:
    expected_asset = symbol.split("/", maxsplit=1)[0].upper()
    invalid_identity = evidence.asset != expected_asset
    future_dated = context is not None and (
        evidence.collected_at > context.collected_at
        or any(item.observed_at > context.collected_at for item in evidence.funding_history)
        or any(item.observed_at > context.collected_at for item in evidence.open_interest_history)
    )
    if not invalid_identity and not future_dated:
        return evidence
    reason = (
        "Derivatives evidence did not match the requested asset."
        if invalid_identity
        else "Future-dated derivatives evidence was excluded."
    )
    return DerivativesEvidence(
        asset=expected_asset,
        status="unavailable",
        collected_at=context.collected_at if context is not None else evidence.collected_at,
        warnings=[*evidence.warnings, reason],
    )


def _build_market_summary(
    market: MarketEvidence,
    technical: TechnicalSnapshot,
    *,
    contextual_timeframes: Sequence[MarketTimeframeEvidence] = (),
) -> str:
    """Build a concrete, dated market briefing from the shared market posture."""

    confirmation = [
        (item.timeframe, item.technical.trend)
        for item in contextual_timeframes
        if item.status == "complete" and item.technical is not None
    ]
    posture = build_market_posture(
        market,
        technical,
        contextual_timeframes=confirmation,
    )
    return _render_market_summary_text(posture)


def _render_market_summary_text(posture: MarketPostureSummary) -> str:
    """Render the bounded summary text from a validated market posture."""

    price_time = posture.as_of.strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        f"As of {price_time}, {posture.symbol} is trading at {format_money(posture.price)} "
        f"on {posture.exchange.title()}"
    ]
    if posture.change_24h_percent is not None:
        parts.append(f"24h {_signed_percent(posture.change_24h_percent)}")
    available_returns = [
        f"{item.label} {_signed_percent(item.return_percent)}"
        for item in posture.window_returns
        if item.status == "available" and item.return_percent is not None
    ]
    if available_returns:
        parts.append("returns " + ", ".join(available_returns))
    parts.append(
        f"range {format_money(posture.low)}-{format_money(posture.high)} "
        f"({posture.range_percent:.1f}%)"
    )
    parts.append(f"quote volume {format_money(posture.quote_volume)}")
    parts.append(f"max drawdown {posture.maximum_drawdown * 100:.1f}%")
    if posture.rsi is not None:
        parts.append(f"RSI {posture.rsi:.1f} ({posture.rsi_band})")
    if posture.macd is not None:
        parts.append(f"MACD {'positive' if posture.macd > 0 else 'negative'}")
    if posture.atr is not None:
        parts.append(f"ATR {format_money(posture.atr)}")
    if posture.volatility is not None:
        parts.append(f"volatility {posture.volatility * 100:.2f}%")
    if posture.support is not None:
        parts.append(f"support {format_money(posture.support)}")
    if posture.resistance is not None:
        parts.append(f"resistance {format_money(posture.resistance)}")
    parts.append(f"trend {posture.trend}")
    if posture.fresh:
        parts.append("data fresh")
    elif posture.data_delay_seconds > 0:
        parts.append(f"data {_delay_label(posture.data_delay_seconds)} old")
    if posture.contextual_confirmation:
        parts.append("higher timeframes " + ", ".join(posture.contextual_confirmation))
    return ". ".join(parts)[:1200] + "."


def _signed_percent(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{value:.1f}%"


def _delay_label(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes / 60
    return f"{hours:.1f} h"


def contextual_timeframes(primary: SupportedTimeframe) -> tuple[SupportedTimeframe, ...]:
    """Return higher-timeframe confirmation evidence for a primary interval."""

    return _CONTEXTUAL_TIMEFRAMES[primary]
