"""Deterministic driving layer for explicit guided research actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from crypto_research.agents.registry import (
    capability_groups as _manifest_capability_groups,
)
from crypto_research.agents.registry import (
    capability_options as _manifest_capability_options,
)
from crypto_research.agents.registry import (
    collectors_for,
    compile_capability_route,
    manifest_for,
)
from crypto_research.agents.registry import (
    expand_capabilities as _expand_capabilities,
)
from crypto_research.agents.registry import (
    guided_capability as _guided_capability,
)
from crypto_research.agents.registry import (
    labels as _manifest_labels,
)
from crypto_research.domain.core import (
    ASSET_ALIASES,
    COIN_ID_BY_ASSET,
    MAX_COMPARISON_ASSETS,
    SUPPORTED_EXCHANGES,
    SUPPORTED_TIMEFRAMES,
    ResearchCapability,
    SupportedExchange,
    SupportedTimeframe,
    build_market_symbol,
    canonical_asset_symbol,
)
from crypto_research.domain.forecast import ForecastSettings
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisRequest,
    ResearchAction,
)
from crypto_research.tools.fundamentals import defillama_slug_for

_GUIDED_CONTEXTUAL_TIMEFRAMES: dict[str, tuple[SupportedTimeframe, ...]] = {
    "1m": ("4h", "1d"),
    "5m": ("4h", "1d"),
    "15m": ("4h", "1d"),
    "30m": ("4h", "1d"),
    "1h": ("4h", "1d"),
    "4h": ("1d",),
    "1d": (),
}
_MARKET_DATA_CAPABILITIES = frozenset(
    {
        ResearchCapability.MARKET,
        ResearchCapability.DISCOVERY,
        ResearchCapability.RISK,
        ResearchCapability.FORECAST,
        ResearchCapability.DERIVATIVES,
    }
)
type GuidedResearchMode = Literal["asset", "discovery"]


@dataclass(frozen=True, slots=True)
class GuidedAgentExecution:
    """One specialist's isolated capability and collector assignment."""

    agent_id: str
    capabilities: tuple[ResearchCapability, ...]
    collectors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuidedResearchPlan:
    action: ResearchAction
    mode: GuidedResearchMode
    scope_label: str
    display_capabilities: tuple[ResearchCapability, ...]
    requested_capabilities: tuple[ResearchCapability, ...]
    expected_agents: tuple[str, ...]
    collectors: tuple[str, ...]
    contextual_timeframes: tuple[SupportedTimeframe, ...] = ()
    uses_market_data: bool = False
    execution: tuple[GuidedAgentExecution, ...] = ()


def compile_guided_research_plan(
    assets: Sequence[str],
    capabilities: Sequence[ResearchCapability | str],
    *,
    mode: GuidedResearchMode | str = "asset",
    exchange: str = "kraken",
    timeframe: str = "1h",
    forecast_settings: ForecastSettings | None = None,
) -> GuidedResearchPlan:
    selected_mode = _normalize_mode(mode)
    normalized_assets = _normalize_guided_assets(assets)
    display_capabilities = _normalize_capabilities(capabilities)
    selected_exchange = _normalize_exchange(exchange)
    selected_timeframe = _normalize_timeframe(timeframe)
    if selected_mode == "discovery":
        _validate_discovery_selection(normalized_assets, display_capabilities)
        requested_capabilities = [ResearchCapability.DISCOVERY]
        analysis_assets: list[AnalysisAsset] = []
        intent = "Market-wide discovery"
        scope_label = "Market-wide discovery"
    else:
        _validate_asset_selection(normalized_assets, display_capabilities)
        expanded = _expand_capabilities(set(display_capabilities))
        requested_capabilities = [cap for cap in ResearchCapability if cap in expanded]
        analysis_assets = _analysis_assets(normalized_assets, selected_exchange)
        intent = _build_intent(normalized_assets, display_capabilities)
        scope_label = "Assets: " + ", ".join(normalized_assets)
    request = AnalysisRequest(
        user_intent=intent,
        assets=analysis_assets,
        exchange=selected_exchange,
        timeframe=selected_timeframe,
        candle_limit=250 if ResearchCapability.DISCOVERY in requested_capabilities else 750,
        symbol=build_market_symbol("BTC", selected_exchange),
        coin_id=None if selected_mode == "discovery" else "bitcoin",
        comparison_symbols=(
            [asset.symbol for asset in analysis_assets] if len(normalized_assets) > 1 else []
        ),
        forecast_settings=(
            forecast_settings if ResearchCapability.FORECAST in requested_capabilities else None
        ),
    )
    route = compile_capability_route(requested_capabilities)
    action = ResearchAction(
        reasoning="Structured research request from guided interface.",
        request=request,
        agents_to_call=route,
        requested_capabilities=requested_capabilities,
    )
    collectors = tuple(
        sorted(
            {
                collector
                for capability in requested_capabilities
                if (definition := _guided_capability(capability)) is not None
                for collector in definition.collectors
            }
        )
    )
    uses_market_data = bool(set(requested_capabilities) & _MARKET_DATA_CAPABILITIES)
    return GuidedResearchPlan(
        action=action,
        mode=selected_mode,
        scope_label=scope_label,
        display_capabilities=tuple(display_capabilities),
        requested_capabilities=tuple(requested_capabilities),
        expected_agents=tuple(route),
        collectors=collectors,
        contextual_timeframes=(
            ()
            if selected_mode == "discovery" or not uses_market_data
            else _GUIDED_CONTEXTUAL_TIMEFRAMES[selected_timeframe]
        ),
        uses_market_data=uses_market_data,
        execution=_compile_execution_matrix(requested_capabilities, route),
    )


def agent_labels() -> dict[str, str]:
    """Return agent_id to label mapping for UI display."""

    return _manifest_labels()


def capability_options() -> tuple[tuple[ResearchCapability, str, str], ...]:
    """Return (capability, label, description) for UI toggles."""

    return _manifest_capability_options()


def defi_eligible_assets(assets: Sequence[str]) -> tuple[str, ...]:
    """Return selected assets that have registered DefiLlama protocol metrics."""

    normalized = _normalize_guided_assets(assets)
    return tuple(
        asset for asset in normalized if defillama_slug_for(COIN_ID_BY_ASSET.get(asset)) is not None
    )


def capability_groups() -> tuple[tuple[str, tuple[ResearchCapability, ...]], ...]:
    """Return (group_title, capabilities) for UI capability grouping."""

    return _manifest_capability_groups()


def available_assets() -> list[str]:
    """Return the predetermined asset list for the UI."""

    return list(ASSET_ALIASES.keys())


def asset_display_name(ticker: str) -> str:
    """Return a friendly 'TICKER — Name' label for a supported asset ticker."""

    aliases = ASSET_ALIASES.get(ticker)
    if aliases:
        name = next((word for word in aliases if word != ticker.casefold()), aliases[0])
        return f"{ticker} — {name.title()}"
    return ticker


def _build_intent(
    assets: Sequence[str],
    capabilities: list[ResearchCapability],
) -> str:
    """Build a concise user-intent string from the selected parameters."""

    labels = {cap.value: label for cap, label, _ in _manifest_capability_options()}
    cap_labels = [labels.get(cap.value, cap.value) for cap in capabilities]
    if len(assets) > 1:
        intent = f"Compare {', '.join(assets)}: {' + '.join(cap_labels)}"
    else:
        intent = f"Research {', '.join(assets)}: {' + '.join(cap_labels)}"
    return intent


def _normalize_capabilities(
    capabilities: Sequence[ResearchCapability | str],
) -> list[ResearchCapability]:
    normalized: list[ResearchCapability] = []
    for value in capabilities:
        try:
            capability = (
                value
                if isinstance(value, ResearchCapability)
                else ResearchCapability(str(value).strip().casefold())
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown research capability: {value!r}") from exc
        definition = _guided_capability(capability)
        if definition is None and capability is not ResearchCapability.DISCOVERY:
            raise ValueError(f"{capability.value.title()} is not available in Guided Research.")
        if capability not in normalized:
            normalized.append(capability)
    return normalized


def _normalize_mode(value: GuidedResearchMode | str) -> GuidedResearchMode:
    normalized = value.strip().casefold() if isinstance(value, str) else ""
    if normalized in {"asset", "discovery"}:
        return cast("GuidedResearchMode", normalized)
    raise ValueError("Guided research mode must be asset research or market discovery.")


def _normalize_exchange(value: str) -> SupportedExchange:
    normalized = value.strip().casefold() if isinstance(value, str) else ""
    if normalized not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Unsupported exchange: {value}")
    return normalized


def _normalize_timeframe(value: str) -> SupportedTimeframe:
    normalized = value.strip().casefold() if isinstance(value, str) else ""
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {value}")
    return normalized


def _normalize_guided_assets(assets: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in assets:
        if not isinstance(value, str):
            raise ValueError(f"Unknown guided asset: {value!r}")
        symbol = canonical_asset_symbol(value.strip())
        if symbol not in ASSET_ALIASES:
            raise ValueError(f"Unknown guided asset: {value!r}")
        if symbol not in normalized:
            normalized.append(symbol)
    return normalized


def _validate_asset_selection(
    assets: Sequence[str],
    capabilities: Sequence[ResearchCapability],
) -> None:
    if not assets:
        raise ValueError("Select at least one supported asset for asset research.")
    if len(assets) > MAX_COMPARISON_ASSETS:
        raise ValueError("Guided research supports at most four assets.")
    if not capabilities:
        raise ValueError("Select at least one research topic.")
    if ResearchCapability.DISCOVERY in capabilities:
        raise ValueError("Use Market discovery mode for a market-wide scan.")
    if ResearchCapability.DEFI in capabilities and not defi_eligible_assets(assets):
        raise ValueError("DeFi metrics are only available for DeFi protocol assets.")


def _validate_discovery_selection(
    assets: Sequence[str],
    capabilities: Sequence[ResearchCapability],
) -> None:
    if assets:
        raise ValueError("Market discovery scans the exchange watchlist and does not use assets.")
    if set(capabilities) != {ResearchCapability.DISCOVERY}:
        raise ValueError("Market discovery runs only the market-wide Discovery evidence set.")


def _analysis_assets(assets: Sequence[str], exchange: SupportedExchange) -> list[AnalysisAsset]:
    return [
        AnalysisAsset(
            requested_name=asset,
            symbol=build_market_symbol(asset, exchange),
            coin_id=COIN_ID_BY_ASSET[asset],
        )
        for asset in assets
    ]


def _compile_execution_matrix(
    capabilities: Sequence[ResearchCapability],
    route: Sequence[str],
) -> tuple[GuidedAgentExecution, ...]:
    """Return explicit specialist ownership without duplicating collector work."""

    requested = set(capabilities)
    executions: list[GuidedAgentExecution] = []
    for agent_id in route:
        owned = tuple(
            capability
            for capability in ResearchCapability
            if capability in requested and capability in manifest_for(agent_id).capabilities
        )
        if not owned:
            continue
        collectors = tuple(sorted(collectors_for(agent_id, set(owned))))
        executions.append(
            GuidedAgentExecution(
                agent_id=agent_id,
                capabilities=owned,
                collectors=collectors,
            )
        )
    return tuple(executions)


__all__ = [
    "agent_labels",
    "asset_display_name",
    "available_assets",
    "compile_guided_research_plan",
    "GuidedAgentExecution",
    "GuidedResearchPlan",
    "GuidedResearchMode",
    "capability_groups",
    "capability_options",
    "defi_eligible_assets",
]
