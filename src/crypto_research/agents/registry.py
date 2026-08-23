"""The small, public agent registry used by the research workspace."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_research.agents.base import AgentAnalyzer, AgentManifest
from crypto_research.agents.forecast.forecast_agent import FORECAST_MANIFEST
from crypto_research.agents.forecast.forecast_analyzer import ANALYZER as FORECAST_ANALYZER
from crypto_research.agents.fundamentals.fundamentals_agent import FUNDAMENTALS_MANIFEST
from crypto_research.agents.fundamentals.fundamentals_analyzer import (
    ANALYZER as FUNDAMENTALS_ANALYZER,
)
from crypto_research.agents.market.market_agent import MARKET_MANIFEST
from crypto_research.agents.market.market_analyzer import ANALYZER as MARKET_ANALYZER
from crypto_research.agents.news.news_agent import NEWS_MANIFEST
from crypto_research.agents.news.news_analyzer import ANALYZER as NEWS_ANALYZER
from crypto_research.agents.onchain.onchain_agent import ONCHAIN_MANIFEST
from crypto_research.agents.onchain.onchain_analyzer import ANALYZER as ONCHAIN_ANALYZER
from crypto_research.domain.core import ResearchCapability


@dataclass(frozen=True, slots=True)
class GuidedCapabilityDefinition:
    """A single user-facing Guided Research selection."""

    capability: ResearchCapability
    label: str
    description: str
    scopes: frozenset[ResearchCapability]
    agent_id: str
    collectors: frozenset[str]


REGISTRY: tuple[AgentManifest, ...] = (
    MARKET_MANIFEST,
    FUNDAMENTALS_MANIFEST,
    ONCHAIN_MANIFEST,
    NEWS_MANIFEST,
    FORECAST_MANIFEST,
)
ANALYZERS: dict[str, AgentAnalyzer] = {
    analyzer.id: analyzer
    for analyzer in (
        MARKET_ANALYZER,
        FUNDAMENTALS_ANALYZER,
        ONCHAIN_ANALYZER,
        NEWS_ANALYZER,
        FORECAST_ANALYZER,
    )
}

# Discovery deliberately is not an asset-research selection.  It is an
# assetless market scan that hands a candidate back to the asset selector.
GUIDED_CAPABILITIES: tuple[GuidedCapabilityDefinition, ...] = (
    GuidedCapabilityDefinition(
        ResearchCapability.MARKET,
        "Market behavior",
        "Price action, trend, momentum, and technical signals.",
        frozenset({ResearchCapability.MARKET}),
        "market_agent",
        frozenset({"market"}),
    ),
    GuidedCapabilityDefinition(
        ResearchCapability.RISK,
        "Risk assessment",
        "Deterministic risk metrics, volatility, and evidence coverage.",
        frozenset({ResearchCapability.RISK}),
        "market_agent",
        frozenset({"market", "risk"}),
    ),
    GuidedCapabilityDefinition(
        ResearchCapability.DERIVATIVES,
        "Derivatives positioning",
        "Public perpetual-futures funding and open-interest context from Binance.",
        frozenset({ResearchCapability.DERIVATIVES}),
        "market_agent",
        frozenset({"market", "derivatives"}),
    ),
    GuidedCapabilityDefinition(
        ResearchCapability.FUNDAMENTALS,
        "Fundamentals",
        "Token fundamentals, supply context, and project metrics.",
        frozenset({ResearchCapability.FUNDAMENTALS}),
        "fundamentals_agent",
        frozenset({"fundamentals"}),
    ),
    GuidedCapabilityDefinition(
        ResearchCapability.DEFI,
        "DeFi activity",
        "Protocol TVL and DeFi metrics for supported protocol tokens.",
        frozenset({ResearchCapability.DEFI}),
        "fundamentals_agent",
        frozenset({"defi"}),
    ),
    GuidedCapabilityDefinition(
        ResearchCapability.ONCHAIN,
        "On-Chain Activity",
        "Active and new addresses, transactions, transfer value, and network fees.",
        frozenset({ResearchCapability.ONCHAIN}),
        "onchain_agent",
        frozenset({"onchain"}),
    ),
    GuidedCapabilityDefinition(
        ResearchCapability.NEWS,
        "Recent news",
        "Recent, relevant reporting and its stated context.",
        frozenset({ResearchCapability.NEWS}),
        "news_agent",
        frozenset({"news"}),
    ),
    GuidedCapabilityDefinition(
        ResearchCapability.FORECAST,
        "Price forecast",
        "Deterministic ML forecasts with an isolated LLM interpretation.",
        frozenset({ResearchCapability.FORECAST}),
        "forecast_agent",
        frozenset({"forecast"}),
    ),
)

_CAPABILITY_BY_ID = {definition.capability: definition for definition in GUIDED_CAPABILITIES}


def expand_capabilities(caps: set[ResearchCapability]) -> set[ResearchCapability]:
    """Translate public choices into the scopes used by collectors."""

    expanded: set[ResearchCapability] = set()
    for capability in caps:
        definition = _CAPABILITY_BY_ID.get(capability)
        expanded.update(definition.scopes if definition else {capability})
    return expanded


def capability_groups() -> tuple[tuple[str, tuple[ResearchCapability, ...]], ...]:
    """Return the stable user-facing research lenses and their topics."""

    return (
        (
            "Market lens",
            (
                ResearchCapability.MARKET,
                ResearchCapability.RISK,
                ResearchCapability.DERIVATIVES,
                ResearchCapability.FORECAST,
            ),
        ),
        (
            "Project & network",
            (
                ResearchCapability.FUNDAMENTALS,
                ResearchCapability.DEFI,
                ResearchCapability.ONCHAIN,
            ),
        ),
        ("Information", (ResearchCapability.NEWS,)),
    )


def agents_for(caps: set[ResearchCapability]) -> list[AgentManifest]:
    return [manifest for manifest in REGISTRY if manifest.capabilities.intersection(caps)]


def compile_capability_route(capabilities: list[ResearchCapability]) -> list[str]:
    """Compile the ordered specialist route for explicit research capabilities."""

    return [manifest.id for manifest in agents_for(set(capabilities))]


def collectors_for(agent_id: str, capabilities: set[ResearchCapability]) -> frozenset[str]:
    return frozenset(
        collector
        for definition in GUIDED_CAPABILITIES
        if definition.agent_id == agent_id and definition.capability in capabilities
        for collector in definition.collectors
    )


def manifest_for(agent_id: str) -> AgentManifest:
    for manifest in REGISTRY:
        if manifest.id == agent_id:
            return manifest
    raise ValueError(f"Unknown agent: {agent_id}")


def analyzer_for(agent_id: str) -> AgentAnalyzer:
    try:
        return ANALYZERS[agent_id]
    except KeyError as exc:
        raise ValueError(f"Unknown agent analyzer: {agent_id}") from exc


def labels() -> dict[str, str]:
    return {manifest.id: manifest.label for manifest in REGISTRY}


def capability_options() -> tuple[tuple[ResearchCapability, str, str], ...]:
    return tuple(
        (definition.capability, definition.label, definition.description)
        for definition in GUIDED_CAPABILITIES
    )


def guided_capability(capability: ResearchCapability) -> GuidedCapabilityDefinition | None:
    return _CAPABILITY_BY_ID.get(capability)


__all__ = [
    "AgentManifest",
    "ANALYZERS",
    "FUNDAMENTALS_MANIFEST",
    "FORECAST_MANIFEST",
    "GUIDED_CAPABILITIES",
    "GuidedCapabilityDefinition",
    "MARKET_MANIFEST",
    "NEWS_MANIFEST",
    "ONCHAIN_MANIFEST",
    "REGISTRY",
    "agents_for",
    "analyzer_for",
    "capability_groups",
    "capability_options",
    "compile_capability_route",
    "expand_capabilities",
    "collectors_for",
    "guided_capability",
    "labels",
    "manifest_for",
]
