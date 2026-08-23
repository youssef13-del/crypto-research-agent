"""Pure presentation models and builders shared by the web UI and PDF export."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from crypto_research.domain.analytics import build_market_posture
from crypto_research.domain.evidence import (
    DefiEvidence,
    DerivativesEvidence,
    FundamentalEvidence,
    normalize_news_items,
)
from crypto_research.domain.forecast import ForecastAgentResult, ForecastFailure
from crypto_research.domain.market import ComparisonMetrics, MarketEvidence
from crypto_research.domain.research import (
    AgentAnswer,
    AgentExecutionStatus,
    AnalysisAsset,
    AssetResearchBundle,
    NewsEvidence,
    NewsItem,
    ResearchCapability,
    ResearchReport,
    TechnicalSnapshot,
)
from crypto_research.orchestration.planning import agent_labels
from crypto_research.shared.formatting import format_compact_number
from crypto_research.shared.security import normalize_http_url, redact_secrets


@dataclass(frozen=True, slots=True)
class AssetPresentation:
    exchange: str
    symbol: str
    timeframe: str
    current_price: float
    technical: TechnicalSnapshot
    comparison_metrics: ComparisonMetrics | None = None
    candle_timestamps: tuple[datetime, ...] = ()
    candle_closes: tuple[float, ...] = ()
    collected_at: datetime | None = None
    first_time: datetime | None = None
    last_time: datetime | None = None
    candle_count: int = 0


@dataclass(frozen=True, slots=True)
class DashboardView:
    snapshots: tuple[tuple[MarketEvidence, TechnicalSnapshot], ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourcePresentation:
    publisher: str
    title: str
    url: str | None
    published_at: datetime
    kind: str = "News"
    time_context: Literal["Published", "Collected"] = "Published"


@dataclass(frozen=True, slots=True)
class AgentClaimPresentation:
    statement: str
    claim_kind: str
    evidence_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class AgentAnalysisSectionPresentation:
    asset: str
    scope: Literal[
        "market",
        "risk",
        "derivatives",
        "fundamentals",
        "defi",
        "news",
        "forecast",
        "onchain",
    ]
    text: str


@dataclass(frozen=True, slots=True)
class StructuredAgentAnalysisPresentation:
    verdict: str
    sections: tuple[AgentAnalysisSectionPresentation, ...]
    comparison: str = ""


@dataclass(frozen=True, slots=True)
class AgentAnswerPresentation:
    agent: str
    title: str
    answer: str
    claims: tuple[AgentClaimPresentation, ...]
    uncertainty: tuple[str, ...]
    limitations: tuple[str, ...]
    confidence: float
    status: Literal["complete", "partial", "unavailable"]
    analysis_state: Literal["live", "evidence_only", "unavailable"] = "live"
    coverage_state: Literal["complete", "partial", "not_applicable"] = "complete"
    technical_terms: tuple[str, ...] = ()
    analysis: str = ""
    structured_analysis: StructuredAgentAnalysisPresentation | None = None


@dataclass(frozen=True, slots=True)
class AgentStatusPresentation:
    agent: str
    status: Literal["complete", "partial", "unavailable"]
    limitation: str | None = None
    capabilities: tuple[str, ...] = ()
    analysis_state: Literal["live", "evidence_only", "unavailable"] = "live"
    coverage_state: Literal["complete", "partial", "not_applicable"] = "complete"
    source_state: Literal["live", "cached", "partial", "unavailable"] = "live"


@dataclass(frozen=True, slots=True)
class CapabilityDataPresentation:
    agent: str
    capability: str
    title: str
    asset: str | None
    facts: tuple[tuple[str, str], ...] = ()
    status: Literal["complete", "partial", "unavailable"] = "complete"
    limitation: str | None = None


@dataclass(frozen=True, slots=True)
class AgentPanelPresentation:
    agent: str
    title: str
    status: Literal["complete", "partial", "unavailable"]
    state_label: str = ""
    capabilities: tuple[str, ...] = ()
    data: tuple[CapabilityDataPresentation, ...] = ()
    answer: AgentAnswerPresentation | None = None
    limitation: str | None = None
    analysis_state: Literal["live", "evidence_only", "unavailable"] = "live"
    coverage_state: Literal["complete", "partial", "not_applicable"] = "complete"
    source_state: Literal["live", "cached", "partial", "unavailable"] = "live"


@dataclass(frozen=True, slots=True)
class DiscoveryCandidatePresentation:
    rank: int
    asset: str
    symbol: str
    current_price: float
    momentum_24h: float
    volatility_24h: float
    score: float
    trend: str
    reason: str


@dataclass(frozen=True, slots=True)
class DiscoveryPresentation:
    exchange: str
    timeframe: str
    collected_at: datetime
    summary: str
    candidates: tuple[DiscoveryCandidatePresentation, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchPresentation:
    status: Literal["complete", "partial"]
    warnings: tuple[str, ...]
    sources: tuple[SourcePresentation, ...]
    disclaimer: str
    assets: tuple[AssetPresentation, ...]
    title: str = "Research brief"
    route: tuple[str, ...] = ()
    verified_facts: tuple[str, ...] = ()
    calculations: tuple[str, ...] = ()
    analysis_points: tuple[str, ...] = ()
    speculation: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    agent_statuses: tuple[AgentStatusPresentation, ...] = ()
    agent_panels: tuple[AgentPanelPresentation, ...] = ()
    discovery: DiscoveryPresentation | None = None
    forecast_result: ForecastAgentResult | None = None
    run_id: str | None = None
    retry_of_run_id: str | None = None
    retried_agents: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchTurn:
    content: str
    research: ResearchPresentation
    agents: tuple[str, ...] = ()
    failed_agents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        content = redact_secrets(self.content.strip())
        if not content:
            raise ValueError("Research messages cannot be empty.")
        object.__setattr__(self, "content", content)
        object.__setattr__(
            self,
            "agents",
            tuple(redact_secrets(agent.strip()) for agent in self.agents if agent.strip()),
        )
        object.__setattr__(
            self,
            "failed_agents",
            tuple(redact_secrets(agent.strip()) for agent in self.failed_agents if agent.strip()),
        )


_AGENT_LABELS = agent_labels()


def build_agent_answer_presentation(answer: AgentAnswer) -> AgentAnswerPresentation:
    """Convert a validated specialist answer into an escaped display model."""

    structured = answer.structured_analysis
    return AgentAnswerPresentation(
        agent=answer.agent,
        title=_AGENT_LABELS.get(answer.agent, "Independent research analysis"),
        answer=redact_secrets(answer.answer),
        technical_terms=tuple(redact_secrets(item) for item in answer.technical_terms),
        analysis=redact_secrets(answer.analysis),
        structured_analysis=(
            StructuredAgentAnalysisPresentation(
                verdict=redact_secrets(structured.verdict),
                sections=tuple(
                    AgentAnalysisSectionPresentation(
                        asset=redact_secrets(section.asset),
                        scope=section.scope,
                        text=redact_secrets(section.text),
                    )
                    for section in structured.sections
                ),
                comparison=redact_secrets(structured.comparison),
            )
            if structured is not None
            else None
        ),
        claims=tuple(
            AgentClaimPresentation(
                statement=redact_secrets(claim.statement),
                claim_kind=claim.claim_kind,
                evidence_ids=tuple(redact_secrets(item) for item in claim.evidence_ids),
                confidence=claim.confidence,
            )
            for claim in answer.evidence
        ),
        uncertainty=tuple(redact_secrets(item) for item in answer.uncertainty),
        limitations=tuple(redact_secrets(item) for item in answer.limitations),
        confidence=answer.confidence,
        status=answer.status,
        analysis_state=answer.analysis_state,
        coverage_state=answer.coverage_state,
    )


CHART_HISTORY_LIMIT = 120


def build_asset_presentations(
    result: ResearchReport, *, history_limit: int = CHART_HISTORY_LIMIT
) -> list[AssetPresentation]:
    """Build chart-ready market cards while retaining the latest complete history."""

    candidates: list[tuple[MarketEvidence, TechnicalSnapshot, ComparisonMetrics | None]] = []
    if result.market_result:
        candidates.append((result.market_result.market, result.market_result.technical, None))
    if result.market_comparison_result:
        candidates.extend(
            (asset.market, asset.technical, asset.metrics)
            for asset in result.market_comparison_result.assets
        )

    selected: dict[
        tuple[str, str, str],
        tuple[MarketEvidence, TechnicalSnapshot, ComparisonMetrics | None],
    ] = {}
    for market, technical, metrics in candidates:
        key = (market.exchange.casefold(), market.symbol.casefold(), market.timeframe.casefold())
        current = selected.get(key)
        candidate_rank = (market.last_time, market.collected_at)
        current_rank = (current[0].last_time, current[0].collected_at) if current else None
        if current_rank is None or candidate_rank > current_rank:
            selected[key] = (market, technical, metrics)

    return [
        _asset_presentation(market, technical, metrics, history_limit=history_limit)
        for market, technical, metrics in selected.values()
    ]


def _asset_presentation(
    market: MarketEvidence,
    technical: TechnicalSnapshot,
    metrics: ComparisonMetrics | None,
    *,
    history_limit: int,
) -> AssetPresentation:
    tail = market.candles[-history_limit:]
    return AssetPresentation(
        exchange=redact_secrets(market.exchange.strip()),
        symbol=redact_secrets(market.symbol.strip()),
        timeframe=redact_secrets(market.timeframe.strip()),
        current_price=market.current_price,
        technical=technical,
        comparison_metrics=metrics,
        candle_timestamps=tuple(candle.timestamp for candle in tail),
        candle_closes=tuple(candle.close for candle in tail),
        collected_at=market.collected_at,
        first_time=market.first_time,
        last_time=market.last_time,
        candle_count=len(market.candles),
    )


def format_price(value: float) -> str:
    absolute = abs(value)
    if absolute == 0 or absolute >= 1:
        return f"${value:,.2f}"
    if absolute < 1e-11:
        return f"${value:.4e}"
    decimals = min(12, max(2, 3 - math.floor(math.log10(absolute))))
    return f"${value:,.{decimals}f}"


def onchain_data_cards(
    result: ResearchReport,
    *,
    cutoff: datetime,
) -> list[CapabilityDataPresentation]:
    """Build compact per-asset network-activity cards."""

    if result.onchain_result is None:
        return []
    cards: list[CapabilityDataPresentation] = []
    for bundle in result.onchain_result.asset_results:
        evidence = bundle.onchain
        if evidence is None:
            continue
        if evidence.collected_at > cutoff:
            cards.append(
                CapabilityDataPresentation(
                    agent="onchain_agent",
                    capability=ResearchCapability.ONCHAIN.value,
                    title=f"{bundle.asset.symbol} on-chain activity",
                    asset=bundle.asset.symbol,
                    status="partial",
                    limitation="Future-dated on-chain data were excluded before display.",
                )
            )
            continue
        facts: list[tuple[str, str]] = []
        for metric in evidence.metrics:
            value = format_compact_number(metric.latest_value, currency=metric.unit == "usd")
            if metric.seven_day_change_pct is not None:
                value += f" ({metric.seven_day_change_pct:+.1f}% vs prior 7d)"
            facts.append((metric.label, value))
        if not facts:
            facts.append(("Data status", "Unavailable"))
        cards.append(
            CapabilityDataPresentation(
                agent="onchain_agent",
                capability=ResearchCapability.ONCHAIN.value,
                title=f"{bundle.asset.symbol} on-chain activity",
                asset=bundle.asset.symbol,
                facts=tuple(facts),
                status=(
                    "complete"
                    if evidence.status == "complete"
                    else "partial"
                    if evidence.metrics
                    else "unavailable"
                ),
                limitation=None,
            )
        )
    return cards


def relative_age_label(value: datetime, *, cutoff: datetime) -> str:
    elapsed = max(cutoff - value.astimezone(UTC), timedelta(0))
    if elapsed < timedelta(hours=1):
        return f"{max(1, int(elapsed.total_seconds() // 60))} min ago"
    if elapsed < timedelta(days=1):
        return f"{max(1, int(elapsed.total_seconds() // 3600))}h ago"
    return f"{max(1, elapsed.days)}d ago"


def forecast_data_cards(result: ResearchReport) -> list[CapabilityDataPresentation]:
    batch = result.forecast_result
    if batch is None:
        return []
    cards: list[CapabilityDataPresentation] = []
    for item in batch.asset_results:
        if isinstance(item, ForecastFailure):
            cards.append(
                CapabilityDataPresentation(
                    agent="forecast_agent",
                    capability=ResearchCapability.FORECAST.value,
                    title=f"{item.request.symbol} forecast",
                    asset=item.request.symbol,
                    facts=(("Quality", "Unavailable"),),
                    status="unavailable",
                    limitation=item.message,
                )
            )
            continue
        point = item.model_output
        cards.append(
            CapabilityDataPresentation(
                agent="forecast_agent",
                capability=ResearchCapability.FORECAST.value,
                title=f"{item.request.symbol} forecast",
                asset=item.request.symbol,
                facts=(
                    ("Current price", format_price(item.market.current_price)),
                    ("Model output", format_price(point.predicted_price)),
                    ("Predicted return", f"{point.predicted_return:+.2%}"),
                    ("Target time", point.timestamp.strftime("%d %b %Y %H:%M UTC")),
                    (
                        "Interval",
                        f"{format_price(point.lower_interval)} to "
                        f"{format_price(point.upper_interval)}",
                    ),
                    ("Model", item.model.display_name),
                    (
                        "Quality",
                        "Validation passed" if item.quality.passed else "Not trusted",
                    ),
                    ("MAE", f"{item.metrics.mae:.4%}"),
                    ("Baseline MAE", f"{item.metrics.baseline_mae:.4%}"),
                    ("Directional accuracy", f"{item.metrics.directional_accuracy:.1%}"),
                ),
                status="complete" if item.quality.passed else "partial",
                limitation=None,
            )
        )
    return cards


SOURCE_ICONS = {
    "Market data": "MKT",
    "News": "NEWS",
    "Fundamentals": "FUND",
    "DeFi": "DEFI",
    "Derivatives": "DERIV",
}

_SPECIALIST_AGENTS = (
    "market_agent",
    "fundamentals_agent",
    "onchain_agent",
    "news_agent",
    "forecast_agent",
)
_RESEARCH_CAPABILITIES = frozenset({ResearchCapability.NEWS})
_FUNDAMENTALS_CAPABILITIES = frozenset({ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI})
type PresentationStatus = Literal["complete", "partial", "unavailable"]


def _capability_label(capability: ResearchCapability | str) -> str:
    value = capability.value if isinstance(capability, ResearchCapability) else str(capability)
    labels = {
        ResearchCapability.MARKET.value: "Market data",
        ResearchCapability.DISCOVERY.value: "Discovery",
        ResearchCapability.NEWS.value: "News",
        ResearchCapability.FUNDAMENTALS.value: "Fundamentals",
        ResearchCapability.DEFI.value: "DeFi",
        ResearchCapability.RISK.value: "Risk",
        ResearchCapability.FORECAST.value: "Forecast",
        ResearchCapability.ONCHAIN.value: "On-chain activity",
        ResearchCapability.DERIVATIVES.value: "Derivatives positioning",
    }
    return labels.get(value, value.replace("_", " ").title())


def build_research_presentation(
    result: ResearchReport,
    *,
    route: Sequence[str] = (),
    run_id: str | None = None,
) -> ResearchPresentation:
    source_answers = [*result.agent_answers]
    sections = _answer_sections(source_answers)
    return ResearchPresentation(
        status=result.status,
        warnings=tuple(redact_secrets(item) for item in collect_warnings(result)),
        sources=tuple(_source_presentations(result)),
        disclaimer=redact_secrets(result.disclaimer.strip()),
        # Market data is represented in the specialist cards.  Avoid carrying
        # chart-ready candle copies through Guided Research presentation state.
        assets=(),
        title=redact_secrets(_research_title(result)),
        route=tuple(redact_secrets(str(item).strip()) for item in route if str(item).strip()),
        verified_facts=sections["observed_fact"],
        calculations=sections["calculation"],
        analysis_points=sections["interpretation"],
        speculation=sections["speculation"],
        risks=sections["risk"],
        limitations=tuple(
            redact_secrets(item)
            for answer in source_answers
            for item in answer.limitations
            if item.strip()
        ),
        agent_statuses=tuple(
            AgentStatusPresentation(
                agent=item.agent,
                status=item.status,
                limitation=redact_secrets(item.limitation) if item.limitation else None,
                capabilities=tuple(capability.value for capability in item.capabilities),
                analysis_state=item.analysis_state,
                coverage_state=item.coverage_state,
                source_state=item.source_state,
            )
            for item in result.agent_statuses
        ),
        agent_panels=_build_agent_panels(result, route=route),
        discovery=_build_discovery_presentation(result),
        forecast_result=result.forecast_result,
        run_id=run_id,
        retry_of_run_id=(result.retry.original_run_id if result.retry is not None else None),
        retried_agents=(tuple(result.retry.retried_agents) if result.retry is not None else ()),
    )


def _build_agent_panels(
    result: ResearchReport,
    *,
    route: Sequence[str],
) -> tuple[AgentPanelPresentation, ...]:
    answers: dict[str, AgentAnswerPresentation] = {
        str(answer.agent): build_agent_answer_presentation(answer)
        for answer in result.agent_answers
        if answer.agent in _SPECIALIST_AGENTS
    }
    statuses: dict[str, AgentExecutionStatus] = {
        str(item.agent): item for item in result.agent_statuses if item.agent in _SPECIALIST_AGENTS
    }
    ordered = _ordered_specialist_agents(result, route=route, answers=answers, statuses=statuses)
    panels: list[AgentPanelPresentation] = []
    for agent in ordered:
        status_item = statuses.get(agent)
        capabilities = _panel_capabilities(result, agent, status_item)
        data = _agent_data_cards(result, agent, capabilities)
        answer = answers.get(agent)
        if status_item is None and answer is None and not data:
            continue
        status = _presentation_status(
            status_item.status if status_item is not None else None,
            has_content=bool(answer or data),
        )
        analysis_state = (
            status_item.analysis_state
            if status_item is not None
            else answer.analysis_state
            if answer is not None
            else "live"
        )
        coverage_state = (
            status_item.coverage_state
            if status_item is not None
            else answer.coverage_state
            if answer is not None
            else "complete"
        )
        source_state = status_item.source_state if status_item is not None else "live"
        limitation = _first_nonempty(
            status_item.limitation if status_item is not None else None,
            *(answer.limitations if answer is not None else ()),
            *(card.limitation for card in data),
        )
        panels.append(
            AgentPanelPresentation(
                agent=agent,
                title=_AGENT_LABELS.get(agent, agent.replace("_", " ").title()),
                status=status,
                state_label=_agent_state_label(status_item, status),
                capabilities=tuple(capability.value for capability in capabilities),
                data=tuple(data),
                answer=answer,
                limitation=redact_secrets(limitation) if limitation else None,
                analysis_state=analysis_state,
                coverage_state=coverage_state,
                source_state=source_state,
            )
        )
    return tuple(panels)


def _ordered_specialist_agents(
    result: ResearchReport,
    *,
    route: Sequence[str],
    answers: dict[str, AgentAnswerPresentation],
    statuses: dict[str, AgentExecutionStatus],
) -> tuple[str, ...]:
    if statuses:
        return tuple(
            dict.fromkeys(
                candidate
                for candidate in (*route, *statuses, *answers)
                if candidate in _SPECIALIST_AGENTS and candidate in statuses
            )
        )
    candidates = [*route, *statuses, *answers]
    if result.opportunity_result or result.market_result or result.market_comparison_result:
        candidates.append("market_agent")
    if result.research_result is not None:
        candidates.append("news_agent")
    if result.risk_result is not None:
        candidates.append("market_agent")
    if result.fundamentals_result is not None:
        candidates.append("fundamentals_agent")
    if result.onchain_result is not None:
        candidates.append("onchain_agent")
    if result.forecast_result is not None:
        candidates.append("forecast_agent")
    return tuple(
        dict.fromkeys(candidate for candidate in candidates if candidate in _SPECIALIST_AGENTS)
    )


def _panel_capabilities(
    result: ResearchReport,
    agent: str,
    status_item: AgentExecutionStatus | None,
) -> tuple[ResearchCapability, ...]:
    explicit = _coerce_capabilities(status_item.capabilities if status_item is not None else ())
    if explicit:
        if agent == "fundamentals_agent":
            return tuple(
                capability for capability in explicit if capability in _FUNDAMENTALS_CAPABILITIES
            )
        if agent == "market_agent":
            return tuple(
                capability
                for capability in explicit
                if capability
                in {
                    ResearchCapability.MARKET,
                    ResearchCapability.DISCOVERY,
                    ResearchCapability.RISK,
                    ResearchCapability.DERIVATIVES,
                }
            )
        return explicit
    if agent == "market_agent":
        capabilities: list[ResearchCapability] = []
        if result.opportunity_result is not None:
            capabilities.append(ResearchCapability.DISCOVERY)
        if result.market_result is not None or result.market_comparison_result is not None:
            capabilities.append(ResearchCapability.MARKET)
        if result.risk_result is not None:
            capabilities.append(ResearchCapability.RISK)
        if (result.market_result is not None and result.market_result.derivatives is not None) or (
            result.market_comparison_result is not None
            and any(item.derivatives is not None for item in result.market_comparison_result.assets)
        ):
            capabilities.append(ResearchCapability.DERIVATIVES)
        return tuple(capabilities)
    if agent == "news_agent" and result.research_result is not None:
        requested = _coerce_capabilities(result.research_result.requested_capabilities)
        return tuple(capability for capability in requested if capability in _RESEARCH_CAPABILITIES)
    fundamentals_result = result.fundamentals_result
    if agent == "fundamentals_agent" and fundamentals_result is not None:
        requested = _coerce_capabilities(fundamentals_result.requested_capabilities)
        return tuple(
            capability for capability in requested if capability in _FUNDAMENTALS_CAPABILITIES
        )
    if agent == "forecast_agent" and result.forecast_result is not None:
        return (ResearchCapability.FORECAST,)
    if agent == "onchain_agent" and result.onchain_result is not None:
        return (ResearchCapability.ONCHAIN,)
    return ()


def _coerce_capabilities(values: object) -> tuple[ResearchCapability, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    capabilities: list[ResearchCapability] = []
    for value in values:
        try:
            capability = (
                value if isinstance(value, ResearchCapability) else ResearchCapability(str(value))
            )
        except ValueError:
            continue
        if capability not in capabilities:
            capabilities.append(capability)
    return tuple(capabilities)


def _presentation_status(value: object, *, has_content: bool) -> PresentationStatus:
    if value == "complete":
        return "complete"
    if value == "partial":
        return "partial"
    return "partial" if has_content else "unavailable"


def _agent_state_label(status_item: AgentExecutionStatus | None, status: str) -> str:
    analysis_state = status_item.analysis_state if status_item is not None else ""
    source_state = status_item.source_state if status_item is not None else ""
    if analysis_state == "live":
        return "Live answer"
    if analysis_state == "evidence_only":
        return "Live evidence answer" if source_state == "live" else "Evidence answer"
    if analysis_state == "unavailable" or status == "unavailable":
        return "Unavailable"
    return "Partial"


def _agent_data_cards(
    result: ResearchReport,
    agent: str,
    capabilities: tuple[ResearchCapability, ...],
) -> list[CapabilityDataPresentation]:
    cutoff = result.collection_context.collected_at
    if agent == "market_agent":
        cards: list[CapabilityDataPresentation] = []
        if any(
            capability in capabilities
            for capability in (ResearchCapability.MARKET, ResearchCapability.DISCOVERY)
        ):
            cards.extend(_market_data_cards(result, cutoff=cutoff))
        if ResearchCapability.RISK in capabilities:
            cards.extend(_risk_data_cards(result, owner=agent))
        if ResearchCapability.DERIVATIVES in capabilities:
            cards.extend(_derivatives_data_cards(result, cutoff=cutoff))
        return cards
    if agent == "news_agent":
        return _research_data_cards(result, capabilities, cutoff=cutoff)
    if agent == "fundamentals_agent":
        return _fundamentals_data_cards(result, capabilities, cutoff=cutoff)
    if agent == "onchain_agent":
        return onchain_data_cards(result, cutoff=cutoff)
    if agent == "forecast_agent":
        return forecast_data_cards(result)
    return []


def _is_current_market(market: MarketEvidence, *, cutoff: datetime) -> bool:
    return market.collected_at <= cutoff and market.last_time <= cutoff


def _is_current_provider_observation(evidence: object, *, cutoff: datetime) -> bool:
    collected_at = getattr(evidence, "collected_at", None)
    return isinstance(collected_at, datetime) and collected_at <= cutoff


def _provider_boundary_limitation(
    evidence: object,
    *,
    capability: ResearchCapability,
    cutoff: datetime,
) -> str:
    collected_at = getattr(evidence, "collected_at", None)
    label = _capability_label(capability).casefold()
    if isinstance(collected_at, datetime) and collected_at > cutoff:
        return f"Future-dated {label} data were excluded before display."
    return f"{_capability_label(capability)} provider data lacked a valid collection timestamp."


def _quarantined_data_card(
    *,
    agent: str,
    capability: ResearchCapability,
    title: str,
    asset: str | None,
    limitation: str,
) -> CapabilityDataPresentation:
    return CapabilityDataPresentation(
        agent=agent,
        capability=capability.value,
        title=title,
        asset=asset,
        facts=(("Data status", "Excluded"),),
        status="partial",
        limitation=limitation,
    )


def _market_quarantine_card(symbol: str) -> CapabilityDataPresentation:
    return _quarantined_data_card(
        agent="market_agent",
        capability=ResearchCapability.MARKET,
        title=f"{symbol} market data",
        asset=symbol,
        limitation="Future-dated market data were excluded before display.",
    )


def _provider_data_card(
    *,
    agent: str,
    capability: ResearchCapability,
    title: str,
    asset: str | None,
    evidence: object,
    cutoff: datetime,
    renderer: Callable[[str | None, object], CapabilityDataPresentation],
) -> CapabilityDataPresentation:
    if not _is_current_provider_observation(evidence, cutoff=cutoff):
        return _quarantined_data_card(
            agent=agent,
            capability=capability,
            title=title,
            asset=asset,
            limitation=_provider_boundary_limitation(
                evidence,
                capability=capability,
                cutoff=cutoff,
            ),
        )
    return renderer(asset, evidence)


def _market_data_cards(
    result: ResearchReport,
    *,
    cutoff: datetime,
) -> list[CapabilityDataPresentation]:
    cards: list[CapabilityDataPresentation] = []
    if result.market_result is not None:
        primary = result.market_result
        cards.append(
            _market_data_card(
                primary.market,
                primary.technical,
                contexts=primary.contextual_timeframes,
                cutoff=cutoff,
            )
            if _is_current_market(primary.market, cutoff=cutoff)
            else _market_quarantine_card(primary.market.symbol)
        )
    if result.market_comparison_result is not None:
        cards.extend(
            (
                _market_data_card(
                    item.market,
                    item.technical,
                    metrics=item.metrics,
                    contexts=item.contextual_timeframes,
                    cutoff=cutoff,
                )
                if _is_current_market(item.market, cutoff=cutoff)
                else _market_quarantine_card(item.market.symbol)
            )
            for item in result.market_comparison_result.assets
        )
    if result.opportunity_result is not None:
        scan = result.opportunity_result
        if scan.collected_at <= cutoff:
            cards.append(
                CapabilityDataPresentation(
                    agent="market_agent",
                    capability=ResearchCapability.DISCOVERY.value,
                    title="Market discovery",
                    asset=None,
                    facts=(
                        ("Exchange", scan.exchange.title()),
                        ("Timeframe", scan.timeframe),
                        ("Ranked candidates", str(len(scan.candidates))),
                        ("Collected", _utc_label(scan.collected_at)),
                    ),
                    status="partial" if scan.warnings else "complete",
                    limitation=_first_nonempty(*scan.warnings),
                )
            )
        else:
            cards.append(
                _quarantined_data_card(
                    agent="market_agent",
                    capability=ResearchCapability.DISCOVERY,
                    title="Market discovery",
                    asset=None,
                    limitation="Future-dated market discovery data were excluded before display.",
                )
            )
    return cards


def _derivatives_data_cards(
    result: ResearchReport,
    *,
    cutoff: datetime,
) -> list[CapabilityDataPresentation]:
    evidence: list[DerivativesEvidence] = []
    if result.market_result is not None and result.market_result.derivatives is not None:
        evidence.append(result.market_result.derivatives)
    if result.market_comparison_result is not None:
        evidence.extend(
            item.derivatives
            for item in result.market_comparison_result.assets
            if item.derivatives is not None
        )
    return [_derivatives_data_card(item, cutoff=cutoff) for item in evidence]


def _derivatives_data_card(
    evidence: DerivativesEvidence,
    *,
    cutoff: datetime,
) -> CapabilityDataPresentation:
    if evidence.collected_at > cutoff:
        return _quarantined_data_card(
            agent="market_agent",
            capability=ResearchCapability.DERIVATIVES,
            title=f"{evidence.asset} derivatives positioning",
            asset=evidence.asset,
            limitation="Future-dated derivatives data were excluded before display.",
        )
    observations = [
        *(item.observed_at for item in evidence.funding_history),
        *(item.observed_at for item in evidence.open_interest_history),
    ]
    if any(value > cutoff for value in observations):
        return _quarantined_data_card(
            agent="market_agent",
            capability=ResearchCapability.DERIVATIVES,
            title=f"{evidence.asset} derivatives positioning",
            asset=evidence.asset,
            limitation="Future-dated derivatives observations were excluded before display.",
        )
    facts: list[tuple[str, str]] = [("Venue", evidence.venue)]
    if evidence.contract_symbol:
        facts.append(("Contract", evidence.contract_symbol))
    if evidence.latest_funding_rate is not None:
        facts.append(("Latest funding", f"{evidence.latest_funding_rate:.4%}"))
    if evidence.average_funding_rate_24h is not None:
        facts.append(("24h average funding", f"{evidence.average_funding_rate_24h:.4%}"))
    if evidence.latest_open_interest_usd is not None:
        facts.append(("Open interest", format_price(evidence.latest_open_interest_usd)))
    if evidence.open_interest_change_24h_pct is not None:
        facts.append(("24h OI change", f"{evidence.open_interest_change_24h_pct:+.2f}%"))
    if observations:
        latest = max(observations)
        facts.extend(
            [
                ("Observed", _utc_label(latest)),
                ("Freshness", relative_age_label(latest, cutoff=cutoff)),
            ]
        )
    facts.append(("Cache state", evidence.source_state.title()))
    if evidence.status in {"unavailable", "not_applicable"}:
        facts.append(("Data status", evidence.status.replace("_", " ").title()))
    return CapabilityDataPresentation(
        agent="market_agent",
        capability=ResearchCapability.DERIVATIVES.value,
        title=f"{evidence.asset} derivatives positioning",
        asset=evidence.asset,
        facts=tuple(facts),
        status=(
            "complete"
            if evidence.status == "complete" and not evidence.warnings
            else "partial"
            if evidence.status in {"complete", "partial"}
            else "unavailable"
        ),
        limitation=_first_nonempty(*evidence.warnings),
    )


def _market_data_card(
    market: MarketEvidence,
    technical: TechnicalSnapshot,
    *,
    metrics: object | None = None,
    contexts: Sequence[object] = (),
    cutoff: datetime,
) -> CapabilityDataPresentation:
    facts: list[tuple[str, str]] = [
        ("Exchange", market.exchange.title()),
        ("Primary timeframe", market.timeframe),
        ("Current price", format_price(market.current_price)),
        ("Trend", str(getattr(technical, "trend", "Unavailable")).title()),
    ]
    change_24h = build_market_posture(market, technical).change_24h_percent
    if change_24h is not None:
        facts.append(("24h change", f"{change_24h:+.2f}%"))
    rsi = getattr(technical, "rsi", None)
    if rsi is not None:
        facts.append(("RSI", f"{float(rsi):.1f}"))
    volatility = getattr(technical, "volatility", None)
    if volatility is not None:
        facts.append(("Technical volatility", f"{float(volatility):.2%}"))
    if metrics is not None:
        price_return = getattr(metrics, "price_return", None)
        if price_return is not None:
            facts.append(("Observed return", f"{float(price_return):+.2%}"))
    context_labels: list[str] = []
    limitations: list[str | None] = []
    partial = getattr(technical, "status", "available") == "unavailable"
    for context in contexts:
        timeframe = str(getattr(context, "timeframe", "context"))
        context_market = getattr(context, "market", None)
        if context_market is not None and (
            not isinstance(context_market, MarketEvidence)
            or not _is_current_market(context_market, cutoff=cutoff)
        ):
            context_labels.append(f"{timeframe}: Unavailable")
            limitations.append(
                f"Future-dated {timeframe} contextual market data were excluded before display."
            )
            partial = True
            continue
        context_status = str(getattr(context, "status", "partial"))
        context_technical = getattr(context, "technical", None)
        trend = getattr(context_technical, "trend", None)
        context_value = str(trend).title() if trend else context_status.title()
        context_labels.append(f"{timeframe}: {context_value}")
        limitations.append(getattr(context, "limitation", None))
        partial = partial or context_status != "complete"
    if context_labels:
        facts.append(("Higher-timeframe context", " | ".join(context_labels)))
    return CapabilityDataPresentation(
        agent="market_agent",
        capability=ResearchCapability.MARKET.value,
        title=f"{market.symbol} market data",
        asset=market.symbol,
        facts=tuple(facts),
        status="partial" if partial else "complete",
        limitation=_first_nonempty(*limitations, getattr(technical, "limitation", None)),
    )


def _research_data_cards(
    result: ResearchReport,
    capabilities: tuple[ResearchCapability, ...],
    *,
    cutoff: datetime,
) -> list[CapabilityDataPresentation]:
    research = result.research_result
    if research is None:
        return []
    selected = set(capabilities) or set(_coerce_capabilities(research.requested_capabilities))
    bundles = list(research.asset_results)
    cards: list[CapabilityDataPresentation] = []
    if bundles:
        for bundle in bundles:
            asset_label = bundle.asset.symbol
            if ResearchCapability.NEWS in selected and bundle.news is not None:
                cards.append(_news_data_card(asset_label, bundle.news, cutoff=cutoff))
        return cards
    primary_asset_label = _primary_asset_label(result)
    if ResearchCapability.NEWS in selected:
        cards.append(_news_data_card(primary_asset_label, research.news, cutoff=cutoff))
    return cards


def _merge_warnings(*groups: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for group in groups for item in group if item.strip()))


def _safe_news_items(
    news: NewsEvidence,
    *,
    cutoff: datetime,
) -> tuple[list[NewsItem], list[str]]:
    if news.collected_at > cutoff:
        return [], _merge_warnings(
            news.warnings,
            ("Future-dated news metadata were excluded before display.",),
        )
    items, warnings = normalize_news_items(list(news.items), collected_at=cutoff)
    return items, _merge_warnings(news.warnings, warnings)


def _news_data_card(
    asset: str | None,
    news: NewsEvidence,
    *,
    cutoff: datetime,
) -> CapabilityDataPresentation:
    items, warnings = _safe_news_items(news, cutoff=cutoff)
    latest = max(items, key=lambda item: item.published_at) if items else None
    facts: list[tuple[str, str]] = [("Validated items", str(len(items)))]
    if latest is not None:
        facts.extend(
            [
                ("Latest publisher", redact_secrets(latest.publisher)),
                ("Latest title", redact_secrets(latest.title)),
                ("Published", _utc_label(latest.published_at)),
                ("Freshness", relative_age_label(latest.published_at, cutoff=cutoff)),
            ]
        )
    return CapabilityDataPresentation(
        agent="news_agent",
        capability=ResearchCapability.NEWS.value,
        title=f"{asset + ' ' if asset else ''}news",
        asset=asset,
        facts=tuple(facts),
        status="complete" if items and not warnings else "partial",
        limitation=_first_nonempty(*warnings) if not items or warnings else None,
    )


def _fundamentals_data_cards(
    result: ResearchReport,
    capabilities: tuple[ResearchCapability, ...],
    *,
    cutoff: datetime,
) -> list[CapabilityDataPresentation]:
    fundamentals_result = result.fundamentals_result
    if fundamentals_result is None:
        return []
    selected = set(capabilities) or set(
        _coerce_capabilities(fundamentals_result.requested_capabilities)
    )
    bundles = list(fundamentals_result.asset_results)

    def card(
        capability: ResearchCapability,
        asset: str | None,
        evidence: object,
        require_evidence: bool,
    ) -> CapabilityDataPresentation | None:
        if capability not in selected or (require_evidence and evidence is None):
            return None
        is_defi = capability == ResearchCapability.DEFI
        if is_defi and str(getattr(evidence, "status", "")).casefold() != "available":
            return None
        return _provider_data_card(
            agent="fundamentals_agent",
            capability=capability,
            title=f"{asset + ' ' if asset else ''}{'DeFi metrics' if is_defi else 'fundamentals'}",
            asset=asset,
            evidence=evidence,
            cutoff=cutoff,
            renderer=_defi_data_card if is_defi else _fundamental_data_card,
        )

    if bundles:
        cards: list[CapabilityDataPresentation] = []
        for bundle in bundles:
            label = bundle.asset.symbol
            cards.extend(
                built
                for built in (
                    card(ResearchCapability.FUNDAMENTALS, label, bundle.fundamentals, True),
                    card(ResearchCapability.DEFI, label, bundle.defi, True),
                )
                if built is not None
            )
        return cards
    fallback_label = _primary_asset_label(result)
    return [
        built
        for built in (
            card(
                ResearchCapability.FUNDAMENTALS,
                fallback_label,
                fundamentals_result.fundamentals,
                False,
            ),
            card(ResearchCapability.DEFI, fallback_label, fundamentals_result.defi, False),
        )
        if built is not None
    ]


def _fundamental_data_card(asset: str | None, evidence: object) -> CapabilityDataPresentation:
    facts: list[tuple[str, str]] = []
    values = (
        ("Market cap", getattr(evidence, "market_cap", None), format_price),
        ("Rank", getattr(evidence, "rank", None), lambda value: f"#{int(value)}"),
        (
            "Circulating supply",
            getattr(evidence, "circulating_supply", None),
            lambda value: f"{float(value):,.0f}",
        ),
        (
            "Total supply",
            getattr(evidence, "total_supply", None),
            lambda value: f"{float(value):,.0f}",
        ),
    )
    for label, value, formatter in values:
        if value is not None:
            facts.append((label, formatter(value)))
    categories = list(getattr(evidence, "categories", ()))
    if categories:
        facts.append(
            ("Categories", ", ".join(redact_secrets(str(item)) for item in categories[:3]))
        )
    developer = getattr(evidence, "developer_activity", None)
    if developer is not None:
        for label, value in (
            ("Commits (4 weeks)", getattr(developer, "commits_4_weeks", None)),
            ("Repository stars", getattr(developer, "stars", None)),
            ("Repository forks", getattr(developer, "forks", None)),
            ("Contributors", getattr(developer, "contributors", None)),
            ("Merged pull requests", getattr(developer, "merged_pull_requests", None)),
        ):
            if value is not None:
                facts.append((label, f"{int(value):,}"))
        updated_at = getattr(developer, "provider_updated_at", None)
        if updated_at is not None:
            facts.append(("Developer data updated", _utc_label(updated_at)))
    status = str(getattr(evidence, "status", "available"))
    collected_at = getattr(evidence, "collected_at", None)
    if isinstance(collected_at, datetime):
        facts.append(("Collected", _utc_label(collected_at)))
    facts.append(
        (
            "Provider snapshot",
            "Cached" if getattr(evidence, "source_state", "live") == "cached" else "Live",
        )
    )
    return CapabilityDataPresentation(
        agent="fundamentals_agent",
        capability=ResearchCapability.FUNDAMENTALS.value,
        title=f"{asset + ' ' if asset else ''}fundamentals",
        asset=asset,
        facts=tuple(facts)
        or (("Source", redact_secrets(str(getattr(evidence, "source", "Provider")))),),
        status="complete" if status == "available" else "partial",
        limitation=_first_nonempty(*getattr(evidence, "warnings", ())),
    )


def _defi_data_card(asset: str | None, evidence: object) -> CapabilityDataPresentation:
    facts: list[tuple[str, str]] = []
    protocol = getattr(evidence, "protocol", None)
    if protocol:
        facts.append(("Protocol", redact_secrets(str(protocol))))
    tvl = getattr(evidence, "tvl_usd", None)
    if tvl is not None:
        facts.append(("TVL", format_price(float(tvl))))
    for label, value in (
        ("1d change", getattr(evidence, "change_1d", None)),
        ("7d change", getattr(evidence, "change_7d", None)),
    ):
        if value is not None:
            facts.append((label, f"{float(value):+.2f}%"))
    chains = list(getattr(evidence, "chains", ()))
    if chains:
        facts.append(("Chains", ", ".join(redact_secrets(str(item)) for item in chains[:4])))
    status = str(getattr(evidence, "status", "available"))
    return CapabilityDataPresentation(
        agent="fundamentals_agent",
        capability=ResearchCapability.DEFI.value,
        title=f"{asset + ' ' if asset else ''}DeFi metrics",
        asset=asset,
        facts=tuple(facts)
        or (("Source", redact_secrets(str(getattr(evidence, "source", "Provider")))),),
        status="complete" if status == "available" else "partial",
        limitation=_first_nonempty(*getattr(evidence, "warnings", ())),
    )


def _risk_data_cards(result: ResearchReport, *, owner: str) -> list[CapabilityDataPresentation]:
    risk = result.risk_result
    if risk is None:
        return []
    entries = (
        [(entry.asset.symbol, entry.assessment) for entry in risk.asset_results]
        if risk.asset_results
        else [(_primary_asset_label(result), risk.assessment)]
    )
    cards: list[CapabilityDataPresentation] = []
    for asset, assessment in entries:
        facts: list[tuple[str, str]] = [
            ("Risk score", f"{assessment.score:.0f}/100"),
            ("Band", assessment.band.replace("_", " ").title()),
            ("Evidence confidence", f"{assessment.evidence_confidence:.0f}%"),
        ]
        if assessment.factors:
            facts.append(
                ("Key factors", "; ".join(redact_secrets(item) for item in assessment.factors[:3]))
            )
        if assessment.coverage_gaps:
            facts.append(
                (
                    "Coverage gaps",
                    "; ".join(redact_secrets(item) for item in assessment.coverage_gaps[:4]),
                )
            )
        cards.append(
            CapabilityDataPresentation(
                agent=owner,
                capability=ResearchCapability.RISK.value,
                title=f"{asset + ' ' if asset else ''}observed risk assessment",
                asset=asset,
                facts=tuple(facts),
                status="partial" if assessment.coverage_gaps else "complete",
                limitation=_first_nonempty(*assessment.coverage_gaps),
            )
        )
    return cards


def _primary_asset_label(result: ResearchReport) -> str | None:
    assets = result.request.ordered_assets()
    return assets[0].symbol if assets else None


def _first_nonempty(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _utc_label(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%d %b %Y %H:%M UTC")
    return "Not recorded"


def _answer_sections(answers: Sequence[AgentAnswer]) -> dict[str, tuple[str, ...]]:
    sections: dict[str, list[str]] = {
        "observed_fact": [],
        "calculation": [],
        "interpretation": [],
        "speculation": [],
        "risk": [],
    }
    for answer in answers:
        for claim in answer.evidence:
            sections.setdefault(claim.claim_kind, []).append(claim.statement)
    return {
        key: tuple(redact_secrets(value.strip()) for value in values if value.strip())
        for key, values in sections.items()
    }


def _research_title(result: ResearchReport) -> str:
    if result.opportunity_result:
        return "Market opportunity scan"
    if result.market_comparison_result:
        return f"{len(result.market_comparison_result.assets)}-asset market comparison"
    if len(result.request.ordered_assets()) > 1:
        return f"{len(result.request.ordered_assets())}-asset research comparison"
    if result.market_result:
        return f"{result.market_result.market.symbol} market review"
    if result.risk_result:
        return "Crypto risk review"
    if result.research_result and result.research_result.news.items:
        return "Crypto news brief"
    if result.research_result:
        return "Fundamentals brief"
    return "Research brief"


def _build_discovery_presentation(result: ResearchReport) -> DiscoveryPresentation | None:
    scan = result.opportunity_result
    cutoff = result.collection_context.collected_at
    if scan is None or scan.collected_at > cutoff:
        return None
    return DiscoveryPresentation(
        exchange=scan.exchange,
        timeframe=scan.timeframe,
        collected_at=scan.collected_at.astimezone(UTC),
        summary=redact_secrets(scan.summary),
        candidates=tuple(
            DiscoveryCandidatePresentation(
                rank=item.rank,
                asset=redact_secrets(item.asset),
                symbol=redact_secrets(item.symbol),
                current_price=item.current_price,
                momentum_24h=item.momentum_24h,
                volatility_24h=item.volatility_24h,
                score=item.score,
                trend=redact_secrets(item.trend),
                reason=redact_secrets(item.reason),
            )
            for item in scan.candidates
        ),
        warnings=tuple(redact_secrets(item) for item in scan.warnings),
    )


def collect_warnings(result: ResearchReport) -> list[str]:
    values = [*result.warnings]
    for answer in result.agent_answers:
        values.extend(answer.uncertainty)
        values.extend(answer.limitations)
    if result.opportunity_result:
        values.extend(result.opportunity_result.warnings)
    if result.market_comparison_result:
        values.extend(result.market_comparison_result.warnings)
        for item in result.market_comparison_result.assets:
            if item.derivatives is not None:
                values.extend(item.derivatives.warnings)
    if result.market_result and result.market_result.derivatives is not None:
        values.extend(result.market_result.derivatives.warnings)
    if result.research_result:
        values.extend(result.research_result.news.warnings)
        values.extend(result.research_result.fundamentals.warnings)
        values.extend(result.research_result.defi.warnings)
        for bundle in result.research_result.asset_results:
            values.extend(bundle.limitations)
    if result.risk_result:
        values.extend(result.risk_result.assessment.coverage_gaps)
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _source_presentations(result: ResearchReport) -> list[SourcePresentation]:
    cutoff = result.collection_context.collected_at
    sources = [
        SourcePresentation(
            publisher=market.exchange.title(),
            title=f"{market.symbol} {market.timeframe} OHLCV market data",
            url=None,
            published_at=market.collected_at.astimezone(UTC),
            kind="Market data",
        )
        for market in _market_sources(result)
        if market.collected_at <= cutoff and market.last_time <= cutoff
    ]
    if result.opportunity_result and result.opportunity_result.collected_at <= cutoff:
        scan = result.opportunity_result
        sources.append(
            SourcePresentation(
                publisher=scan.exchange.title(),
                title=f"{scan.timeframe} opportunity-scan market data",
                url=None,
                published_at=scan.collected_at.astimezone(UTC),
                kind="Market data",
            )
        )
    research = result.research_result
    if research is not None:
        bundles = research.asset_results
        if not bundles:
            assets = result.request.ordered_assets()
            bundles = [
                AssetResearchBundle(
                    asset=(
                        assets[0]
                        if assets
                        else AnalysisAsset(
                            requested_name=research.news.query or "Crypto",
                            symbol="CRYPTO/USD",
                        )
                    ),
                    news=research.news,
                    fundamentals=research.fundamentals,
                    defi=research.defi,
                )
            ]
        sources.extend(_asset_research_sources(bundles, cutoff=cutoff))
    sources.extend(_fundamentals_sources(result, cutoff=cutoff))
    for derivatives_source in _derivatives_sources(result):
        if derivatives_source.collected_at > cutoff or derivatives_source.status not in {
            "complete",
            "partial",
        }:
            continue
        sources.append(
            SourcePresentation(
                publisher=derivatives_source.venue,
                title=(
                    f"{derivatives_source.contract_symbol or derivatives_source.asset} "
                    "funding and open interest"
                ),
                url=normalize_http_url(derivatives_source.source_url),
                published_at=derivatives_source.collected_at.astimezone(UTC),
                kind="Derivatives",
                time_context="Collected",
            )
        )
    if result.onchain_result is not None:
        for bundle in result.onchain_result.asset_results:
            evidence = bundle.onchain
            if evidence is None or not evidence.metrics or evidence.collected_at > cutoff:
                continue
            sources.append(
                SourcePresentation(
                    publisher=evidence.source,
                    title=f"{bundle.asset.symbol} daily network activity metrics",
                    url=evidence.source_url,
                    published_at=evidence.collected_at.astimezone(UTC),
                    kind="On-chain",
                    time_context="Collected",
                )
            )
    return _unique_sources(sources)


def _derivatives_sources(result: ResearchReport) -> list[DerivativesEvidence]:
    sources: list[DerivativesEvidence] = []
    if result.market_result is not None and result.market_result.derivatives is not None:
        sources.append(result.market_result.derivatives)
    if result.market_comparison_result is not None:
        sources.extend(
            item.derivatives
            for item in result.market_comparison_result.assets
            if item.derivatives is not None
        )
    return sources


def _fundamentals_sources(result: ResearchReport, *, cutoff: datetime) -> list[SourcePresentation]:
    fundamentals_result = result.fundamentals_result
    if fundamentals_result is None:
        return []
    bundles = list(fundamentals_result.asset_results)
    if bundles:
        return _asset_research_sources(bundles, cutoff=cutoff)
    asset = _primary_asset_label(result)
    label = f"{asset}: " if asset else ""
    return [
        source
        for evidence, kind in (
            (fundamentals_result.fundamentals, "Fundamentals"),
            (fundamentals_result.defi, "DeFi"),
        )
        if (source := _provider_source(evidence, label=label, kind=kind, cutoff=cutoff)) is not None
    ]


def _provider_source(
    evidence: FundamentalEvidence | DefiEvidence | None,
    *,
    label: str,
    kind: str,
    cutoff: datetime,
) -> SourcePresentation | None:
    if evidence is None or evidence.status != "available" or evidence.collected_at > cutoff:
        return None
    name = (
        f"{getattr(evidence, 'name', None) or 'Asset'} fundamentals"
        if kind == "Fundamentals"
        else f"{getattr(evidence, 'protocol', None) or 'Protocol'} DeFi metrics"
    )
    return SourcePresentation(
        publisher=redact_secrets(evidence.source),
        title=redact_secrets(f"{label}{name}"),
        url=None,
        published_at=evidence.collected_at.astimezone(UTC),
        kind=kind,
    )


def _asset_research_sources(
    bundles: Sequence[AssetResearchBundle],
    *,
    cutoff: datetime,
) -> list[SourcePresentation]:
    multi = len(bundles) > 1
    sources: list[SourcePresentation] = []
    for bundle in bundles:
        label = f"{bundle.asset.symbol}: " if multi else ""
        if bundle.news is not None:
            news_items, _ = _safe_news_items(bundle.news, cutoff=cutoff)
            sources.extend(
                SourcePresentation(
                    publisher=redact_secrets(item.publisher.strip()),
                    title=redact_secrets(f"{label}{item.title.strip()}"),
                    url=normalize_http_url(item.url),
                    published_at=item.published_at.astimezone(UTC),
                    kind="News",
                )
                for item in news_items
            )
        sources.extend(
            source
            for evidence, kind in (
                (bundle.fundamentals, "Fundamentals"),
                (bundle.defi, "DeFi"),
            )
            if (source := _provider_source(evidence, label=label, kind=kind, cutoff=cutoff))
            is not None
        )
    return sources


def _unique_sources(sources: Sequence[SourcePresentation]) -> list[SourcePresentation]:
    unique: dict[tuple[str, str], SourcePresentation] = {}
    for source in sources:
        unique.setdefault((source.url or "", source.title.casefold()), source)
    return list(unique.values())


def _market_sources(result: ResearchReport) -> list[MarketEvidence]:
    markets: list[MarketEvidence] = []
    if result.market_result:
        markets.append(result.market_result.market)
        markets.extend(
            item.market for item in result.market_result.contextual_timeframes if item.market
        )
    if result.market_comparison_result:
        for item in result.market_comparison_result.assets:
            markets.append(item.market)
            markets.extend(
                context.market for context in item.contextual_timeframes if context.market
            )
    unique: dict[tuple[str, str, str], MarketEvidence] = {}
    for market in markets:
        key = (market.exchange, market.symbol, market.timeframe)
        current = unique.get(key)
        if current is None or market.collected_at > current.collected_at:
            unique[key] = market
    return list(unique.values())


__all__ = [
    name
    for name in globals()
    if name.endswith("Presentation")
    or name
    in {
        "CHART_HISTORY_LIMIT",
        "DashboardView",
        "ResearchTurn",
        "build_agent_answer_presentation",
        "build_asset_presentations",
        "build_research_presentation",
        "collect_warnings",
        "format_price",
    }
]
