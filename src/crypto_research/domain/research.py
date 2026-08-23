"""Research-domain contracts shared by routing, agents, services, and the UI."""

import re
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from crypto_research.domain.core import (
    ASSET_ALIASES,
    COIN_ID_BY_ASSET,
    DEFAULT_QUOTE_BY_EXCHANGE,
    MAX_COMPARISON_ASSETS,
    SUPPORTED_EXCHANGES,
    SUPPORTED_TIMEFRAMES,
    OpportunityCandidate,
    OpportunityScanResult,
    ResearchCapability,
    RiskAssessment,
    RiskBand,
    StrictModel,
    SupportedExchange,
    SupportedTimeframe,
    build_market_symbol,
    extract_supported_assets,
    normalize_market_symbol,
    risk_band,
)
from crypto_research.domain.evidence import (
    AgentAnalysisSection,
    AgentAnswer,
    AgentId,
    AnswerState,
    DefiEvidence,
    DerivativesEvidence,
    EvidenceClaim,
    EvidenceRecord,
    FundamentalEvidence,
    NewsEvidence,
    NewsItem,
    OnChainEvidence,
    ResponseStyle,
    StructuredAgentAnalysis,
    TechnicalSnapshot,
)
from crypto_research.domain.forecast import ForecastAgentResult, ForecastSettings
from crypto_research.domain.market import ComparisonMetrics, MarketEvidence

UNAVAILABLE_ANSWER_MESSAGE = "Live analysis is temporarily unavailable."


def _optional_text(value: str | None, *, lower: bool = False) -> str | None:
    normalized = value.strip() if value is not None else ""
    return (normalized.lower() if lower else normalized) or None


def _optional_coin_id(value: str | None) -> str | None:
    normalized = _optional_text(value, lower=True)
    if normalized and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized):
        raise ValueError("Coin ID must be a lowercase slug.")
    return normalized


class CollectionContext(StrictModel):
    """One immutable observation boundary shared by every collector in a run."""

    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("collected_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Collection timestamp must be timezone-aware.")
        return value.astimezone(UTC)


AssetResolutionStatus = Literal["confirmed", "ambiguous", "not_found", "unavailable"]


class AssetCandidate(StrictModel):
    coin_id: str = Field(min_length=1, max_length=200)
    symbol: str = Field(min_length=1, max_length=30)
    name: str = Field(min_length=1, max_length=200)
    market_cap_rank: int | None = Field(default=None, ge=1)
    network: str | None = Field(default=None, max_length=100)
    contract_address: str | None = Field(default=None, max_length=200)

    @field_validator("coin_id", "network")
    @classmethod
    def normalize_slug(cls, value: str | None) -> str | None:
        return _optional_text(value, lower=True)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("name", "contract_address")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _optional_text(value)


class AssetResolution(StrictModel):
    query: str = Field(min_length=1, max_length=200)
    status: AssetResolutionStatus
    selected: AssetCandidate | None = None
    candidates: list[AssetCandidate] = Field(default_factory=list, max_length=5)
    source: str = "CoinGecko"
    resolved_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("resolved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Asset resolution timestamp must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def require_confirmed_selection(self) -> Self:
        if self.status == "confirmed" and self.selected is None:
            raise ValueError("Confirmed asset resolution requires a selected candidate.")
        return self


class AnalysisAsset(StrictModel):
    requested_name: str = Field(min_length=1, max_length=200)
    symbol: str
    coin_id: str | None = None
    name: str | None = Field(default=None, max_length=200)
    network: str | None = Field(default=None, max_length=100)
    contract_address: str | None = Field(default=None, max_length=200)
    resolution: AssetResolution | None = None

    @field_validator("requested_name", "name", "contract_address", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        return _optional_text(value) if isinstance(value, str) else value

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return normalize_market_symbol(value)

    @field_validator("coin_id")
    @classmethod
    def normalize_coin_id(cls, value: str | None) -> str | None:
        return _optional_coin_id(value)

    @field_validator("network")
    @classmethod
    def normalize_network(cls, value: str | None) -> str | None:
        return _optional_text(value, lower=True)

    @property
    def key(self) -> str:
        if self.coin_id:
            return f"coin:{self.coin_id}"
        if self.network and self.contract_address:
            return f"contract:{self.network}:{self.contract_address.casefold()}"
        return f"market:{self.symbol}"

    @property
    def evidence_key(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.symbol.casefold()).strip("-")


class AnalysisRequest(StrictModel):
    user_intent: str = Field(min_length=1, max_length=4000)
    asset_query: str | None = Field(default=None, max_length=200)
    research_topic: str | None = Field(default=None, max_length=200)
    exchange: SupportedExchange = "kraken"
    symbol: str = "BTC/USD"
    timeframe: SupportedTimeframe = "1h"
    candle_limit: int = Field(default=750, ge=20, le=2000)
    coin_id: str | None = "bitcoin"
    network: str | None = None
    contract_address: str | None = None
    asset_resolution: AssetResolution | None = None
    assets: list[AnalysisAsset] = Field(default_factory=list, max_length=MAX_COMPARISON_ASSETS)
    comparison_symbols: list[str] = Field(default_factory=list, max_length=MAX_COMPARISON_ASSETS)
    response_style: ResponseStyle = "balanced"
    forecast_settings: ForecastSettings | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return normalize_market_symbol(value)

    @field_validator("comparison_symbols")
    @classmethod
    def normalize_comparison_symbols(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            symbol = normalize_market_symbol(item)
            if symbol not in seen:
                seen.add(symbol)
                normalized.append(symbol)
        return normalized

    @field_validator("coin_id")
    @classmethod
    def normalize_coin_id(cls, value: str | None) -> str | None:
        return _optional_coin_id(value)

    @field_validator("asset_query", "research_topic")
    @classmethod
    def normalize_optional_query(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("network")
    @classmethod
    def normalize_network(cls, value: str | None) -> str | None:
        return _optional_text(value, lower=True)

    @field_validator("contract_address")
    @classmethod
    def normalize_contract_address(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("user_intent")
    @classmethod
    def normalize_user_intent(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The natural-language request cannot be empty.")
        return value.strip()

    @model_validator(mode="after")
    def mirror_primary_asset(self) -> Self:
        if not self.assets:
            return self
        symbols = [asset.symbol for asset in self.assets]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Analysis assets must have distinct market symbols.")
        first = self.assets[0]
        object.__setattr__(self, "asset_query", first.requested_name)
        object.__setattr__(self, "symbol", first.symbol)
        object.__setattr__(self, "coin_id", first.coin_id)
        object.__setattr__(self, "network", first.network)
        object.__setattr__(self, "contract_address", first.contract_address)
        object.__setattr__(self, "asset_resolution", first.resolution)
        object.__setattr__(self, "comparison_symbols", symbols if len(self.assets) > 1 else [])
        return self

    def ordered_assets(self) -> list[AnalysisAsset]:
        """Return canonical assets for both direct and multi-asset requests."""

        if self.assets:
            return list(self.assets)
        if (
            not self.comparison_symbols
            and self.asset_query is None
            and self.coin_id is None
            and self.contract_address is None
            and self.asset_resolution is None
        ):
            return []
        symbols = self.comparison_symbols or [self.symbol]
        assets: list[AnalysisAsset] = []
        for index, symbol in enumerate(symbols):
            base = symbol.split("/", maxsplit=1)[0]
            primary = index == 0
            resolution = self.asset_resolution if primary else None
            selected = resolution.selected if resolution is not None else None
            coin_id = (
                (selected.coin_id if selected is not None else self.coin_id)
                if primary
                else COIN_ID_BY_ASSET.get(base)
            )
            assets.append(
                AnalysisAsset(
                    requested_name=(
                        self.asset_query
                        if primary and self.asset_query
                        else selected.name
                        if selected is not None
                        else coin_id or base
                    ),
                    name=selected.name if selected is not None else None,
                    symbol=symbol,
                    coin_id=coin_id,
                    network=self.network if primary else None,
                    contract_address=self.contract_address if primary else None,
                    resolution=resolution,
                )
            )
        return assets


class ResearchAction(StrictModel):
    """Validated, explicit research work selected by the UI or CLI."""

    reasoning: str = Field(description="Concise non-sensitive reason for the selected work")
    request: AnalysisRequest
    agents_to_call: list[str] = Field(default_factory=list)
    requested_capabilities: list[ResearchCapability] = Field(default_factory=list)


class MarketTimeframeEvidence(StrictModel):
    """One primary or contextual market timeframe with an explicit availability outcome."""

    timeframe: SupportedTimeframe
    status: Literal["complete", "partial", "unavailable"]
    market: MarketEvidence | None = None
    technical: TechnicalSnapshot | None = None
    limitation: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.status == "complete" and (self.market is None or self.technical is None):
            raise ValueError("Complete timeframe evidence requires market and technical data.")
        if self.status == "unavailable" and not self.limitation:
            raise ValueError("Unavailable timeframe evidence requires a limitation.")
        return self


class MarketAgentResult(StrictModel):
    market: MarketEvidence
    technical: TechnicalSnapshot
    derivatives: DerivativesEvidence | None = None
    summary: str | None = Field(default=None, max_length=1200)
    contextual_timeframes: list[MarketTimeframeEvidence] = Field(default_factory=list, max_length=3)


class MarketComparisonAsset(StrictModel):
    market: MarketEvidence
    technical: TechnicalSnapshot
    metrics: ComparisonMetrics | None = None
    derivatives: DerivativesEvidence | None = None
    contextual_timeframes: list[MarketTimeframeEvidence] = Field(default_factory=list, max_length=3)


class MarketComparisonResult(StrictModel):
    assets: list[MarketComparisonAsset] = Field(
        default_factory=list, max_length=MAX_COMPARISON_ASSETS
    )
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_distinct_assets(self) -> Self:
        symbols = [item.market.symbol for item in self.assets]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Multi-asset results must contain distinct market symbols.")
        return self


class CapabilityCoverage(StrictModel):
    asset: AnalysisAsset
    capability: ResearchCapability
    status: Literal["available", "unavailable", "not_applicable"]
    evidence_kinds: list[str] = Field(default_factory=list, max_length=6)
    limitation: str | None = Field(default=None, max_length=500)

    @field_validator("evidence_kinds")
    @classmethod
    def normalize_evidence_kinds(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))

    @field_validator("limitation")
    @classmethod
    def normalize_limitation(cls, value: str | None) -> str | None:
        return _optional_text(value)


class AgentExecutionStatus(StrictModel):
    """Safe, user-presentable result state for one requested workflow agent."""

    agent: Literal[
        "market_agent",
        "news_agent",
        "fundamentals_agent",
        "forecast_agent",
        "onchain_agent",
    ]
    status: Literal["complete", "partial", "unavailable"]
    # Data provenance is separate from execution completeness: a completed
    # agent can truthfully render a fresh cached result after live verification
    # failed.
    source_state: Literal["live", "cached", "partial", "unavailable"] = "live"
    analysis_state: Literal["live", "evidence_only", "unavailable"] = "live"
    coverage_state: Literal["complete", "partial", "not_applicable"] = "complete"
    capabilities: list[ResearchCapability] = Field(default_factory=list, max_length=12)
    limitation: str | None = Field(default=None, max_length=500)

    @field_validator("capabilities")
    @classmethod
    def normalize_capabilities(cls, value: list[ResearchCapability]) -> list[ResearchCapability]:
        selected = set(value)
        return [capability for capability in ResearchCapability if capability in selected]

    @field_validator("limitation")
    @classmethod
    def normalize_limitation(cls, value: str | None) -> str | None:
        return _optional_text(value)


class EvidenceCoverageEntry(StrictModel):
    """Traceable handling summary for one asset/capability evidence slice."""

    asset: str = Field(min_length=1, max_length=80)
    capability: ResearchCapability
    collected_records: int = Field(default=0, ge=0)
    accepted_records: int = Field(default=0, ge=0)
    excluded_records: int = Field(default=0, ge=0)
    detailed_records: int = Field(default=0, ge=0)
    summarized_records: int = Field(default=0, ge=0)
    providers: list[str] = Field(default_factory=list, max_length=20)
    earliest_observed_at: datetime | None = None
    latest_observed_at: datetime | None = None
    timeframes: list[SupportedTimeframe] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=12)


class EvidenceCoverageSummary(StrictModel):
    """Default-safe audit of evidence detailed and summarized for unified analysis."""

    entries: list[EvidenceCoverageEntry] = Field(default_factory=list, max_length=80)
    detailed_evidence_ids: list[str] = Field(default_factory=list, max_length=120)
    summarized_evidence_ids: list[str] = Field(default_factory=list, max_length=2000)
    total_collected_records: int = Field(default=0, ge=0)
    total_accepted_records: int = Field(default=0, ge=0)
    total_excluded_records: int = Field(default=0, ge=0)


class AssetResearchBundle(StrictModel):
    asset: AnalysisAsset
    market: MarketEvidence | None = None
    technical: TechnicalSnapshot | None = None
    news: NewsEvidence | None = None
    fundamentals: FundamentalEvidence | None = None
    defi: DefiEvidence | None = None
    onchain: OnChainEvidence | None = None
    risk: RiskAssessment | None = None
    limitations: list[str] = Field(default_factory=list)

    @field_validator("limitations")
    @classmethod
    def normalize_limitations(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class ResearchAgentResult(StrictModel):
    news: NewsEvidence
    fundamentals: FundamentalEvidence = Field(
        default_factory=lambda: FundamentalEvidence(status="unavailable")
    )
    defi: DefiEvidence = Field(default_factory=lambda: DefiEvidence(status="unavailable"))
    asset_results: list[AssetResearchBundle] = Field(
        default_factory=list, max_length=MAX_COMPARISON_ASSETS
    )
    requested_capabilities: list[ResearchCapability] = Field(default_factory=list)
    capabilities: list[ResearchCapability] = Field(default_factory=list)
    summary: str | None = Field(default=None, max_length=1200)


class FundamentalsAgentResult(StrictModel):
    fundamentals: FundamentalEvidence = Field(
        default_factory=lambda: FundamentalEvidence(status="unavailable")
    )
    defi: DefiEvidence = Field(default_factory=lambda: DefiEvidence(status="unavailable"))
    asset_results: list[AssetResearchBundle] = Field(
        default_factory=list, max_length=MAX_COMPARISON_ASSETS
    )
    requested_capabilities: list[ResearchCapability] = Field(default_factory=list)
    capabilities: list[ResearchCapability] = Field(default_factory=list)
    summary: str | None = Field(default=None, max_length=1200)


class OnChainAgentResult(StrictModel):
    asset_results: list[AssetResearchBundle] = Field(
        default_factory=list, max_length=MAX_COMPARISON_ASSETS
    )
    requested_capabilities: list[ResearchCapability] = Field(default_factory=list)
    capabilities: list[ResearchCapability] = Field(default_factory=list)
    summary: str | None = Field(default=None, max_length=1200)


class AssetRiskResult(StrictModel):
    asset: AnalysisAsset
    assessment: RiskAssessment


class RiskResult(StrictModel):
    assessment: RiskAssessment
    asset_results: list[AssetRiskResult] = Field(
        default_factory=list, max_length=MAX_COMPARISON_ASSETS
    )


class AgentResults(StrictModel):
    opportunity_result: OpportunityScanResult | None = None
    market_result: MarketAgentResult | None = None
    market_comparison_result: MarketComparisonResult | None = None
    research_result: ResearchAgentResult | None = None
    # Kept separate from the merged research bundle so specialist UI panels can
    # render the exact Fundamentals/DeFi collector result.
    fundamentals_result: FundamentalsAgentResult | None = None
    onchain_result: OnChainAgentResult | None = None
    risk_result: RiskResult | None = None
    forecast_result: ForecastAgentResult | None = None
    capability_coverage: list[CapabilityCoverage] = Field(default_factory=list, max_length=60)
    agent_statuses: list[AgentExecutionStatus] = Field(default_factory=list, max_length=5)
    evidence_coverage_summary: EvidenceCoverageSummary = Field(
        default_factory=EvidenceCoverageSummary
    )
    collection_context: CollectionContext = Field(default_factory=CollectionContext)


class AnalysisInputs(AgentResults):
    """Agent outputs and workflow warnings used by evidence calculations."""

    assets: list[AnalysisAsset] = Field(default_factory=list, max_length=MAX_COMPARISON_ASSETS)
    requested_capabilities: list[ResearchCapability] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, object]] = Field(default_factory=list)


class ResearchRetryMetadata(StrictModel):
    """Trace one immutable combined report back to the run it retried."""

    original_run_id: str = Field(min_length=1, max_length=100)
    retried_agents: list[AgentId] = Field(min_length=1, max_length=5)
    retried_at: datetime

    @field_validator("original_run_id")
    @classmethod
    def normalize_original_run_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Retry metadata requires an original run ID.")
        return normalized

    @field_validator("retried_agents")
    @classmethod
    def normalize_retried_agents(cls, value: list[AgentId]) -> list[AgentId]:
        return list(dict.fromkeys(value))

    @field_validator("retried_at")
    @classmethod
    def normalize_retried_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Retry timestamp must be timezone-aware.")
        return value.astimezone(UTC)


class ResearchReport(AgentResults):
    """Collected research outputs and ordered specialist answers."""

    status: Literal["complete", "partial"] = "complete"
    request: AnalysisRequest
    agent_answers: list[AgentAnswer] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, object]] = Field(default_factory=list)
    disclaimer: str = "Educational research only. Not financial advice."
    answer_state: AnswerState = AnswerState.SUPPORTED
    evidence_confidence: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    retry: ResearchRetryMetadata | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        degraded = bool(
            self.errors
            or self.warnings
            or self.answer_state is not AnswerState.SUPPORTED
            or any(answer.status != "complete" for answer in self.agent_answers)
        )
        if self.status == "complete" and degraded:
            raise ValueError("Complete results cannot contain degraded research output.")
        if self.status == "partial" and not degraded:
            raise ValueError("Partial results must describe degraded research output.")
        return self


# fmt: off
__all__ = [
    "ASSET_ALIASES", "COIN_ID_BY_ASSET", "DEFAULT_QUOTE_BY_EXCHANGE",
    "UNAVAILABLE_ANSWER_MESSAGE",
    "AgentAnalysisSection", "AgentAnswer", "AgentId", "AnalysisAsset",
    "AnalysisRequest", "AnswerState",
    "AssetCandidate", "AssetResearchBundle", "AssetResolution", "AssetRiskResult",
    "AgentExecutionStatus", "CapabilityCoverage", "CollectionContext",
    "EvidenceCoverageEntry", "EvidenceCoverageSummary",
    "DefiEvidence",
    "EvidenceClaim",
    "EvidenceRecord",
    "FundamentalEvidence", "FundamentalsAgentResult", "MarketAgentResult",
    "MarketTimeframeEvidence",
    "MarketComparisonAsset", "MarketComparisonResult", "NewsEvidence", "NewsItem",
    "OnChainAgentResult", "OnChainEvidence",
    "OpportunityCandidate", "OpportunityScanResult", "ResearchAction", "ResearchAgentResult",
    "ResearchCapability", "ResearchReport", "ResearchRetryMetadata",
    "AnalysisInputs",
    "ResponseStyle", "RiskAssessment", "RiskBand", "RiskResult",
    "StructuredAgentAnalysis",
    "SUPPORTED_EXCHANGES", "SUPPORTED_TIMEFRAMES", "StrictModel",
    "SupportedExchange", "SupportedTimeframe", "TechnicalSnapshot",
    "build_market_symbol", "extract_supported_assets", "normalize_market_symbol", "risk_band",
]
# fmt: on
