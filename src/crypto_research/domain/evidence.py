from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import isclose
from typing import Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, field_validator, model_validator

from crypto_research.domain.core import StrictModel
from crypto_research.shared.security import clean_text, normalize_http_url
from crypto_research.shared.text import publisher_quality, unique_strings

ResponseStyle = Literal["concise", "balanced", "detailed", "beginner", "technical"]
AgentId = Literal[
    "market_agent",
    "news_agent",
    "fundamentals_agent",
    "forecast_agent",
    "onchain_agent",
]

_TECHNICAL_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9 +./%_-]{0,59}")
MAX_EVIDENCE_EXCERPT_CHARS = 240
_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "referrer"})
_NEWS_MAX_AGE = timedelta(days=7)


class AnswerState(StrEnum):
    SUPPORTED = "SUPPORTED_ANSWER"
    PARTIAL = "PARTIAL_ANSWER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class EvidenceClaim(StrictModel):
    statement: str = Field(min_length=1, max_length=1200)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    claim_kind: Literal["observed_fact", "calculation", "interpretation", "speculation", "risk"] = (
        "observed_fact"
    )
    confidence: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)

    @field_validator("statement")
    @classmethod
    def normalize_statement(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Evidence statements cannot be empty.")
        return normalized

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: list[str]) -> list[str]:
        normalized = unique_strings(value)
        if not normalized:
            raise ValueError("Evidence claims require at least one evidence ID.")
        return normalized


class AgentAnalysisSection(StrictModel):
    """One visible, asset-owned specialist interpretation."""

    asset: str = Field(min_length=1, max_length=40)
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
    text: str = Field(min_length=1, max_length=500)

    @field_validator("asset", "text")
    @classmethod
    def normalize_section_text(cls, value: str) -> str:
        return " ".join(value.split())


class StructuredAgentAnalysis(StrictModel):
    """UI-ready structure retained from a validated specialist response."""

    verdict: str = Field(min_length=1, max_length=240)
    sections: list[AgentAnalysisSection] = Field(default_factory=list, max_length=12)
    comparison: str = Field(default="", max_length=300)

    @field_validator("verdict", "comparison")
    @classmethod
    def normalize_analysis_text(cls, value: str) -> str:
        return " ".join(value.split())


class AgentAnswer(StrictModel):
    """One grounded answer produced by a specialist analysis step."""

    agent: AgentId
    # Four-asset Guided Research needs room for a short verdict, one compact
    # paragraph per asset, and an automatic comparison.  The previous 1,200
    # character boundary rejected otherwise valid live responses during local
    # Pydantic normalization.
    answer: str = Field(min_length=1, max_length=3600)
    analysis: str = Field(default="", max_length=1800)
    structured_analysis: StructuredAgentAnalysis | None = None
    technical_terms: list[str] = Field(default_factory=list, max_length=12)
    evidence: list[EvidenceClaim] = Field(default_factory=list, max_length=12)
    uncertainty: list[str] = Field(default_factory=list, max_length=6)
    limitations: list[str] = Field(default_factory=list, max_length=6)
    suggested_followups: list[str] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: Literal["complete", "partial", "unavailable"] = "complete"
    analysis_state: Literal["live", "evidence_only", "unavailable"] = "live"
    coverage_state: Literal["complete", "partial", "not_applicable"] = "complete"

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Agent answers cannot be empty.")
        return normalized

    @field_validator("technical_terms")
    @classmethod
    def normalize_technical_terms(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            term = " ".join(item.split())
            if not _TECHNICAL_TERM.fullmatch(term):
                raise ValueError("Technical terms contain unsupported characters.")
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(term)
        return normalized

    @field_validator("analysis")
    @classmethod
    def normalize_analysis(cls, value: str) -> str:
        return " ".join(value.split())[:1800]

    @field_validator("uncertainty", "limitations", "suggested_followups")
    @classmethod
    def normalize_notes(cls, value: list[str]) -> list[str]:
        normalized = (item.strip() for item in value)
        return list(dict.fromkeys(item for item in normalized if item))

    @model_validator(mode="after")
    def validate_status_confidence(self) -> AgentAnswer:
        if self.status == "complete" and self.confidence <= 0:
            raise ValueError("Complete agent answers require positive confidence.")
        if self.status == "unavailable" and self.confidence != 0:
            raise ValueError("Unavailable agent answers must have zero confidence.")
        if self.status == "unavailable" and self.analysis_state != "unavailable":
            raise ValueError("Unavailable agent answers require unavailable analysis state.")
        for term in self.technical_terms:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
            if re.search(pattern, self.answer, flags=re.IGNORECASE) is None:
                raise ValueError(
                    "Every technical term must occur as a complete term in the answer."
                )
        return self


def _safe_optional_url(value: object) -> object:
    return normalize_http_url(value) if isinstance(value, str) else value


class TechnicalSnapshot(StrictModel):
    status: Literal["available", "unavailable"] = "available"
    limitation: str | None = Field(default=None, max_length=300)
    sma: float | None = Field(default=None, allow_inf_nan=False)
    ema: float | None = Field(default=None, allow_inf_nan=False)
    rsi: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    macd: float | None = Field(default=None, allow_inf_nan=False)
    atr: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    volatility: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    support: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    resistance: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    trend: Literal["bullish", "bearish", "neutral"]

    @model_validator(mode="after")
    def validate_status(self) -> TechnicalSnapshot:
        if self.status == "unavailable" and not self.limitation:
            raise ValueError("Unavailable technical evidence requires a limitation.")
        return self


class NewsItem(StrictModel):
    publisher: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    excerpt: str = Field(max_length=1200)
    content: str = Field(default="", max_length=4000)
    assets: list[str] = Field(default_factory=list, max_length=4)
    url: str | None = None
    published_at: datetime
    source_quality: Literal["high", "medium", "low"] = "medium"

    @field_validator("url", mode="before")
    @classmethod
    def discard_unsafe_url(cls, value: object) -> object:
        return _safe_optional_url(value)

    @field_validator("publisher", "title", "excerpt")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized and value != "":
            raise ValueError("News text cannot contain only whitespace.")
        return normalized

    @field_validator("assets")
    @classmethod
    def normalize_assets(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip().upper() for item in value if item.strip()))

    @field_validator("published_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("News timestamp must be timezone-aware.")
        return value.astimezone(UTC)


class NewsEvidence(StrictModel):
    items: list[NewsItem]
    query: str
    collected_at: datetime
    source_state: Literal["live", "cached"] = "live"
    warnings: list[str] = Field(default_factory=list)

    @field_validator("collected_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="News collection")


class DeveloperActivity(StrictModel):
    """Provider-reported public software-development activity."""

    forks: int | None = Field(default=None, ge=0)
    stars: int | None = Field(default=None, ge=0)
    contributors: int | None = Field(default=None, ge=0)
    merged_pull_requests: int | None = Field(default=None, ge=0)
    commits_4_weeks: int | None = Field(default=None, ge=0)
    provider_updated_at: datetime | None = None

    @field_validator("provider_updated_at")
    @classmethod
    def require_provider_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_timezone(value, label="Developer activity")


class FundamentalEvidence(StrictModel):
    name: str | None = None
    symbol: str | None = None
    market_cap: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rank: int | None = Field(default=None, ge=1)
    circulating_supply: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    total_supply: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_supply: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    categories: list[str] = Field(default_factory=list)
    homepage: str | None = None
    genesis_date: str | None = None
    developer_activity: DeveloperActivity | None = None
    status: Literal["available", "unavailable"] = "available"
    source: str = "CoinGecko"
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_state: Literal["live", "cached"] = "live"
    warnings: list[str] = Field(default_factory=list)

    @field_validator("homepage", mode="before")
    @classmethod
    def discard_unsafe_homepage(cls, value: object) -> object:
        return _safe_optional_url(value)

    @field_validator("collected_at")
    @classmethod
    def require_collected_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="Fundamental collection")


class FundingRateObservation(StrictModel):
    observed_at: datetime
    rate: float = Field(ge=-1, le=1, allow_inf_nan=False)

    @field_validator("observed_at")
    @classmethod
    def require_observation_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="Funding-rate observation")


class OpenInterestObservation(StrictModel):
    observed_at: datetime
    value_usd: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("observed_at")
    @classmethod
    def require_observation_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="Open-interest observation")


class DerivativesEvidence(StrictModel):
    """Validated public perpetual-futures positioning evidence."""

    asset: str = Field(min_length=2, max_length=15)
    contract_symbol: str | None = Field(default=None, max_length=31)
    venue: Literal["Binance USD-M Futures"] = "Binance USD-M Futures"
    status: Literal["complete", "partial", "unavailable", "not_applicable"]
    funding_history: list[FundingRateObservation] = Field(default_factory=list, max_length=24)
    open_interest_history: list[OpenInterestObservation] = Field(
        default_factory=list, max_length=48
    )
    latest_funding_rate: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    average_funding_rate_24h: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    latest_open_interest_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    open_interest_change_24h_pct: float | None = Field(default=None, allow_inf_nan=False)
    source: Literal["Binance USD-M Futures"] = "Binance USD-M Futures"
    source_url: str | None = None
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_state: Literal["live", "cached"] = "live"
    warnings: list[str] = Field(default_factory=list)

    @field_validator("asset", "contract_symbol")
    @classmethod
    def normalize_symbols(cls, value: str | None) -> str | None:
        return value.strip().upper() if value is not None else None

    @field_validator("source_url", mode="before")
    @classmethod
    def discard_unsafe_source_url(cls, value: object) -> object:
        return _safe_optional_url(value)

    @field_validator("collected_at")
    @classmethod
    def require_collection_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="Derivatives collection")

    @model_validator(mode="after")
    def validate_evidence_state(self) -> DerivativesEvidence:
        _validate_derivatives_history(self)
        _validate_derivatives_availability(self)
        _validate_funding_aggregates(self)
        _validate_open_interest_aggregates(self)
        return self


def _validate_derivatives_history(evidence: DerivativesEvidence) -> None:
    for observations in (evidence.funding_history, evidence.open_interest_history):
        timestamps = [item.observed_at for item in observations]
        if timestamps != sorted(set(timestamps)):
            raise ValueError("Derivatives observations must be unique and chronological.")
        if timestamps and timestamps[-1] > evidence.collected_at:
            raise ValueError("Derivatives observations cannot be future-dated.")


def _validate_derivatives_availability(evidence: DerivativesEvidence) -> None:
    available = bool(evidence.funding_history or evidence.open_interest_history)
    expected = (
        "complete" if evidence.funding_history and evidence.open_interest_history else "partial"
    )
    if available and (evidence.status != expected or not evidence.contract_symbol):
        raise ValueError(f"{expected.title()} derivatives evidence has an invalid status.")
    if not available and evidence.status not in {"unavailable", "not_applicable"}:
        raise ValueError("Empty derivatives evidence must be unavailable or not applicable.")
    if evidence.contract_symbol and not (
        evidence.contract_symbol.startswith(evidence.asset)
        and evidence.contract_symbol.endswith("USDT")
    ):
        raise ValueError("Derivatives contract identity must match the requested asset.")


def _validate_funding_aggregates(evidence: DerivativesEvidence) -> None:
    if not evidence.funding_history:
        if (
            evidence.latest_funding_rate is not None
            or evidence.average_funding_rate_24h is not None
        ):
            raise ValueError("Funding aggregates require validated funding observations.")
        return
    if evidence.latest_funding_rate != evidence.funding_history[-1].rate:
        raise ValueError("Latest funding rate must match the final observation.")
    rates = [
        item.rate
        for item in evidence.funding_history
        if item.observed_at >= evidence.collected_at - timedelta(hours=24)
    ]
    average = sum(rates) / len(rates)
    if evidence.average_funding_rate_24h is None or not isclose(
        evidence.average_funding_rate_24h, average, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("The 24-hour funding average did not match its observations.")


def _validate_open_interest_aggregates(evidence: DerivativesEvidence) -> None:
    if not evidence.open_interest_history:
        if (
            evidence.latest_open_interest_usd is not None
            or evidence.open_interest_change_24h_pct is not None
        ):
            raise ValueError("Open-interest aggregates require validated observations.")
        return
    latest = evidence.open_interest_history[-1]
    if evidence.latest_open_interest_usd != latest.value_usd:
        raise ValueError("Latest open interest must match the final observation.")
    target = latest.observed_at - timedelta(hours=24)
    previous = next(
        (
            item
            for item in reversed(evidence.open_interest_history[:-1])
            if item.observed_at <= target
        ),
        None,
    )
    expected = (
        (latest.value_usd / previous.value_usd - 1) * 100
        if previous and previous.value_usd > 0
        else None
    )
    if expected is None and evidence.open_interest_change_24h_pct is not None:
        raise ValueError("A 24-hour open-interest change requires a prior observation.")
    if expected is not None and (
        evidence.open_interest_change_24h_pct is None
        or not isclose(
            evidence.open_interest_change_24h_pct, expected, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise ValueError("The 24-hour open-interest change did not match its observations.")


class DefiEvidence(StrictModel):
    protocol: str | None = None
    slug: str | None = None
    category: str | None = None
    chains: list[str] = Field(default_factory=list)
    tvl_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    change_1d: float | None = Field(default=None, allow_inf_nan=False)
    change_7d: float | None = Field(default=None, allow_inf_nan=False)
    homepage: str | None = None
    status: Literal["available", "unavailable", "not_applicable"] = "available"
    source: str = "DefiLlama"
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_state: Literal["live", "cached"] = "live"
    warnings: list[str] = Field(default_factory=list)

    @field_validator("homepage", mode="before")
    @classmethod
    def discard_unsafe_homepage(cls, value: object) -> object:
        return _safe_optional_url(value)

    @field_validator("collected_at")
    @classmethod
    def require_collected_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="DeFi collection")


OnChainMetricId = Literal[
    "AdrActCnt",
    "AdrNewCnt",
    "TxCnt",
    "TxTfrValAdjUSD",
    "FeeTotUSD",
]


class OnChainObservation(StrictModel):
    observed_at: datetime
    value: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("observed_at")
    @classmethod
    def require_observed_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="On-chain observation")


class OnChainMetricSeries(StrictModel):
    metric: OnChainMetricId
    label: str = Field(min_length=1, max_length=80)
    unit: Literal["count", "usd"]
    observations: list[OnChainObservation] = Field(default_factory=list, max_length=30)
    latest_value: float = Field(ge=0, allow_inf_nan=False)
    latest_at: datetime
    seven_day_average: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    previous_seven_day_average: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    seven_day_change_pct: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator("latest_at")
    @classmethod
    def require_latest_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="On-chain metric")

    @model_validator(mode="after")
    def validate_latest_observation(self) -> OnChainMetricSeries:
        if not self.observations:
            raise ValueError("On-chain metric series require observations.")
        timestamps = [item.observed_at for item in self.observations]
        if timestamps != sorted(set(timestamps)):
            raise ValueError("On-chain observations must be unique and chronological.")
        latest = self.observations[-1]
        if self.latest_at != latest.observed_at or self.latest_value != latest.value:
            raise ValueError("On-chain latest values must match the final observation.")
        return self


class OnChainEvidence(StrictModel):
    asset: str = Field(min_length=1, max_length=40)
    provider_asset: str | None = Field(default=None, max_length=80)
    status: Literal["complete", "partial", "unavailable", "not_applicable"]
    metrics: list[OnChainMetricSeries] = Field(default_factory=list, max_length=5)
    source: str = "Coin Metrics Community"
    source_url: str | None = None
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_state: Literal["live", "cached"] = "live"
    warnings: list[str] = Field(default_factory=list)

    @field_validator("source_url", mode="before")
    @classmethod
    def discard_unsafe_source_url(cls, value: object) -> object:
        return _safe_optional_url(value)

    @field_validator("collected_at")
    @classmethod
    def require_collected_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="On-chain collection")

    @model_validator(mode="after")
    def validate_status(self) -> OnChainEvidence:
        metric_ids = [item.metric for item in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("On-chain metric series must be unique.")
        expected = "complete" if len(self.metrics) >= 3 else "partial" if self.metrics else None
        if expected is not None and self.status != expected:
            raise ValueError(f"{expected.title()} on-chain evidence has an invalid status.")
        if not self.metrics and self.status not in {"unavailable", "not_applicable"}:
            raise ValueError("Empty on-chain evidence must be unavailable or not applicable.")
        return self


class EvidenceRecord(StrictModel):
    """Normalized metadata wrapped around evidence exposed to an LLM."""

    evidence_id: str
    claim_type: str
    source: str
    source_tier: Literal["primary", "research", "news", "community"]
    collected_at: datetime
    observed_at: datetime | None = None
    asset: str | None = None
    payload: dict[str, object]

    @field_validator("collected_at", "observed_at")
    @classmethod
    def require_evidence_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_timezone(value, label="Evidence")


@dataclass(frozen=True, slots=True)
class NormalizedSource:
    """Compact source metadata shared by provider and presentation boundaries."""

    publisher: str
    title: str
    excerpt: str
    url: str | None
    observed_at: datetime | None
    source_quality: Literal["high", "medium", "low"]


def normalize_news_items(
    items: list[NewsItem], *, collected_at: datetime
) -> tuple[list[NewsItem], list[str]]:
    """Normalize, deduplicate, and strictly time-bound dated news records."""

    accepted: dict[tuple[str, str], NewsItem] = {}
    future = stale = malformed = 0
    reference = collected_at.astimezone(UTC)
    for item in items:
        normalized = _normalize_source(
            publisher=item.publisher,
            title=item.title,
            excerpt=item.excerpt,
            url=item.url,
            observed_at=item.published_at,
            title_limit=300,
        )
        if normalized is None or normalized.observed_at is None:
            malformed += 1
            continue
        if normalized.observed_at > reference:
            future += 1
            continue
        if normalized.observed_at < reference - _NEWS_MAX_AGE:
            stale += 1
            continue
        rebuilt = NewsItem(
            publisher=normalized.publisher,
            title=normalized.title,
            excerpt=normalized.excerpt,
            assets=item.assets,
            url=normalized.url,
            published_at=normalized.observed_at,
            source_quality=normalized.source_quality,
        )
        key = source_identity(normalized)
        prior = accepted.get(key)
        if prior is None or rebuilt.published_at > prior.published_at:
            accepted[key] = rebuilt
    return list(accepted.values()), _quality_warnings(
        future=future,
        stale=stale,
        malformed=malformed,
        label="news",
    )


def _normalize_source(
    *,
    publisher: object,
    title: object,
    excerpt: object,
    url: object,
    observed_at: datetime | None,
    title_limit: int,
) -> NormalizedSource | None:
    """Convert an untrusted provider row into compact evidence or reject it."""

    normalized_title = clean_text(str(title or ""), max_length=title_limit)
    normalized_excerpt = clean_text(str(excerpt or ""), max_length=MAX_EVIDENCE_EXCERPT_CHARS)
    normalized_publisher = clean_text(str(publisher or ""), max_length=120)
    if not normalized_title or not normalized_excerpt or not normalized_publisher:
        return None
    return NormalizedSource(
        publisher=normalized_publisher,
        title=normalized_title,
        excerpt=normalized_excerpt,
        url=canonical_source_url(url if isinstance(url, str) else None),
        observed_at=observed_at.astimezone(UTC) if observed_at is not None else None,
        source_quality=cast(
            Literal["high", "medium", "low"], publisher_quality(normalized_publisher)
        ),
    )


def canonical_source_url(value: str | None) -> str | None:
    """Remove tracking-only query data while retaining a safe article URL."""

    normalized = normalize_http_url(value)
    if normalized is None:
        return None
    parsed = urlsplit(normalized)
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if key.casefold() not in _TRACKING_QUERY_KEYS and not key.casefold().startswith("utm_")
        ],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), query, ""))


def source_identity(source: NormalizedSource) -> tuple[str, str]:
    """Use canonical URLs, or publisher plus title for URL-less source rows."""

    title_key = " ".join(source.title.casefold().split())
    publisher_key = " ".join(source.publisher.casefold().split())
    if source.url:
        return "url", source.url
    return f"publisher:{publisher_key}", title_key


def _quality_warnings(*, future: int, stale: int, malformed: int, label: str) -> list[str]:
    warnings: list[str] = []
    if future:
        warnings.append(f"{future} future-dated {label} source(s) were excluded.")
    if stale:
        warnings.append(f"{stale} stale {label} source(s) were excluded.")
    if malformed:
        warnings.append(f"{malformed} malformed {label} source(s) were excluded.")
    return warnings


def _require_timezone(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{label} timestamp must be timezone-aware.")
    return value.astimezone(UTC)
