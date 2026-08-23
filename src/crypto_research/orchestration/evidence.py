"""Build allow-listed evidence and provider limitations for agent analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from crypto_research.agents.registry import analyzer_for
from crypto_research.domain.analytics import build_market_posture, calculate_market_features
from crypto_research.domain.evidence import DerivativesEvidence
from crypto_research.domain.market import MarketEvidence
from crypto_research.domain.research import (
    AnalysisInputs,
    DefiEvidence,
    EvidenceCoverageSummary,
    EvidenceRecord,
    FundamentalEvidence,
    FundamentalsAgentResult,
    NewsEvidence,
    NewsItem,
    ResearchAgentResult,
    ResearchCapability,
    RiskAssessment,
    TechnicalSnapshot,
)
from crypto_research.orchestration.coverage import CAPABILITY_LIMITATIONS as _CAPABILITY_LIMITATIONS
from crypto_research.orchestration.coverage import (
    build_analysis_evidence_digest,
    build_capability_coverage,
    build_complete_evidence_digest,
    build_evidence_coverage_summary,
    is_current_market,
    select_detailed_evidence,
)
from crypto_research.shared.text import unique_strings as _unique

_CAPABILITY_EVIDENCE_KINDS: dict[ResearchCapability, frozenset[str]] = {
    ResearchCapability.DISCOVERY: frozenset({"market_screen"}),
    ResearchCapability.MARKET: frozenset({"market_snapshot", "technical_calculation"}),
    ResearchCapability.NEWS: frozenset({"recent_news"}),
    ResearchCapability.FUNDAMENTALS: frozenset({"project_fundamentals"}),
    ResearchCapability.DEFI: frozenset({"defi_protocol_metrics"}),
    ResearchCapability.RISK: frozenset({"deterministic_risk_assessment", "recent_news"}),
    ResearchCapability.FORECAST: frozenset(),
    ResearchCapability.ONCHAIN: frozenset({"onchain_activity"}),
    ResearchCapability.DERIVATIVES: frozenset({"derivatives_positioning"}),
}
type SpecialistAgentId = Literal[
    "market_agent",
    "news_agent",
    "fundamentals_agent",
    "onchain_agent",
]


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    """Allow-listed evidence and provider limitations for one agent LLM call."""

    available_evidence: Mapping[str, object] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    complete_data_digest: Mapping[str, object] = field(default_factory=dict)
    coverage_summary: EvidenceCoverageSummary | None = None
    analysis_data_digest: Mapping[str, object] = field(default_factory=dict)
    agent_id: SpecialistAgentId | None = None
    requested_capabilities: tuple[ResearchCapability, ...] = ()


def build_research_evidence(inputs: AnalysisInputs) -> EvidenceSpec:
    """Build the combined evidence ledger passed to unified analysis."""

    ledger = _build_evidence_ledger(inputs)
    evidence = select_detailed_evidence(ledger)
    coverage_summary = build_evidence_coverage_summary(inputs, ledger, list(evidence))
    return EvidenceSpec(
        available_evidence=evidence,
        limitations=tuple(_research_limitations(inputs)),
        complete_data_digest=build_complete_evidence_digest(ledger, coverage_summary),
        coverage_summary=coverage_summary,
        analysis_data_digest=build_analysis_evidence_digest(ledger, coverage_summary),
        requested_capabilities=tuple(_requested_capability_set(inputs)),
    )


def build_evidence_records(inputs: AnalysisInputs) -> tuple[EvidenceRecord, ...]:
    """Return the complete validated ledger for durable run provenance."""

    ledger = _build_evidence_ledger(inputs)
    return tuple(
        EvidenceRecord.model_validate(value) for value in ledger.values() if isinstance(value, dict)
    )


def build_specialist_evidence(
    inputs: AnalysisInputs,
    *,
    agent: SpecialistAgentId,
    capabilities: Sequence[ResearchCapability],
) -> EvidenceSpec:
    """Build one specialist's isolated, auditable evidence view.

    Each LLM-analyzed guided capability has a single specialist owner. Risk
    remains a deterministic structured overlay and is not sent as a separate
    specialist-analysis capability. The full ledger stays in the coverage digest;
    the caller may subsequently choose a smaller citation-ready prompt subset.
    """

    selected = _specialist_capabilities(capabilities)
    policy = analyzer_for(agent).evidence_policy
    scoped_inputs = inputs.model_copy(
        update={"requested_capabilities": list(policy.expand(selected))}
    )
    ledger = _build_specialist_evidence_ledger(scoped_inputs, agent)
    evidence = select_detailed_evidence(ledger)
    coverage_summary = build_evidence_coverage_summary(scoped_inputs, ledger, list(evidence))
    return EvidenceSpec(
        available_evidence=evidence,
        limitations=tuple(_specialist_limitations(scoped_inputs, agent, selected)),
        complete_data_digest=build_complete_evidence_digest(ledger, coverage_summary),
        coverage_summary=coverage_summary,
        analysis_data_digest=build_analysis_evidence_digest(ledger, coverage_summary),
        agent_id=agent,
        requested_capabilities=selected,
    )


def _specialist_capabilities(
    capabilities: Sequence[ResearchCapability],
) -> tuple[ResearchCapability, ...]:
    selected = set(capabilities)
    return tuple(capability for capability in ResearchCapability if capability in selected)


def _build_specialist_evidence_ledger(
    inputs: AnalysisInputs,
    agent: SpecialistAgentId,
) -> dict[str, object]:
    """Filter the canonical ledger through the selected agent's evidence policy."""

    ledger = _build_evidence_ledger(inputs)
    allowed = analyzer_for(agent).evidence_policy.allowed_kinds
    return {
        evidence_id: record
        for evidence_id, record in ledger.items()
        if isinstance(record, Mapping) and record.get("claim_type") in allowed
    }


def _specialist_limitations(
    inputs: AnalysisInputs,
    agent: SpecialistAgentId,
    capabilities: Sequence[ResearchCapability],
) -> list[str]:
    """Keep provider gaps scoped to the specialist rather than the whole run."""

    # Workflow warnings are intentionally not copied wholesale into every
    # specialist prompt.  Each branch below derives its own validated provider
    # notes and capability coverage gaps, so an unrelated collector outage
    # cannot dilute the requested specialist's conclusion.
    values = analyzer_for(agent).evidence_policy.limitations(inputs, capabilities)
    values.extend(_specialist_coverage_limitations(inputs, capabilities))
    return _unique(values)


def _specialist_coverage_limitations(
    inputs: AnalysisInputs,
    capabilities: Sequence[ResearchCapability],
) -> list[str]:
    """Filter coverage gaps so one specialist never inherits another's outage."""

    selected = set(capabilities)
    coverage = inputs.capability_coverage or build_capability_coverage(inputs)
    multi_asset = len(inputs.assets) > 1
    return [
        f"{item.asset.symbol}: {item.limitation}" if multi_asset else item.limitation
        for item in coverage
        if (
            item.capability in selected
            and item.status != "available"
            and item.limitation is not None
        )
    ]


def _build_evidence_ledger(inputs: AnalysisInputs) -> dict[str, object]:
    """Build the allow-listed evidence records available to agent analysis.

    The ledger is filtered by requested capabilities so the LLM only sees
    evidence types relevant to the request. This prevents overloading the model
    with unrelated market/news/fundamentals data when only a subset was asked.
    """

    requested = _requested_capability_set(inputs)
    evidence: dict[str, object] = {}

    if not requested:
        _add_market_ledger(evidence, inputs)
        _add_research_ledger(evidence, inputs)
        _add_risk_ledger(evidence, inputs)
        _add_onchain_ledger(evidence, inputs)
        return evidence

    wanted_kinds = set().union(
        *(_CAPABILITY_EVIDENCE_KINDS[cap] for cap in requested if cap in _CAPABILITY_EVIDENCE_KINDS)
    )
    if ResearchCapability.DISCOVERY in requested:
        _add_market_ledger(evidence, inputs, kinds={"market_screen"})
    if ResearchCapability.MARKET in requested:
        _add_market_ledger(
            evidence,
            inputs,
            kinds={"market_snapshot", "technical_calculation"},
        )
    if ResearchCapability.DERIVATIVES in requested:
        _add_market_ledger(evidence, inputs, kinds={"derivatives_positioning"})
    _add_research_ledger(
        evidence,
        inputs,
        wanted_kinds=wanted_kinds,
    )
    if ResearchCapability.RISK in requested:
        _add_risk_ledger(evidence, inputs)
    if ResearchCapability.ONCHAIN in requested:
        _add_onchain_ledger(evidence, inputs)
    return evidence


def _requested_capability_set(inputs: AnalysisInputs) -> set[ResearchCapability]:
    if inputs.requested_capabilities:
        return set(inputs.requested_capabilities)
    if inputs.research_result is not None:
        return set(inputs.research_result.requested_capabilities)
    if inputs.fundamentals_result is not None:
        return set(inputs.fundamentals_result.requested_capabilities)
    if inputs.onchain_result is not None:
        return set(inputs.onchain_result.requested_capabilities)
    return set()


def _add_market_ledger(
    evidence: dict[str, object],
    inputs: AnalysisInputs,
    *,
    kinds: set[str] | frozenset[str] | None = None,
) -> None:
    """Add market, comparison, and opportunity records (optionally by kind)."""

    wanted = set(kinds) if kinds is not None else None
    _add_opportunity_records(evidence, inputs, wanted)
    _add_derivatives_records(evidence, inputs, wanted)
    if wanted is None or {"market_snapshot", "technical_calculation"} & wanted:
        _add_market_records(evidence, inputs)


def _add_opportunity_records(
    evidence: dict[str, object], inputs: AnalysisInputs, wanted: set[str] | None
) -> None:
    result = inputs.opportunity_result
    if result is None or result.collected_at > _reference_time(inputs):
        return
    if wanted is not None and "market_screen" not in wanted:
        return
    for candidate in result.candidates:
        evidence_id = f"opportunity.{candidate.symbol}"
        evidence[evidence_id] = _record(
            evidence_id=evidence_id,
            claim_type="market_screen",
            source="exchange market data and deterministic scoring",
            source_tier="primary",
            collected_at=result.collected_at,
            asset=candidate.symbol,
            payload=candidate.model_dump(mode="json"),
        )


def _add_derivatives_records(
    evidence: dict[str, object], inputs: AnalysisInputs, wanted: set[str] | None
) -> None:
    if wanted is not None and "derivatives_positioning" not in wanted:
        return
    comparison = inputs.market_comparison_result
    values = (
        [item.derivatives for item in comparison.assets if item.derivatives is not None]
        if comparison is not None
        else []
    )
    if inputs.market_result is not None and inputs.market_result.derivatives is not None:
        values.append(inputs.market_result.derivatives)
    for derivative in values:
        _add_derivatives_evidence(evidence, derivative, reference_time=_reference_time(inputs))


def _add_market_records(evidence: dict[str, object], inputs: AnalysisInputs) -> None:
    comparison = inputs.market_comparison_result
    if comparison is not None:
        for asset in comparison.assets:
            _add_market_evidence(
                evidence, asset.market, asset.technical, reference_time=_reference_time(inputs)
            )
            _add_contextual_market_records(evidence, asset.contextual_timeframes, inputs)
    result = inputs.market_result
    if result is not None:
        _add_market_evidence(
            evidence, result.market, result.technical, reference_time=_reference_time(inputs)
        )
        _add_contextual_market_records(evidence, result.contextual_timeframes, inputs)


def _add_contextual_market_records(
    evidence: dict[str, object], contextual_timeframes: Sequence[object], inputs: AnalysisInputs
) -> None:
    for contextual in contextual_timeframes:
        market = getattr(contextual, "market", None)
        if market is not None:
            _add_market_evidence(
                evidence,
                market,
                getattr(contextual, "technical", None),
                reference_time=_reference_time(inputs),
            )


def _add_research_ledger(
    evidence: dict[str, object],
    inputs: AnalysisInputs,
    *,
    wanted_kinds: set[str] | frozenset[str] | None = None,
) -> None:
    research = inputs.research_result
    fundamentals = inputs.fundamentals_result or inputs.research_result
    if research is None and fundamentals is None:
        return
    wanted = set(wanted_kinds) if wanted_kinds is not None else None

    def wants(kind: str) -> bool:
        return wanted is None or kind in wanted

    if fundamentals is not None and wants("project_fundamentals"):
        _add_fundamentals_records(evidence, inputs, fundamentals, kind="fundamentals")
    if fundamentals is not None and wants("defi_protocol_metrics"):
        _add_fundamentals_records(evidence, inputs, fundamentals, kind="defi")
    if research is None or wanted in (
        {"project_fundamentals"},
        {"defi_protocol_metrics"},
    ):
        return

    # Aggregators return the same stories for several assets, so a shared
    # seen-set keeps each story in the ledger once instead of once per asset.
    assets_with_news: set[str] = set()
    seen_news: dict[tuple[str, str], str] = {}

    if len(inputs.assets) > 1 and research.asset_results:
        for bundle in research.asset_results:
            key = bundle.asset.evidence_key
            if bundle.news is not None and wants("recent_news"):
                _add_news_records(
                    evidence,
                    bundle.news,
                    prefix=f"news.{key}",
                    asset=bundle.asset.symbol,
                    reference_time=_reference_time(inputs),
                    seen=seen_news,
                )
                assets_with_news.add(bundle.asset.symbol)
        # Fallback: shared news covers assets missing their own bundle news.
        if wants("recent_news") and research.news.items:
            for asset in inputs.assets:
                if asset.symbol not in assets_with_news:
                    _add_news_records(
                        evidence,
                        research.news,
                        prefix=f"news.{asset.evidence_key}",
                        asset=asset.symbol,
                        reference_time=_reference_time(inputs),
                        seen=seen_news,
                    )
        return

    # Single asset case - use research.news
    if wants("recent_news"):
        _add_news_records(
            evidence,
            research.news,
            prefix="news",
            asset=inputs.assets[0].symbol if inputs.assets else None,
            reference_time=_reference_time(inputs),
            seen=seen_news,
        )


def _add_fundamentals_records(
    evidence: dict[str, object],
    inputs: AnalysisInputs,
    result: ResearchAgentResult | FundamentalsAgentResult,
    *,
    kind: Literal["fundamentals", "defi"],
) -> None:
    if len(inputs.assets) > 1 and result.asset_results:
        for bundle in result.asset_results:
            key = bundle.asset.evidence_key
            if kind == "fundamentals" and bundle.fundamentals is not None:
                _add_fundamental_record(
                    evidence,
                    bundle.fundamentals,
                    evidence_id=f"fundamentals.{key}",
                    asset=bundle.asset.symbol,
                    reference_time=_reference_time(inputs),
                )
            elif kind == "defi" and bundle.defi is not None:
                _add_defi_record(
                    evidence,
                    bundle.defi,
                    evidence_id=f"defi.{key}",
                    asset=bundle.asset.symbol,
                    reference_time=_reference_time(inputs),
                )
        return
    if kind == "fundamentals":
        _add_fundamental_record(
            evidence,
            result.fundamentals,
            evidence_id="fundamentals",
            asset=inputs.assets[0].symbol if inputs.assets else None,
            reference_time=_reference_time(inputs),
        )
    else:
        _add_defi_record(
            evidence,
            result.defi,
            evidence_id="defi",
            asset=inputs.assets[0].symbol if inputs.assets else None,
            reference_time=_reference_time(inputs),
        )


def _add_onchain_ledger(evidence: dict[str, object], inputs: AnalysisInputs) -> None:
    if inputs.onchain_result is None:
        return
    reference = _reference_time(inputs)
    for bundle in inputs.onchain_result.asset_results:
        onchain = bundle.onchain
        if onchain is None or onchain.collected_at > reference:
            continue
        for metric in onchain.metrics:
            evidence_id = f"onchain.{bundle.asset.evidence_key}.{metric.metric.casefold()}"
            evidence[evidence_id] = _record(
                evidence_id=evidence_id,
                claim_type="onchain_activity",
                source=onchain.source,
                source_tier="primary",
                collected_at=onchain.collected_at,
                observed_at=metric.latest_at,
                asset=bundle.asset.symbol,
                payload={
                    "metric": metric.metric,
                    "label": metric.label,
                    "unit": metric.unit,
                    "latest_value": metric.latest_value,
                    "seven_day_average": metric.seven_day_average,
                    "previous_seven_day_average": metric.previous_seven_day_average,
                    "seven_day_change_pct": metric.seven_day_change_pct,
                },
            )


def _add_risk_ledger(evidence: dict[str, object], inputs: AnalysisInputs) -> None:
    if inputs.risk_result is None:
        return
    risk = inputs.risk_result
    if len(risk.asset_results) > 1:
        for result in risk.asset_results:
            evidence_id = f"risk.{result.asset.evidence_key}"
            evidence[evidence_id] = _risk_record(
                evidence_id,
                result.assessment,
                asset=result.asset.symbol,
                collected_at=_reference_time(inputs),
            )
        evidence["risk.aggregate"] = _risk_record(
            "risk.aggregate", risk.assessment, collected_at=_reference_time(inputs)
        )
    else:
        evidence["risk"] = _risk_record(
            "risk", risk.assessment, collected_at=_reference_time(inputs)
        )


def _research_limitations(inputs: AnalysisInputs) -> list[str]:
    """Collect already-sanitized collector and evidence coverage warnings once."""

    values = [
        *inputs.warnings,
        *[
            str(item.get("message", "")).strip()
            for item in inputs.errors
            if str(item.get("message", "")).strip()
        ],
    ]
    if inputs.opportunity_result is not None:
        values.extend(inputs.opportunity_result.warnings)
        if inputs.opportunity_result.collected_at > _reference_time(inputs):
            values.append("Future-dated opportunity data were excluded before analysis.")
    if inputs.market_comparison_result is not None:
        values.extend(inputs.market_comparison_result.warnings)
        for item in inputs.market_comparison_result.assets:
            if item.derivatives is not None:
                values.extend(item.derivatives.warnings)
        if any(
            not is_current_market(item.market, context=inputs.collection_context)
            for item in inputs.market_comparison_result.assets
        ):
            values.append("Future-dated market data were excluded before analysis.")
    if inputs.market_result is not None and not is_current_market(
        inputs.market_result.market, context=inputs.collection_context
    ):
        values.append("Future-dated market data were excluded before analysis.")
    if inputs.market_result is not None and inputs.market_result.derivatives is not None:
        values.extend(inputs.market_result.derivatives.warnings)
    if inputs.research_result is not None:
        research = inputs.research_result
        values.extend(research.news.warnings)
        values.extend(research.fundamentals.warnings)
        values.extend(research.defi.warnings)
        for bundle in research.asset_results:
            prefix = f"{bundle.asset.symbol}: " if len(research.asset_results) > 1 else ""
            values.extend(f"{prefix}{warning}" for warning in bundle.limitations)
            values.extend(
                f"{prefix}{warning}"
                for warning in _future_news_limitations(
                    bundle.news, reference_time=_reference_time(inputs)
                )
            )
        values.extend(
            _future_news_limitations(research.news, reference_time=_reference_time(inputs))
        )
        values.extend(
            _future_collection_limitations(research, reference_time=_reference_time(inputs))
        )
    if inputs.risk_result is not None:
        values.extend(inputs.risk_result.assessment.coverage_gaps)
    values.extend(_coverage_limitations(inputs))
    return _unique(values)


def _coverage_limitations(inputs: AnalysisInputs) -> list[str]:
    coverage = inputs.capability_coverage or build_capability_coverage(inputs)
    if not coverage:
        requested = _requested_capability_set(inputs)
        research = inputs.research_result
        resolved = set(research.capabilities) if research is not None else set()
        if inputs.opportunity_result is not None:
            resolved.add(ResearchCapability.DISCOVERY)
        if inputs.market_result is not None or (
            inputs.market_comparison_result is not None and inputs.market_comparison_result.assets
        ):
            resolved.add(ResearchCapability.MARKET)
        if inputs.risk_result is not None:
            resolved.add(ResearchCapability.RISK)
        derivatives = []
        if inputs.market_result is not None and inputs.market_result.derivatives is not None:
            derivatives.append(inputs.market_result.derivatives)
        if inputs.market_comparison_result is not None:
            derivatives.extend(
                item.derivatives
                for item in inputs.market_comparison_result.assets
                if item.derivatives is not None
            )
        if any(item.status in {"complete", "partial"} for item in derivatives):
            resolved.add(ResearchCapability.DERIVATIVES)
        return [
            _CAPABILITY_LIMITATIONS[capability]
            for capability in ResearchCapability
            if capability in requested - resolved
        ]
    multi_asset = len(inputs.assets) > 1
    return [
        f"{item.asset.symbol}: {item.limitation}" if multi_asset else item.limitation
        for item in coverage
        if item.status != "available" and item.limitation is not None
    ]


def _repeats_title(title: str, excerpt: str) -> bool:
    title_key = " ".join(title.casefold().split())
    excerpt_key = " ".join(excerpt.casefold().split())
    return bool(title_key and excerpt_key) and (
        excerpt_key.startswith(title_key) or title_key.startswith(excerpt_key)
    )


def _add_news_records(
    evidence: dict[str, object],
    news: NewsEvidence,
    *,
    prefix: str,
    asset: str | None = None,
    reference_time: datetime,
    seen: dict[tuple[str, str], str],
) -> None:
    for index, item in enumerate(_current_news_items(news, reference_time=reference_time)):
        key = (item.publisher.casefold(), " ".join(item.title.casefold().split()))
        if key in seen:
            continue
        excerpt = "" if item.excerpt and _repeats_title(item.title, item.excerpt) else item.excerpt
        evidence_id = f"{prefix}.{index}"
        evidence[evidence_id] = _record(
            evidence_id=evidence_id,
            claim_type="recent_news",
            source=item.publisher,
            source_tier="news",
            collected_at=_safe_collected_at(news.collected_at),
            observed_at=item.published_at,
            asset=asset or (", ".join(item.assets) if item.assets else news.query),
            payload={
                "publisher": item.publisher,
                "title": item.title,
                "excerpt": excerpt,
                "assets": item.assets,
                "published_at": item.published_at,
                "source_quality": item.source_quality,
            },
        )
        seen[key] = evidence_id


def _current_news_items(
    news: NewsEvidence | None,
    *,
    reference_time: datetime | None = None,
) -> list[NewsItem]:
    if news is None:
        return []
    reference = min(news.collected_at, reference_time or datetime.now(UTC))
    return [item for item in news.items if item.published_at <= reference]


def _has_current_news(news: NewsEvidence | None) -> bool:
    return bool(_current_news_items(news))


def _future_news_limitations(news: NewsEvidence | None, *, reference_time: datetime) -> list[str]:
    if news is None:
        return []
    reference = min(news.collected_at, reference_time)
    count = sum(item.published_at > reference for item in news.items)
    if not count:
        return []
    return [
        f"{count} future-dated news item"
        f"{' was' if count == 1 else 's were'} excluded before analysis."
    ]


def _future_collection_limitations(
    research: ResearchAgentResult, *, reference_time: datetime
) -> list[str]:
    values = [research.fundamentals.collected_at, research.defi.collected_at]
    if any(value > reference_time for value in values):
        return ["Future-dated provider metadata were excluded before analysis."]
    return []


def _add_fundamental_record(
    evidence: dict[str, object],
    fundamentals: FundamentalEvidence,
    *,
    evidence_id: str,
    asset: str | None = None,
    reference_time: datetime,
) -> None:
    if fundamentals.status == "available" and fundamentals.collected_at <= reference_time:
        evidence[evidence_id] = _record(
            evidence_id=evidence_id,
            claim_type="project_fundamentals",
            source=fundamentals.source,
            source_tier="research",
            collected_at=_safe_collected_at(fundamentals.collected_at),
            asset=asset or fundamentals.symbol,
            payload=fundamentals.model_dump(mode="json", exclude={"warnings", "homepage"}),
        )


def _add_defi_record(
    evidence: dict[str, object],
    defi: DefiEvidence,
    *,
    evidence_id: str,
    asset: str | None = None,
    reference_time: datetime,
) -> None:
    if defi.status == "available" and defi.collected_at <= reference_time:
        evidence[evidence_id] = _record(
            evidence_id=evidence_id,
            claim_type="defi_protocol_metrics",
            source=defi.source,
            source_tier="research",
            collected_at=_safe_collected_at(defi.collected_at),
            asset=asset or defi.slug,
            payload=defi.model_dump(mode="json", exclude={"warnings", "homepage"}),
        )


def _risk_record(
    evidence_id: str,
    assessment: RiskAssessment,
    *,
    asset: str | None = None,
    collected_at: datetime,
) -> dict[str, object]:
    return _record(
        evidence_id=evidence_id,
        claim_type="deterministic_risk_assessment",
        source="ChainScope deterministic risk model",
        source_tier="research",
        collected_at=collected_at,
        asset=asset,
        payload=assessment.model_dump(mode="json"),
    )


def _add_market_evidence(
    evidence: dict[str, object],
    market: MarketEvidence,
    technical: TechnicalSnapshot | None,
    *,
    reference_time: datetime,
) -> None:
    if market.collected_at > reference_time or market.last_time > reference_time:
        return
    symbol = market.symbol
    features = calculate_market_features(market)
    suffix = "" if market.timeframe == "1h" else f".{market.timeframe}"
    market_id = f"market.{symbol}{suffix}"
    payload: dict[str, object] = {
        "snapshot": market.model_dump(mode="json", exclude={"candles"}),
        "ohlcv_features": features.model_dump(mode="json"),
    }
    if technical is not None:
        payload["market_posture"] = build_market_posture(market, technical).model_dump(mode="json")
    evidence[market_id] = _record(
        evidence_id=market_id,
        claim_type="market_snapshot",
        source=market.data_source,
        source_tier="primary",
        collected_at=_safe_collected_at(market.collected_at),
        observed_at=market.last_time,
        asset=symbol,
        payload=payload,
    )
    if technical is not None and technical.status == "available":
        technical_id = f"technical.{symbol}{suffix}"
        evidence[technical_id] = _record(
            evidence_id=technical_id,
            claim_type="technical_calculation",
            source="ChainScope deterministic indicators",
            source_tier="research",
            collected_at=_safe_collected_at(market.collected_at),
            observed_at=market.last_time,
            asset=symbol,
            payload=technical.model_dump(mode="json"),
        )


def _add_derivatives_evidence(
    evidence: dict[str, object],
    derivatives: DerivativesEvidence,
    *,
    reference_time: datetime,
) -> None:
    if derivatives.status not in {"complete", "partial"}:
        return
    observations = [
        *(item.observed_at for item in derivatives.funding_history),
        *(item.observed_at for item in derivatives.open_interest_history),
    ]
    if derivatives.collected_at > reference_time or any(
        observed_at > reference_time for observed_at in observations
    ):
        return
    evidence_id = f"derivatives.{derivatives.asset}"
    payload = derivatives.model_dump(mode="json")
    payload["research_capability"] = ResearchCapability.DERIVATIVES.value
    evidence[evidence_id] = _record(
        evidence_id=evidence_id,
        claim_type="derivatives_positioning",
        source=derivatives.source,
        source_tier="primary",
        collected_at=_safe_collected_at(derivatives.collected_at),
        observed_at=max(observations, default=derivatives.collected_at),
        asset=derivatives.asset,
        payload=payload,
    )


def _record(**values: object) -> dict[str, object]:
    return EvidenceRecord.model_validate(values).model_dump(mode="json")


def _safe_collected_at(value: datetime) -> datetime:
    return min(value, datetime.now(UTC))


def _reference_time(inputs: AnalysisInputs) -> datetime:
    if inputs.collection_context is not None:
        return inputs.collection_context.collected_at
    return datetime.now(UTC)


__all__ = [
    "EvidenceSpec",
    "SpecialistAgentId",
    "build_evidence_records",
    "build_capability_coverage",
    "build_research_evidence",
    "build_specialist_evidence",
]
