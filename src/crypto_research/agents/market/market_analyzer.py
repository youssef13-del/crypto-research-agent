"""Response schema and analysis policy for the Market & Risk Agent."""

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, field_validator

from crypto_research.agents.base import AgentAnalyzer, AgentEvidencePolicy, compact_model_text
from crypto_research.agents.guardrails import (
    AnswerRequirements,
    asset_key,
    asset_note,
    compact_prose,
    deterministic_claims,
    facts_for_asset,
    finish_composition,
    grounded_text,
    join_paragraphs,
    merge_clauses,
    ordered_sections,
    safe_generated_text,
    useful_comparison,
    useful_section,
    useful_verdict,
)
from crypto_research.domain.analytics import build_market_posture
from crypto_research.domain.core import StrictModel
from crypto_research.domain.market import MarketEvidence
from crypto_research.domain.research import (
    AgentAnalysisSection,
    AgentAnswer,
    AnalysisInputs,
    ResearchCapability,
    TechnicalSnapshot,
)
from crypto_research.llm.client import LLMRole

from .market_prompts import (
    SYSTEM_PROMPT,
    compact_briefs,
    evidence_limits,
    output_contract,
    prompt_budget,
    structured_instruction,
)

ROLE = LLMRole.MARKET
REQUIRED_SCOPES = ("market", "risk", "derivatives", "discovery")


def evidence_limitations(
    inputs: AnalysisInputs, capabilities: Sequence[ResearchCapability]
) -> list[str]:
    values = list(inputs.opportunity_result.warnings) if inputs.opportunity_result else []
    comparison = inputs.market_comparison_result
    if comparison is not None:
        values.extend(comparison.warnings)
        values.extend(
            warning
            for item in comparison.assets
            if item.derivatives is not None
            for warning in item.derivatives.warnings
        )
    if inputs.market_result is not None:
        values.extend(inputs.market_result.market.data_quality.warnings)
        if inputs.market_result.derivatives is not None:
            values.extend(inputs.market_result.derivatives.warnings)
    if ResearchCapability.RISK in capabilities and inputs.risk_result is not None:
        values.extend(inputs.risk_result.assessment.coverage_gaps)
    return values


EVIDENCE_POLICY = AgentEvidencePolicy(
    allowed_kinds=frozenset(
        {
            "market_screen",
            "market_snapshot",
            "technical_calculation",
            "derivatives_positioning",
            "deterministic_risk_assessment",
        }
    ),
    limitations=evidence_limitations,
    capability_expansions={
        ResearchCapability.RISK: frozenset({ResearchCapability.MARKET}),
    },
    fallback_scopes=(
        ("market", ("market_screen", "market_snapshot", "technical_calculation")),
        ("derivatives", ("derivatives_positioning",)),
        ("risk", ("deterministic_risk_assessment",)),
    ),
    fallback_verdict="Validated market evidence remains available for review.",
    no_evidence_message=(
        "Verified market or risk evidence was not available for the selected coin scope."
    ),
    fallback_claim_kinds=(
        "market_snapshot",
        "technical_calculation",
        "derivatives_positioning",
        "deterministic_risk_assessment",
    ),
    fallback_intro="The verified market read is limited but still usable.",
    fallback_multi_intro="The verified market read is mixed across the selected coins.",
    fallback_comparison_note=(
        "On comparison, the better-looking coin is the one with stronger price momentum, "
        "cleaner trend evidence, and fewer risk coverage gaps; these records do not imply "
        "a trade recommendation."
    ),
    compare_fallback_always=True,
)


class MarketAssetAnalysis(StrictModel):
    symbol: str = Field(min_length=1, max_length=40)
    market_analysis: str = Field(min_length=1)
    risk_analysis: str = Field(min_length=1)

    @field_validator("market_analysis", mode="before")
    @classmethod
    def compact_market_analysis(cls, value: object) -> object:
        return compact_model_text(value, max_chars=220)

    @field_validator("risk_analysis", mode="before")
    @classmethod
    def compact_risk_analysis(cls, value: object) -> object:
        return compact_model_text(value, max_chars=180)


class MarketLiveOutput(StrictModel):
    verdict: str = Field(min_length=1)
    assets: list[MarketAssetAnalysis] = Field(min_length=1, max_length=4)
    comparison: str
    limitations: list[str] = Field(max_length=3)
    confidence: Literal["low", "medium", "high"]

    @field_validator("verdict", "comparison", mode="before")
    @classmethod
    def compact_summary(cls, value: object) -> object:
        return compact_model_text(value, max_chars=220)

    @field_validator("limitations", mode="before")
    @classmethod
    def compact_limitations(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [compact_model_text(item, max_chars=160) for item in value[:3]]


def summarize(inputs: AnalysisInputs) -> dict[str, object]:
    discovery = inputs.opportunity_result.candidates[:4] if inputs.opportunity_result else []
    return {
        "market_summary": (
            inputs.market_result.summary if inputs.market_result is not None else None
        ),
        "per_asset_market": _asset_summaries(inputs),
        "comparison_brief": _comparison_brief(inputs),
        "risk_brief": _risk_brief(inputs),
        "comparison_warnings": (
            inputs.market_comparison_result.warnings
            if inputs.market_comparison_result is not None
            else []
        ),
        "discovery_candidates": [
            {
                "rank": candidate.rank,
                "symbol": candidate.symbol,
                "score": candidate.score,
                "momentum_24h": candidate.momentum_24h,
                "trend": candidate.trend,
                "volatility_24h": candidate.volatility_24h,
                "screen_reason": candidate.reason,
            }
            for candidate in discovery
        ],
    }


def _market_brief(market: MarketEvidence, technical: TechnicalSnapshot) -> str:
    posture = build_market_posture(market, technical)
    parts = [f"{market.symbol}: {market.current_price:,.0f}".replace(",", "")]
    if posture.change_24h_percent is not None:
        sign = "+" if posture.change_24h_percent >= 0 else ""
        parts.append(f"{sign}{posture.change_24h_percent:.1f}%")
    parts.append(f"trend {technical.trend}")
    if technical.rsi is not None:
        parts.append(f"rsi {technical.rsi:.1f} ({posture.rsi_band})")
    return " ".join(parts)


def _comparison_brief(inputs: AnalysisInputs) -> str | None:
    comparison = inputs.market_comparison_result
    if comparison is not None and comparison.assets:
        entries = [_market_brief(item.market, item.technical) for item in comparison.assets]
    elif inputs.market_result is not None:
        entries = [_market_brief(inputs.market_result.market, inputs.market_result.technical)]
    else:
        return None
    return " | ".join(entries)[:300]


def _risk_brief(inputs: AnalysisInputs) -> str | None:
    risk = inputs.risk_result
    if risk is None:
        return None
    assessments = (
        [(result.asset.symbol, result.assessment) for result in risk.asset_results]
        if risk.asset_results
        else [("", risk.assessment)]
    )
    entries: list[str] = []
    for symbol, assessment in assessments:
        detail = f"score {assessment.score:.0f} ({assessment.band})"
        if symbol:
            detail = f"{symbol}: {detail}, confidence {assessment.evidence_confidence:.0f}/100"
        if factors := ", ".join(assessment.factors[: 3 if not symbol else 2]):
            detail += f" factors {factors}"
        if gaps := ", ".join(assessment.coverage_gaps[:2]):
            detail += f" gaps {gaps}"
        entries.append(detail)
    return " | ".join(entries)[:240]


def _asset_summaries(inputs: AnalysisInputs) -> list[dict[str, object]]:
    risk_by_symbol = {
        result.asset.symbol: result.assessment
        for result in (inputs.risk_result.asset_results if inputs.risk_result is not None else [])
    }
    comparison = inputs.market_comparison_result
    assets = (
        [(item.market, item.technical) for item in comparison.assets]
        if comparison is not None and comparison.assets
        else (
            [(inputs.market_result.market, inputs.market_result.technical)]
            if inputs.market_result is not None
            else []
        )
    )
    summaries: list[dict[str, object]] = []
    for market, technical in assets:
        posture = build_market_posture(market, technical)
        risk = risk_by_symbol.get(market.symbol)
        brief = (
            f"{market.symbol} trades around {market.current_price:g} with a {technical.trend} trend"
        )
        if posture.change_24h_percent is not None:
            brief += f"; 24h change is {posture.change_24h_percent:+.2f}%"
        if risk is not None:
            brief += f"; risk is {risk.band} ({risk.score:.0f}/100)"
        summaries.append(
            {
                "symbol": market.symbol,
                "price": market.current_price,
                "trend": technical.trend,
                "rsi": technical.rsi,
                "change_24h_percent": posture.change_24h_percent,
                "risk": (
                    None
                    if risk is None
                    else {
                        "score": risk.score,
                        "band": risk.band,
                        "evidence_confidence": risk.evidence_confidence,
                        "factors": risk.factors[:3],
                        "coverage_gaps": risk.coverage_gaps[:3],
                    }
                ),
                "analysis_brief": brief + ".",
            }
        )
    return summaries


def compose(
    generated: MarketLiveOutput,
    *,
    selected_evidence: Mapping[str, object],
    limitations: Sequence[str],
    requirements: AnswerRequirements,
) -> AgentAnswer:
    expected = [item.label for item in requirements.assets]
    selected_scopes = set(requirements.scopes)
    explicit = bool(selected_scopes & {"market", "risk", "derivatives"})
    discovery = "discovery" in selected_scopes
    include_derivatives = "derivatives" in selected_scopes
    include_market = discovery or "market" in selected_scopes or not explicit
    include_risk = not discovery and ("risk" in selected_scopes or not explicit)
    recovery_notes: list[str] = []
    sections = ordered_sections(generated.assets, expected, recovery_notes=recovery_notes)
    default_verdict = _default_verdict(requirements.scopes, len(expected))
    verdict = useful_verdict(
        generated.verdict,
        default=default_verdict,
        asset_count=len(expected),
    )
    structured: list[AgentAnalysisSection] = []
    paragraphs = [compact_prose(verdict, max_chars=180, max_sentences=1)]
    for symbol in expected:
        section = sections.get(asset_key(symbol))
        market_facts = facts_for_asset(
            symbol,
            selected_evidence,
            kinds=(
                ("market_screen",) if discovery else ("market_snapshot", "technical_calculation")
            ),
            max_chars=260,
        )
        derivatives_facts = facts_for_asset(
            symbol,
            selected_evidence,
            kinds=("derivatives_positioning",),
            max_chars=230,
        )
        risk_facts = facts_for_asset(
            symbol,
            selected_evidence,
            kinds=("deterministic_risk_assessment",),
            max_chars=230,
        )
        facts = compact_prose(
            " ".join(
                value
                for value, included in (
                    (market_facts, include_market),
                    (derivatives_facts, include_derivatives),
                    (risk_facts, include_risk),
                )
                if included and value
            ),
            max_chars=380,
            max_sentences=4,
        )
        market_view = useful_section(
            section.market_analysis if isinstance(section, MarketAssetAnalysis) else "",
            default="The verified market metrics provide the clearest view of current posture.",
            avoid=(verdict,),
        )
        risk_view = useful_section(
            section.risk_analysis if isinstance(section, MarketAssetAnalysis) else "",
            default="Observed risk should be read alongside the available evidence coverage.",
            avoid=(verdict, market_view),
        )
        interpretations: list[str] = []
        if include_market:
            structured.append(
                AgentAnalysisSection(
                    asset=symbol,
                    scope="market",
                    text=grounded_text(market_facts, market_view, max_chars=430),
                )
            )
            interpretations.append(market_view)
        if include_derivatives:
            derivatives_view = (
                "Funding and open interest describe positioning, not liquidations or future "
                "price direction."
            )
            structured.append(
                AgentAnalysisSection(
                    asset=symbol,
                    scope="derivatives",
                    text=grounded_text(
                        derivatives_facts,
                        derivatives_view,
                        max_chars=390,
                    ),
                )
            )
            interpretations.append(derivatives_view)
        if include_risk:
            structured.append(
                AgentAnalysisSection(
                    asset=symbol,
                    scope="risk",
                    text=grounded_text(risk_facts, risk_view, max_chars=390),
                )
            )
            interpretations.append(risk_view)
        fallback = (
            "Verified market evidence is limited for this asset."
            if include_market
            else "Verified derivatives evidence is limited for this asset."
            if include_derivatives
            else "Verified risk evidence is limited for this asset."
        )
        paragraphs.append(
            asset_note(
                symbol,
                facts or fallback,
                merge_clauses(interpretations),
                max_chars=470,
            )
        )
    comparison = useful_comparison(
        generated.comparison,
        asset_count=len(expected),
        avoid=(verdict,),
    )
    if len(expected) > 1:
        paragraphs.append(
            compact_prose(
                comparison
                or "Relative comparison is limited to the verified per-coin differences above.",
                max_chars=220,
                max_sentences=1,
            )
        )
    return finish_composition(
        agent="market_agent",
        answer=join_paragraphs(paragraphs, max_chars=2_400),
        analysis="",
        verdict=verdict,
        sections=structured,
        comparison=comparison,
        claims=deterministic_claims(
            expected,
            selected_evidence,
            kind_groups=(
                {
                    "market_screen",
                    "market_snapshot",
                    "technical_calculation",
                    "derivatives_positioning",
                    "deterministic_risk_assessment",
                },
            ),
        ),
        limitations=limitations,
        generated_limitations=[
            value for item in generated.limitations if (value := safe_generated_text(str(item)))
        ],
        recovery_notes=recovery_notes,
        confidence=generated.confidence,
        asset_count=len(expected),
    )


def _default_verdict(scopes: Sequence[str], asset_count: int) -> str:
    selected = set(scopes)
    if selected & {"market", "risk"} == {"market"}:
        return "The verified market evidence provides a focused view of current conditions."
    if selected & {"market", "risk"} == {"risk"}:
        return "The deterministic evidence provides a focused view of observed risk."
    return (
        "The verified market and risk evidence provides a focused current read."
        if asset_count == 1
        else "The verified market and risk evidence provides a usable comparative read."
    )


ANALYZER = AgentAnalyzer(
    id="market_agent",
    role=ROLE,
    system_prompt=SYSTEM_PROMPT,
    output_schema=MarketLiveOutput,
    prompt_budget=prompt_budget,
    structured_instruction=structured_instruction,
    output_contract=output_contract,
    evidence_policy=EVIDENCE_POLICY,
    evidence_limits=evidence_limits,
    summarize=summarize,
    compose=compose,
    compact_briefs=compact_briefs,
)

__all__ = [
    "MarketAssetAnalysis",
    "MarketLiveOutput",
    "ANALYZER",
    "REQUIRED_SCOPES",
    "ROLE",
    "SYSTEM_PROMPT",
    "prompt_budget",
]
