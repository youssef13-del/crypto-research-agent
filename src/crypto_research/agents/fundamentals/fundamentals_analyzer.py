"""Response schema and analysis policy for the Fundamentals Agent."""

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
    evidence_ids_for_asset,
    facts_for_asset,
    finish_composition,
    grounded_text,
    join_paragraphs,
    ordered_sections,
    safe_generated_text,
    useful_comparison,
    useful_section,
    useful_verdict,
)
from crypto_research.domain.core import StrictModel
from crypto_research.domain.research import (
    AgentAnalysisSection,
    AgentAnswer,
    AnalysisInputs,
    ResearchCapability,
)
from crypto_research.llm.client import LLMRole

from .fundamentals_prompts import (
    SYSTEM_PROMPT,
    compact_briefs,
    evidence_limits,
    output_contract,
    prompt_budget,
    structured_instruction,
)

ROLE = LLMRole.FUNDAMENTALS
REQUIRED_SCOPES = ("fundamentals", "defi")


def evidence_limitations(
    inputs: AnalysisInputs, capabilities: Sequence[ResearchCapability]
) -> list[str]:
    data = inputs.fundamentals_result or inputs.research_result
    if data is None:
        return []
    values = (
        list(data.fundamentals.warnings) if ResearchCapability.FUNDAMENTALS in capabilities else []
    )
    if ResearchCapability.DEFI in capabilities:
        values.extend(data.defi.warnings)
    for bundle in data.asset_results:
        if ResearchCapability.FUNDAMENTALS in capabilities and bundle.fundamentals is not None:
            values.extend(bundle.fundamentals.warnings)
        if ResearchCapability.DEFI in capabilities and bundle.defi is not None:
            values.extend(bundle.defi.warnings)
    return values


EVIDENCE_POLICY = AgentEvidencePolicy(
    allowed_kinds=frozenset({"project_fundamentals", "defi_protocol_metrics"}),
    limitations=evidence_limitations,
    fallback_scopes=(
        ("fundamentals", ("project_fundamentals",)),
        ("defi", ("defi_protocol_metrics",)),
    ),
    fallback_verdict="Validated project evidence remains available for review.",
    no_evidence_message=(
        "Verified fundamentals or DeFi provider evidence was not available for the selected "
        "coin scope."
    ),
    fallback_claim_kinds=("project_fundamentals", "defi_protocol_metrics"),
    fallback_intro=(
        "The verified fundamentals read focuses on project size, supply context, and available "
        "protocol metrics."
    ),
    fallback_comparison_note=(
        "DeFi coverage is included only where protocol metrics were verified."
    ),
)


class FundamentalAssetAnalysis(StrictModel):
    symbol: str = Field(min_length=1, max_length=40)
    analysis: str = Field(min_length=1)

    @field_validator("analysis", mode="before")
    @classmethod
    def compact_analysis(cls, value: object) -> object:
        return compact_model_text(value, max_chars=260)


class DefiAssetAnalysis(StrictModel):
    symbol: str = Field(min_length=1, max_length=40)
    analysis: str = Field(min_length=1)

    @field_validator("analysis", mode="before")
    @classmethod
    def compact_analysis(cls, value: object) -> object:
        return compact_model_text(value, max_chars=220)


class FundamentalsLiveOutput(StrictModel):
    verdict: str = Field(min_length=1)
    assets: list[FundamentalAssetAnalysis] = Field(min_length=1, max_length=4)
    defi_assets: list[DefiAssetAnalysis] = Field(max_length=4)
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
    fundamentals = inputs.fundamentals_result or inputs.research_result
    return {
        "fundamentals_summary": fundamentals.summary if fundamentals is not None else None,
        "asset_count": len(fundamentals.asset_results) if fundamentals is not None else 0,
        "per_asset_fundamentals": _asset_summaries(inputs),
    }


def _asset_summaries(inputs: AnalysisInputs) -> list[dict[str, object]]:
    fundamentals = inputs.fundamentals_result or inputs.research_result
    if fundamentals is None:
        return []
    summaries: list[dict[str, object]] = []
    for bundle in fundamentals.asset_results:
        fundamental = bundle.fundamentals
        defi = bundle.defi
        summaries.append(
            {
                "symbol": bundle.asset.symbol,
                "fundamentals": (
                    None
                    if fundamental is None
                    else {
                        "status": fundamental.status,
                        "name": fundamental.name,
                        "market_cap": fundamental.market_cap,
                        "rank": fundamental.rank,
                        "circulating_supply": fundamental.circulating_supply,
                        "categories": fundamental.categories[:4],
                        "developer_activity": (
                            fundamental.developer_activity.model_dump(mode="json")
                            if fundamental.developer_activity is not None
                            else None
                        ),
                        "analysis_signals": _analysis_signals(fundamental),
                        "analysis_brief": _fundamentals_brief(bundle.asset.symbol, fundamental),
                    }
                ),
                "defi": (
                    None
                    if defi is None or defi.status != "available"
                    else {
                        "protocol": defi.protocol,
                        "tvl_usd": defi.tvl_usd,
                        "change_1d": defi.change_1d,
                        "change_7d": defi.change_7d,
                        "chains": defi.chains[:4],
                        "analysis_brief": _defi_brief(bundle.asset.symbol, defi),
                    }
                ),
            }
        )
    return summaries


def _analysis_signals(evidence: object) -> dict[str, str]:
    def field(name: str) -> object:
        if isinstance(evidence, Mapping):
            return evidence.get(name)
        return getattr(evidence, name, None)

    rank = field("rank")
    market_position = (
        "unknown"
        if not isinstance(rank, int)
        else "leading"
        if rank <= 10
        else "established"
        if rank <= 100
        else "smaller"
    )
    circulating = field("circulating_supply")
    maximum = field("max_supply") or field("total_supply")
    if (
        not isinstance(circulating, int | float)
        or not isinstance(maximum, int | float)
        or maximum <= 0
    ):
        supply_profile = "unknown"
    else:
        ratio = min(max(float(circulating) / float(maximum), 0.0), 1.0)
        supply_profile = (
            "limited_remaining_issuance"
            if ratio >= 0.9
            else "moderate_remaining_issuance"
            if ratio >= 0.6
            else "substantial_remaining_issuance"
        )
    developer = field("developer_activity")
    commits = (
        developer.get("commits_4_weeks")
        if isinstance(developer, Mapping)
        else getattr(developer, "commits_4_weeks", None)
    )
    return {
        "market_position": market_position,
        "supply_profile": supply_profile,
        "development_activity": (
            "unknown" if not isinstance(commits, int) else "active" if commits > 0 else "inactive"
        ),
    }


def _fundamentals_brief(symbol: str, evidence: object) -> str:
    if getattr(evidence, "status", "unavailable") != "available":
        return f"{symbol} has no available fundamentals evidence in this run."
    parts = [f"{symbol} fundamentals are available"]
    if (market_cap := getattr(evidence, "market_cap", None)) is not None:
        parts.append(f"market cap {float(market_cap):g}")
    if (rank := getattr(evidence, "rank", None)) is not None:
        parts.append(f"rank {rank}")
    if categories := list(getattr(evidence, "categories", ()))[:2]:
        parts.append("categories " + ", ".join(str(item) for item in categories))
    commits = getattr(getattr(evidence, "developer_activity", None), "commits_4_weeks", None)
    if commits is not None:
        parts.append(f"{int(commits)} commits over four weeks")
    return "; ".join(parts) + "."


def _defi_brief(symbol: str, evidence: object) -> str:
    protocol = getattr(evidence, "protocol", None) or symbol
    parts = [f"{symbol} DeFi metrics are available for {protocol}"]
    if (tvl := getattr(evidence, "tvl_usd", None)) is not None:
        parts.append(f"TVL {float(tvl):g}")
    if (change := getattr(evidence, "change_7d", None)) is not None:
        parts.append(f"7d TVL change {float(change):+.2f}%")
    return "; ".join(parts) + "."


def compose(
    generated: FundamentalsLiveOutput,
    *,
    selected_evidence: Mapping[str, object],
    limitations: Sequence[str],
    requirements: AnswerRequirements,
) -> AgentAnswer:
    expected = [item.label for item in requirements.assets]
    selected_scopes = set(requirements.scopes)
    explicit = bool(selected_scopes & {"fundamentals", "defi"})
    include_fundamentals = "fundamentals" in selected_scopes or not explicit
    include_defi = "defi" in selected_scopes
    recovery_notes: list[str] = []
    sections = ordered_sections(generated.assets, expected, recovery_notes=recovery_notes)
    verdict = useful_verdict(
        generated.verdict,
        default=(
            "The current provider snapshot supports a focused fundamentals read for this asset."
            if len(expected) == 1
            else "The verified fundamentals evidence differs across the selected assets."
        ),
        asset_count=len(expected),
    )
    structured: list[AgentAnalysisSection] = []
    paragraphs = [compact_prose(verdict, max_chars=180, max_sentences=1)]
    for symbol in expected:
        section = sections.get(asset_key(symbol))
        facts = facts_for_asset(
            symbol,
            selected_evidence,
            kinds=("project_fundamentals",),
            max_chars=320,
        )
        interpretation = useful_section(
            section.analysis if isinstance(section, FundamentalAssetAnalysis) else "",
            default=_fallback_interpretation(symbol, selected_evidence),
            avoid=(verdict,),
        )
        if include_fundamentals:
            structured.append(
                AgentAnalysisSection(
                    asset=symbol,
                    scope="fundamentals",
                    text=grounded_text(facts, interpretation, max_chars=480),
                )
            )
            paragraphs.append(
                asset_note(
                    symbol,
                    facts or "Verified fundamentals coverage is unavailable for this asset.",
                    interpretation,
                    max_chars=520,
                )
            )
    comparison = useful_comparison(
        generated.comparison,
        asset_count=len(expected),
        avoid=(verdict,),
    )
    if len(expected) > 1 and comparison:
        paragraphs.append(compact_prose(comparison, max_chars=220, max_sentences=1))

    defi_expected = (
        [
            symbol
            for symbol in expected
            if evidence_ids_for_asset(
                symbol,
                selected_evidence,
                kinds={"defi_protocol_metrics"},
            )
        ]
        if include_defi
        else []
    )
    defi_sections = ordered_sections(
        generated.defi_assets,
        defi_expected,
        recovery_notes=recovery_notes,
        label="DeFi analysis",
    )
    defi_paragraphs: list[str] = []
    for symbol in defi_expected:
        section = defi_sections.get(asset_key(symbol))
        facts = facts_for_asset(
            symbol,
            selected_evidence,
            kinds=("defi_protocol_metrics",),
            max_chars=150,
        )
        interpretation = useful_section(
            section.analysis if isinstance(section, DefiAssetAnalysis) else "",
            default="The verified protocol metrics provide a limited DeFi activity read.",
            avoid=(verdict,),
        )
        structured.append(
            AgentAnalysisSection(
                asset=symbol,
                scope="defi",
                text=grounded_text(facts, interpretation, max_chars=360),
            )
        )
        defi_paragraphs.append(asset_note(symbol, facts, interpretation, max_chars=240))
    return finish_composition(
        agent="fundamentals_agent",
        answer=join_paragraphs(paragraphs, max_chars=2_400),
        analysis=join_paragraphs(defi_paragraphs, max_chars=1_000),
        verdict=verdict,
        sections=structured,
        comparison=comparison,
        claims=deterministic_claims(
            expected,
            selected_evidence,
            kind_groups=({"project_fundamentals"}, {"defi_protocol_metrics"}),
        ),
        limitations=limitations,
        generated_limitations=[
            value for item in generated.limitations if (value := safe_generated_text(str(item)))
        ],
        recovery_notes=recovery_notes,
        confidence=generated.confidence,
        asset_count=len(expected),
    )


def _fallback_interpretation(symbol: str, evidence: Mapping[str, object]) -> str:
    payload: Mapping[str, object] = {}
    for evidence_id in evidence_ids_for_asset(
        symbol,
        evidence,
        kinds={"project_fundamentals"},
    ):
        record = evidence[evidence_id]
        candidate = record.get("payload") if isinstance(record, Mapping) else None
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not payload:
        return "The available provider coverage supports only a limited fundamental read."
    signals = _analysis_signals(payload)
    strengths: list[str] = []
    watch_items: list[str] = []
    if signals["market_position"] == "leading":
        strengths.append("an established market position")
    elif signals["market_position"] == "established":
        strengths.append("a meaningful market presence")
    elif signals["market_position"] == "smaller":
        watch_items.append("its smaller market position")
    if signals["development_activity"] == "active":
        strengths.append("visible recent code activity")
    elif signals["development_activity"] == "inactive":
        watch_items.append("limited recent code activity")
    if signals["supply_profile"] == "substantial_remaining_issuance":
        watch_items.append("substantial remaining issuance")
    elif signals["supply_profile"] == "moderate_remaining_issuance":
        watch_items.append("remaining supply issuance")
    lead = (
        " and ".join(strengths[:2]) + " support a stronger maturity signal"
        if strengths
        else "The provider snapshot supports only a limited maturity signal"
    )
    lead = lead[0].upper() + lead[1:]
    if watch_items:
        return lead + ", while " + " and ".join(watch_items[:2]) + " remain key watch items."
    if signals["development_activity"] == "active":
        return lead + ", though repository activity alone does not establish adoption or value."
    return lead + ", but market position alone does not establish project durability."


ANALYZER = AgentAnalyzer(
    id="fundamentals_agent",
    role=ROLE,
    system_prompt=SYSTEM_PROMPT,
    output_schema=FundamentalsLiveOutput,
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
    "DefiAssetAnalysis",
    "FundamentalAssetAnalysis",
    "FundamentalsLiveOutput",
    "ANALYZER",
    "REQUIRED_SCOPES",
    "ROLE",
    "SYSTEM_PROMPT",
    "prompt_budget",
]
