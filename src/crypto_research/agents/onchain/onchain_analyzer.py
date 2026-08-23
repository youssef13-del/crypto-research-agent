"""Response schema and analysis policy for the On-Chain Activity Agent."""

from collections.abc import Mapping, Sequence

from crypto_research.agents.base import (
    AgentAnalyzer,
    AgentEvidencePolicy,
    NarrativeAssetAnalysis,
    NarrativeLiveOutput,
)
from crypto_research.agents.guardrails import AnswerRequirements, compose_narrative_answer
from crypto_research.domain.research import AgentAnswer, AnalysisInputs, ResearchCapability
from crypto_research.llm.client import LLMRole

from .onchain_prompts import (
    SYSTEM_PROMPT,
    compact_briefs,
    evidence_limits,
    output_contract,
    prompt_budget,
    structured_instruction,
)

ROLE = LLMRole.RESEARCH
REQUIRED_SCOPES = ("onchain",)


def evidence_limitations(
    inputs: AnalysisInputs, _capabilities: Sequence[ResearchCapability]
) -> list[str]:
    if inputs.onchain_result is None:
        return []
    return [
        warning
        for bundle in inputs.onchain_result.asset_results
        for warning in (*bundle.limitations, *(bundle.onchain.warnings if bundle.onchain else ()))
    ]


EVIDENCE_POLICY = AgentEvidencePolicy(
    allowed_kinds=frozenset({"onchain_activity"}),
    limitations=evidence_limitations,
    fallback_scopes=(("onchain", ("onchain_activity",)),),
    fallback_verdict="Validated network activity remains available for review.",
    no_evidence_message=(
        "No current verified on-chain activity metrics were available for this coin scope."
    ),
    fallback_claim_kinds=("onchain_activity",),
    fallback_intro="Verified network activity records remain available for review.",
)


OnchainAssetAnalysis = NarrativeAssetAnalysis


class OnchainLiveOutput(NarrativeLiveOutput):
    pass


def summarize(inputs: AnalysisInputs) -> dict[str, object]:
    onchain = inputs.onchain_result
    return {
        "onchain_summary": onchain.summary if onchain is not None else None,
        "per_asset_onchain": [
            {
                "symbol": bundle.asset.symbol,
                "status": bundle.onchain.status if bundle.onchain is not None else "unavailable",
                "metrics": [
                    {
                        "metric": metric.metric,
                        "latest_value": metric.latest_value,
                        "seven_day_change_pct": metric.seven_day_change_pct,
                    }
                    for metric in (bundle.onchain.metrics if bundle.onchain is not None else [])
                ],
            }
            for bundle in (onchain.asset_results if onchain is not None else [])
        ],
    }


def compose(
    generated: OnchainLiveOutput,
    *,
    selected_evidence: Mapping[str, object],
    limitations: Sequence[str],
    requirements: AnswerRequirements,
) -> AgentAnswer:
    return compose_narrative_answer(
        generated,
        agent="onchain_agent",
        scope="onchain",
        evidence_kind="onchain_activity",
        selected_evidence=selected_evidence,
        limitations=limitations,
        requirements=requirements,
        default_verdict="The verified network metrics provide a limited view of current activity.",
        default_interpretation=(
            "The available network metrics support only a limited activity read."
        ),
        empty_evidence="No current validated on-chain metric was available for this asset.",
        include_facts_in_section=True,
    )


ANALYZER = AgentAnalyzer(
    id="onchain_agent",
    role=ROLE,
    system_prompt=SYSTEM_PROMPT,
    output_schema=OnchainLiveOutput,
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
    "OnchainAssetAnalysis",
    "OnchainLiveOutput",
    "ANALYZER",
    "REQUIRED_SCOPES",
    "ROLE",
    "SYSTEM_PROMPT",
    "prompt_budget",
]
