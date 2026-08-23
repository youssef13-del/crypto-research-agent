"""Response schema and analysis policy for the News Agent."""

from collections.abc import Mapping, Sequence

from crypto_research.agents.base import (
    AgentAnalyzer,
    AgentEvidencePolicy,
    NarrativeAssetAnalysis,
    NarrativeLiveOutput,
)
from crypto_research.agents.guardrails import AnswerRequirements, compose_narrative_answer
from crypto_research.domain.evidence import NewsEvidence, NewsItem
from crypto_research.domain.research import AgentAnswer, AnalysisInputs, ResearchCapability
from crypto_research.llm.client import LLMRole

from .news_prompts import (
    SYSTEM_PROMPT,
    compact_briefs,
    evidence_limits,
    output_contract,
    prompt_budget,
    structured_instruction,
)

ROLE = LLMRole.RESEARCH
REQUIRED_SCOPES = ("news",)


def evidence_limitations(
    inputs: AnalysisInputs, capabilities: Sequence[ResearchCapability]
) -> list[str]:
    if inputs.research_result is None or ResearchCapability.NEWS not in capabilities:
        return []
    return list(inputs.research_result.news.warnings)


EVIDENCE_POLICY = AgentEvidencePolicy(
    allowed_kinds=frozenset({"recent_news"}),
    limitations=evidence_limitations,
    fallback_scopes=(("news", ("recent_news",)),),
    fallback_verdict="Validated recent coverage remains available for review.",
    no_evidence_message=(
        "No recent verified articles were found for that request. Try a more specific asset or "
        "topic name."
    ),
    fallback_claim_kinds=("recent_news",),
    fallback_intro=(
        "The verified news read is based only on recent source records that survived validation."
    ),
    fallback_comparison_note=(
        "Comparison is limited to the supplied coverage; missing records remain a coverage gap."
    ),
    structured_section_override=(
        "The validated reports provide current context, but live interpretation is unavailable "
        "and the coverage does not establish market impact."
    ),
)


NewsAssetAnalysis = NarrativeAssetAnalysis


class NewsLiveOutput(NarrativeLiveOutput):
    pass


def summarize(inputs: AnalysisInputs) -> dict[str, object]:
    research = inputs.research_result
    return {
        "news_summary": research.summary if research is not None else None,
        "news_items": len(research.news.items) if research is not None else 0,
        "per_asset_news": _asset_summaries(inputs),
    }


def _asset_summaries(inputs: AnalysisInputs) -> list[dict[str, object]]:
    research = inputs.research_result
    if research is None:
        return []
    summaries: list[dict[str, object]] = []
    for bundle in research.asset_results:
        relevant = _relevant_items(bundle.asset.symbol, bundle.news)
        all_items = list(bundle.news.items) if bundle.news is not None else []
        summaries.append(
            {
                "symbol": bundle.asset.symbol,
                "coverage": {
                    "validated_items": len(all_items),
                    "publisher_count": len({item.publisher.casefold() for item in all_items}),
                },
                "items": [
                    {
                        "publisher": item.publisher,
                        "title": item.title,
                        "excerpt": item.excerpt,
                        "published_at": item.published_at.isoformat(),
                        "quality": item.source_quality,
                    }
                    for item in relevant
                ],
            }
        )
    return summaries


def _relevant_items(symbol: str, news: NewsEvidence | None) -> list[NewsItem]:
    items = list(news.items) if news is not None else []
    base = symbol.split("/", maxsplit=1)[0].upper()
    quality = {"high": 2, "medium": 1, "low": 0}
    return sorted(
        items,
        key=lambda item: (
            base in {str(asset).upper() for asset in item.assets},
            quality.get(str(item.source_quality), 0),
            item.published_at,
        ),
        reverse=True,
    )[:2]


def compose(
    generated: NewsLiveOutput,
    *,
    selected_evidence: Mapping[str, object],
    limitations: Sequence[str],
    requirements: AnswerRequirements,
) -> AgentAnswer:
    return compose_narrative_answer(
        generated,
        agent="news_agent",
        scope="news",
        evidence_kind="recent_news",
        selected_evidence=selected_evidence,
        limitations=limitations,
        requirements=requirements,
        default_verdict=(
            "Recent verified coverage identifies a current development for this asset, while "
            "market impact remains unproven."
            if len(requirements.assets) == 1
            else (
                "Recent verified coverage highlights different developments across the "
                "selected assets."
            )
        ),
        default_interpretation=(
            "Recent coverage confirms ongoing project activity, but the supplied sources do not "
            "contain enough qualitative signal to support a directional interpretation for this "
            "asset."
        ),
        empty_evidence="No recent validated news record was available for this asset.",
        include_facts_in_section=False,
        reject_news_causality=True,
    )


ANALYZER = AgentAnalyzer(
    id="news_agent",
    role=ROLE,
    system_prompt=SYSTEM_PROMPT,
    output_schema=NewsLiveOutput,
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
    "NewsAssetAnalysis",
    "NewsLiveOutput",
    "ANALYZER",
    "REQUIRED_SCOPES",
    "ROLE",
    "SYSTEM_PROMPT",
    "prompt_budget",
]
