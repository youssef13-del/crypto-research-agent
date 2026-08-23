"""Shared helpers for bounded specialist collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from crypto_research.domain.core import ResearchCapability, StrictModel
from crypto_research.domain.research import (
    AgentAnswer,
    AnalysisAsset,
    AnalysisInputs,
    AssetResearchBundle,
    CollectionContext,
)
from crypto_research.llm.client import LLMRole
from crypto_research.shared.text import clean_generated_text


@dataclass(frozen=True, slots=True)
class AgentManifest:
    id: str
    label: str
    capabilities: frozenset[ResearchCapability]


@dataclass(frozen=True, slots=True)
class AgentEvidencePolicy:
    allowed_kinds: frozenset[str]
    limitations: Callable[[AnalysisInputs, Sequence[ResearchCapability]], list[str]]
    capability_expansions: Mapping[ResearchCapability, frozenset[ResearchCapability]] = field(
        default_factory=dict
    )
    fallback_scopes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    fallback_verdict: str = "Validated evidence remains available for review."
    no_evidence_message: str = "Verified evidence was not available for this research scope."
    fallback_claim_kinds: tuple[str, ...] = ()
    fallback_intro: str = "Verified records remain available for review."
    fallback_multi_intro: str | None = None
    fallback_comparison_note: str | None = None
    compare_fallback_always: bool = False
    structured_section_override: str | None = None

    def expand(self, capabilities: Sequence[ResearchCapability]) -> tuple[ResearchCapability, ...]:
        selected = set(capabilities)
        for capability in tuple(selected):
            selected.update(self.capability_expansions.get(capability, ()))
        return tuple(item for item in ResearchCapability if item in selected)


@dataclass(frozen=True, slots=True)
class AgentAnalyzer:
    """Typed configuration consumed by the shared analysis runner."""

    id: str
    role: LLMRole
    system_prompt: str
    output_schema: type[BaseModel]
    prompt_budget: Callable[..., int]
    structured_instruction: Callable[[Sequence[str]], str]
    output_contract: Callable[[Sequence[str]], dict[str, str]]
    evidence_policy: AgentEvidencePolicy
    evidence_limits: Callable[[int], tuple[int, int, int]]
    summarize: Callable[[AnalysisInputs], dict[str, object]] | None = None
    compose: Callable[..., AgentAnswer] | None = None
    compact_briefs: Callable[[Mapping[str, object]], dict[str, object]] | None = None


def compact_model_text(value: object, *, max_chars: int) -> object:
    if not isinstance(value, str):
        return value
    return clean_generated_text(value, max_chars=max_chars, max_sentences=2)


class NarrativeAssetAnalysis(StrictModel):
    symbol: str = Field(min_length=1, max_length=40)
    analysis: str = Field(min_length=1)

    @field_validator("analysis", mode="before")
    @classmethod
    def compact_analysis(cls, value: object) -> object:
        return compact_model_text(value, max_chars=320)


class NarrativeLiveOutput(StrictModel):
    verdict: str = Field(min_length=1)
    assets: list[NarrativeAssetAnalysis] = Field(min_length=1, max_length=4)
    comparison: str
    limitations: list[str] = Field(max_length=3)
    confidence: Literal["low", "medium", "high"]

    @field_validator("verdict", mode="before")
    @classmethod
    def compact_verdict(cls, value: object) -> object:
        return compact_model_text(value, max_chars=180)

    @field_validator("comparison", mode="before")
    @classmethod
    def compact_comparison(cls, value: object) -> object:
        return compact_model_text(value, max_chars=220)

    @field_validator("limitations", mode="before")
    @classmethod
    def compact_limitations(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [compact_model_text(item, max_chars=160) for item in value[:3]]


def requested_capabilities(
    requested: list[ResearchCapability | str] | None,
) -> set[ResearchCapability]:
    return {
        value if isinstance(value, ResearchCapability) else ResearchCapability(value)
        for value in (requested or [])
    }


def requested_list(requested: set[ResearchCapability]) -> list[ResearchCapability]:
    return [capability for capability in ResearchCapability if capability in requested]


def collection_kwargs(context: CollectionContext | None) -> dict[str, object]:
    return {"collected_at": context.collected_at} if context is not None else {}


def collect_asset_jobs(
    assets: list[AnalysisAsset],
    jobs: list[tuple[int, str, Callable[[], object]]],
) -> list[AssetResearchBundle]:
    values: list[dict[str, object]] = [{} for _ in assets]
    if jobs:
        with ThreadPoolExecutor(max_workers=min(5, len(jobs))) as pool:
            futures = [(index, field, pool.submit(job)) for index, field, job in jobs]
            for index, field, future in futures:
                try:
                    values[index][field] = future.result()
                except Exception as exc:
                    limitations = values[index].setdefault("limitations", [])
                    if not isinstance(limitations, list):
                        limitations = []
                        values[index]["limitations"] = limitations
                    limitations.append(
                        f"{field.replace('_', ' ').title()} was unavailable ({type(exc).__name__})."
                    )
    bundles: list[AssetResearchBundle] = []
    for asset, fields in zip(assets, values, strict=True):
        raw_limitations = fields.pop("limitations", [])
        limitations = (
            [str(item) for item in raw_limitations] if isinstance(raw_limitations, list) else []
        )
        limitations.extend(
            warning for result in fields.values() for warning in getattr(result, "warnings", [])
        )
        bundles.append(AssetResearchBundle(asset=asset, limitations=limitations, **fields))
    return bundles
