"""Ordered per-asset on-chain activity collector."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from crypto_research.agents.base import AgentManifest, collect_asset_jobs
from crypto_research.domain.evidence import OnChainEvidence
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisRequest,
    CollectionContext,
    OnChainAgentResult,
    ResearchCapability,
)
from crypto_research.tools.types import OnChainServices

from .onchain_collector import collection_kwargs

ONCHAIN_MANIFEST = AgentManifest(
    id="onchain_agent",
    label="On-Chain Activity Agent",
    capabilities=frozenset({ResearchCapability.ONCHAIN}),
)


class OnChainAgent:
    """Collect Coin Metrics network-activity evidence without guessing identities."""

    def __init__(
        self,
        *,
        services: OnChainServices,
        base_url: str,
    ) -> None:
        self._services = services
        self._base_url = base_url

    def run(
        self,
        request: AnalysisRequest,
        *,
        collection_context: CollectionContext | None = None,
    ) -> OnChainAgentResult:
        assets = request.ordered_assets()
        jobs: list[tuple[int, str, Callable[[], object]]] = [
            (index, "onchain", partial(self._fetch, asset, collection_context))
            for index, asset in enumerate(assets)
        ]
        bundles = collect_asset_jobs(assets, jobs)
        available = [
            bundle.onchain
            for bundle in bundles
            if bundle.onchain is not None and bundle.onchain.metrics
        ]
        return OnChainAgentResult(
            asset_results=bundles,
            requested_capabilities=[ResearchCapability.ONCHAIN],
            capabilities=[ResearchCapability.ONCHAIN] if available else [],
            summary=_summary(available),
        )

    def _fetch(
        self,
        asset: AnalysisAsset,
        context: CollectionContext | None,
    ) -> OnChainEvidence:
        result = self._services.collect(
            asset=asset,
            base_url=self._base_url,
            **collection_kwargs(context),
        )
        if not isinstance(result, OnChainEvidence):
            raise TypeError("On-chain provider returned an unsupported result.")
        return result


def _summary(values: list[OnChainEvidence]) -> str | None:
    parts: list[str] = []
    for evidence in values:
        directions = [
            f"{metric.label} {metric.seven_day_change_pct:+.1f}%"
            for metric in evidence.metrics
            if metric.seven_day_change_pct is not None
        ]
        if directions:
            parts.append(f"{evidence.asset}: " + ", ".join(directions[:3]))
    return "; ".join(parts)[:1200] or None


__all__ = ["OnChainAgent"]
