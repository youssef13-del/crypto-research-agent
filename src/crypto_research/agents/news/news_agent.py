"""Focused collector for the News Agent.

The former research collector mixed current reporting with topic and web
research.  The public product now has one News Agent, so this module accepts
only the news scope and never performs a secondary topic-research pass.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from crypto_research.agents.base import AgentManifest
from crypto_research.domain.evidence import normalize_news_items
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisRequest,
    AssetResearchBundle,
    CollectionContext,
    NewsEvidence,
    ResearchAgentResult,
    ResearchCapability,
)
from crypto_research.tools.types import NewsServices

from .news_collector import news_requested

NEWS_MANIFEST = AgentManifest(
    id="news_agent",
    label="News Agent",
    capabilities=frozenset({ResearchCapability.NEWS}),
)


class NewsAgent:
    """Collect current, relevant news for one to four selected assets."""

    def __init__(
        self,
        *,
        services: NewsServices,
    ) -> None:
        self._services = services

    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability | str] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> ResearchAgentResult:
        collected_at = (
            collection_context.collected_at if collection_context is not None else datetime.now(UTC)
        )
        if not news_requested(requested_capabilities):
            return ResearchAgentResult(
                news=NewsEvidence(items=[], query="", collected_at=collected_at),
                requested_capabilities=[],
            )
        assets = request.ordered_assets()
        if assets:
            bundles = self._collect_assets(assets, request.user_intent, collected_at)
            news = self._combine(
                [item.news for item in bundles if item.news is not None],
                collected_at,
            )
        else:
            news = self._fetch_global(request, collected_at)
            bundles = []
        return ResearchAgentResult(
            news=news,
            asset_results=bundles,
            requested_capabilities=[ResearchCapability.NEWS],
            capabilities=[ResearchCapability.NEWS] if news.items else [],
            summary=self._summary(news),
        )

    def _collect_assets(
        self,
        assets: list[AnalysisAsset],
        query_context: str,
        collected_at: datetime,
    ) -> list[AssetResearchBundle]:
        with ThreadPoolExecutor(max_workers=min(4, len(assets))) as pool:
            futures = [
                pool.submit(self._fetch, asset, query_context, collected_at) for asset in assets
            ]
            evidence = [future.result() for future in futures]
        return [
            AssetResearchBundle(asset=asset, news=item, limitations=list(item.warnings))
            for asset, item in zip(assets, evidence, strict=True)
        ]

    def _fetch(
        self,
        asset: AnalysisAsset,
        query_context: str,
        collected_at: datetime,
    ) -> NewsEvidence:
        try:
            value = self._services.collect(
                asset_name=asset.coin_id or asset.name or asset.requested_name,
                symbol=asset.symbol,
                query_context=query_context,
                collected_at=collected_at,
            )
            items, warnings = normalize_news_items(value.items, collected_at=collected_at)
            return value.model_copy(
                update={
                    "items": [
                        item.model_copy(
                            update={"assets": list(dict.fromkeys((*item.assets, asset.symbol)))}
                        )
                        for item in items
                    ],
                    "collected_at": min(value.collected_at, collected_at),
                    "warnings": list(dict.fromkeys((*value.warnings, *warnings))),
                }
            )
        except Exception as exc:
            return NewsEvidence(
                items=[],
                query=asset.requested_name,
                collected_at=collected_at,
                warnings=[f"News provider was unavailable ({type(exc).__name__})."],
            )

    def _fetch_global(self, request: AnalysisRequest, collected_at: datetime) -> NewsEvidence:
        try:
            name, symbol = self._services.resolve_query(
                request.coin_id, request.symbol, request.user_intent, request.research_topic
            )
            return self._fetch(
                AnalysisAsset(requested_name=name, symbol=symbol, coin_id=request.coin_id),
                request.user_intent,
                collected_at,
            )
        except Exception as exc:
            return NewsEvidence(
                items=[],
                query=request.research_topic or request.user_intent,
                collected_at=collected_at,
                warnings=[f"News query resolution failed ({type(exc).__name__})."],
            )

    @staticmethod
    def _combine(values: list[NewsEvidence], collected_at: datetime) -> NewsEvidence:
        seen: set[tuple[str, str]] = set()
        items = []
        warnings: list[str] = []
        for value in values:
            warnings.extend(value.warnings)
            for item in value.items:
                key = (item.url or "", item.title.casefold())
                if key not in seen:
                    seen.add(key)
                    items.append(item)
        items.sort(key=lambda item: item.published_at, reverse=True)
        return NewsEvidence(
            items=items,
            query="; ".join(value.query for value in values if value.query),
            collected_at=collected_at,
            source_state=(
                "cached"
                if values and all(value.source_state == "cached" for value in values)
                else "live"
            ),
            warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _summary(news: NewsEvidence) -> str | None:
        if not news.items:
            return None
        return "Recent coverage: " + "; ".join(
            f"{item.publisher}: {item.title}" for item in news.items[:4]
        )


__all__ = ["NewsAgent"]
