"""Ordered per-asset fundamentals and DeFi evidence collector."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from pydantic import SecretStr

from crypto_research.agents.base import AgentManifest, collect_asset_jobs, collection_kwargs
from crypto_research.agents.base import (
    requested_list as _requested_list,
)
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisRequest,
    AssetResearchBundle,
    CollectionContext,
    DefiEvidence,
    FundamentalEvidence,
    FundamentalsAgentResult,
    ResearchCapability,
)
from crypto_research.shared.formatting import format_compact_number, format_money
from crypto_research.tools.fundamentals import defillama_slug_for
from crypto_research.tools.types import FundamentalsServices

from .fundamentals_collector import requested_capabilities as parse_requested_capabilities

_FUNDAMENTAL_CAPABILITIES = {
    ResearchCapability.FUNDAMENTALS,
    ResearchCapability.RISK,
}
FUNDAMENTALS_MANIFEST = AgentManifest(
    id="fundamentals_agent",
    label="Fundamentals Agent",
    capabilities=frozenset({ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI}),
)


class FundamentalsAgent:
    """Collect ordered per-asset fundamentals and DeFi evidence."""

    def __init__(
        self,
        *,
        services: FundamentalsServices,
        coingecko_api_key: SecretStr | None = None,
        defillama_base_url: str = "https://api.llama.fi",
    ) -> None:
        self._services = services
        self._coingecko_api_key = coingecko_api_key
        self._defillama_base_url = defillama_base_url

    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability | str] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> FundamentalsAgentResult:
        requested = parse_requested_capabilities(requested_capabilities)
        need_fundamentals = not requested or bool(requested & _FUNDAMENTAL_CAPABILITIES)
        need_defi = ResearchCapability.DEFI in requested
        assets = request.ordered_assets()
        asset_results = self._collect_assets(
            assets,
            need_fundamentals=need_fundamentals,
            need_defi=need_defi,
            collection_context=collection_context,
        )
        primary = asset_results[0] if asset_results else None
        fundamentals = (
            primary.fundamentals
            if primary is not None and primary.fundamentals is not None
            else FundamentalEvidence(status="unavailable")
        )
        defi = (
            primary.defi
            if primary is not None and primary.defi is not None
            else DefiEvidence(status="unavailable")
        )

        selected: list[ResearchCapability] = []
        if need_fundamentals and any(
            bundle.fundamentals is not None and bundle.fundamentals.status == "available"
            for bundle in asset_results
        ):
            selected.append(ResearchCapability.FUNDAMENTALS)
        if need_defi and any(
            bundle.defi is not None and bundle.defi.status == "available"
            for bundle in asset_results
        ):
            selected.append(ResearchCapability.DEFI)
        return FundamentalsAgentResult(
            fundamentals=fundamentals,
            defi=defi,
            asset_results=asset_results,
            requested_capabilities=_requested_list(requested),
            capabilities=selected,
            summary=_build_fundamentals_summary(fundamentals, defi),
        )

    def _collect_assets(
        self,
        assets: list[AnalysisAsset],
        *,
        need_fundamentals: bool,
        need_defi: bool,
        collection_context: CollectionContext | None,
    ) -> list[AssetResearchBundle]:
        jobs: list[tuple[int, str, Callable[[], object]]] = []
        for index, asset in enumerate(assets):
            if need_fundamentals:
                jobs.append(
                    (
                        index,
                        "fundamentals",
                        partial(self._fetch_fundamentals, asset, collection_context),
                    )
                )
            if need_defi and defillama_slug_for(asset.coin_id) is not None:
                jobs.append((index, "defi", partial(self._fetch_defi, asset, collection_context)))
        return collect_asset_jobs(assets, jobs)

    def _fetch_fundamentals(
        self,
        asset: AnalysisAsset,
        collection_context: CollectionContext | None,
    ) -> FundamentalEvidence:
        try:
            result = self._services.fundamentals(
                symbol=asset.symbol,
                coin_id=asset.coin_id,
                api_key=(
                    self._coingecko_api_key.get_secret_value()
                    if self._coingecko_api_key is not None
                    else None
                ),
                **collection_kwargs(collection_context),
            )
            return _sanitize_fundamentals(result, collection_context)
        except Exception as exc:
            return FundamentalEvidence(
                status="unavailable",
                **collection_kwargs(collection_context),
                warnings=[
                    "Fundamentals provider failed "
                    f"({type(exc).__name__}); fundamentals are unavailable."
                ],
            )

    def _fetch_defi(
        self,
        asset: AnalysisAsset,
        collection_context: CollectionContext | None,
    ) -> DefiEvidence:
        slug = defillama_slug_for(asset.coin_id)
        if slug is None:
            return DefiEvidence(
                slug=asset.coin_id,
                status="not_applicable",
                **collection_kwargs(collection_context),
                warnings=[
                    "DeFi metrics are not applicable to this asset; DefiLlama tracks lending and "
                    "yield protocols and no registered protocol was found for this coin."
                ],
            )
        try:
            result = self._services.defi(
                protocol_slug=slug,
                base_url=self._defillama_base_url,
                **collection_kwargs(collection_context),
            )
            return _sanitize_defi(result, collection_context)
        except Exception as exc:
            return DefiEvidence(
                slug=slug,
                status="unavailable",
                **collection_kwargs(collection_context),
                warnings=[
                    f"DeFi provider failed ({type(exc).__name__}); DeFi data is unavailable."
                ],
            )


def _sanitize_fundamentals(
    evidence: FundamentalEvidence,
    context: CollectionContext | None,
) -> FundamentalEvidence:
    if context is None or evidence.collected_at <= context.collected_at:
        return evidence
    return evidence.model_copy(
        update={
            "status": "unavailable",
            "collected_at": context.collected_at,
            "warnings": [
                *evidence.warnings,
                "Future-dated fundamentals data were excluded before analysis.",
            ],
        }
    )


def _sanitize_defi(evidence: DefiEvidence, context: CollectionContext | None) -> DefiEvidence:
    if context is None or evidence.collected_at <= context.collected_at:
        return evidence
    return evidence.model_copy(
        update={
            "status": "unavailable",
            "collected_at": context.collected_at,
            "warnings": [
                *evidence.warnings,
                "Future-dated DeFi data were excluded before analysis.",
            ],
        }
    )


def _build_fundamentals_summary(
    fundamentals: FundamentalEvidence,
    defi: DefiEvidence,
) -> str | None:
    """Build a concrete fundamentals summary."""
    parts: list[str] = []
    name = fundamentals.name or fundamentals.symbol or "The asset"
    if fundamentals.market_cap is not None:
        parts.append(f"market cap {format_money(fundamentals.market_cap)}")
    if fundamentals.circulating_supply is not None:
        parts.append(f"circulating supply {format_compact_number(fundamentals.circulating_supply)}")
    if fundamentals.max_supply is not None:
        parts.append(f"max supply {format_compact_number(fundamentals.max_supply)}")
    if fundamentals.rank is not None:
        parts.append(f"rank #{fundamentals.rank}")
    if defi.tvl_usd is not None:
        parts.append(f"TVL {format_money(defi.tvl_usd)}")
    if not parts:
        return None
    return f"{name}: " + ", ".join(parts) + "."
