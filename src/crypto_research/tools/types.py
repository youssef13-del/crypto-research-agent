"""Typed service dependencies consumed by research specialists."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from crypto_research.domain.evidence import (
    DefiEvidence,
    DerivativesEvidence,
    FundamentalEvidence,
    NewsEvidence,
    OnChainEvidence,
)
from crypto_research.domain.market import ComparisonMetrics, MarketEvidence
from crypto_research.domain.research import OpportunityScanResult, TechnicalSnapshot

MarketSnapshotResult = tuple[
    list[tuple[MarketEvidence, TechnicalSnapshot]],
    list[str],
]


@dataclass(frozen=True, slots=True)
class MarketServices:
    """Market data and deterministic calculation functions used by the market agent."""

    market_evidence: Callable[..., MarketEvidence]
    indicators: Callable[..., TechnicalSnapshot]
    snapshots: Callable[..., MarketSnapshotResult]
    comparison: Callable[..., MarketSnapshotResult]
    comparison_metrics: Callable[..., ComparisonMetrics]
    discovery: Callable[..., OpportunityScanResult]
    derivatives: Callable[..., DerivativesEvidence]


@dataclass(frozen=True, slots=True)
class FundamentalsServices:
    """Fundamental and DeFi evidence functions used by the fundamentals agent."""

    fundamentals: Callable[..., FundamentalEvidence]
    defi: Callable[..., DefiEvidence]


@dataclass(frozen=True, slots=True)
class NewsServices:
    """News collection and query-resolution functions used by the news agent."""

    collect: Callable[..., NewsEvidence]
    resolve_query: Callable[..., tuple[str, str]]


@dataclass(frozen=True, slots=True)
class OnChainServices:
    """Network evidence function used by the on-chain agent."""

    collect: Callable[..., OnChainEvidence]
