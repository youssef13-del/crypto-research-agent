"""Public contracts for saved research history and management outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from crypto_research.domain.research import ResearchReport


@dataclass(frozen=True, slots=True)
class ResearchRunSummary:
    id: str
    created_at: datetime
    completed_at: datetime | None
    state: str
    question: str
    assets: tuple[str, ...]
    capabilities: tuple[str, ...]
    exchange: str | None
    timeframe: str | None
    pinned: bool
    evidence_count: int


@dataclass(frozen=True, slots=True)
class StoredResearchRun:
    summary: ResearchRunSummary
    report: ResearchReport


@dataclass(frozen=True, slots=True)
class ResearchComparison:
    current: StoredResearchRun
    previous: StoredResearchRun | None


@dataclass(frozen=True, slots=True)
class BulkDeleteResult:
    """Tenant-safe outcome for one bounded bulk research deletion."""

    requested_count: int
    deleted_count: int
    protected_count: int


__all__ = [
    "BulkDeleteResult",
    "ResearchComparison",
    "ResearchRunSummary",
    "StoredResearchRun",
]
