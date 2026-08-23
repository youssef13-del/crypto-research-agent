from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from crypto_research.domain.research import (
    AgentAnswer,
    ResearchCapability,
    ResearchReport,
)


class WorkflowNode(StrEnum):
    MARKET_AGENT = "market_agent"
    NEWS_AGENT = "news_agent"
    FUNDAMENTALS_AGENT = "fundamentals_agent"
    ONCHAIN_AGENT = "onchain_agent"
    FORECAST_AGENT = "forecast_agent"

    @property
    def progress_label(self) -> str:
        return _PROGRESS_LABELS[self]


_PROGRESS_LABELS = {
    WorkflowNode.MARKET_AGENT: "Fetching market, risk, and derivatives evidence...",
    WorkflowNode.NEWS_AGENT: "Collecting relevant news...",
    WorkflowNode.FUNDAMENTALS_AGENT: "Collecting fundamentals and DeFi evidence...",
    WorkflowNode.ONCHAIN_AGENT: "Collecting network activity from Coin Metrics...",
    WorkflowNode.FORECAST_AGENT: "Training and validating forecast models...",
}


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    node: WorkflowNode
    route: tuple[WorkflowNode, ...] = ()

    @property
    def label(self) -> str:
        return self.node.progress_label


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    research_report: ResearchReport
    agents: tuple[WorkflowNode, ...]
    route: tuple[WorkflowNode, ...]
    requested_capabilities: tuple[ResearchCapability, ...]
    warnings: tuple[str, ...]
    errors: tuple[dict[str, object], ...]
    agent_answers: tuple[AgentAnswer, ...] = ()
    run_id: str | None = None

    @property
    def failed(self) -> bool:
        return bool(self.errors)

    def public_payload(self) -> dict[str, object]:
        """Return the safe CLI/API payload without internal workflow metadata."""

        return dict(self.research_report.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class ResultEvent:
    result: ResearchOutcome


type WorkflowEvent = ProgressEvent | ResultEvent
