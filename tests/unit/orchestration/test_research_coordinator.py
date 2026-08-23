from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

import pytest
from tests.support.fakes import fake_market_service

from crypto_research.agents.shared_analysis import SpecialistAgentId, SpecialistAnalysisRunner
from crypto_research.domain.evidence import EvidenceRecord
from crypto_research.domain.forecast import ForecastAgentResult
from crypto_research.domain.history import ResearchRunSummary, StoredResearchRun
from crypto_research.domain.research import (
    AgentAnswer,
    AgentExecutionStatus,
    AnalysisInputs,
    AnalysisRequest,
    CollectionContext,
    DefiEvidence,
    FundamentalEvidence,
    FundamentalsAgentResult,
    MarketAgentResult,
    MarketComparisonAsset,
    MarketComparisonResult,
    NewsEvidence,
    OnChainAgentResult,
    OpportunityScanResult,
    ResearchAgentResult,
    ResearchCapability,
    ResearchReport,
)
from crypto_research.orchestration import runtime as executor
from crypto_research.orchestration.planning import compile_guided_research_plan
from crypto_research.orchestration.runtime import ResearchRuntime
from crypto_research.tools.market import calculate_indicators


def test_coordinator_runs_selected_agents_in_deterministic_display_order() -> None:
    specialist = RecordingSpecialist()
    runtime = _runtime(specialist=specialist)
    plan = compile_guided_research_plan(
        ["BTC", "ETH"],
        [
            ResearchCapability.NEWS,
            ResearchCapability.FUNDAMENTALS,
            ResearchCapability.MARKET,
        ],
    )

    outcome = runtime.ask(plan.action)

    assert [node.value for node in outcome.route] == [
        "market_agent",
        "fundamentals_agent",
        "news_agent",
    ]
    assert [answer.agent for answer in outcome.agent_answers] == [
        "market_agent",
        "fundamentals_agent",
        "news_agent",
    ]
    assert plan.action.request is not None
    assert plan.action.request.comparison_symbols == ["BTC/USD", "ETH/USD"]


@pytest.mark.parametrize(
    ("route", "message"),
    [
        (["unknown_agent"], "unknown agent"),
        (["market_agent", "market_agent"], "duplicate"),
        (["news_agent"], "does not own"),
    ],
)
def test_invalid_planned_route_fails_before_collection(
    route: list[str],
    message: str,
) -> None:
    market = MarketAgentStub()
    runtime = _runtime(market=market)
    plan = compile_guided_research_plan(["BTC"], [ResearchCapability.MARKET])

    with pytest.raises(ValueError, match=message):
        runtime.ask(plan.action.model_copy(update={"agents_to_call": route}))

    assert market.calls == 0


def test_empty_legacy_route_is_derived_from_capability_ownership() -> None:
    runtime = _runtime()
    plan = compile_guided_research_plan(["BTC"], [ResearchCapability.MARKET])

    outcome = runtime.ask(plan.action.model_copy(update={"agents_to_call": []}))

    assert [node.value for node in outcome.route] == ["market_agent"]


def test_coordinator_isolates_one_agent_failure() -> None:
    runtime = _runtime(news=FailingNewsAgent(), specialist=RecordingSpecialist())
    plan = compile_guided_research_plan(
        ["BTC"],
        [ResearchCapability.MARKET, ResearchCapability.NEWS],
    )

    outcome = runtime.ask(plan.action)

    assert outcome.research_report is not None
    statuses = {item.agent: item.status for item in outcome.research_report.agent_statuses}
    assert statuses["market_agent"] == "complete"
    assert statuses["news_agent"] == "unavailable"
    assert [answer.agent for answer in outcome.agent_answers] == [
        "market_agent",
        "news_agent",
    ]
    assert outcome.agent_answers[1].analysis_state == "unavailable"


def test_market_risk_guided_scope_completes_without_specialist_scope_errors() -> None:
    runtime = ResearchRuntime(
        market_agent=MarketWithComparisonStub(),
        news_agent=NewsAgentStub(),
        fundamentals_agent=FundamentalsAgentStub(),
        onchain_agent=OnChainAgentStub(),
        forecast_agent=ForecastAgentStub(),
        specialist_analysis=SpecialistAnalysisRunner(),
    )
    plan = compile_guided_research_plan(
        ["BTC"],
        [ResearchCapability.MARKET],
    )

    outcome = runtime.ask(plan.action)

    assert outcome.research_report is not None
    assert outcome.research_report.status in {"complete", "partial"}
    assert outcome.errors == ()
    assert outcome.research_report.agent_statuses[0].agent == "market_agent"
    assert outcome.research_report.agent_statuses[0].status in {"complete", "partial"}


def test_guided_market_and_risk_topics_route_independently_to_one_agent() -> None:
    market = compile_guided_research_plan(["BTC"], [ResearchCapability.MARKET])
    risk = compile_guided_research_plan(["BTC"], [ResearchCapability.RISK])
    combined = compile_guided_research_plan(
        ["BTC"],
        [ResearchCapability.MARKET, ResearchCapability.RISK],
    )

    assert market.requested_capabilities == (ResearchCapability.MARKET,)
    assert risk.requested_capabilities == (ResearchCapability.RISK,)
    assert (
        market.expected_agents
        == risk.expected_agents
        == combined.expected_agents
        == ("market_agent",)
    )
    assert combined.execution[0].capabilities == (
        ResearchCapability.MARKET,
        ResearchCapability.RISK,
    )


def test_guided_onchain_topic_has_complete_execution_metadata() -> None:
    plan = compile_guided_research_plan(["BTC"], [ResearchCapability.ONCHAIN])

    assert plan.expected_agents == ("onchain_agent",)
    assert plan.collectors == ("onchain",)
    assert plan.execution[0].agent_id == "onchain_agent"
    assert plan.execution[0].capabilities == (ResearchCapability.ONCHAIN,)
    assert plan.execution[0].collectors == ("onchain",)


def test_explicit_topics_preserve_capabilities_and_intent() -> None:
    plan = compile_guided_research_plan(
        ["BTC"],
        [ResearchCapability.MARKET, ResearchCapability.NEWS],
    )

    assert plan.display_capabilities == (ResearchCapability.MARKET, ResearchCapability.NEWS)
    assert plan.action.request is not None
    assert plan.action.request.user_intent == "Research BTC: Market behavior + Recent news"


def test_guided_research_enforces_four_coin_boundary() -> None:
    with pytest.raises(ValueError, match="at most four"):
        compile_guided_research_plan(
            ["BTC", "ETH", "SOL", "ADA", "DOGE"],
            [ResearchCapability.MARKET],
        )


def test_coordinator_persists_completed_research_without_changing_the_result() -> None:
    history = RecordingHistory()
    runtime = _runtime(history=history)
    plan = compile_guided_research_plan(["BTC"], [ResearchCapability.MARKET])

    outcome = runtime.ask(plan.action)

    assert outcome.research_report is not None
    assert history.created == ["Research BTC: Market behavior"]
    assert history.completed == [outcome.research_report]


def test_retry_reuses_validated_evidence_and_carries_successful_cards_forward() -> None:
    stored = _stored_retry_run(analysis_state="evidence_only")
    history = RetryHistory(stored)
    market = MarketAgentStub()
    news = NewsAgentStub()
    specialist = RecordingSpecialist()
    runtime = _runtime(
        history=history,
        market=market,
        news=news,
        specialist=specialist,
    )
    original = stored.report.model_dump(mode="json")

    outcome = runtime.retry_failed_agents(stored.summary.id)

    assert market.calls == 0
    assert news.calls == 0
    assert specialist.calls == [("news_agent", (ResearchCapability.NEWS,))]
    assert outcome.run_id == "retry-run-1"
    assert [node.value for node in outcome.route] == ["news_agent"]
    assert [node.value for node in outcome.agents] == ["market_agent", "news_agent"]
    assert [answer.agent for answer in outcome.agent_answers] == [
        "market_agent",
        "news_agent",
    ]
    assert outcome.research_report is not None
    assert outcome.research_report.retry is not None
    assert outcome.research_report.retry.original_run_id == stored.summary.id
    assert outcome.research_report.retry.retried_agents == ["news_agent"]
    assert outcome.research_report.agent_answers[0] == stored.report.agent_answers[0]
    assert stored.report.model_dump(mode="json") == original
    assert history.completed_ids == ["retry-run-1"]


def test_retry_recollects_only_agents_with_unavailable_collectors() -> None:
    stored = _stored_retry_run(analysis_state="unavailable")
    history = RetryHistory(stored)
    market = MarketAgentStub()
    news = NewsAgentStub()
    specialist = RecordingSpecialist()
    runtime = _runtime(
        history=history,
        market=market,
        news=news,
        specialist=specialist,
    )

    outcome = runtime.retry_failed_agents(stored.summary.id)

    assert market.calls == 0
    assert news.calls == 1
    assert specialist.calls == [("news_agent", (ResearchCapability.NEWS,))]
    assert outcome.research_report is not None
    assert outcome.research_report.research_result is not None
    assert outcome.research_report.agent_answers[-1].analysis_state == "live"


def test_retry_preserves_an_unavailable_card_when_collection_fails_again() -> None:
    stored = _stored_retry_run(analysis_state="unavailable")
    history = RetryHistory(stored)
    runtime = _runtime(history=history, news=FailingNewsAgent())

    outcome = runtime.retry_failed_agents(stored.summary.id)

    assert outcome.research_report is not None
    assert outcome.research_report.status == "partial"
    assert outcome.research_report.agent_answers[-1].analysis_state == "unavailable"
    assert outcome.research_report.retry is not None
    assert history.completed_ids == ["retry-run-1"]


def test_retry_rejects_reports_without_failed_analysis() -> None:
    stored = _stored_retry_run(analysis_state="live")
    history = RetryHistory(stored)
    runtime = _runtime(history=history)

    with pytest.raises(ValueError, match="no failed agent analysis"):
        runtime.retry_failed_agents(stored.summary.id)

    assert history.created == []


def test_retry_recomputes_deterministic_risk_when_risk_is_in_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _stored_retry_run(analysis_state="evidence_only")
    report = base.report.model_copy(
        update={
            "research_result": None,
            "agent_answers": [
                AgentAnswer(
                    agent="market_agent",
                    answer="Market analysis needs another attempt",
                    confidence=0.3,
                    status="partial",
                    analysis_state="evidence_only",
                    coverage_state="partial",
                )
            ],
            "agent_statuses": [
                AgentExecutionStatus(
                    agent="market_agent",
                    status="partial",
                    analysis_state="evidence_only",
                    coverage_state="partial",
                    capabilities=[ResearchCapability.MARKET, ResearchCapability.RISK],
                )
            ],
            "errors": [{"agent": "market_agent", "message": "RuntimeError"}],
        }
    )
    stored = StoredResearchRun(
        summary=base.summary,
        report=report,
    )
    calls: list[object] = []

    def evaluate(**kwargs: object) -> None:
        calls.append(kwargs)
        return

    monkeypatch.setattr(executor, "evaluate_research_risk", evaluate)
    runtime = _runtime(history=RetryHistory(stored))

    outcome = runtime.retry_failed_agents(stored.summary.id)

    assert len(calls) == 1
    assert outcome.research_report is not None
    assert outcome.research_report.retry is not None


def _runtime(
    *,
    market: MarketAgentStub | None = None,
    news: NewsAgentStub | FailingNewsAgent | None = None,
    fundamentals: FundamentalsAgentStub | None = None,
    specialist: RecordingSpecialist | None = None,
    history: RecordingHistory | None = None,
) -> ResearchRuntime:
    return ResearchRuntime(
        market_agent=market or MarketAgentStub(),
        news_agent=news or NewsAgentStub(),
        fundamentals_agent=fundamentals or FundamentalsAgentStub(),
        onchain_agent=OnChainAgentStub(),
        forecast_agent=ForecastAgentStub(),
        specialist_analysis=specialist or RecordingSpecialist(),
        history=history,
    )


class RecordingHistory:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.completed: list[ResearchReport] = []

    def create_run(
        self,
        *,
        request: AnalysisRequest,
        capabilities: Sequence[ResearchCapability],
        question: str,
    ) -> str:
        del request, capabilities
        self.created.append(question)
        return "run-1"

    def complete_run(
        self,
        run_id: str,
        report: ResearchReport,
        evidence: Sequence[EvidenceRecord] = (),
    ) -> None:
        del run_id, evidence
        self.completed.append(report)

    def fail_run(self, run_id: str, failure: str) -> None:
        raise AssertionError(f"Unexpected persistence failure for {run_id}: {failure}")

    def get_run(self, run_id: str) -> StoredResearchRun | None:
        del run_id
        return None


class RetryHistory(RecordingHistory):
    def __init__(self, stored: StoredResearchRun) -> None:
        super().__init__()
        self.stored = stored
        self.completed_ids: list[str] = []

    def create_run(
        self,
        *,
        request: AnalysisRequest,
        capabilities: Sequence[ResearchCapability],
        question: str,
    ) -> str:
        del request, capabilities
        self.created.append(question)
        return "retry-run-1"

    def complete_run(
        self,
        run_id: str,
        report: ResearchReport,
        evidence: Sequence[EvidenceRecord] = (),
    ) -> None:
        del evidence
        self.completed_ids.append(run_id)
        self.completed.append(report)

    def get_run(self, run_id: str) -> StoredResearchRun | None:
        return self.stored if run_id == self.stored.summary.id else None


def _stored_retry_run(
    *,
    analysis_state: Literal["live", "evidence_only", "unavailable"],
) -> StoredResearchRun:
    plan = compile_guided_research_plan(
        ["BTC"],
        [ResearchCapability.MARKET, ResearchCapability.NEWS],
    )
    assert plan.action.request is not None
    news_status = (
        "unavailable"
        if analysis_state == "unavailable"
        else ("partial" if analysis_state == "evidence_only" else "complete")
    )
    news_confidence = 0.0 if analysis_state == "unavailable" else 0.6
    report_status = "complete" if analysis_state == "live" else "partial"
    report = ResearchReport(
        request=plan.action.request,
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[],
                query="BTC",
                collected_at=datetime(2026, 8, 16, 12, tzinfo=UTC),
            )
        ),
        agent_answers=[
            AgentAnswer(agent="market_agent", answer="Market complete", confidence=0.8),
            AgentAnswer(
                agent="news_agent",
                answer="News needs another attempt",
                confidence=news_confidence,
                status=news_status,
                analysis_state=analysis_state,
                coverage_state="partial" if analysis_state != "live" else "complete",
            ),
        ],
        agent_statuses=[
            AgentExecutionStatus(
                agent="market_agent",
                status="complete",
                capabilities=[ResearchCapability.MARKET],
            ),
            AgentExecutionStatus(
                agent="news_agent",
                status=news_status,
                source_state="unavailable" if analysis_state == "unavailable" else "live",
                analysis_state=analysis_state,
                coverage_state="partial" if analysis_state != "live" else "complete",
                capabilities=[ResearchCapability.NEWS],
            ),
        ],
        status=report_status,
        errors=(
            [{"agent": "news_agent", "message": "RuntimeError"}] if analysis_state != "live" else []
        ),
    )
    summary = ResearchRunSummary(
        id="original-run-1",
        created_at=datetime(2026, 8, 16, 12, tzinfo=UTC),
        completed_at=datetime(2026, 8, 16, 12, 1, tzinfo=UTC),
        state="partial" if analysis_state != "live" else "complete",
        question="Research BTC",
        assets=("BTC/USD",),
        capabilities=("market", "news"),
        exchange="kraken",
        timeframe="1h",
        pinned=False,
        evidence_count=1,
    )
    return StoredResearchRun(summary=summary, report=report)


class MarketAgentStub:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> OpportunityScanResult | MarketAgentResult | MarketComparisonResult:
        del request, requested_capabilities, collection_context
        self.calls += 1
        from crypto_research.domain.research import MarketComparisonResult

        return MarketComparisonResult()


class MarketWithComparisonStub(MarketAgentStub):
    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> OpportunityScanResult | MarketAgentResult | MarketComparisonResult:
        del requested_capabilities, collection_context
        self.calls += 1
        market = fake_market_service()
        if request.ordered_assets():
            market = market.model_copy(update={"symbol": request.ordered_assets()[0].symbol})
        asset = MarketComparisonAsset(
            market=market,
            technical=calculate_indicators(market.candles),
        )
        return MarketComparisonResult(assets=[asset])


class NewsAgentStub:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        *_args: object,
        requested_capabilities: list[ResearchCapability | str] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> ResearchAgentResult:
        del requested_capabilities
        self.calls += 1
        collected_at = collection_context.collected_at if collection_context else datetime.now(UTC)
        return ResearchAgentResult(
            news=NewsEvidence(items=[], query="BTC", collected_at=collected_at)
        )


class FailingNewsAgent(NewsAgentStub):
    def run(self, *_args: object, **_kwargs: object) -> ResearchAgentResult:
        raise RuntimeError("news failed")


class FundamentalsAgentStub:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        *_args: object,
        requested_capabilities: list[ResearchCapability | str] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> FundamentalsAgentResult:
        del requested_capabilities, collection_context
        self.calls += 1
        return FundamentalsAgentResult(
            fundamentals=FundamentalEvidence(status="unavailable"),
            defi=DefiEvidence(status="unavailable"),
        )


class OnChainAgentStub:
    def run(
        self,
        request: AnalysisRequest,
        *,
        collection_context: CollectionContext | None = None,
    ) -> OnChainAgentResult:
        del request, collection_context
        raise AssertionError("On-chain collection was not selected.")


class ForecastAgentStub:
    def run(self, request: AnalysisRequest) -> ForecastAgentResult:
        del request
        raise AssertionError("Forecasting was not selected.")

    def analyze(self, question: str, result: ForecastAgentResult) -> AgentAnswer:
        del question, result
        raise AssertionError("Forecasting was not selected.")


class RecordingSpecialist:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ResearchCapability, ...]]] = []

    def run(
        self,
        question: str,
        inputs: AnalysisInputs,
        *,
        agent: SpecialistAgentId,
        capabilities: Sequence[ResearchCapability],
    ) -> AgentAnswer:
        del question, inputs
        self.calls.append((agent, tuple(capabilities)))
        return AgentAnswer(agent=agent, answer=f"{agent} complete", confidence=0.8)
