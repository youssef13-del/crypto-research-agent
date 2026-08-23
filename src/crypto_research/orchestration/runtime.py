"""Deterministic coordinator for explicit guided research."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Protocol, cast

from crypto_research.agents.registry import agents_for, manifest_for
from crypto_research.agents.shared_analysis import SpecialistAgentId
from crypto_research.domain.analytics import evaluate_research_risk
from crypto_research.domain.evidence import AgentId, EvidenceRecord
from crypto_research.domain.forecast import ForecastAgentResult
from crypto_research.domain.history import StoredResearchRun
from crypto_research.domain.research import (
    AgentAnswer,
    AgentExecutionStatus,
    AnalysisInputs,
    AnalysisRequest,
    CollectionContext,
    FundamentalsAgentResult,
    MarketAgentResult,
    MarketComparisonResult,
    OnChainAgentResult,
    OpportunityScanResult,
    ResearchAction,
    ResearchAgentResult,
    ResearchCapability,
    ResearchReport,
    ResearchRetryMetadata,
)
from crypto_research.orchestration.events import (
    ProgressEvent,
    ResearchOutcome,
    ResultEvent,
    WorkflowEvent,
    WorkflowNode,
)

type MarketRunnerOutput = OpportunityScanResult | MarketAgentResult | MarketComparisonResult

logger = logging.getLogger(__name__)


class MarketRunner(Protocol):
    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> MarketRunnerOutput: ...


class NewsRunner(Protocol):
    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability | str] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> ResearchAgentResult: ...


class FundamentalsRunner(Protocol):
    def run(
        self,
        request: AnalysisRequest,
        *,
        requested_capabilities: list[ResearchCapability | str] | None = None,
        collection_context: CollectionContext | None = None,
    ) -> FundamentalsAgentResult: ...


class OnChainRunner(Protocol):
    def run(
        self,
        request: AnalysisRequest,
        *,
        collection_context: CollectionContext | None = None,
    ) -> OnChainAgentResult: ...


class ForecastRunner(Protocol):
    def run(self, request: AnalysisRequest) -> ForecastAgentResult: ...

    def analyze(self, question: str, result: ForecastAgentResult) -> AgentAnswer: ...


class SpecialistAnalysisProtocol(Protocol):
    def run(
        self,
        question: str,
        inputs: AnalysisInputs,
        *,
        agent: SpecialistAgentId,
        capabilities: Sequence[ResearchCapability],
    ) -> AgentAnswer: ...


class ResearchHistoryProtocol(Protocol):
    def create_run(
        self,
        *,
        request: AnalysisRequest,
        capabilities: Sequence[ResearchCapability],
        question: str,
    ) -> str: ...

    def complete_run(
        self,
        run_id: str,
        report: ResearchReport,
        evidence: Sequence[EvidenceRecord] = (),
    ) -> None: ...

    def fail_run(self, run_id: str, failure: str) -> None: ...

    def get_run(self, run_id: str) -> StoredResearchRun | None: ...


class ResearchRuntime:
    """Run explicit guided research actions."""

    def __init__(
        self,
        *,
        market_agent: MarketRunner,
        news_agent: NewsRunner,
        fundamentals_agent: FundamentalsRunner,
        onchain_agent: OnChainRunner,
        forecast_agent: ForecastRunner,
        specialist_analysis: SpecialistAnalysisProtocol,
        history: ResearchHistoryProtocol | None = None,
    ) -> None:
        self._market_agent = market_agent
        self._news_agent = news_agent
        self._fundamentals_agent = fundamentals_agent
        self._onchain_agent = onchain_agent
        self._forecast_agent = forecast_agent
        self._specialist_analysis = specialist_analysis
        self._history = history

    def ask(self, action: ResearchAction) -> ResearchOutcome:
        return _last_result(self.stream(action))

    def stream(self, action: ResearchAction) -> Iterator[WorkflowEvent]:
        yield from self._stream_research(action)

    def retry_failed_agents(
        self,
        run_id: str,
    ) -> ResearchOutcome:
        return _last_result(self.stream_retry_failed_agents(run_id))

    def stream_retry_failed_agents(
        self,
        run_id: str,
    ) -> Iterator[WorkflowEvent]:
        if self._history is None:
            raise RuntimeError("Failed-agent retry requires durable research history.")
        stored = self._history.get_run(run_id)
        if stored is None:
            raise KeyError("The original research run is unavailable in this workspace.")
        yield from self._stream_retry(stored)

    def _stream_research(
        self,
        action: ResearchAction,
    ) -> Iterator[WorkflowEvent]:
        request = action.request
        capabilities = _ordered_capabilities(action.requested_capabilities)
        history_run_id = _begin_history_run(
            self._history,
            request=request,
            capabilities=capabilities,
            question=request.user_intent,
        )
        route = _validated_route(action, capabilities)
        context = CollectionContext()
        inputs = AnalysisInputs(
            assets=request.ordered_assets(),
            requested_capabilities=list(capabilities),
            collection_context=context,
        )
        errors: list[dict[str, object]] = []

        with ThreadPoolExecutor(max_workers=max(1, len(route))) as pool:
            future_by_node = {
                pool.submit(self._run_collector, node, request, capabilities, context): node
                for node in route
            }
            for future in as_completed(future_by_node):
                node = future_by_node[future]
                yield ProgressEvent(node, route=route)
                try:
                    _apply_collector_output(inputs, node, future.result())
                except Exception as exc:
                    errors.append({"agent": node.value, "message": type(exc).__name__})
        if ResearchCapability.RISK in capabilities:
            inputs.risk_result = evaluate_research_risk(
                market_result=inputs.market_result,
                market_comparison_result=inputs.market_comparison_result,
                research_result=inputs.research_result,
                fundamentals_result=inputs.fundamentals_result,
                assets=inputs.assets,
            )

        answers: list[AgentAnswer] = []
        statuses: list[AgentExecutionStatus] = []
        for node in route:
            node_caps = _node_capabilities(node, capabilities)
            agent_error = next((item for item in errors if item.get("agent") == node.value), None)
            if agent_error is not None:
                answer = AgentAnswer(
                    agent=_specialist_id(node),
                    answer="The selected data provider failed before analysis could begin.",
                    uncertainty=["No live specialist analysis was produced for this card."],
                    limitations=["The provider path failed before validated evidence was ready."],
                    confidence=0.0,
                    status="unavailable",
                    analysis_state="unavailable",
                    coverage_state="partial",
                )
                answers.append(answer)
                statuses.append(
                    AgentExecutionStatus(
                        agent=_status_agent_id(node),
                        status="unavailable",
                        source_state="unavailable",
                        analysis_state="unavailable",
                        coverage_state="partial",
                        capabilities=list(node_caps),
                        limitation="The selected provider path failed before analysis.",
                    )
                )
                continue
            try:
                if node is WorkflowNode.FORECAST_AGENT:
                    if inputs.forecast_result is None:
                        raise RuntimeError("Forecast Agent is not configured.")
                    answer = self._forecast_agent.analyze(
                        request.user_intent, inputs.forecast_result
                    )
                else:
                    answer = self._specialist_analysis.run(
                        request.user_intent,
                        inputs,
                        agent=cast(SpecialistAgentId, _specialist_id(node)),
                        capabilities=node_caps,
                    )
            except Exception as exc:
                errors.append({"agent": node.value, "message": type(exc).__name__})
                answer = AgentAnswer(
                    agent=_specialist_id(node),
                    answer=(
                        "Live specialist analysis was interrupted before a validated response "
                        "could be produced for this card."
                    ),
                    analysis=(
                        "Review the resources below; collected evidence remains isolated to "
                        "this agent."
                    ),
                    uncertainty=["The live LLM response could not be normalized for this agent."],
                    limitations=[
                        "Specialist analysis failed safely without cancelling the research run."
                    ],
                    confidence=0.3,
                    status="partial",
                    analysis_state="evidence_only",
                    coverage_state="partial",
                )
            answers.append(answer)
            statuses.append(
                AgentExecutionStatus(
                    agent=_status_agent_id(node),
                    status=answer.status,
                    source_state=_source_state(inputs, node, answer.status),
                    analysis_state=answer.analysis_state,
                    coverage_state=answer.coverage_state,
                    capabilities=list(node_caps),
                    limitation=next(iter(answer.limitations), None),
                )
            )

        warnings = _collect_warnings(inputs)
        report_status = (
            "partial"
            if errors or warnings or any(answer.status != "complete" for answer in answers)
            else "complete"
        )
        report = ResearchReport(
            request=request,
            opportunity_result=inputs.opportunity_result,
            market_result=inputs.market_result,
            market_comparison_result=inputs.market_comparison_result,
            research_result=inputs.research_result,
            fundamentals_result=inputs.fundamentals_result,
            onchain_result=inputs.onchain_result,
            risk_result=inputs.risk_result,
            forecast_result=inputs.forecast_result,
            collection_context=context,
            agent_statuses=statuses,
            agent_answers=answers,
            status=report_status,
            warnings=warnings,
            errors=errors,
        )
        persisted = _complete_history_run(self._history, history_run_id, report, inputs)
        yield ResultEvent(
            ResearchOutcome(
                research_report=report,
                agents=route,
                route=route,
                requested_capabilities=capabilities,
                warnings=tuple(warnings),
                errors=tuple(errors),
                agent_answers=tuple(answers),
                run_id=history_run_id if persisted else None,
            )
        )

    def _stream_retry(
        self,
        stored: StoredResearchRun,
    ) -> Iterator[WorkflowEvent]:
        previous = stored.report
        retry_statuses = [
            status
            for status in previous.agent_statuses
            if status.analysis_state in {"unavailable", "evidence_only"}
        ]
        if not retry_statuses:
            raise ValueError("The selected report has no failed agent analysis to retry.")
        retry_nodes = tuple(WorkflowNode(status.agent) for status in retry_statuses)
        capabilities = _ordered_capabilities(
            tuple(
                capability
                for status in previous.agent_statuses
                for capability in status.capabilities
            )
        )
        full_route = tuple(WorkflowNode(status.agent) for status in previous.agent_statuses)
        new_run_id = _begin_history_run(
            self._history,
            request=previous.request,
            capabilities=capabilities,
            question=previous.request.user_intent,
        )
        if new_run_id is None:
            raise RuntimeError("The retry report could not be created in research history.")
        inputs = _analysis_inputs_from_report(previous, capabilities)
        retry_context = CollectionContext()
        inputs.collection_context = retry_context
        retried_agent_ids = {_specialist_id(node) for node in retry_nodes}
        errors = [
            dict(item)
            for item in previous.errors
            if str(item.get("agent", "")) not in retried_agent_ids
        ]
        collector_nodes = tuple(
            node
            for node, status in zip(retry_nodes, retry_statuses, strict=True)
            if status.analysis_state == "unavailable"
        )
        for node in collector_nodes:
            _clear_collector_output(inputs, node)
        with ThreadPoolExecutor(max_workers=max(1, len(collector_nodes))) as pool:
            future_by_node = {
                pool.submit(
                    self._run_collector,
                    node,
                    previous.request,
                    capabilities,
                    retry_context,
                ): node
                for node in collector_nodes
            }
            for future in as_completed(future_by_node):
                node = future_by_node[future]
                yield ProgressEvent(node, route=retry_nodes)
                try:
                    _apply_collector_output(inputs, node, future.result())
                except Exception as exc:
                    errors.append({"agent": node.value, "message": type(exc).__name__})
        if ResearchCapability.RISK in capabilities:
            inputs.risk_result = evaluate_research_risk(
                market_result=inputs.market_result,
                market_comparison_result=inputs.market_comparison_result,
                research_result=inputs.research_result,
                fundamentals_result=inputs.fundamentals_result,
                assets=inputs.assets,
            )

        answers_by_agent = {answer.agent: answer for answer in previous.agent_answers}
        statuses_by_agent = {status.agent: status for status in previous.agent_statuses}
        for node in retry_nodes:
            node_caps = _node_capabilities(node, capabilities)
            collector_error = next(
                (item for item in errors if item.get("agent") == node.value),
                None,
            )
            if collector_error is not None:
                answer = _collector_unavailable_answer(node)
            else:
                try:
                    answer = self._analyze_node(
                        node, previous.request.user_intent, inputs, node_caps
                    )
                except Exception as exc:
                    errors.append({"agent": node.value, "message": type(exc).__name__})
                    answer = _analysis_unavailable_answer(node)
            agent_id = _specialist_id(node)
            answers_by_agent[agent_id] = answer
            statuses_by_agent[agent_id] = AgentExecutionStatus(
                agent=agent_id,
                status=answer.status,
                source_state=_source_state(inputs, node, answer.status),
                analysis_state=answer.analysis_state,
                coverage_state=answer.coverage_state,
                capabilities=list(node_caps),
                limitation=next(iter(answer.limitations), None),
            )

        answers = [
            answers_by_agent[_specialist_id(node)]
            for node in full_route
            if _specialist_id(node) in answers_by_agent
        ]
        statuses = [
            statuses_by_agent[_specialist_id(node)]
            for node in full_route
            if _specialist_id(node) in statuses_by_agent
        ]
        warnings = _collect_warnings(inputs)
        report_status = (
            "partial"
            if errors or warnings or any(answer.status != "complete" for answer in answers)
            else "complete"
        )
        report = ResearchReport(
            request=previous.request,
            opportunity_result=inputs.opportunity_result,
            market_result=inputs.market_result,
            market_comparison_result=inputs.market_comparison_result,
            research_result=inputs.research_result,
            fundamentals_result=inputs.fundamentals_result,
            onchain_result=inputs.onchain_result,
            risk_result=inputs.risk_result,
            forecast_result=inputs.forecast_result,
            collection_context=inputs.collection_context,
            agent_statuses=statuses,
            agent_answers=answers,
            status=report_status,
            warnings=warnings,
            errors=errors,
            retry=ResearchRetryMetadata(
                original_run_id=stored.summary.id,
                retried_agents=[_specialist_id(node) for node in retry_nodes],
                retried_at=datetime.now(UTC),
            ),
        )
        if not _complete_history_run(self._history, new_run_id, report, inputs):
            raise RuntimeError("The combined retry report could not be saved.")
        yield ResultEvent(
            ResearchOutcome(
                research_report=report,
                agents=full_route,
                route=retry_nodes,
                requested_capabilities=capabilities,
                warnings=tuple(warnings),
                errors=tuple(errors),
                agent_answers=tuple(answers),
                run_id=new_run_id,
            )
        )

    def _analyze_node(
        self,
        node: WorkflowNode,
        question: str,
        inputs: AnalysisInputs,
        capabilities: tuple[ResearchCapability, ...],
    ) -> AgentAnswer:
        if node is WorkflowNode.FORECAST_AGENT:
            if inputs.forecast_result is None:
                raise RuntimeError("Forecast Agent is not configured.")
            return self._forecast_agent.analyze(question, inputs.forecast_result)
        return self._specialist_analysis.run(
            question,
            inputs,
            agent=cast(SpecialistAgentId, _specialist_id(node)),
            capabilities=capabilities,
        )

    def _run_collector(
        self,
        node: WorkflowNode,
        request: AnalysisRequest,
        capabilities: tuple[ResearchCapability, ...],
        context: CollectionContext,
    ) -> object:
        selected = list(_node_capabilities(node, capabilities))
        if node is WorkflowNode.MARKET_AGENT:
            return self._market_agent.run(
                request,
                requested_capabilities=selected,
                collection_context=context,
            )
        if node is WorkflowNode.NEWS_AGENT:
            return self._news_agent.run(
                request,
                requested_capabilities=cast(list[ResearchCapability | str], selected),
                collection_context=context,
            )
        if node is WorkflowNode.FUNDAMENTALS_AGENT:
            return self._fundamentals_agent.run(
                request,
                requested_capabilities=cast(list[ResearchCapability | str], selected),
                collection_context=context,
            )
        if node is WorkflowNode.ONCHAIN_AGENT:
            return self._onchain_agent.run(
                request,
                collection_context=context,
            )
        if node is WorkflowNode.FORECAST_AGENT:
            return self._forecast_agent.run(request)
        raise ValueError(f"Unsupported research node: {node.value}")


def _apply_collector_output(inputs: AnalysisInputs, node: WorkflowNode, output: object) -> None:
    if node is WorkflowNode.MARKET_AGENT:
        if isinstance(output, OpportunityScanResult):
            inputs.opportunity_result = output
        elif isinstance(output, MarketAgentResult):
            inputs.market_result = output
        elif isinstance(output, MarketComparisonResult):
            inputs.market_comparison_result = output
        else:
            raise TypeError("Market Agent returned an unsupported result.")
    elif node is WorkflowNode.NEWS_AGENT:
        if not isinstance(output, ResearchAgentResult):
            raise TypeError("News Agent returned an unsupported result.")
        inputs.research_result = output
    elif node is WorkflowNode.FUNDAMENTALS_AGENT:
        if not isinstance(output, FundamentalsAgentResult):
            raise TypeError("Fundamentals Agent returned an unsupported result.")
        inputs.fundamentals_result = output
    elif node is WorkflowNode.ONCHAIN_AGENT:
        if not isinstance(output, OnChainAgentResult):
            raise TypeError("On-Chain Activity Agent returned an unsupported result.")
        inputs.onchain_result = output
    elif node is WorkflowNode.FORECAST_AGENT:
        if not isinstance(output, ForecastAgentResult):
            raise TypeError("Forecast Agent returned an unsupported result.")
        inputs.forecast_result = output
    else:
        raise ValueError(f"Unsupported research node: {node.value}")


def _analysis_inputs_from_report(
    report: ResearchReport,
    capabilities: tuple[ResearchCapability, ...],
) -> AnalysisInputs:
    return AnalysisInputs(
        assets=report.request.ordered_assets(),
        requested_capabilities=list(capabilities),
        opportunity_result=report.opportunity_result,
        market_result=report.market_result,
        market_comparison_result=report.market_comparison_result,
        research_result=report.research_result,
        fundamentals_result=report.fundamentals_result,
        onchain_result=report.onchain_result,
        risk_result=report.risk_result,
        forecast_result=report.forecast_result,
        capability_coverage=report.capability_coverage,
        evidence_coverage_summary=report.evidence_coverage_summary,
        collection_context=report.collection_context,
    )


def _clear_collector_output(inputs: AnalysisInputs, node: WorkflowNode) -> None:
    if node is WorkflowNode.MARKET_AGENT:
        inputs.opportunity_result = None
        inputs.market_result = None
        inputs.market_comparison_result = None
        inputs.risk_result = None
    elif node is WorkflowNode.NEWS_AGENT:
        inputs.research_result = None
    elif node is WorkflowNode.FUNDAMENTALS_AGENT:
        inputs.fundamentals_result = None
    elif node is WorkflowNode.ONCHAIN_AGENT:
        inputs.onchain_result = None
    elif node is WorkflowNode.FORECAST_AGENT:
        inputs.forecast_result = None
    else:
        raise ValueError(f"Unsupported retry node: {node.value}")


def _collector_unavailable_answer(node: WorkflowNode) -> AgentAnswer:
    return AgentAnswer(
        agent=_specialist_id(node),
        answer="The selected data provider failed before analysis could begin.",
        uncertainty=["No live specialist analysis was produced for this card."],
        limitations=["The provider path failed before validated evidence was ready."],
        confidence=0.0,
        status="unavailable",
        analysis_state="unavailable",
        coverage_state="partial",
    )


def _analysis_unavailable_answer(node: WorkflowNode) -> AgentAnswer:
    return AgentAnswer(
        agent=_specialist_id(node),
        answer=(
            "Live specialist analysis was interrupted before a validated response "
            "could be produced for this card."
        ),
        analysis="Review the resources below; collected evidence remains isolated to this agent.",
        uncertainty=["The live LLM response could not be normalized for this agent."],
        limitations=["Specialist analysis failed safely without cancelling the research run."],
        confidence=0.3,
        status="partial",
        analysis_state="evidence_only",
        coverage_state="partial",
    )


def _route_for(capabilities: tuple[ResearchCapability, ...]) -> tuple[WorkflowNode, ...]:
    return tuple(WorkflowNode(manifest.id) for manifest in agents_for(set(capabilities)))


def _validated_route(
    action: ResearchAction,
    capabilities: tuple[ResearchCapability, ...],
) -> tuple[WorkflowNode, ...]:
    """Validate the planned route before any provider or specialist work begins."""

    if not action.agents_to_call:
        return _route_for(capabilities)
    if len(action.agents_to_call) != len(set(action.agents_to_call)):
        raise ValueError("The execution route contains duplicate agents.")
    try:
        route = tuple(WorkflowNode(agent_id) for agent_id in action.agents_to_call)
    except ValueError as exc:
        raise ValueError("The execution route contains an unknown agent.") from exc
    selected = set(capabilities)
    covered: set[ResearchCapability] = set()
    for node in route:
        owned = set(manifest_for(node.value).capabilities)
        if not selected.intersection(owned):
            raise ValueError(f"{node.value} does not own a requested capability.")
        covered.update(selected.intersection(owned))
    if covered != selected:
        missing = ", ".join(sorted(capability.value for capability in selected - covered))
        raise ValueError(f"The execution route does not cover: {missing}.")
    return route


def _node_capabilities(
    node: WorkflowNode,
    capabilities: tuple[ResearchCapability, ...],
) -> tuple[ResearchCapability, ...]:
    owned = set(manifest_for(node.value).capabilities)
    return tuple(capability for capability in capabilities if capability in owned)


def _ordered_capabilities(
    values: Sequence[ResearchCapability | str],
) -> tuple[ResearchCapability, ...]:
    selected: set[ResearchCapability] = set()
    for value in values:
        try:
            selected.add(
                value if isinstance(value, ResearchCapability) else ResearchCapability(value)
            )
        except ValueError:
            continue
    return tuple(capability for capability in ResearchCapability if capability in selected)


def _specialist_id(node: WorkflowNode) -> AgentId:
    return cast(
        AgentId,
        {
            WorkflowNode.MARKET_AGENT: "market_agent",
            WorkflowNode.NEWS_AGENT: "news_agent",
            WorkflowNode.FUNDAMENTALS_AGENT: "fundamentals_agent",
            WorkflowNode.ONCHAIN_AGENT: "onchain_agent",
            WorkflowNode.FORECAST_AGENT: "forecast_agent",
        }[node],
    )


def _status_agent_id(node: WorkflowNode) -> AgentId:
    return _specialist_id(node)


def _source_state(
    inputs: AnalysisInputs,
    node: WorkflowNode,
    fallback_status: str,
) -> str:
    if fallback_status == "unavailable":
        return "unavailable"
    states: list[str] = []
    if node is WorkflowNode.NEWS_AGENT and inputs.research_result is not None:
        states.append(inputs.research_result.news.source_state)
    if node is WorkflowNode.FUNDAMENTALS_AGENT and inputs.fundamentals_result is not None:
        states.extend(
            evidence.source_state
            for bundle in inputs.fundamentals_result.asset_results
            for evidence in (bundle.fundamentals, bundle.defi)
            if evidence is not None
        )
    if node is WorkflowNode.ONCHAIN_AGENT and inputs.onchain_result is not None:
        states.extend(
            bundle.onchain.source_state
            for bundle in inputs.onchain_result.asset_results
            if bundle.onchain is not None
        )
    if node is WorkflowNode.MARKET_AGENT:
        if inputs.market_result is not None:
            states.append(inputs.market_result.market.source_state)
            if inputs.market_result.derivatives is not None:
                states.append(inputs.market_result.derivatives.source_state)
        if inputs.market_comparison_result is not None:
            for item in inputs.market_comparison_result.assets:
                states.append(item.market.source_state)
                if item.derivatives is not None:
                    states.append(item.derivatives.source_state)
    if node is WorkflowNode.FORECAST_AGENT:
        states.append("live")
    if states and all(state == "cached" for state in states):
        return "cached"
    return "partial" if "cached" in states else "live"


def _collect_warnings(inputs: AnalysisInputs) -> list[str]:
    values = [*inputs.warnings]
    if inputs.opportunity_result is not None:
        values.extend(inputs.opportunity_result.warnings)
    if inputs.market_comparison_result is not None:
        values.extend(inputs.market_comparison_result.warnings)
        for item in inputs.market_comparison_result.assets:
            if item.derivatives is not None:
                values.extend(item.derivatives.warnings)
    if inputs.market_result is not None and inputs.market_result.derivatives is not None:
        values.extend(inputs.market_result.derivatives.warnings)
    if inputs.research_result is not None:
        values.extend(inputs.research_result.news.warnings)
        for bundle in inputs.research_result.asset_results:
            values.extend(bundle.limitations)
    if inputs.fundamentals_result is not None:
        for bundle in inputs.fundamentals_result.asset_results:
            values.extend(bundle.limitations)
    if inputs.onchain_result is not None:
        for bundle in inputs.onchain_result.asset_results:
            values.extend(bundle.limitations)
            if bundle.onchain is not None:
                values.extend(bundle.onchain.warnings)
    if inputs.risk_result is not None:
        values.extend(inputs.risk_result.assessment.coverage_gaps)
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _last_result(events: Iterator[WorkflowEvent]) -> ResearchOutcome:
    outcome: ResearchOutcome | None = None
    for event in events:
        if isinstance(event, ResultEvent):
            outcome = event.result
    if outcome is None:
        raise ValueError("The research runtime returned no completion event.")
    return outcome


def _begin_history_run(
    history: ResearchHistoryProtocol | None,
    *,
    request: AnalysisRequest,
    capabilities: Sequence[ResearchCapability],
    question: str,
) -> str | None:
    if history is None:
        return None
    try:
        return history.create_run(
            request=request,
            capabilities=capabilities,
            question=question,
        )
    except Exception:
        logger.warning("Could not create the research history record", exc_info=True)
        return None


def _complete_history_run(
    history: ResearchHistoryProtocol | None,
    run_id: str | None,
    report: ResearchReport,
    inputs: AnalysisInputs,
) -> bool:
    if history is None or run_id is None:
        return False
    try:
        from crypto_research.orchestration.evidence import build_evidence_records

        history.complete_run(run_id, report, build_evidence_records(inputs))
        return True
    except Exception:
        logger.warning("Could not complete the research history record", exc_info=True)
        try:
            history.fail_run(run_id, "Research completed, but durable persistence failed.")
        except Exception:
            logger.warning("Could not mark the research history record failed", exc_info=True)
        return False


__all__ = ["ResearchOutcome", "ResearchRuntime"]
