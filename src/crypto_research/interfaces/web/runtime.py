from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import streamlit as st

from crypto_research.bootstrap import (
    calculate_dashboard_metrics as calculate_dashboard_metrics,
)
from crypto_research.bootstrap import (
    create_research_runtime as load_research_runtime,
)
from crypto_research.bootstrap import (
    load_home_market_overview as load_home_market_overview,
)
from crypto_research.bootstrap import (
    load_market_dashboard as load_market_dashboard,
)
from crypto_research.config import Settings
from crypto_research.domain.account import UserWorkspace
from crypto_research.domain.history import StoredResearchRun
from crypto_research.domain.research import ResearchAction
from crypto_research.interfaces.web.presentation import (
    AgentAnalysisSectionPresentation as AgentAnalysisSectionPresentation,
)
from crypto_research.interfaces.web.presentation import (
    AgentAnswerPresentation as AgentAnswerPresentation,
)
from crypto_research.interfaces.web.presentation import (
    AgentClaimPresentation as AgentClaimPresentation,
)
from crypto_research.interfaces.web.presentation import (
    AgentPanelPresentation as AgentPanelPresentation,
)
from crypto_research.interfaces.web.presentation import (
    AgentStatusPresentation as AgentStatusPresentation,
)
from crypto_research.interfaces.web.presentation import (
    AssetPresentation as AssetPresentation,
)
from crypto_research.interfaces.web.presentation import (
    CapabilityDataPresentation as CapabilityDataPresentation,
)
from crypto_research.interfaces.web.presentation import DashboardView as DashboardView
from crypto_research.interfaces.web.presentation import (
    DiscoveryCandidatePresentation as DiscoveryCandidatePresentation,
)
from crypto_research.interfaces.web.presentation import (
    DiscoveryPresentation as DiscoveryPresentation,
)
from crypto_research.interfaces.web.presentation import (
    ResearchPresentation as ResearchPresentation,
)
from crypto_research.interfaces.web.presentation import ResearchTurn as ResearchTurn
from crypto_research.interfaces.web.presentation import (
    SourcePresentation as SourcePresentation,
)
from crypto_research.interfaces.web.presentation import (
    StructuredAgentAnalysisPresentation as StructuredAgentAnalysisPresentation,
)
from crypto_research.orchestration.events import ProgressEvent, ResultEvent, WorkflowNode
from crypto_research.orchestration.runtime import ResearchOutcome, ResearchRuntime
from crypto_research.shared.security import escape_markdown, redact_secrets


def _agent_labels() -> dict[str, str]:
    from crypto_research.orchestration.planning import agent_labels

    return agent_labels()


AGENT_LABELS = _agent_labels()
RUNTIME_STATE_KEY = "research_runtime"
LATEST_RESEARCH_TURN_STATE_KEY = "latest_research_turn"
CURRENT_OWNER_STATE_KEY = "current_workspace_owner"
CURRENT_WORKSPACE_STATE_KEY = "current_user_workspace"


@dataclass(frozen=True, slots=True)
class _RuntimeSession:
    settings: Settings
    owner_id: str
    runtime: ResearchRuntime


class StatusDisplay(Protocol):
    def write(self, *args: Any) -> None: ...

    def update(
        self,
        *,
        label: str | None = None,
        expanded: bool | None = None,
        state: Literal["running", "complete", "error"] | None = None,
    ) -> None: ...


def load_runtime_settings() -> Settings:
    """Read provider settings for each request so credentials never become stale."""

    try:
        raw = st.secrets.to_dict()
    except (AttributeError, FileNotFoundError, OSError):  # fmt: skip
        raw = {}
    settings = Settings.load_application()
    if not raw:
        return settings
    merged = settings.model_dump(mode="python")
    merged.update(raw)
    return Settings.from_mapping(merged)


def turn_from_outcome(outcome: ResearchOutcome) -> ResearchTurn:
    """Convert a completed workflow into UI-safe research state."""

    from crypto_research.interfaces.web.presentation import build_research_presentation

    agents = _agent_names(outcome.agents)
    return ResearchTurn(
        content="Specialist research results are ready.",
        research=build_research_presentation(
            outcome.research_report,
            route=agents or _agent_names(outcome.route),
            run_id=outcome.run_id,
        ),
        agents=agents,
        failed_agents=_failed_agent_names(outcome),
    )


def turn_from_stored_research(stored: StoredResearchRun) -> ResearchTurn:
    """Rehydrate a saved tenant report without provider or model calls."""

    from crypto_research.interfaces.web.presentation import build_research_presentation

    route = tuple(item.agent for item in stored.report.agent_statuses)
    return ResearchTurn(
        content="Latest saved Guided Research report.",
        research=build_research_presentation(
            stored.report,
            route=route,
            run_id=stored.summary.id,
        ),
        agents=route,
    )


def execute_research_with_progress(
    research_runtime: ResearchRuntime,
    action: ResearchAction,
    *,
    status: StatusDisplay,
) -> ResearchOutcome:
    outcome: ResearchOutcome | None = None
    for event in research_runtime.stream(action):
        if isinstance(event, ProgressEvent):
            show_progress(status, event)
        elif isinstance(event, ResultEvent):
            outcome = event.result
    if outcome is None:
        raise ValueError("The workflow returned no completion event.")
    return outcome


def show_progress(status: StatusDisplay, event: ProgressEvent) -> None:
    status.update(label=event.label, state="running")
    public = public_agent_summary(_agent_names(event.route))
    if public:
        status.write(public)


def _agent_names(nodes: tuple[WorkflowNode, ...]) -> tuple[str, ...]:
    return tuple(node.value for node in nodes if node.value in AGENT_LABELS)


def _failed_agent_names(outcome: ResearchOutcome) -> tuple[str, ...]:
    return tuple(
        item.agent for item in outcome.research_report.agent_statuses if item.status != "complete"
    )


def public_agent_summary(route: tuple[str, ...]) -> str:
    labels = [AGENT_LABELS[name] for name in route if name in AGENT_LABELS]
    return "Using live crypto research: " + ", ".join(labels) if labels else ""


def initialize_state() -> None:
    """Initialize only state retained by the research-only interface."""

    st.session_state.setdefault(LATEST_RESEARCH_TURN_STATE_KEY, None)


def session_runtime(settings: Settings, *, key: str = RUNTIME_STATE_KEY) -> ResearchRuntime:
    cached = st.session_state.get(key)
    owner_id = current_owner_id()
    if (
        isinstance(cached, _RuntimeSession)
        and cached.settings == settings
        and cached.owner_id == owner_id
    ):
        return cached.runtime
    research_runtime = load_research_runtime(settings, owner_id=owner_id)
    st.session_state[key] = _RuntimeSession(
        settings=settings,
        owner_id=owner_id,
        runtime=research_runtime,
    )
    return research_runtime


def current_owner_id() -> str:
    value = st.session_state.get(CURRENT_OWNER_STATE_KEY)
    workspace = current_workspace()
    if (
        not isinstance(value, str)
        or not value.strip()
        or workspace is None
        or workspace.profile.id != value
    ):
        raise RuntimeError("An authenticated workspace is required for this operation.")
    return value


def current_workspace() -> UserWorkspace | None:
    value = st.session_state.get(CURRENT_WORKSPACE_STATE_KEY)
    return value if isinstance(value, UserWorkspace) else None


def initialize_workspace(owner_id: str, workspace: UserWorkspace) -> None:
    normalized = owner_id.strip()
    if not normalized or normalized != workspace.profile.id:
        raise ValueError("The workspace owner binding is invalid.")
    previous = st.session_state.get(CURRENT_OWNER_STATE_KEY)
    if isinstance(previous, str) and previous != normalized:
        clear_user_session_state()
    st.session_state[CURRENT_OWNER_STATE_KEY] = normalized
    st.session_state[CURRENT_WORKSPACE_STATE_KEY] = workspace
    initialize_state()


def update_current_workspace(workspace: UserWorkspace) -> None:
    if current_owner_id() != workspace.profile.id:
        raise ValueError("The updated workspace does not match the authenticated owner.")
    st.session_state[CURRENT_OWNER_STATE_KEY] = workspace.profile.id
    st.session_state[CURRENT_WORKSPACE_STATE_KEY] = workspace


def clear_user_session_state() -> None:
    st.session_state.clear()


def safe_markdown(value: str, *, preserve_paragraphs: bool = False) -> str:
    return escape_markdown(redact_secrets(value), preserve_paragraphs=preserve_paragraphs)
