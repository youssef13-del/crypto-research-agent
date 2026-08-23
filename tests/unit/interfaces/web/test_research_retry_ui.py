from streamlit.testing.v1 import AppTest

from crypto_research.interfaces.web.pages.research import _retryable_agents
from crypto_research.interfaces.web.runtime import (
    AgentPanelPresentation,
    ResearchPresentation,
    ResearchTurn,
)


def _presentation(*panels: AgentPanelPresentation) -> ResearchPresentation:
    return ResearchPresentation(
        status="partial",
        warnings=(),
        sources=(),
        disclaimer="Research only.",
        assets=(),
        agent_panels=panels,
        run_id="run-1",
    )


def test_only_failed_analysis_states_are_retryable() -> None:
    turn = ResearchTurn(
        content="Research cards",
        research=_presentation(
            AgentPanelPresentation(
                agent="market_agent",
                title="Market",
                status="complete",
                source_state="cached",
            ),
            AgentPanelPresentation(
                agent="fundamentals_agent",
                title="Fundamentals",
                status="partial",
                coverage_state="partial",
            ),
            AgentPanelPresentation(
                agent="news_agent",
                title="News",
                status="partial",
                analysis_state="evidence_only",
            ),
            AgentPanelPresentation(
                agent="onchain_agent",
                title="On-chain",
                status="unavailable",
                source_state="unavailable",
                analysis_state="unavailable",
            ),
        ),
    )

    assert _retryable_agents(turn) == ("news_agent", "onchain_agent")


def test_research_cards_show_compact_execution_summary() -> None:
    source = """
from crypto_research.interfaces.web.components.research import render_research_response
from crypto_research.interfaces.web.runtime import AgentPanelPresentation, ResearchPresentation

render_research_response(
    "Structured research",
    ResearchPresentation(
        status="partial",
        warnings=(),
        sources=(),
        disclaimer="Research only.",
        assets=(),
        retry_of_run_id="original-run",
        retried_agents=("news_agent",),
        agent_panels=(
            AgentPanelPresentation(agent="market_agent", title="Market", status="complete"),
            AgentPanelPresentation(
                agent="fundamentals_agent", title="Fundamentals", status="complete",
                source_state="cached",
            ),
            AgentPanelPresentation(
                agent="news_agent", title="News", status="partial",
                analysis_state="evidence_only",
            ),
            AgentPanelPresentation(
                agent="onchain_agent", title="On-chain", status="unavailable",
                source_state="unavailable", analysis_state="unavailable",
            ),
        ),
    ),
)
"""
    app = AppTest.from_string(source, default_timeout=30).run()

    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Complete", "2"),
        ("Live", "2"),
        ("Cached", "1"),
        ("Evidence only", "1"),
        ("Unavailable", "1"),
    ]
    assert any("Combined retry report" in caption.value for caption in app.caption)


def test_latest_saved_guided_research_is_restored_once_and_cannot_be_cleared() -> None:
    source = """
from datetime import UTC, datetime
import streamlit as st
from crypto_research.domain.account import UserProfile, UserWorkspace
from crypto_research.domain.history import ResearchRunSummary, StoredResearchRun
from crypto_research.domain.research import AnalysisRequest, ResearchReport
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.pages import research as page

now = datetime.now(UTC)
workspace = UserWorkspace(profile=UserProfile(
    id="saved-user", provider="auth0", email="saved@example.com", provider_name="Saved",
    created_at=now, updated_at=now, last_login_at=now,
))
runtime.initialize_workspace(workspace.profile.id, workspace)
st.session_state.setdefault("restore-calls", 0)
stored = StoredResearchRun(
    summary=ResearchRunSummary(
        id="saved-run", created_at=now, completed_at=now, state="complete",
        question="Saved research", assets=(), capabilities=(), exchange=None,
        timeframe=None, pinned=False, evidence_count=0,
    ),
    report=ResearchReport(
        request=AnalysisRequest(user_intent="Saved research", asset_query="Bitcoin"),
        status="complete",
    ),
)
class Scoped:
    def latest_run(self):
        st.session_state["restore-calls"] += 1
        return stored
class Repository:
    def for_owner(self, owner_id):
        assert owner_id == workspace.profile.id
        return Scoped()
page.load_research_repository = lambda *args: Repository()
page.research_page()
"""
    app = AppTest.from_string(source, default_timeout=30).run()

    assert not app.exception
    assert app.session_state["restore-calls"] == 1
    restored = app.session_state["latest_research_turn"]
    assert restored.research.run_id == "saved-run"
    assert any("restored when you return" in item.value for item in app.caption)
    assert all(button.label != "Clear" for button in app.button)

    app.run()

    assert not app.exception
    assert app.session_state["restore-calls"] == 1
