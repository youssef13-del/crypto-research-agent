"""Authenticated application shell and sidebar navigation chrome."""

from __future__ import annotations

import html

import streamlit as st

from crypto_research.config import LLMProvider, Settings
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.auth import logout
from crypto_research.interfaces.web.components.layout import render_brand
from crypto_research.interfaces.web.presentation import ResearchTurn
from crypto_research.interfaces.web.research_jobs import ResearchJobSnapshot, latest_research_job
from crypto_research.interfaces.web.runtime import (
    LATEST_RESEARCH_TURN_STATE_KEY,
)

_SYNCED_RESEARCH_JOB_STATE_KEY = "synced_research_job_id"


def render_sidebar(settings: Settings) -> None:
    """Render compact account identity and session controls around native navigation."""

    with st.sidebar:
        render_brand(compact=True)
        workspace = runtime.current_workspace()
        if workspace is not None:
            st.markdown(
                '<div class="cs-sidebar-profile">'
                f"<span>{html.escape(workspace.profile.effective_name[:1].upper())}</span><div>"
                f"<strong>{html.escape(workspace.profile.effective_name)}</strong>"
                f"<small>{html.escape(workspace.profile.email)}</small></div></div>",
                unsafe_allow_html=True,
            )
        status_class, status_label = _workspace_status(settings)
        st.markdown(
            f'<div class="cs-sidebar-status"><span class="cs-status-dot {status_class}"></span>'
            f"{status_label}</div>",
            unsafe_allow_html=True,
        )
        if workspace is not None:
            render_research_activity(workspace.profile.id)
        if workspace is not None and st.button(
            "Sign out",
            width="stretch",
            icon=":material/logout:",
        ):
            logout()
        st.markdown(
            '<div class="cs-footer">Research only. Market observations and forecasts are not '
            "trading instructions.</div>",
            unsafe_allow_html=True,
        )


def _workspace_status(settings: Settings) -> tuple[str, str]:
    if settings.llm_provider is LLMProvider.DISABLED:
        return "cs-status-dot-muted", "Deterministic analysis"
    return "", "Live analysis available"


def render_research_activity(owner_id: str) -> None:
    """Keep current research progress and its latest result visible across pages."""

    snapshot = latest_research_job(owner_id)
    if snapshot is not None and snapshot.active:
        _render_active_research_activity(owner_id)
        return
    synchronized = _sync_completed_research(snapshot)
    if snapshot is not None:
        _render_job_snapshot(snapshot)
    else:
        _render_session_result()
    if synchronized:
        st.rerun()


@st.fragment(run_every=1.0)
def _render_active_research_activity(owner_id: str) -> None:
    snapshot = latest_research_job(owner_id)
    synchronized = _sync_completed_research(snapshot)
    if snapshot is not None:
        _render_job_snapshot(snapshot)
    if synchronized:
        st.rerun()


def _sync_completed_research(snapshot: ResearchJobSnapshot | None) -> bool:
    if (
        snapshot is None
        or snapshot.state != "complete"
        or snapshot.outcome is None
        or st.session_state.get(_SYNCED_RESEARCH_JOB_STATE_KEY) == snapshot.id
    ):
        return False
    turn = runtime.turn_from_outcome(snapshot.outcome)
    st.session_state[LATEST_RESEARCH_TURN_STATE_KEY] = turn
    st.session_state[_SYNCED_RESEARCH_JOB_STATE_KEY] = snapshot.id
    return True


def _render_job_snapshot(snapshot: ResearchJobSnapshot) -> None:
    labels = [runtime.AGENT_LABELS.get(agent, agent) for agent in snapshot.agents]
    if snapshot.active:
        _render_activity_heading("running", "Research in progress")
        st.caption(snapshot.question)
        st.caption(snapshot.label)
        if labels:
            st.caption("Agents: " + ", ".join(labels))
        st.caption("You can change pages while this continues.")
        return
    if snapshot.state == "complete":
        _render_activity_heading("complete", "Latest report ready")
        st.caption(snapshot.question)
        report = snapshot.outcome.research_report if snapshot.outcome is not None else None
        if report is not None:
            st.caption(f"Status: {report.status.title()}")
        st.caption("Open Research to review the saved result.")
        return
    _render_activity_heading("error", "Research did not finish")
    st.caption(snapshot.question)
    st.caption(snapshot.error_message or "Research is temporarily unavailable.")


def _render_session_result() -> None:
    turn = st.session_state.get(LATEST_RESEARCH_TURN_STATE_KEY)
    if not isinstance(turn, ResearchTurn):
        return
    _render_activity_heading("complete", "Latest report available")
    st.caption(f"Status: {turn.research.status.title()}")
    st.caption("Open Research to review the saved result.")


def _render_activity_heading(state: str, label: str) -> None:
    st.markdown(
        '<div class="cs-research-activity-heading">'
        f'<span class="cs-research-activity-dot is-{html.escape(state)}"></span>'
        f"<strong>{html.escape(label)}</strong></div>",
        unsafe_allow_html=True,
    )


__all__ = ["render_research_activity", "render_sidebar"]
