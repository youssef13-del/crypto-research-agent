from __future__ import annotations

import time
from collections.abc import Iterator
from threading import Event

import pytest
from pytest import MonkeyPatch
from streamlit.testing.v1 import AppTest

from crypto_research.config import Settings
from crypto_research.domain.research import ResearchCapability, ResearchReport
from crypto_research.interfaces.web import research_jobs
from crypto_research.interfaces.web.research_jobs import (
    ResearchJobAlreadyRunning,
    ResearchJobSnapshot,
)
from crypto_research.orchestration.events import (
    ProgressEvent,
    ResearchOutcome,
    ResultEvent,
    WorkflowEvent,
    WorkflowNode,
)
from crypto_research.orchestration.planning import compile_guided_research_plan


@pytest.fixture(autouse=True)
def _empty_job_registry(monkeypatch: MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(research_jobs, "_JOBS_BY_OWNER", {})
    yield


def test_guided_job_continues_and_remains_owner_scoped_during_navigation(
    monkeypatch: MonkeyPatch,
) -> None:
    started = Event()
    release = Event()
    plan = compile_guided_research_plan(
        ["BTC"],
        [
            ResearchCapability.MARKET,
            ResearchCapability.FUNDAMENTALS,
            ResearchCapability.NEWS,
        ],
    )
    assert plan.action.request is not None
    outcome = ResearchOutcome(
        research_report=ResearchReport(request=plan.action.request, status="complete"),
        agents=(
            WorkflowNode.MARKET_AGENT,
            WorkflowNode.FUNDAMENTALS_AGENT,
            WorkflowNode.NEWS_AGENT,
        ),
        route=(
            WorkflowNode.MARKET_AGENT,
            WorkflowNode.FUNDAMENTALS_AGENT,
            WorkflowNode.NEWS_AGENT,
        ),
        requested_capabilities=tuple(plan.requested_capabilities),
        warnings=(),
        errors=(),
        run_id="saved-run",
    )

    class FakeRuntime:
        def stream(self, *_args: object, **_kwargs: object) -> Iterator[WorkflowEvent]:
            yield ProgressEvent(
                WorkflowNode.MARKET_AGENT,
                route=(
                    WorkflowNode.MARKET_AGENT,
                    WorkflowNode.FUNDAMENTALS_AGENT,
                    WorkflowNode.NEWS_AGENT,
                ),
            )
            started.set()
            assert release.wait(3)
            yield ProgressEvent(WorkflowNode.MARKET_AGENT)
            yield ResultEvent(outcome)

    captured: dict[str, str] = {}

    def create(_settings: Settings, *, owner_id: str) -> FakeRuntime:
        captured["owner_id"] = owner_id
        return FakeRuntime()

    monkeypatch.setattr(research_jobs, "create_research_runtime", create)
    first = research_jobs.start_guided_research_job(
        owner_id="owner-1",
        settings=Settings(_env_file=None),
        action=plan.action,
        question="Quick overview: Research BTC",
    )

    assert started.wait(3)
    running = research_jobs.latest_research_job("owner-1")
    assert running is not None
    assert running.id == first.id
    assert running.active is True
    assert running.state == "running"
    assert running.agents == ("market_agent", "fundamentals_agent", "news_agent")
    assert research_jobs.latest_research_job("owner-2") is None
    assert captured["owner_id"] == "owner-1"
    with pytest.raises(ResearchJobAlreadyRunning):
        research_jobs.start_guided_research_job(
            owner_id="owner-1",
            settings=Settings(_env_file=None),
            action=plan.action,
            question="Another request",
        )

    release.set()
    completed = _wait_for_terminal_job("owner-1")

    assert completed.state == "complete"
    assert completed.outcome is outcome
    assert completed.completed_at is not None


def test_background_failure_becomes_safe_persistent_job_state(
    monkeypatch: MonkeyPatch,
) -> None:
    plan = compile_guided_research_plan(
        ["BTC"],
        [
            ResearchCapability.MARKET,
            ResearchCapability.FUNDAMENTALS,
            ResearchCapability.NEWS,
        ],
    )

    class FailingRuntime:
        def stream(self, *_args: object, **_kwargs: object) -> Iterator[WorkflowEvent]:
            raise RuntimeError("provider secret must never reach the UI")
            yield ProgressEvent(WorkflowNode.MARKET_AGENT)  # pragma: no cover

    monkeypatch.setattr(
        research_jobs,
        "create_research_runtime",
        lambda *_args, **_kwargs: FailingRuntime(),
    )
    research_jobs.start_guided_research_job(
        owner_id="owner-1",
        settings=Settings(_env_file=None),
        action=plan.action,
        question="Quick overview: Research BTC",
    )

    failed = _wait_for_terminal_job("owner-1")

    assert failed.state == "error"
    assert failed.outcome is None
    assert failed.error_message == "The research services are temporarily unavailable."
    assert "secret" not in failed.error_message


def test_sidebar_syncs_completed_job_into_latest_research_across_pages() -> None:
    source = """
from datetime import UTC, datetime
from crypto_research.config import Settings
from crypto_research.domain.account import UserProfile, UserWorkspace
from crypto_research.domain.research import AnalysisRequest, ResearchReport
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.components import shell
from crypto_research.interfaces.web.research_jobs import ResearchJobSnapshot
from crypto_research.orchestration.events import ResearchOutcome

now = datetime.now(UTC)
workspace = UserWorkspace(profile=UserProfile(
    id="owner-1", provider="auth0", email="owner@example.com", provider_name="Owner",
    created_at=now, updated_at=now, last_login_at=now,
))
runtime.initialize_workspace("owner-1", workspace)
report = ResearchReport(
    request=AnalysisRequest(user_intent="Quick overview", asset_query="Bitcoin"),
    status="complete",
)
outcome = ResearchOutcome(
    research_report=report,
    agents=(), route=(), requested_capabilities=(), warnings=(), errors=(), run_id="job-run",
)
snapshot = ResearchJobSnapshot(
    id="job-1", owner_id="owner-1", kind="guided", question="Quick overview: BTC",
    state="complete", label="Specialist results ready", agents=(), started_at=now,
    updated_at=now, completed_at=now, outcome=outcome,
)
shell.latest_research_job = lambda owner_id: snapshot if owner_id == "owner-1" else None
shell.render_sidebar(Settings(_env_file=None))
"""
    app = AppTest.from_string(source, default_timeout=30).run()

    assert not app.exception
    latest = app.session_state["latest_research_turn"]
    assert latest.research.run_id == "job-run"
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Latest report ready" in rendered
    assert any("Open Research" in item.value for item in app.caption)


def test_sidebar_fragment_renders_active_job_on_every_page() -> None:
    source = """
from datetime import UTC, datetime
from crypto_research.config import Settings
from crypto_research.domain.account import UserProfile, UserWorkspace
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.components import shell
from crypto_research.interfaces.web.research_jobs import ResearchJobSnapshot

now = datetime.now(UTC)
workspace = UserWorkspace(profile=UserProfile(
    id="owner-1", provider="auth0", email="owner@example.com", provider_name="Owner",
    created_at=now, updated_at=now, last_login_at=now,
))
runtime.initialize_workspace("owner-1", workspace)
snapshot = ResearchJobSnapshot(
    id="job-1", owner_id="owner-1", kind="guided", question="Due diligence: BTC",
    state="running", label="Collecting relevant news...", agents=("news_agent",),
    started_at=now, updated_at=now,
)
original_latest = shell.latest_research_job
shell.latest_research_job = lambda owner_id: snapshot if owner_id == "owner-1" else None
shell.render_sidebar(Settings(_env_file=None))
shell.latest_research_job = original_latest
"""
    app = AppTest.from_string(source, default_timeout=30).run()

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Research in progress" in rendered
    captions = [item.value for item in app.caption]
    assert "Due diligence: BTC" in captions
    assert "Collecting relevant news..." in captions
    assert "Agents: News Agent" in captions
    assert any("change pages" in caption for caption in captions)


def test_guided_submit_schedules_job_instead_of_blocking_the_page() -> None:
    source = """
from datetime import UTC, datetime
import streamlit as st
from crypto_research.config import Settings
from crypto_research.domain.account import UserProfile, UserWorkspace
from crypto_research.domain.core import ResearchCapability
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.pages import research as page
from crypto_research.orchestration.planning import compile_guided_research_plan

now = datetime.now(UTC)
workspace = UserWorkspace(profile=UserProfile(
    id="owner-1", provider="auth0", email="owner@example.com", provider_name="Owner",
    created_at=now, updated_at=now, last_login_at=now,
))
runtime.initialize_workspace("owner-1", workspace)
plan = compile_guided_research_plan(
    ["BTC"],
    [ResearchCapability.MARKET, ResearchCapability.FUNDAMENTALS, ResearchCapability.NEWS],
)
original_settings = runtime.load_runtime_settings
original_start = page.start_guided_research_job
runtime.load_runtime_settings = lambda: Settings(_env_file=None)
def schedule(**kwargs):
    st.session_state["scheduled-guided-job"] = kwargs
page.start_guided_research_job = schedule
page._submit_guided_research(plan)
runtime.load_runtime_settings = original_settings
page.start_guided_research_job = original_start
"""
    app = AppTest.from_string(source, default_timeout=30).run()

    assert not app.exception
    scheduled = app.session_state["scheduled-guided-job"]
    assert scheduled["owner_id"] == "owner-1"
    assert scheduled["action"].agents_to_call == [
        "market_agent",
        "fundamentals_agent",
        "news_agent",
    ]
    assert scheduled["question"] == "Research BTC - Market behavior + Fundamentals + Recent news"


def _wait_for_terminal_job(owner_id: str) -> ResearchJobSnapshot:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        snapshot = research_jobs.latest_research_job(owner_id)
        if snapshot is not None and not snapshot.active:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("Background research job did not finish in time.")
