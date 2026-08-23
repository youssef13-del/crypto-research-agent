"""Owner-scoped background execution for navigation-safe Guided Research."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Literal
from uuid import uuid4

from crypto_research.bootstrap import create_research_runtime
from crypto_research.config import Settings
from crypto_research.domain.research import ResearchAction
from crypto_research.orchestration.events import (
    ProgressEvent,
    ResearchOutcome,
    ResultEvent,
    WorkflowEvent,
)

type ResearchJobState = Literal["queued", "running", "complete", "error"]
type ResearchJobKind = Literal["guided", "retry"]

LOGGER = logging.getLogger(__name__)
_COMPLETED_JOB_RETENTION = timedelta(hours=1)
_MAX_RETAINED_JOBS = 128


class ResearchJobAlreadyRunning(RuntimeError):
    """Raised when one owner tries to start overlapping research jobs."""


@dataclass(frozen=True, slots=True)
class ResearchJobSnapshot:
    """Immutable, secret-free state safe for rendering on any authenticated page."""

    id: str
    owner_id: str
    kind: ResearchJobKind
    question: str
    state: ResearchJobState
    label: str
    agents: tuple[str, ...]
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    outcome: ResearchOutcome | None = None
    error_message: str | None = None

    @property
    def active(self) -> bool:
        return self.state in {"queued", "running"}


@dataclass(slots=True)
class _ResearchJobRecord:
    id: str
    owner_id: str
    kind: ResearchJobKind
    question: str
    state: ResearchJobState
    label: str
    agents: tuple[str, ...]
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    outcome: ResearchOutcome | None = None
    error_message: str | None = None

    def snapshot(self) -> ResearchJobSnapshot:
        return ResearchJobSnapshot(
            id=self.id,
            owner_id=self.owner_id,
            kind=self.kind,
            question=self.question,
            state=self.state,
            label=self.label,
            agents=self.agents,
            started_at=self.started_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            outcome=self.outcome,
            error_message=self.error_message,
        )


_LOCK = RLock()
_JOBS_BY_OWNER: dict[str, _ResearchJobRecord] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chainscope-research")


def start_guided_research_job(
    *,
    owner_id: str,
    settings: Settings,
    action: ResearchAction,
    question: str,
) -> ResearchJobSnapshot:
    """Start one Guided Research run that survives Streamlit page navigation."""

    record = _reserve_job(
        owner_id=owner_id,
        kind="guided",
        question=question,
        agents=tuple(action.agents_to_call),
    )
    try:
        _EXECUTOR.submit(_run_guided_job, record, settings, action)
    except RuntimeError:
        _release_failed_submission(record)
        raise
    with _LOCK:
        return record.snapshot()


def start_retry_research_job(
    *,
    owner_id: str,
    settings: Settings,
    run_id: str,
    question: str,
    agents: tuple[str, ...],
) -> ResearchJobSnapshot:
    """Start a failed-agent retry without blocking the active Streamlit page."""

    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("Retry requires a stored research run ID.")
    record = _reserve_job(
        owner_id=owner_id,
        kind="retry",
        question=question,
        agents=agents,
    )
    try:
        _EXECUTOR.submit(_run_retry_job, record, settings, normalized_run_id)
    except RuntimeError:
        _release_failed_submission(record)
        raise
    with _LOCK:
        return record.snapshot()


def latest_research_job(owner_id: str) -> ResearchJobSnapshot | None:
    """Return only the latest job belonging to the requested internal owner ID."""

    normalized = _normalize_owner(owner_id)
    with _LOCK:
        _purge_completed_jobs(datetime.now(UTC))
        record = _JOBS_BY_OWNER.get(normalized)
        return record.snapshot() if record is not None else None


def _reserve_job(
    *,
    owner_id: str,
    kind: ResearchJobKind,
    question: str,
    agents: tuple[str, ...],
) -> _ResearchJobRecord:
    normalized_owner = _normalize_owner(owner_id)
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise ValueError("Research requires a visible question or scope.")
    now = datetime.now(UTC)
    with _LOCK:
        _purge_completed_jobs(now)
        existing = _JOBS_BY_OWNER.get(normalized_owner)
        if existing is not None and existing.state in {"queued", "running"}:
            raise ResearchJobAlreadyRunning("Research is already running for this workspace.")
        record = _ResearchJobRecord(
            id=uuid4().hex,
            owner_id=normalized_owner,
            kind=kind,
            question=normalized_question[:500],
            state="queued",
            label="Queued for research...",
            agents=tuple(dict.fromkeys(agents)),
            started_at=now,
            updated_at=now,
        )
        _JOBS_BY_OWNER[normalized_owner] = record
        return record


def _run_guided_job(
    record: _ResearchJobRecord,
    settings: Settings,
    action: ResearchAction,
) -> None:
    try:
        research_runtime = create_research_runtime(settings, owner_id=record.owner_id)
        events = research_runtime.stream(action)
        _consume_events(record, events)
    except Exception as exc:
        LOGGER.error("Background Guided Research failed (%s).", type(exc).__name__)
        _fail_job(record)


def _run_retry_job(
    record: _ResearchJobRecord,
    settings: Settings,
    run_id: str,
) -> None:
    try:
        research_runtime = create_research_runtime(settings, owner_id=record.owner_id)
        events = research_runtime.stream_retry_failed_agents(run_id)
        _consume_events(record, events)
    except Exception as exc:
        LOGGER.error("Background research retry failed (%s).", type(exc).__name__)
        _fail_job(record)


def _consume_events(
    record: _ResearchJobRecord,
    events: Iterable[WorkflowEvent],
) -> None:
    outcome: ResearchOutcome | None = None
    _update_record(record, state="running", label="Starting specialist research...")
    for event in events:
        if isinstance(event, ProgressEvent):
            agents = tuple(node.value for node in event.route) or record.agents
            _update_record(record, state="running", label=event.label, agents=agents)
        elif isinstance(event, ResultEvent):
            outcome = event.result
    if outcome is None:
        raise ValueError("The background workflow returned no completion event.")
    now = datetime.now(UTC)
    with _LOCK:
        if _JOBS_BY_OWNER.get(record.owner_id) is not record:
            return
        record.state = "complete"
        record.label = "Specialist results ready"
        record.outcome = outcome
        record.updated_at = now
        record.completed_at = now


def _update_record(
    record: _ResearchJobRecord,
    *,
    state: ResearchJobState,
    label: str,
    agents: tuple[str, ...] | None = None,
) -> None:
    with _LOCK:
        if _JOBS_BY_OWNER.get(record.owner_id) is not record:
            return
        record.state = state
        record.label = label
        if agents:
            record.agents = tuple(dict.fromkeys(agents))
        record.updated_at = datetime.now(UTC)


def _fail_job(record: _ResearchJobRecord) -> None:
    now = datetime.now(UTC)
    with _LOCK:
        if _JOBS_BY_OWNER.get(record.owner_id) is not record:
            return
        record.state = "error"
        record.label = "Research unavailable"
        record.error_message = "The research services are temporarily unavailable."
        record.updated_at = now
        record.completed_at = now


def _release_failed_submission(record: _ResearchJobRecord) -> None:
    with _LOCK:
        if _JOBS_BY_OWNER.get(record.owner_id) is record:
            del _JOBS_BY_OWNER[record.owner_id]


def _normalize_owner(owner_id: str) -> str:
    normalized = owner_id.strip() if isinstance(owner_id, str) else ""
    if not normalized:
        raise ValueError("An authenticated owner is required for background research.")
    return normalized


def _purge_completed_jobs(now: datetime) -> None:
    expired = [
        owner_id
        for owner_id, record in _JOBS_BY_OWNER.items()
        if record.completed_at is not None and now - record.completed_at > _COMPLETED_JOB_RETENTION
    ]
    for owner_id in expired:
        del _JOBS_BY_OWNER[owner_id]
    overflow = len(_JOBS_BY_OWNER) - _MAX_RETAINED_JOBS
    if overflow <= 0:
        return
    terminal = sorted(
        (record for record in _JOBS_BY_OWNER.values() if record.completed_at is not None),
        key=lambda record: record.completed_at or record.updated_at,
    )
    for record in terminal[:overflow]:
        _JOBS_BY_OWNER.pop(record.owner_id, None)


__all__ = [
    "ResearchJobAlreadyRunning",
    "ResearchJobSnapshot",
    "latest_research_job",
    "start_guided_research_job",
    "start_retry_research_job",
]
