"""SQL-backed research history and persistent cache implementation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, event, func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from crypto_research.domain.account import (
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserWorkspace,
    WorkspaceStats,
)
from crypto_research.domain.evidence import EvidenceRecord
from crypto_research.domain.history import (
    BulkDeleteResult,
    ResearchComparison,
    ResearchRunSummary,
    StoredResearchRun,
)
from crypto_research.domain.research import (
    AnalysisRequest,
    ResearchCapability,
    ResearchReport,
)
from crypto_research.storage.models import (
    CacheEntryRecord,
    EvidenceSnapshotRecord,
    ResearchRunRecord,
    StoredCacheEntry,
    UserPreferenceRecord,
    UserRecord,
    WatchlistAssetRecord,
)
from crypto_research.storage.serialization import (
    PAYLOAD_SCHEMA_VERSION,
    application_version,
    deserialize_report_payload,
    payload_checksum,
)

_MIGRATION_LOCK = Lock()


class ResearchRepository:
    """Synchronous repository shared by CLI and Streamlit runtimes."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    def for_owner(self, owner_id: str) -> ScopedResearchRepository:
        """Bind all user-owned research operations to one internal owner ID."""

        return ScopedResearchRepository(self, _validated_owner_id(owner_id))

    def upsert_user(self, identity: UserIdentity) -> UserWorkspace:
        """Create or refresh one verified OIDC-backed workspace."""

        if not identity.email_verified:
            raise ValueError("A verified email is required for a ChainScope workspace.")
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            record = session.scalar(
                select(UserRecord).where(
                    UserRecord.issuer == identity.issuer,
                    UserRecord.subject == identity.subject,
                )
            )
            if record is None:
                record = UserRecord(
                    id=str(uuid4()),
                    provider=identity.provider,
                    issuer=identity.issuer,
                    subject=identity.subject,
                    email=identity.email,
                    email_verified=True,
                    provider_name=identity.provider_name,
                    avatar_url=identity.avatar_url,
                    created_at=now,
                    updated_at=now,
                    last_login_at=now,
                )
                session.add(record)
                session.flush()
                session.add(UserPreferenceRecord(user_id=record.id))
                session.add_all(
                    WatchlistAssetRecord(
                        user_id=record.id,
                        asset=asset,
                        position=position,
                        created_at=now,
                    )
                    for position, asset in enumerate(("BTC", "ETH", "SOL", "XRP"))
                )
            else:
                record.provider = identity.provider
                record.email = identity.email
                record.email_verified = True
                record.provider_name = identity.provider_name
                record.avatar_url = identity.avatar_url
                record.updated_at = now
                record.last_login_at = now
            user_id = record.id
        workspace = self.get_workspace(user_id)
        if workspace is None:
            raise RuntimeError("The authenticated workspace could not be loaded.")
        return workspace

    def get_workspace(self, user_id: str) -> UserWorkspace | None:
        owner_id = _validated_owner_id(user_id)
        with self._sessions() as session:
            record = session.get(UserRecord, owner_id)
            if record is None:
                return None
            preference = session.get(UserPreferenceRecord, owner_id)
            watchlist = tuple(
                session.scalars(
                    select(WatchlistAssetRecord.asset)
                    .where(WatchlistAssetRecord.user_id == owner_id)
                    .order_by(WatchlistAssetRecord.position)
                ).all()
            )
            run_count = int(
                session.scalar(
                    select(func.count(ResearchRunRecord.id)).where(
                        ResearchRunRecord.owner_id == owner_id
                    )
                )
                or 0
            )
            pinned_count = int(
                session.scalar(
                    select(func.count(ResearchRunRecord.id)).where(
                        ResearchRunRecord.owner_id == owner_id,
                        ResearchRunRecord.pinned.is_(True),
                    )
                )
                or 0
            )
            evidence_count = int(
                session.scalar(
                    select(func.count(EvidenceSnapshotRecord.id))
                    .join(ResearchRunRecord, ResearchRunRecord.id == EvidenceSnapshotRecord.run_id)
                    .where(ResearchRunRecord.owner_id == owner_id)
                )
                or 0
            )
            last_research_at = session.scalar(
                select(func.max(ResearchRunRecord.completed_at)).where(
                    ResearchRunRecord.owner_id == owner_id
                )
            )
            return UserWorkspace(
                profile=_user_profile(record),
                preferences=_user_preferences(preference),
                watchlist=watchlist,
                stats=WorkspaceStats(
                    research_runs=run_count,
                    pinned_runs=pinned_count,
                    evidence_records=evidence_count,
                    last_research_at=_utc(last_research_at) if last_research_at else None,
                ),
            )

    def delete_workspace(self, user_id: str) -> bool:
        owner_id = _validated_owner_id(user_id)
        with self._sessions.begin() as session:
            session.execute(delete(ResearchRunRecord).where(ResearchRunRecord.owner_id == owner_id))
            result = session.execute(delete(UserRecord).where(UserRecord.id == owner_id))
        return bool(getattr(result, "rowcount", 0))

    def create_run(
        self,
        *,
        request: AnalysisRequest,
        capabilities: Sequence[ResearchCapability],
        question: str,
        owner_id: str = "local",
    ) -> str:
        run_id = str(uuid4())
        owner_id = _validated_owner_id(owner_id)
        capability_values = [item.value for item in capabilities]
        assets = [item.symbol for item in request.ordered_assets()]
        record = ResearchRunRecord(
            id=run_id,
            owner_id=owner_id,
            created_at=datetime.now(UTC),
            state="running",
            question=question.strip() or "Guided research",
            assets=assets,
            capabilities=capability_values,
            exchange=request.exchange,
            timeframe=request.timeframe,
            scope_key=_scope_key(assets, capability_values, request.exchange, request.timeframe),
            request_payload=request.model_dump(mode="json"),
            application_version=application_version(),
        )
        with self._sessions.begin() as session:
            session.add(record)
        return run_id

    def complete_run(
        self,
        run_id: str,
        report: ResearchReport,
        evidence: Sequence[EvidenceRecord] = (),
        *,
        owner_id: str = "local",
    ) -> None:
        owner_id = _validated_owner_id(owner_id)
        completed_at = datetime.now(UTC)
        snapshots = [_snapshot(run_id, item) for item in evidence]
        with self._sessions.begin() as session:
            record = session.scalar(
                select(ResearchRunRecord).where(
                    ResearchRunRecord.id == run_id,
                    ResearchRunRecord.owner_id == owner_id,
                )
            )
            if record is None:
                raise KeyError(f"Unknown research run: {run_id}")
            record.state = report.status
            record.completed_at = completed_at
            record.report_payload = report.model_dump(mode="json")
            record.failure = None
            session.add_all(snapshots)

    def fail_run(self, run_id: str, failure: str, *, owner_id: str = "local") -> None:
        owner_id = _validated_owner_id(owner_id)
        with self._sessions.begin() as session:
            record = session.scalar(
                select(ResearchRunRecord).where(
                    ResearchRunRecord.id == run_id,
                    ResearchRunRecord.owner_id == owner_id,
                )
            )
            if record is None:
                return
            record.state = "failed"
            record.completed_at = datetime.now(UTC)
            record.failure = " ".join(failure.split())[:500]

    def list_runs(
        self,
        *,
        asset: str | None = None,
        capability: str | None = None,
        state: str | None = None,
        limit: int = 100,
        owner_id: str = "local",
    ) -> list[ResearchRunSummary]:
        owner_id = _validated_owner_id(owner_id)
        evidence_counts = (
            select(
                EvidenceSnapshotRecord.run_id,
                func.count(EvidenceSnapshotRecord.id).label("evidence_count"),
            )
            .group_by(EvidenceSnapshotRecord.run_id)
            .subquery()
        )
        statement = (
            select(ResearchRunRecord, func.coalesce(evidence_counts.c.evidence_count, 0))
            .outerjoin(evidence_counts, evidence_counts.c.run_id == ResearchRunRecord.id)
            .where(ResearchRunRecord.owner_id == owner_id)
            .order_by(ResearchRunRecord.pinned.desc(), ResearchRunRecord.created_at.desc())
        )
        if state:
            statement = statement.where(ResearchRunRecord.state == state)
        with self._sessions() as session:
            rows = session.execute(statement).all()
        summaries = [_summary(record, int(count)) for record, count in rows]
        if asset:
            folded = asset.strip().casefold()
            summaries = [
                item
                for item in summaries
                if any(value.casefold() == folded for value in item.assets)
            ]
        if capability:
            summaries = [item for item in summaries if capability in item.capabilities]
        return summaries[: max(1, min(limit, 500))]

    def get_run(self, run_id: str, *, owner_id: str = "local") -> StoredResearchRun | None:
        owner_id = _validated_owner_id(owner_id)
        with self._sessions() as session:
            record = session.scalar(
                select(ResearchRunRecord).where(
                    ResearchRunRecord.id == run_id,
                    ResearchRunRecord.owner_id == owner_id,
                )
            )
            if record is None or record.report_payload is None:
                return None
            count = session.scalar(
                select(func.count(EvidenceSnapshotRecord.id)).where(
                    EvidenceSnapshotRecord.run_id == run_id
                )
            )
            payload = deserialize_report_payload(
                record.report_payload,
                version=record.payload_schema_version,
            )
            return StoredResearchRun(
                summary=_summary(record, int(count or 0)),
                report=ResearchReport.model_validate(payload),
            )

    def latest_run(self, *, owner_id: str) -> StoredResearchRun | None:
        """Return the newest completed report for one owner, independent of pin order."""

        owner_id = _validated_owner_id(owner_id)
        with self._sessions() as session:
            run_id = session.scalar(
                select(ResearchRunRecord.id)
                .where(
                    ResearchRunRecord.owner_id == owner_id,
                    ResearchRunRecord.report_payload.is_not(None),
                    ResearchRunRecord.state.in_(("complete", "partial")),
                )
                .order_by(
                    ResearchRunRecord.completed_at.desc(),
                    ResearchRunRecord.created_at.desc(),
                )
                .limit(1)
            )
        return self.get_run(run_id, owner_id=owner_id) if run_id is not None else None

    def compare(self, run_id: str, *, owner_id: str = "local") -> ResearchComparison | None:
        owner_id = _validated_owner_id(owner_id)
        current = self.get_run(run_id, owner_id=owner_id)
        if current is None:
            return None
        with self._sessions() as session:
            scope_key = session.scalar(
                select(ResearchRunRecord.scope_key).where(ResearchRunRecord.id == run_id)
            )
            prior_id = session.scalar(
                select(ResearchRunRecord.id)
                .where(
                    ResearchRunRecord.scope_key == scope_key,
                    ResearchRunRecord.owner_id == owner_id,
                    ResearchRunRecord.id != run_id,
                    ResearchRunRecord.report_payload.is_not(None),
                    ResearchRunRecord.created_at < current.summary.created_at,
                )
                .order_by(ResearchRunRecord.created_at.desc())
                .limit(1)
            )
        return ResearchComparison(
            current=current,
            previous=(self.get_run(prior_id, owner_id=owner_id) if prior_id is not None else None),
        )

    def pin(self, run_id: str, *, pinned: bool, owner_id: str = "local") -> bool:
        owner_id = _validated_owner_id(owner_id)
        with self._sessions.begin() as session:
            result = session.execute(
                update(ResearchRunRecord)
                .where(
                    ResearchRunRecord.id == run_id,
                    ResearchRunRecord.owner_id == owner_id,
                )
                .values(pinned=pinned)
            )
        return bool(getattr(result, "rowcount", 0))

    def delete(self, run_id: str, *, owner_id: str = "local") -> bool:
        owner_id = _validated_owner_id(owner_id)
        with self._sessions.begin() as session:
            result = session.execute(
                delete(ResearchRunRecord).where(
                    ResearchRunRecord.id == run_id,
                    ResearchRunRecord.owner_id == owner_id,
                )
            )
        return bool(getattr(result, "rowcount", 0))

    def delete_many(
        self,
        run_ids: Sequence[str],
        *,
        owner_id: str = "local",
    ) -> BulkDeleteResult:
        """Delete up to 100 owned unpinned runs in one transaction."""

        owner_id = _validated_owner_id(owner_id)
        normalized_ids = _validated_run_ids(run_ids)
        if not normalized_ids:
            return BulkDeleteResult(0, 0, 0)
        with self._sessions.begin() as session:
            rows = session.execute(
                select(ResearchRunRecord.id, ResearchRunRecord.pinned).where(
                    ResearchRunRecord.owner_id == owner_id,
                    ResearchRunRecord.id.in_(normalized_ids),
                )
            ).all()
            protected_count = sum(bool(pinned) for _, pinned in rows)
            deletable_ids = tuple(run_id for run_id, pinned in rows if not pinned)
            if not deletable_ids:
                return BulkDeleteResult(
                    requested_count=len(normalized_ids),
                    deleted_count=0,
                    protected_count=protected_count,
                )
            result = session.execute(
                delete(ResearchRunRecord).where(
                    ResearchRunRecord.owner_id == owner_id,
                    ResearchRunRecord.pinned.is_(False),
                    ResearchRunRecord.id.in_(deletable_ids),
                )
            )
        return BulkDeleteResult(
            requested_count=len(normalized_ids),
            deleted_count=int(getattr(result, "rowcount", 0) or 0),
            protected_count=protected_count,
        )

    def prune(
        self,
        *,
        retention_days: int = 365,
        now: datetime | None = None,
        owner_id: str = "local",
    ) -> int:
        owner_id = _validated_owner_id(owner_id)
        cutoff = _utc(now or datetime.now(UTC)) - timedelta(days=retention_days)
        with self._sessions.begin() as session:
            result = session.execute(
                delete(ResearchRunRecord).where(
                    ResearchRunRecord.owner_id == owner_id,
                    ResearchRunRecord.pinned.is_(False),
                    ResearchRunRecord.created_at < cutoff,
                )
            )
        return int(getattr(result, "rowcount", 0) or 0)

    def recover_interrupted_runs(self) -> int:
        """Mark runs left open by a prior process as interrupted."""

        with self._sessions.begin() as session:
            result = session.execute(
                update(ResearchRunRecord)
                .where(ResearchRunRecord.state == "running")
                .values(
                    state="failed",
                    completed_at=datetime.now(UTC),
                    failure="The application stopped before this research run completed.",
                )
            )
        return int(getattr(result, "rowcount", 0) or 0)

    # Structural implementation of services.cache.CacheBackend.
    def get_cache_entry(
        self,
        namespace: str,
        cache_key: str,
        *,
        allow_stale: bool = False,
    ) -> StoredCacheEntry | None:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            record = session.get(CacheEntryRecord, (namespace, cache_key))
            if record is None:
                return None
            expires = record.stale_until if allow_stale else record.fresh_until
            if _utc(expires) < now or record.payload_schema_version != PAYLOAD_SCHEMA_VERSION:
                session.delete(record)
                return None
            if payload_checksum(record.payload) != record.checksum:
                session.delete(record)
                return None
            return StoredCacheEntry(
                payload=record.payload,
                age_seconds=max((now - _utc(record.created_at)).total_seconds(), 0),
                fresh=_utc(record.fresh_until) >= now,
                negative=record.negative,
            )

    def set_cache_entry(
        self,
        namespace: str,
        cache_key: str,
        payload: Mapping[str, object],
        *,
        fresh_seconds: float,
        stale_seconds: float,
        negative: bool = False,
    ) -> None:
        now = datetime.now(UTC)
        normalized = dict(payload)
        record = CacheEntryRecord(
            namespace=namespace,
            cache_key=cache_key,
            payload=normalized,
            checksum=payload_checksum(normalized),
            payload_schema_version=PAYLOAD_SCHEMA_VERSION,
            created_at=now,
            fresh_until=now + timedelta(seconds=fresh_seconds),
            stale_until=now + timedelta(seconds=max(fresh_seconds, stale_seconds)),
            negative=negative,
        )
        with self._sessions.begin() as session:
            session.merge(record)

    def delete_cache_entry(self, namespace: str, cache_key: str) -> None:
        with self._sessions.begin() as session:
            session.execute(
                delete(CacheEntryRecord).where(
                    CacheEntryRecord.namespace == namespace,
                    CacheEntryRecord.cache_key == cache_key,
                )
            )

    def prune_cache(self, *, now: datetime | None = None) -> int:
        cutoff = _utc(now or datetime.now(UTC))
        with self._sessions.begin() as session:
            result = session.execute(
                delete(CacheEntryRecord).where(CacheEntryRecord.stale_until < cutoff)
            )
        return int(getattr(result, "rowcount", 0) or 0)

    def acquire_cache_lease(
        self,
        namespace: str,
        cache_key: str,
        *,
        ttl_seconds: float,
    ) -> str:
        del namespace, cache_key, ttl_seconds
        return "sql-local"

    def release_cache_lease(self, namespace: str, cache_key: str, token: str) -> None:
        del namespace, cache_key, token


class ScopedResearchRepository:
    """Tenant-safe research history facade bound to one internal user ID."""

    def __init__(self, repository: ResearchRepository, owner_id: str) -> None:
        self._repository = repository
        self.owner_id = _validated_owner_id(owner_id)

    def get_workspace(self) -> UserWorkspace | None:
        return self._repository.get_workspace(self.owner_id)

    def delete_workspace(self) -> bool:
        return self._repository.delete_workspace(self.owner_id)

    def create_run(
        self,
        *,
        request: AnalysisRequest,
        capabilities: Sequence[ResearchCapability],
        question: str,
    ) -> str:
        return self._repository.create_run(
            request=request,
            capabilities=capabilities,
            question=question,
            owner_id=self.owner_id,
        )

    def complete_run(
        self,
        run_id: str,
        report: ResearchReport,
        evidence: Sequence[EvidenceRecord] = (),
    ) -> None:
        self._repository.complete_run(
            run_id,
            report,
            evidence,
            owner_id=self.owner_id,
        )

    def fail_run(self, run_id: str, failure: str) -> None:
        self._repository.fail_run(run_id, failure, owner_id=self.owner_id)

    def list_runs(
        self,
        *,
        asset: str | None = None,
        capability: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> list[ResearchRunSummary]:
        return self._repository.list_runs(
            asset=asset,
            capability=capability,
            state=state,
            limit=limit,
            owner_id=self.owner_id,
        )

    def get_run(self, run_id: str) -> StoredResearchRun | None:
        return self._repository.get_run(run_id, owner_id=self.owner_id)

    def latest_run(self) -> StoredResearchRun | None:
        return self._repository.latest_run(owner_id=self.owner_id)

    def compare(self, run_id: str) -> ResearchComparison | None:
        return self._repository.compare(run_id, owner_id=self.owner_id)

    def pin(self, run_id: str, *, pinned: bool) -> bool:
        return self._repository.pin(run_id, pinned=pinned, owner_id=self.owner_id)

    def delete(self, run_id: str) -> bool:
        return self._repository.delete(run_id, owner_id=self.owner_id)

    def delete_many(self, run_ids: Sequence[str]) -> BulkDeleteResult:
        return self._repository.delete_many(run_ids, owner_id=self.owner_id)

    def prune(self, *, retention_days: int = 365, now: datetime | None = None) -> int:
        return self._repository.prune(
            retention_days=retention_days,
            now=now,
            owner_id=self.owner_id,
        )

    def prune_cache(self, *, now: datetime | None = None) -> int:
        return self._repository.prune_cache(now=now)


def create_repository(database_url: str) -> ResearchRepository:
    if not database_url.strip().casefold().startswith("sqlite"):
        raise ValueError("Research storage requires a SQLite DATABASE_URL.")
    _ensure_sqlite_parent(database_url)
    migrate_database(database_url)
    engine = create_engine(database_url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        _configure_sqlite(engine)
    return ResearchRepository(engine)


def migrate_database(database_url: str) -> None:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).with_name("migrations")))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    with _MIGRATION_LOCK:
        command.upgrade(config, "head")


def _configure_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def set_pragmas(dbapi_connection: Any, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def _ensure_sqlite_parent(database_url: str) -> None:
    prefix = "sqlite+pysqlite:///"
    if database_url.startswith(prefix) and ":memory:" not in database_url:
        database_path = Path(database_url.removeprefix(prefix)).expanduser()
        database_path.parent.mkdir(parents=True, exist_ok=True)


def _snapshot(run_id: str, evidence: EvidenceRecord) -> EvidenceSnapshotRecord:
    payload = evidence.model_dump(mode="json")
    return EvidenceSnapshotRecord(
        run_id=run_id,
        evidence_id=evidence.evidence_id,
        claim_type=evidence.claim_type,
        asset=evidence.asset,
        source=evidence.source,
        collected_at=evidence.collected_at,
        observed_at=evidence.observed_at,
        payload_hash=payload_checksum(payload),
        payload=payload,
    )


def _summary(record: ResearchRunRecord, evidence_count: int) -> ResearchRunSummary:
    return ResearchRunSummary(
        id=record.id,
        created_at=_utc(record.created_at),
        completed_at=_utc(record.completed_at) if record.completed_at else None,
        state=record.state,
        question=record.question,
        assets=tuple(record.assets),
        capabilities=tuple(record.capabilities),
        exchange=record.exchange,
        timeframe=record.timeframe,
        pinned=record.pinned,
        evidence_count=evidence_count,
    )


def _user_profile(record: UserRecord) -> UserProfile:
    return UserProfile(
        id=record.id,
        provider=record.provider,
        email=record.email,
        provider_name=record.provider_name,
        display_name=record.display_name,
        avatar_url=record.avatar_url,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        last_login_at=_utc(record.last_login_at),
    )


def _user_preferences(record: UserPreferenceRecord | None) -> UserPreferences:
    if record is None:
        return UserPreferences()
    return UserPreferences.model_validate(
        {
            "default_exchange": record.default_exchange,
            "default_timeframe": record.default_timeframe,
        }
    )


def _validated_owner_id(value: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 64
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("Owner ID must be 1-64 printable characters.")
    return normalized


def _validated_run_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        run_id = value.strip()
        if not run_id or len(run_id) > 64 or any(ord(character) < 32 for character in run_id):
            raise ValueError("Research run IDs must be 1-64 printable characters.")
        if run_id not in seen:
            seen.add(run_id)
            normalized.append(run_id)
    if len(normalized) > 100:
        raise ValueError("Bulk research deletion is limited to 100 runs.")
    return tuple(normalized)


def _scope_key(
    assets: Sequence[str],
    capabilities: Sequence[str],
    exchange: str | None,
    timeframe: str | None,
) -> str:
    payload = {
        "assets": sorted(item.casefold() for item in assets),
        "capabilities": sorted(capabilities),
        "exchange": exchange,
        "timeframe": timeframe,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "BulkDeleteResult",
    "ResearchComparison",
    "ResearchRepository",
    "ResearchRunSummary",
    "ScopedResearchRepository",
    "StoredResearchRun",
    "create_repository",
    "migrate_database",
]
