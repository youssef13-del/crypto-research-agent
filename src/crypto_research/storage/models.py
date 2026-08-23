"""SQLAlchemy models for ChainScope's SQLite schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from crypto_research.storage.serialization import PAYLOAD_SCHEMA_VERSION


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(40))
    issuer: Mapped[str] = mapped_column(String(500))
    subject: Mapped[str] = mapped_column(String(500))
    email: Mapped[str] = mapped_column(String(320), index=True)
    email_verified: Mapped[bool] = mapped_column(Boolean)
    provider_name: Mapped[str] = mapped_column(String(120))
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("issuer", "subject", name="ux_users_oidc_identity"),)


class UserPreferenceRecord(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    default_exchange: Mapped[str] = mapped_column(String(40), default="kraken")
    default_timeframe: Mapped[str] = mapped_column(String(20), default="1h")


class WatchlistAssetRecord(Base):
    __tablename__ = "watchlist_assets"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    asset: Mapped[str] = mapped_column(String(16), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ResearchRunRecord(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(64), default="local", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    state: Mapped[str] = mapped_column(String(20), index=True)
    question: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(24), default="research")
    assets: Mapped[list[str]] = mapped_column(JSON, default=list)
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)
    exchange: Mapped[str | None] = mapped_column(String(40), nullable=True)
    timeframe: Mapped[str | None] = mapped_column(String(20), nullable=True)
    scope_key: Mapped[str] = mapped_column(String(64), index=True)
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    report_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    payload_schema_version: Mapped[int] = mapped_column(Integer, default=PAYLOAD_SCHEMA_VERSION)
    application_version: Mapped[str] = mapped_column(String(40))
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    failure: Mapped[str | None] = mapped_column(String(500), nullable=True)

    __table_args__ = (Index("ix_research_runs_scope_completed", "scope_key", "completed_at"),)


class EvidenceSnapshotRecord(Base):
    __tablename__ = "evidence_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("research_runs.id", ondelete="CASCADE"), index=True
    )
    evidence_id: Mapped[str] = mapped_column(String(240))
    claim_type: Mapped[str] = mapped_column(String(80), index=True)
    asset: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(240))
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)

    __table_args__ = (Index("ux_evidence_run_id", "run_id", "evidence_id", unique=True),)


class CacheEntryRecord(Base):
    __tablename__ = "cache_entries"

    namespace: Mapped[str] = mapped_column(String(80), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    checksum: Mapped[str] = mapped_column(String(64))
    payload_schema_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fresh_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    stale_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    negative: Mapped[bool] = mapped_column(Boolean, default=False)


@dataclass(frozen=True, slots=True)
class StoredCacheEntry:
    payload: Mapping[str, object]
    age_seconds: float
    fresh: bool
    negative: bool
