"""Typed account and personal-workspace contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from crypto_research.domain.core import StrictModel, SupportedExchange, SupportedTimeframe


class UserIdentity(StrictModel):
    """Verified identity claims supplied by the configured OIDC provider."""

    provider: str = Field(min_length=1, max_length=40)
    issuer: str = Field(min_length=1, max_length=500)
    subject: str = Field(min_length=1, max_length=500)
    email: str = Field(min_length=3, max_length=320)
    email_verified: bool
    provider_name: str = Field(min_length=1, max_length=120)
    avatar_url: str | None = Field(default=None, max_length=2048)

    @field_validator("provider", "issuer", "subject", "email", "provider_name")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("Identity text must be printable.")
        return normalized

    @field_validator("email")
    @classmethod
    def validate_email_shape(cls, value: str) -> str:
        local, separator, domain = value.casefold().partition("@")
        if not separator or not local or not domain:
            raise ValueError("Identity email must be valid.")
        return value.casefold()

    @field_validator("email_verified")
    @classmethod
    def require_verified_email(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Identity email must be verified.")
        return value


class UserProfile(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    provider: str = Field(min_length=1, max_length=40)
    email: str = Field(min_length=3, max_length=320)
    provider_name: str = Field(min_length=1, max_length=120)
    display_name: str | None = Field(default=None, max_length=120)
    avatar_url: str | None = Field(default=None, max_length=2048)
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime

    @property
    def effective_name(self) -> str:
        return self.display_name or self.provider_name


class UserPreferences(StrictModel):
    default_exchange: SupportedExchange = "kraken"
    default_timeframe: SupportedTimeframe = "1h"


class WorkspaceStats(StrictModel):
    research_runs: int = Field(default=0, ge=0)
    pinned_runs: int = Field(default=0, ge=0)
    evidence_records: int = Field(default=0, ge=0)
    last_research_at: datetime | None = None


class UserWorkspace(StrictModel):
    profile: UserProfile
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    watchlist: tuple[str, ...] = ("BTC", "ETH", "SOL", "XRP")
    stats: WorkspaceStats = Field(default_factory=WorkspaceStats)


__all__ = [
    "UserIdentity",
    "UserPreferences",
    "UserProfile",
    "UserWorkspace",
    "WorkspaceStats",
]
