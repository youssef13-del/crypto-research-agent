"""Shared HTTP client construction for external providers."""

from collections.abc import Mapping
from datetime import UTC, datetime

import httpx


def make_http_client(
    timeout_seconds: float,
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx.Client:
    return httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers=headers,
    )


def normalize_collection_time(value: datetime | None) -> datetime:
    collected_at = value or datetime.now(UTC)
    if collected_at.tzinfo is None or collected_at.tzinfo.utcoffset(collected_at) is None:
        raise ValueError("collected_at must be timezone-aware.")
    return collected_at.astimezone(UTC)
