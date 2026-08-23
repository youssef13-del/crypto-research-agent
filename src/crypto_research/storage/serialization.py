"""Versioned payload serialization and integrity helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

PAYLOAD_SCHEMA_VERSION = 2


def payload_checksum(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def deserialize_report_payload(payload: Mapping[str, object], *, version: int) -> dict[str, Any]:
    if version != PAYLOAD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported research payload schema: {version}")
    return {str(key): item for key, item in payload.items()}


def application_version() -> str:
    try:
        return version("crypto-research-agent")
    except PackageNotFoundError:
        return "0+unknown"
