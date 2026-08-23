from __future__ import annotations

import re
from ipaddress import ip_address
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup

_ASSIGNED_SECRET = re.compile(
    r"""(?ix)
    (?P<key_quote>["']?)
    (?P<key>(?:[a-z0-9]+[_-])*(?:api[_-]?key|api[_-]?token|access[_-]?token|
        auth[_-]?token|refresh[_-]?token|authorization|client[_-]?secret|password|secret))
    (?P=key_quote)
    (?P<separator>\s*[:=]\s*)
    (?:
        "(?P<double_value>(?:\\.|[^"\\\r\n])*)"
        |'(?P<single_value>(?:\\.|[^'\\\r\n])*)'
        |(?P<bare_value>(?:bearer\s+)?[^\s,;&}\]"'<>]+)
    )
    """
)
_BEARER_SECRET = re.compile(
    r"""(?ix)\bbearer\s+(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^\s,;&}\]"'<>]+)"""
)
_MARKDOWN_CHARACTER = re.compile(r"([\\`*_[\]{}()#!|<>])")
_MAX_USER_QUESTION_CHARS = 2000
_ALTERNATE_NUMERIC_HOST = re.compile(r"(?i)^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*$")
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "accesstoken",
        "apikey",
        "auth",
        "authorization",
        "authtoken",
        "clientsecret",
        "password",
        "secret",
        "signature",
        "sig",
        "token",
    }
)
_SENSITIVE_QUERY_SUFFIXES = (
    "accesstoken",
    "apikey",
    "authorization",
    "authtoken",
    "clientsecret",
    "password",
    "secret",
    "signature",
    "token",
)


def clean_text(value: str, *, max_length: int = 1200) -> str:
    """Strip provider HTML and collapse unsafe or repeated whitespace."""

    text = BeautifulSoup(value or "", "html.parser").get_text(" ")
    collapsed = " ".join(text.replace("\x00", " ").split())
    return collapsed[:max_length]


def normalize_user_question(value: str, *, max_length: int = _MAX_USER_QUESTION_CHARS) -> str:
    """Collapse unsafe whitespace and bound a user question before it enters prompts."""

    normalized = " ".join((value or "").replace("\x00", " ").split())
    return normalized[:max_length]


def redact_secrets(value: str) -> str:
    """Redact common credential forms without exposing provider exception details."""

    def redact_assignment(match: re.Match[str]) -> str:
        matched_value = (
            match.group("double_value")
            or match.group("single_value")
            or match.group("bare_value")
            or ""
        )
        if "redacted" in matched_value.casefold():
            return match.group(0)
        quote_character = match.group("key_quote")
        value_quote = '"' if match.group("double_value") is not None else ""
        if match.group("single_value") is not None:
            value_quote = "'"
        return (
            f"{quote_character}{match.group('key')}{quote_character}"
            f"{match.group('separator')}{value_quote}[redacted]{value_quote}"
        )

    def redact_bearer(match: re.Match[str]) -> str:
        if "redacted" in match.group(0).casefold():
            return match.group(0)
        return "Bearer [redacted]"

    assigned = _ASSIGNED_SECRET.sub(redact_assignment, value)
    return _BEARER_SECRET.sub(redact_bearer, assigned)


def normalize_http_url(value: str | None) -> str | None:
    """Return a safe public HTTP(S) URL, or ``None`` for an unsafe value."""

    if value is None:
        return None
    candidate = value.strip()
    if not candidate or any(character.isspace() or ord(character) < 32 for character in candidate):
        return None
    if "\\" in candidate:
        return None

    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None

    raw_hostname = parsed.hostname.casefold().rstrip(".")
    if (
        not raw_hostname
        or "%" in raw_hostname
        or raw_hostname == "localhost"
        or raw_hostname.endswith((".localhost", ".local", ".internal"))
    ):
        return None
    try:
        address = ip_address(raw_hostname)
    except ValueError:
        if "." not in raw_hostname or _ALTERNATE_NUMERIC_HOST.fullmatch(raw_hostname):
            return None
        try:
            hostname = raw_hostname.encode("idna").decode("ascii")
        except UnicodeError:
            return None
    else:
        if not address.is_global:
            return None
        hostname = f"[{address.compressed}]" if address.version == 6 else address.compressed
    netloc = hostname if port is None else f"{hostname}:{port}"
    path = quote(parsed.path, safe="/%:@!$&'*,;=~+._-")
    safe_query = [
        (key, query_value)
        for key, query_value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_sensitive_query_key(key)
    ]
    query = urlencode(safe_query, doseq=True)
    return urlunsplit((parsed.scheme.lower(), netloc, path, query, ""))


def escape_markdown(value: str, *, preserve_paragraphs: bool = False) -> str:
    """Escape untrusted text used inside Markdown content or link labels.

    Paragraph breaks are collapsed by default; set ``preserve_paragraphs`` to
    keep blank-line-separated paragraphs intact for multi-line user content.
    """

    if preserve_paragraphs:
        normalized = "\n\n".join(
            " ".join(paragraph.split())
            for paragraph in re.split(r"\n\s*\n", value)
            if paragraph.strip()
        )
    else:
        normalized = " ".join(value.split())
    return _MARKDOWN_CHARACTER.sub(r"\\\1", normalized)


def _is_sensitive_query_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    return normalized in _SENSITIVE_QUERY_KEYS or normalized.endswith(_SENSITIVE_QUERY_SUFFIXES)
