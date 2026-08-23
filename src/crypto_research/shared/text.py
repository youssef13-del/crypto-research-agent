"""Text normalization and lightweight request-size helpers."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from math import ceil

_MESSAGE_OVERHEAD_TOKENS = 12
_GENERATED_PREFIX = re.compile(
    r"^(?:verdict|analysis|summary|takeaway|overall(?: view)?|market(?: analysis| view)?|"
    r"risk(?: analysis| view)?|news(?: analysis| context)?|why it matters|comparison|model view)"
    r"\s*[:\-–—]\s*",
    flags=re.IGNORECASE,
)

PUBLISHER_PRIORITY = {
    "coindesk": 30,
    "cointelegraph": 28,
    "decrypt": 26,
    "googlenews": 18,
}
_HIGH_QUALITY_PUBLISHERS = frozenset(
    {
        "coindesk",
        "theblock",
        "cointelegraph",
        "decrypt",
        "bitcoinmagazine",
    }
)
_MEDIUM_QUALITY_PUBLISHERS = frozenset(
    {
        "reuters",
        "bloomberg",
        "cnbc",
        "forbes",
        "techcrunch",
        "theverge",
        "coingecko",
        "binance",
        "coinbase",
    }
)


def publisher_key(publisher: str) -> str:
    """Normalize a publisher name into a stable comparison key."""

    return re.sub(r"[^a-z0-9]+", "", publisher.casefold())


def publisher_quality(publisher: str) -> str:
    """Classify a publisher into a high, medium, or low quality tier."""

    key = publisher_key(publisher)
    if key in _HIGH_QUALITY_PUBLISHERS:
        return "high"
    if key in _MEDIUM_QUALITY_PUBLISHERS:
        return "medium"
    return "low"


def unique_strings(values: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving order and stripping whitespace."""

    normalized = (value.strip() for value in values)
    return list(dict.fromkeys(value for value in normalized if value))


def normalize_text(value: object) -> str | None:
    """Return a stripped string or ``None`` for an empty value."""

    normalized = value.strip() if isinstance(value, str) else ""
    return normalized or None


def clean_generated_text(
    value: str,
    *,
    max_chars: int,
    max_sentences: int | None = None,
    ensure_sentence: bool = False,
) -> str:
    """Normalize bounded model prose without cutting words."""

    clean = re.sub(r"(?m)^(?:```.*|#{1,6}\s*|[-*•]\s+|\d+[.)]\s+)", "", html.unescape(value))
    clean = re.sub(r"<[^>\n]{1,120}>", "", clean)
    clean = " ".join(clean.replace("\r", "\n").split())
    clean = _GENERATED_PREFIX.sub(
        "", clean.replace("**", "").replace("__", "").replace("`", "")
    ).strip()
    sentences: list[str] = []
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", clean):
        key = re.sub(r"[^a-z0-9]+", " ", sentence.casefold()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
        if max_sentences is not None and len(sentences) >= max_sentences:
            break
    clean = " ".join(sentences)
    if len(clean) > max_chars:
        shortened = clean[: max_chars - 1].rsplit(" ", maxsplit=1)[0].rstrip(",;:")
        clean = (shortened or clean[: max_chars - 1]).rstrip(".!?") + "."
    if ensure_sentence and clean:
        clean = clean.rstrip(" ,;:") + ("" if clean.endswith((".", "!", "?")) else ".")
    return clean


def estimate_tokens(*texts: str, output_tokens: int = 0) -> int:
    """Estimate complete request tokens conservatively from UTF-8 bytes."""

    encoded_bytes = sum(len(text.encode("utf-8")) for text in texts)
    return (
        max(1, ceil(encoded_bytes / 3))
        + _MESSAGE_OVERHEAD_TOKENS * len(texts)
        + max(output_tokens, 0)
    )
