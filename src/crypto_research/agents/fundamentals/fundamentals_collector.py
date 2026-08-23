"""Fundamentals collection selection owned by the Fundamentals Agent."""

from collections.abc import Iterable

from crypto_research.domain.research import ResearchCapability


def requested_capabilities(
    values: Iterable[ResearchCapability | str] | None,
) -> set[ResearchCapability]:
    return {
        value if isinstance(value, ResearchCapability) else ResearchCapability(value)
        for value in values or ()
    }
