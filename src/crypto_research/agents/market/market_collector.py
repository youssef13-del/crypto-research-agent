"""Market collection selection owned by the Market & Risk Agent."""

from collections.abc import Iterable

from crypto_research.domain.research import ResearchCapability


def selected_capabilities(
    values: Iterable[ResearchCapability | str] | None,
) -> set[ResearchCapability]:
    return {
        value if isinstance(value, ResearchCapability) else ResearchCapability(value)
        for value in values or ()
    }
