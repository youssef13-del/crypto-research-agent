"""News collection selection owned by the News Agent."""

from collections.abc import Iterable

from crypto_research.domain.research import ResearchCapability


def news_requested(values: Iterable[ResearchCapability | str] | None) -> bool:
    return ResearchCapability.NEWS in {
        value if isinstance(value, ResearchCapability) else ResearchCapability(value)
        for value in values or ()
    }
