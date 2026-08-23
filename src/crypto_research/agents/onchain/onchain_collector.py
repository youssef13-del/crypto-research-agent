"""On-chain collection selection owned by the On-Chain Activity Agent."""

from crypto_research.domain.research import CollectionContext


def collection_kwargs(context: CollectionContext | None) -> dict[str, object]:
    return {} if context is None else {"collected_at": context.collected_at}
