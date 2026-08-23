from crypto_research.interfaces.web.components.research import _visible_metrics
from crypto_research.interfaces.web.presentation import CapabilityDataPresentation


def test_news_card_metrics_show_coverage_source_and_freshness() -> None:
    card = CapabilityDataPresentation(
        agent="news_agent",
        capability="news",
        title="BTC/USD news",
        asset="BTC/USD",
        facts=(
            ("Validated items", "4"),
            ("Latest publisher", "CoinDesk"),
            ("Latest title", "Bitcoin network update"),
            ("Published", "17 Aug 2026 12:30 UTC"),
            ("Freshness", "8 min ago"),
        ),
    )

    assert _visible_metrics("news_agent", {"news": card}) == (
        ("Validated items", "4"),
        ("Latest publisher", "CoinDesk"),
        ("Published", "17 Aug 2026 12:30 UTC"),
        ("Freshness", "8 min ago"),
    )
