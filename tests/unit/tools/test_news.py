from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
import pytest

from crypto_research.domain.research import NewsEvidence, NewsItem
from crypto_research.tools import news
from crypto_research.tools.cache import TTLCache
from crypto_research.tools.news import fetch_rss_items

NOW = datetime(2026, 8, 22, 12, 30, tzinfo=UTC)


def test_default_news_sources_keep_cointelegraph_and_generated_google_news() -> None:
    feed_urls = news._feed_urls(
        news.DEFAULT_RSS_URLS,
        asset_name="Bitcoin",
        search_terms={"bitcoin", "btc"},
    )

    assert news.DEFAULT_RSS_URLS == ("https://cointelegraph.com/rss",)
    assert feed_urls[0] == "https://cointelegraph.com/rss"
    assert any(url.startswith("https://news.google.com/rss/search?") for url in feed_urls)
    assert all("coindesk.com/arc/outboundfeeds/rss" not in url for url in feed_urls)


def test_rss_provider_boundary_rejects_entries_without_excerpts() -> None:
    feed = _rss(
        "Provider",
        """
        <item>
          <title>Bitcoin headline without supporting text</title>
          <link>https://publisher.test/headline-only</link>
          <pubDate>Sat, 22 Aug 2026 10:00:00 +0000</pubDate>
        </item>
        <item>
          <title>Bitcoin headline with supporting text</title>
          <description>Bitcoin evidence remains available for validation.</description>
          <link>https://publisher.test/usable</link>
          <pubDate>Sat, 22 Aug 2026 10:30:00 +0000</pubDate>
        </item>
        """,
    )

    with _client({"publisher.test": feed}) as client:
        items, warnings = fetch_rss_items(
            http=client,
            feed_urls=("https://publisher.test/rss",),
            search_patterns=(re.compile(r"\bbitcoin\b", re.IGNORECASE),),
            generic_query=False,
        )

    assert warnings == []
    assert [item.title for item in items] == ["Bitcoin headline with supporting text"]


def test_default_news_sources_normalize_deduplicate_and_keep_syndicated_coindesk() -> None:
    duplicate_title = "Bitcoin adoption expands across payment networks"
    cointelegraph = _rss(
        "Cointelegraph",
        f"""
        <item>
          <title>{duplicate_title}</title>
          <description>Bitcoin payment adoption continues across several networks.</description>
          <link>https://cointelegraph.com/news/payment-adoption</link>
          <pubDate>Sat, 22 Aug 2026 10:00:00 +0000</pubDate>
        </item>
        """,
    )
    google_news = _rss(
        "Bitcoin crypto - Google News",
        f"""
        <item>
          <title>CoinDesk examines Bitcoin payment adoption</title>
          <description>
            CoinDesk reporting supplies supporting context for the payment story.
          </description>
          <link>https://news.google.com/rss/articles/coindesk-story</link>
          <source url="https://www.coindesk.com/">CoinDesk</source>
          <pubDate>Sat, 22 Aug 2026 11:00:00 +0000</pubDate>
        </item>
        <item>
          <title>{duplicate_title}</title>
          <description>Reuters supplies another version of the adoption story.</description>
          <link>https://news.google.com/rss/articles/duplicate-story</link>
          <source url="https://www.reuters.com/">Reuters</source>
          <pubDate>Sat, 22 Aug 2026 10:30:00 +0000</pubDate>
        </item>
        <item>
          <title>Bitcoin miners publish a fresh operating update</title>
          <description>Reuters reports new operating evidence from listed miners.</description>
          <link>https://news.google.com/rss/articles/reuters-story</link>
          <source url="https://www.reuters.com/">Reuters</source>
          <pubDate>Sat, 22 Aug 2026 08:00:00 +0000</pubDate>
        </item>
        """,
    )

    with _client(
        {
            "cointelegraph.com": cointelegraph,
            "news.google.com": google_news,
        }
    ) as client:
        result = news.fetch_news_evidence(
            asset_name="Bitcoin",
            symbol="BTC/USD",
            max_items=10,
            client=client,
            collected_at=NOW,
        )

    assert result.source_state == "live"
    assert len(result.items) == 3
    assert {item.publisher for item in result.items} == {
        "CoinDesk",
        "Cointelegraph",
        "Reuters",
    }
    assert sum(item.title == duplicate_title for item in result.items) == 1
    assert next(item for item in result.items if item.title == duplicate_title).publisher == (
        "Cointelegraph"
    )


def test_news_cache_remains_a_safe_fallback_after_live_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = TTLCache[tuple[str, str, tuple[str, ...], int, tuple[str, ...]], NewsEvidence](
        300,
        clone=lambda result: result.model_copy(deep=True),
    )
    calls = 0

    def collect(**_kwargs: object) -> tuple[list[NewsItem], list[str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                NewsItem(
                    publisher="Cointelegraph",
                    title="Cache Token publishes a verified update",
                    excerpt="Cache Token evidence is available for the research record.",
                    url="https://cointelegraph.com/news/cache-token",
                    published_at=datetime(2026, 8, 22, 10, tzinfo=UTC),
                )
            ], []
        return [], ["RSS source unavailable (ReadTimeout)."]

    monkeypatch.setattr(news, "_NEWS_CACHE", cache)
    monkeypatch.setattr(news, "fetch_rss_items", collect)

    first = news.fetch_news_evidence(
        asset_name="Cache Token",
        symbol="CACHE/USD",
        collected_at=NOW,
    )
    second = news.fetch_news_evidence(
        asset_name="Cache Token",
        symbol="CACHE/USD",
        collected_at=NOW,
    )

    assert first.source_state == "live"
    assert second.source_state == "cached"
    assert second.items == first.items
    assert any("cached matching sources" in warning for warning in second.warnings)


def _rss(title: str, items: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>{title}</title>{items}</channel></rss>"""


def _client(feeds: dict[str, str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=feeds[request.url.host], request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))
