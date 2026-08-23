import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus, urlsplit, urlunsplit

import httpx

from crypto_research.domain.evidence import MAX_EVIDENCE_EXCERPT_CHARS, normalize_news_items
from crypto_research.domain.research import (
    ASSET_ALIASES,
    COIN_ID_BY_ASSET,
    NewsEvidence,
    NewsItem,
)
from crypto_research.shared.security import clean_text, normalize_http_url
from crypto_research.shared.text import PUBLISHER_PRIORITY, publisher_key
from crypto_research.tools.cache import TTLCache
from crypto_research.tools.http import make_http_client

DEFAULT_RSS_URLS = ("https://cointelegraph.com/rss",)
GENERIC_CRYPTO_TERMS = {"crypto", "cryptocurrency", "cryptocurrencies", "digital assets"}
_GENERIC_NEWS_QUALIFIERS = {
    "asset",
    "assets",
    "broad",
    "ecosystem",
    "general",
    "important",
    "industry",
    "key",
    "major",
    "market",
    "markets",
    "narrative",
    "narratives",
    "overall",
    "sector",
    "theme",
    "themes",
    "trend",
    "trends",
}
_TOPIC_STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "any",
    "for",
    "from",
    "get",
    "give",
    "i",
    "is",
    "latest",
    "me",
    "news",
    "of",
    "on",
    "related",
    "show",
    "tell",
    "the",
    "to",
    "today",
    "updates",
    "what",
}

NEWS_ALIASES: dict[str, tuple[str, ...]] = {
    coin_id: ASSET_ALIASES[asset] for asset, coin_id in COIN_ID_BY_ASSET.items()
}
_NEWS_CACHE_TTL_SECONDS = 300.0
_NEWS_MAX_AGE = timedelta(days=7)
_NEWS_CACHE = TTLCache[tuple[str, str, tuple[str, ...], int, tuple[str, ...]], NewsEvidence](
    _NEWS_CACHE_TTL_SECONDS,
    clone=lambda result: result.model_copy(deep=True),
    namespace="news",
    serialize=lambda result: result.model_dump(mode="json"),
    deserialize=NewsEvidence.model_validate,
)
_NEWS_INTENT_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("sentiment", "bullish", "bearish", "tone"), ("sentiment", "bullish", "bearish")),
    (("catalyst", "catalysts"), ("catalyst", "upgrade", "launch", "approval")),
    (
        (
            "security",
            "secure",
            "exploit",
            "exploits",
            "hack",
            "hacks",
            "audit",
            "audits",
            "vulnerability",
            "vulnerabilities",
            "unaudited",
        ),
        ("security", "exploit", "hack", "audit", "vulnerability"),
    ),
    (("governance", "proposal", "vote", "voting"), ("governance", "proposal", "vote")),
    (
        ("regulation", "regulated", "legal", "tax", "taxed", "taxation"),
        ("regulation", "legal", "tax", "policy"),
    ),
    (("risk", "risks", "risky", "safe", "safety"), ("risk", "reserve", "disclosure")),
    (
        (
            "on-chain",
            "onchain",
            "whale",
            "whales",
            "holder",
            "holders",
            "inflow",
            "outflow",
            "wallet activity",
        ),
        ("whale", "holder", "inflow", "outflow", "on-chain"),
    ),
    (
        (
            "derivatives",
            "funding rate",
            "funding rates",
            "open interest",
            "liquidation",
            "liquidations",
            "futures",
            "perpetuals",
        ),
        ("funding rate", "open interest", "liquidation", "futures"),
    ),
)
_STRICT_NEWS_INTENT_MARKERS = (
    "security",
    "secure",
    "exploit",
    "exploits",
    "hack",
    "hacks",
    "audit",
    "audits",
    "vulnerability",
    "vulnerabilities",
    "unaudited",
    "governance",
    "proposal",
    "vote",
    "regulation",
    "regulated",
    "legal",
    "tax",
    "taxed",
    "taxation",
)


def fetch_news_evidence(
    *,
    asset_name: str,
    symbol: str,
    rss_urls: tuple[str, ...] = DEFAULT_RSS_URLS,
    max_items: int = 20,
    timeout_seconds: float = 20,
    query_context: str | None = None,
    client: httpx.Client | None = None,
    collected_at: datetime | None = None,
) -> NewsEvidence:
    intent_terms = _news_intent_terms(query_context)
    user_query = " ".join((asset_name, symbol, *intent_terms)).strip()
    use_cache = client is None
    collected_at = collected_at or datetime.now(UTC)
    if collected_at.tzinfo is None or collected_at.tzinfo.utcoffset(collected_at) is None:
        raise ValueError("collected_at must be timezone-aware.")
    collected_at = collected_at.astimezone(UTC)
    if max_items <= 0:
        return NewsEvidence(items=[], query=user_query, collected_at=collected_at, warnings=[])
    cache_key = (
        asset_name.strip().lower(),
        symbol.strip().upper(),
        rss_urls,
        max_items,
        intent_terms,
    )
    cached = _NEWS_CACHE.get(cache_key, allow_stale=True) if use_cache else None

    owns_client = client is None
    http = client or make_http_client(timeout_seconds)
    try:
        items_by_key: dict[str, NewsItem] = {}
        warnings: list[str] = []
        search_terms = _search_terms(asset_name=asset_name, symbol=symbol)
        generic_query = bool(search_terms & GENERIC_CRYPTO_TERMS)
        feed_urls = _feed_urls(
            rss_urls,
            asset_name=asset_name,
            search_terms=search_terms,
            intent_terms=intent_terms,
        )
        search_patterns = tuple(
            re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in search_terms
        )
        intent_patterns = tuple(
            re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in intent_terms
        )
        rss_items, rss_warnings = fetch_rss_items(
            http=http,
            feed_urls=feed_urls,
            search_patterns=search_patterns,
            generic_query=generic_query,
        )
        warnings.extend(rss_warnings)
        recent_rss_items, excluded_future = _recent_items(rss_items, collected_at=collected_at)
        _merge_items(items_by_key, recent_rss_items)
        _future_data_warning(warnings, excluded_future, source="RSS")

        if (
            not items_by_key
            and not generic_query
            and not any(
                term
                in NEWS_ALIASES.get(COIN_ID_BY_ASSET.get(symbol.split("/", maxsplit=1)[0], ""), ())
                for term in _search_terms(asset_name=asset_name, symbol=symbol)
            )
        ):
            loose_terms = {
                term
                for term in (
                    asset_name.strip().lower(),
                    symbol.split("/", maxsplit=1)[0].lower(),
                )
                if term and term not in GENERIC_CRYPTO_TERMS
            }
            if loose_terms:
                loose_patterns = tuple(
                    re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in loose_terms
                )
                loose_items, _ = fetch_rss_items(
                    http=http,
                    feed_urls=feed_urls,
                    search_patterns=loose_patterns,
                    generic_query=False,
                )
                recent_loose_items, excluded_future = _recent_items(
                    loose_items,
                    collected_at=collected_at,
                )
                _merge_items(items_by_key, recent_loose_items)
                _future_data_warning(warnings, excluded_future, source="RSS fallback")

        if _requires_intent_match(query_context):
            intent_matched = {
                key: item
                for key, item in items_by_key.items()
                if _matches_intent(item, intent_patterns)
            }
            if intent_matched:
                items_by_key = intent_matched
            else:
                warnings.append(
                    "No strictly matching coverage was found; "
                    "the closest general coverage is shown."
                )
        items = sorted(
            items_by_key.values(),
            key=lambda item: _rank_item(
                item,
                search_patterns=search_patterns,
                intent_patterns=intent_patterns,
                generic_query=generic_query,
                collected_at=collected_at,
            ),
            reverse=True,
        )
        items, normalization_warnings = normalize_news_items(items, collected_at=collected_at)
        warnings.extend(normalization_warnings)
        items = _dedupe_near_duplicates(items)
        items = _round_robin_stories(items)
        if not items:
            warnings.append("No recent matching crypto news items were found across providers.")
        result = NewsEvidence(
            items=items[:max_items],
            query=user_query,
            collected_at=collected_at,
            source_state="live",
            warnings=warnings,
        )
        if use_cache and result.items:
            _NEWS_CACHE.set(cache_key, result)
        if (
            not result.items
            and cached is not None
            and _news_cache_is_current(cached, collected_at=collected_at)
            and _news_live_verification_failed(result)
        ):
            return _cached_news_evidence(cached)
        return result
    except Exception:
        if cached is not None and _news_cache_is_current(cached, collected_at=collected_at):
            return _cached_news_evidence(cached)
        raise
    finally:
        if owns_client:
            with suppress(Exception):
                http.close()


def _news_cache_is_current(
    evidence: NewsEvidence,
    *,
    collected_at: datetime,
) -> bool:
    """Accept a fresh cache entry only when it predates this run's cutoff."""

    return evidence.collected_at <= collected_at and all(
        item.published_at <= collected_at for item in evidence.items
    )


def _news_live_verification_failed(evidence: NewsEvidence) -> bool:
    return any(
        marker in warning.casefold()
        for warning in evidence.warnings
        for marker in ("unavailable", "failed", "request", "http ")
    )


def _cached_news_evidence(evidence: NewsEvidence) -> NewsEvidence:
    return evidence.model_copy(
        update={
            "source_state": "cached",
            "warnings": list(
                dict.fromkeys(
                    (
                        *evidence.warnings,
                        "Live news verification failed; fresh cached matching sources are shown.",
                    )
                )
            ),
        }
    )


def _merge_items(items_by_key: dict[str, NewsItem], items: list[NewsItem]) -> None:
    for item in items:
        key = _news_item_key(item)
        prior_item = items_by_key.get(key)
        if prior_item is None or item.published_at > prior_item.published_at:
            items_by_key[key] = item


def _search_terms(*, asset_name: str, symbol: str) -> set[str]:
    normalized_asset = asset_name.strip().lower()
    base_symbol = symbol.split("/", maxsplit=1)[0].strip().lower()
    if normalized_asset in GENERIC_CRYPTO_TERMS or base_symbol in {"", "crypto"}:
        return set(GENERIC_CRYPTO_TERMS)

    terms = {term for term in (normalized_asset, base_symbol) if term}
    terms.update(NEWS_ALIASES.get(normalized_asset, ()))
    for coin_id, aliases in NEWS_ALIASES.items():
        if base_symbol in aliases:
            terms.add(coin_id)
            terms.update(aliases)
    return terms


def _feed_urls(
    rss_urls: tuple[str, ...],
    *,
    asset_name: str,
    search_terms: set[str],
    intent_terms: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if rss_urls != DEFAULT_RSS_URLS:
        return rss_urls
    base_query = (
        "crypto cryptocurrency" if search_terms & GENERIC_CRYPTO_TERMS else f"{asset_name} crypto"
    )
    query = f"{base_query} {_intent_query(intent_terms)}".strip()
    google_news = (
        f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    return (*rss_urls, google_news)


def _rank_item(
    item: NewsItem,
    *,
    search_patterns: tuple[re.Pattern[str], ...],
    intent_patterns: tuple[re.Pattern[str], ...] = (),
    generic_query: bool,
    collected_at: datetime,
) -> tuple[int, int, float, float]:
    title = item.title.lower()
    excerpt = item.excerpt.lower()
    title_hits = sum(1 for pattern in search_patterns if pattern.search(title))
    excerpt_hits = sum(1 for pattern in search_patterns if pattern.search(excerpt))
    intent_hits = sum(
        3 if pattern.search(title) else 1 if pattern.search(excerpt) else 0
        for pattern in intent_patterns
    )
    relevance = (1 if generic_query else title_hits * 5 + excerpt_hits * 2) + intent_hits * 4
    publisher_score = PUBLISHER_PRIORITY.get(publisher_key(item.publisher), 0)
    age_hours = max((collected_at - item.published_at).total_seconds() / 3600, 0.0)
    freshness = 2 ** (-age_hours / _NEWS_RECENCY_HALF_LIFE_HOURS)
    return relevance, publisher_score, freshness, item.published_at.timestamp()


def _news_intent_terms(query_context: str | None) -> tuple[str, ...]:
    lowered = clean_text(query_context or "", max_length=400).casefold()
    selected: list[str] = []
    for markers, terms in _NEWS_INTENT_GROUPS:
        if any(_contains_news_term(lowered, marker) for marker in markers):
            selected.extend(terms)
    return tuple(dict.fromkeys(selected))


def _contains_news_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE) is not None


def _requires_intent_match(query_context: str | None) -> bool:
    lowered = (query_context or "").casefold()
    return any(_contains_news_term(lowered, term) for term in _STRICT_NEWS_INTENT_MARKERS)


def _matches_intent(
    item: NewsItem,
    patterns: tuple[re.Pattern[str], ...],
) -> bool:
    content = f"{item.title} {item.excerpt}"
    return any(pattern.search(content) for pattern in patterns)


def _intent_query(intent_terms: tuple[str, ...]) -> str:
    if not intent_terms:
        return ""
    return "(" + " OR ".join(f'"{term}"' for term in intent_terms) + ")"


def _news_item_key(item: NewsItem) -> str:
    if item.url:
        parsed = urlsplit(item.url)
        return urlunsplit(
            (
                parsed.scheme.casefold(),
                parsed.netloc.casefold(),
                parsed.path.rstrip("/"),
                "",
                "",
            )
        )
    return re.sub(r"[^a-z0-9]+", " ", item.title.casefold()).strip()


def _dedupe_near_duplicates(items: list[NewsItem]) -> list[NewsItem]:
    """Keep only the highest-ranked title of each near-duplicate cluster.

    Syndicated coverage often reaches different URLs with equivalent headlines; collapsing
    identical normalized word sets keeps the top-N records diverse and cuts redundant tokens.
    """

    seen: set[str] = set()
    kept: list[NewsItem] = []
    for item in items:
        signature = _title_signature(item.title)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(item)
    return kept


def _title_signature(title: str) -> str:
    words = [word for word in re.findall(r"[a-z0-9]+", title.casefold()) if len(word) >= 4]
    return " ".join(sorted(set(words)))


_STORY_CLUSTER_JACCARD = 0.4
_NEWS_RECENCY_HALF_LIFE_HOURS = 24.0


def _title_tokens(title: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", title.casefold()) if len(word) >= 4}


def _token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _round_robin_stories(items: list[NewsItem]) -> list[NewsItem]:
    """Interleave ranked items across story clusters so distinct stories lead.

    Syndicated coverage of the same event reaches different URLs with reworded
    headlines. Grouping items whose headline tokens overlap into story clusters
    and round-robining across those clusters keeps the top-N records diverse
    while preserving the overall ranking order within each cluster.
    """

    clusters: list[list[NewsItem]] = []
    cluster_tokens: list[set[str]] = []
    for item in items:
        tokens = _title_tokens(item.title)
        for index, signature in enumerate(cluster_tokens):
            if _token_jaccard(tokens, signature) >= _STORY_CLUSTER_JACCARD:
                clusters[index].append(item)
                cluster_tokens[index] |= tokens
                break
        else:
            clusters.append([item])
            cluster_tokens.append(tokens)

    selected: list[NewsItem] = []
    cursor = 0
    while len(selected) < len(items):
        progressed = False
        for cluster in clusters:
            if cursor < len(cluster):
                selected.append(cluster[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return selected


def _is_recent(published_at: datetime, *, collected_at: datetime) -> bool:
    return collected_at - _NEWS_MAX_AGE <= published_at <= collected_at


def _recent_items(
    items: list[NewsItem],
    *,
    collected_at: datetime,
) -> tuple[list[NewsItem], int]:
    future_count = sum(item.published_at > collected_at for item in items)
    return (
        [item for item in items if _is_recent(item.published_at, collected_at=collected_at)],
        future_count,
    )


def _future_data_warning(warnings: list[str], count: int, *, source: str) -> None:
    if count:
        plural = "item was" if count == 1 else "items were"
        warnings.append(f"{count} future-dated news {plural} excluded from {source} data.")


def _extract_news_topic(user_intent: str) -> str | None:
    lowered = user_intent.lower()
    if not any(term in lowered for term in ("news", "update", "updates")):
        return None
    topic_words = [
        word for word in re.findall(r"[a-z0-9]+", lowered) if word not in _TOPIC_STOPWORDS
    ]
    if any(term in lowered for term in GENERIC_CRYPTO_TERMS):
        topic_words = [word for word in topic_words if word not in GENERIC_CRYPTO_TERMS]
    return " ".join(topic_words) if topic_words else "crypto"


def resolve_news_query(
    request_coin_id: str | None,
    symbol: str,
    user_intent: str,
    research_topic: str | None = None,
) -> tuple[str, str]:
    if request_coin_id is not None:
        return request_coin_id, symbol

    topic = (research_topic or "").strip().lower() or _extract_news_topic(user_intent)
    if topic is None:
        return "crypto", "CRYPTO/USD"
    if _is_generic_news_topic(topic, user_intent=user_intent):
        return "crypto", "CRYPTO/USD"
    slug = re.sub(r"[^a-z0-9]+", "-", topic).strip("-").upper() or "CRYPTO"
    return topic, f"{slug}/USD"


def _is_generic_news_topic(topic: str, *, user_intent: str) -> bool:
    """Normalize broad router phrases such as ``crypto market themes``."""

    topic_tokens = set(re.findall(r"[a-z0-9]+", topic.casefold()))
    generic_tokens = {
        token for phrase in GENERIC_CRYPTO_TERMS for token in re.findall(r"[a-z0-9]+", phrase)
    }
    ignored = set(_TOPIC_STOPWORDS) | _GENERIC_NEWS_QUALIFIERS | generic_tokens
    has_crypto_context = bool(topic_tokens & generic_tokens) or any(
        term in user_intent.casefold() for term in GENERIC_CRYPTO_TERMS
    )
    return has_crypto_context and not (topic_tokens - ignored)


def fetch_rss_items(
    *,
    http: httpx.Client,
    feed_urls: tuple[str, ...],
    search_patterns: tuple[re.Pattern[str], ...],
    generic_query: bool,
) -> tuple[list[NewsItem], list[str]]:
    """Collect dated, relevant articles from RSS providers."""

    items: list[NewsItem] = []
    warnings: list[str] = []
    for url in feed_urls:
        try:
            response = http.get(url)
            response.raise_for_status()
            parsed = _parse_feed(response.text)
        except Exception as exc:
            warnings.append(f"RSS source unavailable ({type(exc).__name__}).")
            continue
        for entry in parsed.entries:
            title = clean_text(str(getattr(entry, "title", "")), max_length=300)
            excerpt = clean_text(
                str(getattr(entry, "summary", "")),
                max_length=MAX_EVIDENCE_EXCERPT_CHARS,
            )
            if not title or not excerpt:
                continue
            haystack = f"{title} {excerpt} {_entry_tags(entry)}".lower()
            if not generic_query and not any(
                pattern.search(haystack) for pattern in search_patterns
            ):
                continue
            published_at = _rss_published_at(entry)
            if published_at is None:
                continue
            feed_title = str(getattr(parsed.feed, "title", "unknown"))
            items.append(
                NewsItem(
                    publisher=_entry_publisher(entry, feed_title),
                    title=title,
                    excerpt=excerpt,
                    url=normalize_http_url(str(getattr(entry, "link", ""))),
                    published_at=published_at,
                )
            )
    return items, warnings


def _entry_publisher(entry: object, feed_title: str) -> str:
    source = getattr(entry, "source", None)
    source_title = clean_text(str(getattr(source, "title", "")), max_length=120)
    return source_title or clean_text(feed_title, max_length=120) or "unknown"


def _rss_published_at(entry: object) -> datetime | None:
    raw = str(getattr(entry, "published", "") or getattr(entry, "updated", ""))
    try:
        parsed = parsedate_to_datetime(raw)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except TypeError, ValueError, OverflowError:
        return None


def _entry_tags(entry: object) -> str:
    raw_tags = getattr(entry, "tags", [])
    if not isinstance(raw_tags, list):
        return ""
    return " ".join(
        term for item in raw_tags if isinstance(term := getattr(item, "term", None), str)
    )


def _parse_feed(value: str) -> Any:
    import feedparser  # type: ignore[import-untyped]

    return feedparser.parse(value)
