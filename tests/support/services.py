from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import httpx

from crypto_research.domain.market import Candle


def candles_for_prices(prices: list[float], *, start: datetime | None = None) -> list[Candle]:
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Candle(
            timestamp=start + timedelta(hours=index),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1_000,
        )
        for index, price in enumerate(prices)
    ]


def ohlcv_row(timestamp: datetime, price: float) -> list[float]:
    return [timestamp.timestamp() * 1_000, price, price + 1, price - 1, price, 1_000]


def rss_entry(title: str, link: str, published: str) -> SimpleNamespace:
    safe_link = link if link.startswith(("http://", "https://")) else f"https://example.test/{link}"
    return SimpleNamespace(
        title=title,
        summary="Bitcoin market update",
        link=safe_link,
        published=published,
    )


class NoCallClient:
    def get(self, *_: object, **__: object) -> Any:
        raise AssertionError("RSS sources should not be fetched when max_items is zero")


class RSSClient:
    def __init__(self, responses: dict[str, str | Exception]) -> None:
        self._responses = responses

    def get(self, url: str) -> Any:
        response = self._responses[url]
        if isinstance(response, Exception):
            raise response
        return RSSResponse(response)


class RSSResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class HTTPResponse:
    def __init__(
        self,
        *,
        text: str = "",
        payload: object | None = None,
        status_code: int = 200,
    ) -> None:
        self.text = text
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(self.status_code),
            )


class MultiProviderClient:
    def __init__(self, responses: dict[str, HTTPResponse]) -> None:
        self._responses = responses

    def get(self, url: str, **_: object) -> HTTPResponse:
        return self._responses[url]


class FundamentalClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def get(self, *args: object, **kwargs: object) -> Any:
        del args, kwargs
        return FundamentalResponse(self._payload)


class FundamentalResponse:
    def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload


class CapturingFundamentalClient:
    def __init__(
        self,
        response: dict[str, object] | dict[str, FundamentalResponse],
    ) -> None:
        self._response = response
        self.urls: list[str] = []

    def get(self, url: str, **_: object) -> FundamentalResponse:
        self.urls.append(url)
        if all(isinstance(value, FundamentalResponse) for value in self._response.values()):
            return cast(dict[str, FundamentalResponse], self._response)[url]
        return FundamentalResponse(cast(dict[str, object], self._response))
