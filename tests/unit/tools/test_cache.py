from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import sleep

from crypto_research.tools.cache import TTLCache, canonical_cache_key, configure_cache_backend


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    payload: Mapping[str, object]
    age_seconds: float = 0
    fresh: bool = True
    negative: bool = False


class MemoryBackend:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], MemoryEntry] = {}

    def get_cache_entry(
        self,
        namespace: str,
        cache_key: str,
        *,
        allow_stale: bool = False,
    ) -> MemoryEntry | None:
        del allow_stale
        return self.values.get((namespace, cache_key))

    def set_cache_entry(
        self,
        namespace: str,
        cache_key: str,
        payload: Mapping[str, object],
        *,
        fresh_seconds: float,
        stale_seconds: float,
        negative: bool = False,
    ) -> None:
        del fresh_seconds, stale_seconds
        self.values[(namespace, cache_key)] = MemoryEntry(
            payload=dict(payload),
            negative=negative,
        )

    def delete_cache_entry(self, namespace: str, cache_key: str) -> None:
        self.values.pop((namespace, cache_key), None)

    def prune_cache(self) -> int:
        return 0

    def acquire_cache_lease(
        self,
        namespace: str,
        cache_key: str,
        *,
        ttl_seconds: float,
    ) -> str:
        del namespace, cache_key, ttl_seconds
        return "memory"

    def release_cache_lease(self, namespace: str, cache_key: str, token: str) -> None:
        del namespace, cache_key, token


def test_named_cache_round_trips_through_persistent_backend() -> None:
    backend = MemoryBackend()
    configure_cache_backend(backend)
    first = TTLCache[str, int](
        30,
        namespace="test",
        serialize=lambda value: {"value": value},
        deserialize=lambda payload: _integer(payload["value"]),
    )
    second = TTLCache[str, int](
        30,
        namespace="test",
        serialize=lambda value: {"value": value},
        deserialize=lambda payload: _integer(payload["value"]),
    )

    first.set("btc", 42)

    assert second.get("btc") == 42
    assert second.stats.hits == 1
    assert canonical_cache_key("test", "btc") == canonical_cache_key("test", "btc")
    configure_cache_backend(None)


def test_single_flight_loads_one_value_for_concurrent_callers() -> None:
    configure_cache_backend(None)
    cache = TTLCache[str, int](30)
    calls = 0
    calls_lock = Lock()

    def loader() -> int:
        nonlocal calls
        with calls_lock:
            calls += 1
        sleep(0.02)
        return 7

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cache.get_or_load("btc", loader), range(8)))

    assert results == [7] * 8
    assert calls == 1
    assert cache.stats.hits >= 7


def test_persistent_hydration_preserves_stale_entry_age() -> None:
    backend = MemoryBackend()
    key = canonical_cache_key("onchain", "btc")
    backend.values[("onchain", key)] = MemoryEntry(
        payload={"value": 42},
        age_seconds=172_790,
        fresh=False,
    )
    now = [200_000.0]
    cache = TTLCache[str, int](
        30,
        namespace="onchain",
        deserialize=lambda payload: _integer(payload["value"]),
        clock=lambda: now[0],
    )
    configure_cache_backend(backend)

    assert cache.get("btc", allow_stale=True) == 42
    backend.values.clear()
    now[0] += 11

    assert cache.get("btc", allow_stale=True) is None
    configure_cache_backend(None)


def test_negative_cache_is_bounded() -> None:
    cache = TTLCache[str, int](30)
    cache.remember_failure("provider", ttl_seconds=0.01)

    assert cache.failure_cached("provider")
    sleep(0.02)
    assert not cache.failure_cached("provider")
    assert cache.stats.failures == 1


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Expected an integer cache payload.")
    return value
