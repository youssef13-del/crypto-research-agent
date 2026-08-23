"""Bounded process cache with optional durable cache-aside storage."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Hashable, Mapping, Set
from dataclasses import dataclass
from threading import Lock
from time import monotonic, sleep
from typing import Literal, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CachePolicy:
    fresh_seconds: float
    stale_seconds: float
    max_entries: int = 256


CACHE_POLICIES: dict[str, CachePolicy] = {
    "market": CachePolicy(15, 3_600, 256),
    "asset_resolution": CachePolicy(86_400, 604_800, 256),
    "fundamentals": CachePolicy(1_800, 86_400, 256),
    "defi": CachePolicy(600, 7_200, 256),
    "derivatives": CachePolicy(300, 3_600, 256),
    "news": CachePolicy(300, 1_800, 128),
    "onchain": CachePolicy(21_600, 172_800, 64),
}


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    stale_hits: int = 0
    writes: int = 0
    evictions: int = 0
    failures: int = 0


class CacheEntry(Protocol):
    @property
    def payload(self) -> Mapping[str, object]: ...

    @property
    def age_seconds(self) -> float: ...

    @property
    def fresh(self) -> bool: ...

    @property
    def negative(self) -> bool: ...


class CacheBackend(Protocol):
    def get_cache_entry(
        self,
        namespace: str,
        cache_key: str,
        *,
        allow_stale: bool = False,
    ) -> CacheEntry | None: ...

    def set_cache_entry(
        self,
        namespace: str,
        cache_key: str,
        payload: Mapping[str, object],
        *,
        fresh_seconds: float,
        stale_seconds: float,
        negative: bool = False,
    ) -> None: ...

    def delete_cache_entry(self, namespace: str, cache_key: str) -> None: ...

    def prune_cache(self) -> int: ...

    def acquire_cache_lease(
        self,
        namespace: str,
        cache_key: str,
        *,
        ttl_seconds: float,
    ) -> str | None: ...

    def release_cache_lease(self, namespace: str, cache_key: str, token: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CacheLoadResult[ValueT]:
    value: ValueT | None
    source: Literal["memory", "persistent", "miss"]
    stale: bool = False
    age_seconds: float | None = None
    fallback_reason: str | None = None


_BACKEND: CacheBackend | None = None
_BACKEND_LOCK = Lock()


def configure_cache_backend(backend: CacheBackend | None) -> None:
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = backend


def canonical_cache_key(namespace: str, key: object) -> str:
    serialized = json.dumps(
        {"namespace": namespace, "key": key},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


class TTLCache[KeyT: Hashable, ValueT]:
    def __init__(
        self,
        ttl_seconds: float | None = None,
        *,
        clone: Callable[[ValueT], ValueT] | None = None,
        max_entries: int | None = None,
        namespace: str | None = None,
        serialize: Callable[[ValueT], Mapping[str, object]] | None = None,
        deserialize: Callable[[Mapping[str, object]], ValueT] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        policy = CACHE_POLICIES.get(namespace or "")
        effective_ttl = (
            ttl_seconds if ttl_seconds is not None else policy.fresh_seconds if policy else 0
        )
        effective_max = (
            max_entries if max_entries is not None else policy.max_entries if policy else 256
        )
        if effective_ttl <= 0:
            raise ValueError("ttl_seconds must be positive.")
        if effective_max <= 0:
            raise ValueError("max_entries must be positive.")
        self._ttl_seconds = effective_ttl
        self._stale_seconds = max(effective_ttl, policy.stale_seconds if policy else effective_ttl)
        self._clone = clone
        self._max_entries = effective_max
        self._namespace = namespace
        self._serialize = serialize
        self._deserialize = deserialize
        if clock is None:
            from time import monotonic

            clock = monotonic
        self._clock = clock
        self._items: dict[KeyT, tuple[float, ValueT]] = {}
        self._negative: dict[KeyT, float] = {}
        self._lock = Lock()
        self._loader_locks: dict[KeyT, Lock] = {}
        self._stats = CacheStats()

    def get(
        self,
        key: KeyT,
        *,
        ttl_seconds: float | None = None,
        allow_stale: bool = False,
    ) -> ValueT | None:
        return self.lookup(
            key,
            ttl_seconds=ttl_seconds,
            allow_stale=allow_stale,
        ).value

    def lookup(
        self,
        key: KeyT,
        *,
        ttl_seconds: float | None = None,
        allow_stale: bool = False,
    ) -> CacheLoadResult[ValueT]:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        ttl = self._ttl_seconds if ttl_seconds is None else ttl_seconds
        now = self._clock()
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                stored_at, value = cached
                maximum_age = self._stale_seconds if allow_stale else ttl
                if now - stored_at <= maximum_age:
                    self._increment("stale_hits" if now - stored_at > ttl else "hits")
                    return CacheLoadResult(
                        value=self._copy(value),
                        source="memory",
                        stale=now - stored_at > ttl,
                        age_seconds=max(now - stored_at, 0),
                    )
                self._items.pop(key, None)
            self._purge_expired(now)
        persistent = self._persistent_get(key, allow_stale=allow_stale)
        if persistent is not None:
            value, age_seconds, backend_fresh = persistent
            fresh = backend_fresh and age_seconds <= ttl
            if not fresh and not allow_stale:
                with self._lock:
                    self._increment("misses")
                return CacheLoadResult(value=None, source="miss")
            if age_seconds > self._stale_seconds:
                with self._lock:
                    self._increment("misses")
                return CacheLoadResult(value=None, source="miss")
            with self._lock:
                self._items[key] = (now - age_seconds, self._copy(value))
                self._increment("hits" if fresh else "stale_hits")
            return CacheLoadResult(
                value=self._copy(value),
                source="persistent",
                stale=not fresh,
                age_seconds=age_seconds,
            )
        with self._lock:
            self._increment("misses")
        return CacheLoadResult(value=None, source="miss")

    def __len__(self) -> int:
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            return len(self._items)

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return self._stats

    def set(self, key: KeyT, value: ValueT) -> None:
        stored_at = self._clock()
        stored_value = self._copy(value)
        with self._lock:
            self._purge_expired(stored_at)
            self._items.pop(key, None)
            self._items[key] = (stored_at, stored_value)
            self._negative.pop(key, None)
            self._increment("writes")
            while len(self._items) > self._max_entries:
                oldest_key = next(iter(self._items))
                self._items.pop(oldest_key, None)
                self._increment("evictions")
        self._persistent_set(key, value)

    def get_or_load(
        self,
        key: KeyT,
        loader: Callable[[], ValueT],
        *,
        cache_if: Callable[[ValueT], bool] | None = None,
    ) -> ValueT:
        cached = self.get(key)
        if cached is not None:
            return cached
        with self._lock:
            loader_lock = self._loader_locks.setdefault(key, Lock())
        try:
            with loader_lock:
                cached = self.get(key)
                if cached is not None:
                    return cached
                lease = self._acquire_persistent_lease(key)
                if lease is None and _BACKEND is not None and self._namespace:
                    deadline = monotonic() + 2
                    while monotonic() < deadline:
                        sleep(0.05)
                        cached = self.get(key)
                        if cached is not None:
                            return cached
                try:
                    value = loader()
                    if cache_if is None or cache_if(value):
                        self.set(key, value)
                    return self._copy(value)
                finally:
                    if lease is not None:
                        self._release_persistent_lease(key, lease)
        finally:
            with self._lock:
                self._loader_locks.pop(key, None)

    def remember_failure(self, key: KeyT, *, ttl_seconds: float = 60) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        with self._lock:
            self._negative[key] = self._clock() + ttl_seconds
            self._increment("failures")
        self._persistent_remember_failure(key, ttl_seconds=ttl_seconds)

    def failure_cached(self, key: KeyT) -> bool:
        with self._lock:
            expires = self._negative.get(key)
            if expires is None:
                return self._persistent_failure_cached(key)
            if expires <= self._clock():
                self._negative.pop(key, None)
                return False
            return True

    def _persistent_remember_failure(self, key: KeyT, *, ttl_seconds: float) -> None:
        backend = _BACKEND
        if backend is None or not self._namespace:
            return
        try:
            backend.set_cache_entry(
                self._namespace,
                self._persistent_failure_key(key),
                {},
                fresh_seconds=ttl_seconds,
                stale_seconds=ttl_seconds,
                negative=True,
            )
        except Exception:
            logger.warning("Persistent negative cache write failed", exc_info=True)

    def _persistent_failure_cached(self, key: KeyT) -> bool:
        backend = _BACKEND
        if backend is None or not self._namespace:
            return False
        try:
            entry = backend.get_cache_entry(
                self._namespace,
                self._persistent_failure_key(key),
                allow_stale=True,
            )
            return entry is not None and entry.negative
        except Exception:
            logger.warning("Persistent negative cache read failed", exc_info=True)
            return False

    def _acquire_persistent_lease(self, key: KeyT) -> str | None:
        backend = _BACKEND
        if backend is None or not self._namespace:
            return None
        try:
            return backend.acquire_cache_lease(
                self._namespace,
                canonical_cache_key(self._namespace, key),
                ttl_seconds=10,
            )
        except Exception:
            logger.warning("Persistent cache lease failed", exc_info=True)
            return None

    def _release_persistent_lease(self, key: KeyT, token: str) -> None:
        backend = _BACKEND
        if backend is None or not self._namespace:
            return
        try:
            backend.release_cache_lease(
                self._namespace,
                canonical_cache_key(self._namespace, key),
                token,
            )
        except Exception:
            logger.warning("Persistent cache lease release failed", exc_info=True)

    def _persistent_get(self, key: KeyT, *, allow_stale: bool) -> tuple[ValueT, float, bool] | None:
        backend = _BACKEND
        if backend is None or not self._namespace or self._deserialize is None:
            return None
        try:
            entry = backend.get_cache_entry(
                self._namespace,
                canonical_cache_key(self._namespace, key),
                allow_stale=allow_stale,
            )
            if entry is None or entry.negative:
                return None
            return self._deserialize(entry.payload), max(entry.age_seconds, 0), entry.fresh
        except Exception:
            logger.warning("Persistent cache read failed", exc_info=True)
            return None

    def _persistent_set(self, key: KeyT, value: ValueT) -> None:
        backend = _BACKEND
        if backend is None or not self._namespace or self._serialize is None:
            return
        try:
            backend.set_cache_entry(
                self._namespace,
                canonical_cache_key(self._namespace, key),
                self._serialize(value),
                fresh_seconds=self._ttl_seconds,
                stale_seconds=self._stale_seconds,
            )
            backend.delete_cache_entry(self._namespace, self._persistent_failure_key(key))
        except Exception:
            logger.warning("Persistent cache write failed", exc_info=True)

    def _persistent_failure_key(self, key: KeyT) -> str:
        assert self._namespace is not None
        return canonical_cache_key(self._namespace, ("failure", key))

    def _copy(self, value: ValueT) -> ValueT:
        return self._clone(value) if self._clone is not None else value

    def _purge_expired(self, now: float, *, exclude: Set[KeyT] | None = None) -> None:
        excluded: Set[KeyT] = exclude or frozenset()
        expired = [
            key
            for key, (stored_at, _) in self._items.items()
            if key not in excluded and now - stored_at > self._stale_seconds
        ]
        for key in expired:
            self._items.pop(key, None)
            self._increment("evictions")
        expired_failures = [key for key, expires in self._negative.items() if expires <= now]
        for key in expired_failures:
            self._negative.pop(key, None)

    def _increment(self, field: str) -> None:
        values = {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "stale_hits": self._stats.stale_hits,
            "writes": self._stats.writes,
            "evictions": self._stats.evictions,
            "failures": self._stats.failures,
        }
        values[field] += 1
        self._stats = CacheStats(**values)


__all__ = [
    "CACHE_POLICIES",
    "CacheBackend",
    "CacheEntry",
    "CacheLoadResult",
    "CachePolicy",
    "CacheStats",
    "TTLCache",
    "canonical_cache_key",
    "configure_cache_backend",
]
