"""Tests for RedisSessionStore using fakeredis (no real Redis required)."""
import json
import time

import fakeredis
import pytest
from redis.asyncio import Redis

from app.storage.redis_session import RedisSessionStore

# Short values so tests stay readable — not the production defaults
_TTL = 10
_HARD_CAP = 50


@pytest.fixture
def redis_client() -> Redis:
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(redis_client: Redis) -> RedisSessionStore:
    return RedisSessionStore(redis=redis_client, ttl=_TTL, hard_cap=_HARD_CAP)


# ---------------------------------------------------------------------------
# Basic save / get
# ---------------------------------------------------------------------------


class TestSaveAndGet:
    async def test_round_trip(self, store: RedisSessionStore) -> None:
        await store.save("abc123", {"intent": "schedule", "step": 2})
        result = await store.get("abc123")
        assert result == {"intent": "schedule", "step": 2}

    async def test_nonexistent_returns_none(self, store: RedisSessionStore) -> None:
        assert await store.get("nope") is None

    async def test_internal_keys_not_exposed(self, store: RedisSessionStore) -> None:
        await store.save("abc123", {"x": 1})
        result = await store.get("abc123")
        assert result is not None
        assert "__created_at" not in result

    async def test_overwrite_updates_state(self, store: RedisSessionStore) -> None:
        await store.save("user1", {"step": "a"})
        await store.save("user1", {"step": "b", "extra": True})
        result = await store.get("user1")
        assert result == {"step": "b", "extra": True}


# ---------------------------------------------------------------------------
# Sliding TTL
# ---------------------------------------------------------------------------


class TestSlidingTTL:
    async def test_initial_ttl_equals_configured(
        self, store: RedisSessionStore, redis_client: Redis
    ) -> None:
        await store.save("p1", {"a": 1})
        ttl = await redis_client.ttl("session:p1")
        # fakeredis (and real Redis) truncates to integer seconds; allow ±1
        assert _TTL - 1 <= ttl <= _TTL

    async def test_get_renews_ttl(
        self, store: RedisSessionStore, redis_client: Redis
    ) -> None:
        await store.save("p2", {"a": 1})
        # Simulate TTL draining to near-zero
        await redis_client.expire("session:p2", 2)
        assert await redis_client.ttl("session:p2") == 2

        # get() must reset TTL back to the full window (±1s integer truncation)
        await store.get("p2")
        assert await redis_client.ttl("session:p2") >= _TTL - 1

    async def test_save_does_not_preserve_low_ttl(
        self, store: RedisSessionStore, redis_client: Redis
    ) -> None:
        await store.save("p3", {"a": 1})
        await redis_client.expire("session:p3", 1)

        # Saving again must restore full TTL (±1s integer truncation)
        await store.save("p3", {"a": 2})
        assert await redis_client.ttl("session:p3") >= _TTL - 1

    async def test_created_at_preserved_across_saves(
        self, store: RedisSessionStore, redis_client: Redis
    ) -> None:
        await store.save("p4", {"step": "a"})
        raw1 = await redis_client.get("session:p4")
        first_ts = json.loads(raw1)["__created_at"]

        await store.save("p4", {"step": "b"})
        raw2 = await redis_client.get("session:p4")
        second_ts = json.loads(raw2)["__created_at"]

        assert first_ts == second_ts
        assert json.loads(raw2)["step"] == "b"


# ---------------------------------------------------------------------------
# Hard cap
# ---------------------------------------------------------------------------


class TestHardCap:
    async def test_session_past_hard_cap_returns_none(
        self,
        store: RedisSessionStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = time.time()
        # Save as if session was created HARD_CAP + 1 seconds ago
        monkeypatch.setattr(
            "app.storage.redis_session._now", lambda: now - _HARD_CAP - 1
        )
        await store.save("old_user", {"data": "stale"})

        # Restore clock to now — hard cap diff exceeds limit
        monkeypatch.setattr("app.storage.redis_session._now", lambda: now)
        assert await store.get("old_user") is None

    async def test_session_past_hard_cap_removes_key(
        self,
        store: RedisSessionStore,
        redis_client: Redis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = time.time()
        monkeypatch.setattr(
            "app.storage.redis_session._now", lambda: now - _HARD_CAP - 1
        )
        await store.save("gone_user", {"x": 1})

        monkeypatch.setattr("app.storage.redis_session._now", lambda: now)
        await store.get("gone_user")

        assert await redis_client.exists("session:gone_user") == 0

    async def test_session_within_hard_cap_returns_data(
        self,
        store: RedisSessionStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = time.time()
        # Session created 5 seconds before the hard cap — still valid
        monkeypatch.setattr(
            "app.storage.redis_session._now", lambda: now - _HARD_CAP + 5
        )
        await store.save("fresh_user", {"data": "fresh"})

        monkeypatch.setattr("app.storage.redis_session._now", lambda: now)
        result = await store.get("fresh_user")
        assert result == {"data": "fresh"}

    async def test_hard_cap_boundary_exactly_at_limit_is_expired(
        self,
        store: RedisSessionStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = time.time()
        # Exactly at the hard cap boundary → expired (> not >=)
        monkeypatch.setattr(
            "app.storage.redis_session._now", lambda: now - _HARD_CAP
        )
        await store.save("boundary_user", {"d": 1})

        monkeypatch.setattr("app.storage.redis_session._now", lambda: now)
        # diff == _HARD_CAP is NOT > _HARD_CAP, so it's still valid
        result = await store.get("boundary_user")
        assert result == {"d": 1}


# ---------------------------------------------------------------------------
# Delete (LGPD erasure)
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_removes_session(self, store: RedisSessionStore) -> None:
        await store.save("to_delete", {"lgpd": True})
        await store.delete("to_delete")
        assert await store.get("to_delete") is None

    async def test_delete_nonexistent_is_noop(self, store: RedisSessionStore) -> None:
        # Must not raise even if key doesn't exist
        await store.delete("ghost_key")

    async def test_deleted_key_gone_from_redis(
        self, store: RedisSessionStore, redis_client: Redis
    ) -> None:
        await store.save("bye", {"x": 1})
        await store.delete("bye")
        assert await redis_client.exists("session:bye") == 0
