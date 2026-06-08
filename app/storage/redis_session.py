"""Redis-backed primitives used by the agent.

This module exposes two unrelated capabilities that share a Redis client:

1. ``RedisSessionStore`` — sliding-TTL key/value store for conversation state.
   Used today by the LGPD route ``DELETE /leads/{phone}`` to evict the
   session on erasure requests. The agent graph itself still uses LangGraph's
   in-memory ``MemorySaver``; migrating the checkpointer to Redis is tracked
   as a roadmap item, but the store class is kept ready for that swap.

2. Message-level idempotency helpers (``is_message_processed`` /
   ``mark_message_processed``) — consumed by the WhatsApp webhook to dedupe
   Evolution API redeliveries by ``message_id``.
"""
import json
import logging
import time as _time_module
from typing import Any

from redis.asyncio import Redis

from app.config import get_settings
from app.storage.redis_resilient import make_resilient_redis

logger = logging.getLogger(__name__)

_SESSION_PREFIX = "session"
_CREATED_AT_KEY = "__created_at"

_redis_client: Redis | None = None


def _now() -> float:
    """Thin wrapper around time.time() — patchable in tests."""
    return _time_module.time()


def _get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        # Sprint 2.6.5 — resilient client (keepalive + health_check + retry).
        _redis_client = make_resilient_redis(get_settings().redis_url)
    return _redis_client


async def reset_redis_client() -> None:
    """Sprint 2.6.5 — discard the singleton so the next call rebuilds it.

    Called by the webhook's retry layer when the active client is dead
    (idle-kill / closed socket / "'NoneType' object is not callable" from
    a half-broken connection pool). Safe no-op when no client exists.
    """
    global _redis_client
    client = _redis_client
    _redis_client = None
    if client is not None:
        try:
            await client.aclose()
        except Exception as exc:
            logger.info("redis_client_close_failed (ignored): %s", exc)


class RedisSessionStore:
    """Manages LangGraph conversation state in Redis with sliding TTL and hard cap.

    Args:
        redis: Async Redis client (injected for testing via fakeredis).
        ttl:  Sliding window in seconds — reset on every read/write.
        hard_cap: Absolute maximum session lifetime in seconds.
    """

    def __init__(self, redis: Redis, ttl: int, hard_cap: int) -> None:
        self._redis = redis
        self._ttl = ttl
        self._hard_cap = hard_cap

    @staticmethod
    def _key(phone_hash: str) -> str:
        return f"{_SESSION_PREFIX}:{phone_hash}"

    async def get(self, phone_hash: str) -> dict[str, Any] | None:
        """Return session state or None if missing / hard-cap exceeded."""
        raw = await self._redis.get(self._key(phone_hash))
        if raw is None:
            return None

        envelope: dict[str, Any] = json.loads(raw)
        created_at: float = envelope.get(_CREATED_AT_KEY, 0.0)

        if _now() - created_at > self._hard_cap:
            await self._redis.delete(self._key(phone_hash))
            logger.info("session hard_cap exceeded, deleted (hash=%.8s)", phone_hash)
            return None

        # Renew sliding window
        await self._redis.expire(self._key(phone_hash), self._ttl)

        # Strip internal bookkeeping keys before returning
        return {k: v for k, v in envelope.items() if not k.startswith("__")}

    async def save(self, phone_hash: str, state: dict[str, Any]) -> None:
        """Persist state with sliding TTL, preserving original created_at."""
        key = self._key(phone_hash)
        existing_raw = await self._redis.get(key)

        if existing_raw is not None:
            existing: dict[str, Any] = json.loads(existing_raw)
            created_at = existing.get(_CREATED_AT_KEY, _now())
        else:
            created_at = _now()

        envelope = {_CREATED_AT_KEY: created_at, **state}
        await self._redis.setex(key, self._ttl, json.dumps(envelope))

    async def delete(self, phone_hash: str) -> None:
        """Remove session — called on LGPD erasure requests."""
        await self._redis.delete(self._key(phone_hash))
        logger.info("session deleted (hash=%.8s)", phone_hash)


def get_store() -> RedisSessionStore:
    """Return a RedisSessionStore bound to the singleton Redis client."""
    settings = get_settings()
    return RedisSessionStore(
        redis=_get_redis_client(),
        ttl=settings.session_ttl_seconds,
        hard_cap=settings.session_hard_cap_seconds,
    )


# ── Message-level idempotency ─────────────────────────────────────────────────

_MSG_PREFIX = "processed_msg"


async def is_message_processed(message_id: str) -> bool:
    """Return True if this Evolution message_id was already handled."""
    return await _get_redis_client().exists(f"{_MSG_PREFIX}:{message_id}") > 0


async def mark_message_processed(message_id: str, ttl: int = 86400) -> None:
    """Mark message_id as processed for ttl seconds (default 24 h)."""
    await _get_redis_client().setex(f"{_MSG_PREFIX}:{message_id}", ttl, "1")
