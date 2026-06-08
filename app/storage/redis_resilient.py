"""Sprint 2.6.5 — resilient Redis client factory.

Centralizes the connection kwargs every ``Redis.from_url`` call should use
so the agent survives the two real-world failure modes seen in production
on free-tier hosting:

1. **Idle-kill** — the server closes a connection that's been idle for
   ~4 min. Fix: ``health_check_interval=30`` (redis-py sends PING on the
   idle connection every 30 s, preventing the close).
2. **Transient blip** — single TCP packet loss / server hiccup. Fix:
   ``retry`` with exponential backoff, ``retry_on_error`` covering
   ``ConnectionError`` and ``TimeoutError``.

Used by every direct client creation in the app. The LangGraph
``AsyncRedisSaver`` checkpointer also takes ``connection_args=`` derived
from ``resilient_connection_kwargs()``.
"""
from __future__ import annotations

from typing import Any

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError


def resilient_connection_kwargs() -> dict[str, Any]:
    """Return the kwargs every Redis client in the app should be created with."""
    return {
        # Keep TCP socket alive across idle periods (free-tier hosts kill
        # silent connections after a few minutes).
        "socket_keepalive": True,
        # PING the server every 30 s when the connection is otherwise idle.
        # This is the SINGLE most effective mitigation for the idle-kill
        # symptom observed in production.
        "health_check_interval": 30,
        # Retry transient errors automatically with exponential backoff.
        "retry_on_timeout": True,
        "retry_on_error": [RedisConnectionError, RedisTimeoutError],
        "retry": Retry(ExponentialBackoff(cap=10, base=1), retries=3),
    }


def make_resilient_redis(url: str, *, decode_responses: bool = True) -> Redis:
    """Create a Redis client with idle-kill protection + auto-retry."""
    return Redis.from_url(
        url,
        decode_responses=decode_responses,
        **resilient_connection_kwargs(),
    )
