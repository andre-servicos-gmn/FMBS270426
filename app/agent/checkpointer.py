"""Singleton AsyncRedisSaver for LangGraph checkpoint persistence.

Conversations survive process restarts because checkpoints live in Redis,
keyed by ``thread_id`` (the customer's phone_hash). TTL is set to 7 days
so old sessions self-evict — after that the customer effectively starts
a fresh conversation.

Lifecycle:
    - ``init_checkpointer()`` is awaited once at app startup (FastAPI lifespan
      or before the first graph invocation in scripts).
    - ``get_checkpointer()`` returns the singleton — call sites compile the
      graph with it.
    - ``close_checkpointer()`` is awaited at shutdown for clean teardown.
"""
import logging

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config import get_settings

logger = logging.getLogger(__name__)

# Redis TTL is configured in MINUTES, not seconds. 7 days = 10080 minutes.
_TTL_MINUTES_7D = 60 * 24 * 7

_saver: AsyncRedisSaver | None = None


async def init_checkpointer() -> AsyncRedisSaver:
    """Initialize the singleton AsyncRedisSaver. Safe to call multiple times."""
    global _saver
    if _saver is not None:
        return _saver

    saver = AsyncRedisSaver(
        redis_url=get_settings().redis_url,
        ttl={"default_ttl": _TTL_MINUTES_7D, "refresh_on_read": True},
    )
    await saver.asetup()
    await saver.aset_client_info()
    _saver = saver
    logger.info("redis_checkpointer initialized ttl_minutes=%d", _TTL_MINUTES_7D)
    return _saver


def get_checkpointer() -> AsyncRedisSaver:
    """Return the initialized singleton. Raises if init was not awaited."""
    if _saver is None:
        raise RuntimeError(
            "Checkpointer not initialized — call init_checkpointer() at startup."
        )
    return _saver


async def close_checkpointer() -> None:
    """Close the Redis client at shutdown."""
    global _saver
    if _saver is None:
        return
    if getattr(_saver, "_owns_its_client", False):
        try:
            await _saver._redis.aclose()
        except Exception as exc:
            logger.warning("redis_checkpointer close failed: %s", exc)
    _saver = None
    logger.info("redis_checkpointer closed")
