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
    """Initialize the singleton AsyncRedisSaver. Safe to call multiple times.

    Sprint 2.6.5 — the AsyncRedisSaver constructor accepts ``connection_args``
    that are forwarded to the internal Redis client. We pass the same
    resilient kwargs used everywhere else (keepalive + health_check +
    retry) so the checkpointer survives idle-kill on free-tier Redis.
    """
    global _saver
    if _saver is not None:
        return _saver

    # Local import keeps app/storage out of import cycle at module load.
    from app.storage.redis_resilient import resilient_connection_kwargs

    saver = AsyncRedisSaver(
        redis_url=get_settings().redis_url,
        connection_args=resilient_connection_kwargs(),
        ttl={"default_ttl": _TTL_MINUTES_7D, "refresh_on_read": True},
    )
    await saver.asetup()
    await saver.aset_client_info()
    _saver = saver
    logger.info(
        "redis_checkpointer initialized ttl_minutes=%d (resilient client)",
        _TTL_MINUTES_7D,
    )
    return _saver


def get_checkpointer() -> AsyncRedisSaver:
    """Return the initialized singleton. Raises if init was not awaited."""
    if _saver is None:
        raise RuntimeError(
            "Checkpointer not initialized — call init_checkpointer() at startup."
        )
    return _saver


async def reset_checkpointer() -> None:
    """Sprint 2.6.5 — tear down + recreate the checkpointer singleton.

    Called by the webhook's retry layer when the graph's Redis-backed
    checkpointer is broken (the agent caught the "'NoneType' object is
    not callable" symptom we saw in production after idle-kill). After
    this, a subsequent ``get_checkpointer()`` would fail; the caller is
    expected to also call ``init_checkpointer()`` to rebuild.
    """
    global _saver
    saver = _saver
    _saver = None
    if saver is not None:
        if getattr(saver, "_owns_its_client", False):
            try:
                await saver._redis.aclose()
            except Exception as exc:
                logger.info("redis_checkpointer close failed (ignored): %s", exc)
    logger.info("redis_checkpointer reset (next request rebuilds)")


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
