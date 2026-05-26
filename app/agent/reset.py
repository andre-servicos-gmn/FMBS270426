"""⚠️ DEV/PILOT ONLY — REMOVE BEFORE PRODUCTION ⚠️

This module powers the ``/reset`` magic command on WhatsApp during manual
piloting. A customer types ``/reset`` and we wipe every Redis key tied to
their phone_hash (LangGraph checkpoints, blobs, RedisVL index entries,
the legacy RedisSessionStore key). The result is a clean conversation: the
next message starts a fresh thread.

This exists ONLY to make hands-on QA loops painless. End customers must
never see this. Before deploying to a real franchise:

    1. DELETE this file (app/agent/reset.py)
    2. REMOVE the `/reset` branch in app/api/webhook.py — search for the
       grep marker ``DEV_RESET_HOOK`` to find every site.
    3. DROP the corresponding tests (tests/test_reset.py).

Grep marker for sweep: ``DEV_RESET_HOOK``.
"""
import logging

from app.storage.redis_session import _get_redis_client

logger = logging.getLogger(__name__)

# DEV_RESET_HOOK — see module docstring for removal checklist.
_RESET_TRIGGER = "/reset"
_SCAN_COUNT = 200


def is_reset_command(text: str) -> bool:
    """Return True for ``/reset`` (case-insensitive, with leading/trailing whitespace).

    Examples that match: "/reset", " /reset ", "/Reset", "/RESET\n".
    Examples that do NOT match: "reset" (no slash), "/reset agora" (suffix),
    "preciso /reset" (prefix).
    """
    return text.strip().lower() == _RESET_TRIGGER


async def reset_conversation(phone_hash: str) -> int:
    """Delete every Redis key referencing ``phone_hash``. Returns count removed.

    Implementation: SCAN with pattern ``*{phone_hash}*`` against the shared
    Redis client (no new connection opened). The phone_hash is a 64-char
    HMAC-SHA256 digest — substring collisions with unrelated keys are
    astronomically unlikely, so the wildcard match is safe.

    Keys NOT affected:
    - ``processed_msg:{message_id}`` — those are per-message idempotency
      tokens, not bound to phone_hash. Leaving them intact prevents the
      reset itself from being reprocessed if Evolution retries the webhook.
    """
    client = _get_redis_client()
    pattern = f"*{phone_hash}*"

    deleted = 0
    cursor: int = 0
    while True:
        cursor, keys = await client.scan(cursor=cursor, match=pattern, count=_SCAN_COUNT)
        if keys:
            deleted += await client.delete(*keys)
        if cursor == 0:
            break

    logger.info("reset_conversation phone_hash=%.8s keys_deleted=%d", phone_hash, deleted)
    return deleted
