import functools
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


async def log_access(
    actor: str,
    action: str,
    target_hash: str,
    ip: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert an audit log entry. Swallows DB errors so callers are never blocked."""
    from app.storage.db import get_session
    from app.storage.models import AccessLog

    try:
        async with get_session() as session:
            entry = AccessLog(
                id=uuid4(),
                actor=actor,
                action=action,
                target_hash=target_hash,
                created_at=datetime.now(timezone.utc),
                ip=ip,
                metadata_=metadata,
            )
            session.add(entry)
            await session.commit()
    except Exception as exc:
        logger.warning(
            "audit log write failed (action=%s target=%.8s): %s", action, target_hash, exc
        )


def audited(action: str, default_actor: str = "system") -> Callable[..., Any]:
    """Decorator that appends an audit log entry after the wrapped async handler runs.

    Inspects kwargs for ``target_hash`` > ``phone_hash`` > ``phone`` (auto-hashed)
    to determine the audit target. Falls back to ``"unknown"``.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)

            target_hash: str = "unknown"
            if "target_hash" in kwargs:
                target_hash = str(kwargs["target_hash"])
            elif "phone_hash" in kwargs:
                target_hash = str(kwargs["phone_hash"])
            elif "phone" in kwargs:
                from app.security.pii_masker import hash_phone

                target_hash = hash_phone(str(kwargs["phone"]))

            actor: str = str(kwargs.get("actor", default_actor))

            ip: str | None = None
            for arg in args:
                if hasattr(arg, "client") and hasattr(arg.client, "host"):
                    ip = arg.client.host
                    break

            await log_access(actor=actor, action=action, target_hash=target_hash, ip=ip)
            return result

        return wrapper

    return decorator
