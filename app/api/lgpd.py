"""LGPD compliance routes — data erasure and portability."""
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update

from app.security.audit_log import log_access
from app.security.pii_masker import hash_phone

router = APIRouter(prefix="/lgpd", tags=["lgpd"])
logger = logging.getLogger(__name__)


class _PhoneRequest(BaseModel):
    phone: str


# ── DELETE /lgpd/lead ─────────────────────────────────────────────────────────

@router.delete("/lead", status_code=200)
async def delete_lead(body: _PhoneRequest) -> dict[str, Any]:
    """Erase all personal data for a phone number (LGPD Art. 18 VI).

    - Soft-deletes the Lead row (sets deleted_at)
    - Zeroes ConversationLog.content_masked for all rows of that hash
    - Deletes the Redis session
    - Writes an audit entry with deleted=True flag (kept for compliance trail)
    """
    from app.storage.db import get_session
    from app.storage.models import ConversationLog, Lead
    from app.storage.redis_session import get_store

    phone_hash = hash_phone(body.phone)
    now = datetime.now(timezone.utc)

    async with get_session() as session:
        lead = (await session.execute(
            select(Lead).where(Lead.phone_hash == phone_hash)
        )).scalar_one_or_none()

        if lead is None:
            raise HTTPException(status_code=404, detail="Lead not found")

        # Soft-delete lead
        lead.deleted_at = now

        # Zero out conversation content
        await session.execute(
            update(ConversationLog)
            .where(ConversationLog.phone_hash == phone_hash)
            .values(content_masked="[DELETED]")
        )

        await session.commit()

    # Delete Redis session
    try:
        store = get_store()
        await store.delete(phone_hash)
    except Exception as exc:
        logger.warning("lgpd_delete redis session removal failed (hash=%.8s): %s", phone_hash, exc)

    await log_access(
        actor="lgpd",
        action="delete_lead",
        target_hash=phone_hash,
        metadata={"deleted": True, "deleted_at": now.isoformat()},
    )

    logger.info("lgpd_delete completed phone_hash=%.8s", phone_hash)
    return {"status": "deleted", "phone_hash": phone_hash}


# ── POST /lgpd/lead/export ────────────────────────────────────────────────────

@router.post("/lead/export", status_code=200)
async def export_lead(body: _PhoneRequest) -> dict[str, Any]:
    """Return all stored data for a phone number (LGPD Art. 18 II — portability).

    Content is returned as-is (already masked); raw PII is never stored.
    """
    from app.storage.db import get_session
    from app.storage.models import AccessLog, ConversationLog, Lead

    phone_hash = hash_phone(body.phone)

    async with get_session() as session:
        lead = (await session.execute(
            select(Lead).where(Lead.phone_hash == phone_hash)
        )).scalar_one_or_none()

        if lead is None:
            raise HTTPException(status_code=404, detail="Lead not found")

        conv_rows = (await session.execute(
            select(ConversationLog)
            .where(ConversationLog.phone_hash == phone_hash)
            .order_by(ConversationLog.created_at.asc())
        )).scalars().all()

        audit_rows = (await session.execute(
            select(AccessLog)
            .where(AccessLog.target_hash == phone_hash)
            .order_by(AccessLog.created_at.asc())
        )).scalars().all()

    await log_access(
        actor="lgpd",
        action="export_lead",
        target_hash=phone_hash,
    )

    return {
        "phone_hash": phone_hash,
        "lead": {
            "created_at": lead.created_at.isoformat(),
            "last_interaction_at": lead.last_interaction_at.isoformat(),
            "deleted_at": lead.deleted_at.isoformat() if lead.deleted_at else None,
            "profile": lead.profile or {},
        },
        "conversations": [
            {
                "id": str(c.id),
                "role": c.message_role,
                "content_masked": c.content_masked,
                "created_at": c.created_at.isoformat(),
            }
            for c in conv_rows
        ],
        "audit_trail": [
            {
                "actor": a.actor,
                "action": a.action,
                "created_at": a.created_at.isoformat(),
            }
            for a in audit_rows
        ],
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
