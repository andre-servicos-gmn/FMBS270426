"""Admin routes — protected by X-Admin-Key header."""
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select

from app.config import get_settings
from app.security.audit_log import log_access

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)

_ADMIN_KEY_HEADER = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def _require_admin(api_key: str | None = Depends(_ADMIN_KEY_HEADER)) -> str:
    key = get_settings().admin_api_key
    if not api_key or not key or api_key != key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return api_key


# ── Leads ─────────────────────────────────────────────────────────────────────

@router.get("/leads")
async def list_leads(
    limit: int = Query(default=50, ge=1, le=200),
    _key: str = Depends(_require_admin),
) -> dict[str, Any]:
    from app.storage.db import get_session
    from app.storage.models import Lead

    async with get_session() as session:
        rows = (await session.execute(
            select(Lead.phone_hash, Lead.profile, Lead.created_at, Lead.last_interaction_at)
            .where(Lead.deleted_at.is_(None))
            .order_by(Lead.last_interaction_at.desc())
            .limit(limit)
        )).all()

    leads = [
        {
            "phone_hash": r.phone_hash,
            "profile": r.profile or {},
            "created_at": r.created_at.isoformat(),
            "last_interaction_at": r.last_interaction_at.isoformat(),
        }
        for r in rows
    ]
    await log_access(actor="admin", action="list_leads", target_hash="bulk")
    return {"leads": leads, "count": len(leads)}


@router.get("/leads/{phone_hash}")
async def get_lead(
    phone_hash: str,
    _key: str = Depends(_require_admin),
) -> dict[str, Any]:
    from app.storage.db import get_session
    from app.storage.models import ConversationLog, Lead

    async with get_session() as session:
        lead_row = (await session.execute(
            select(Lead).where(Lead.phone_hash == phone_hash, Lead.deleted_at.is_(None))
        )).scalar_one_or_none()

        if lead_row is None:
            raise HTTPException(status_code=404, detail="Lead not found")

        conv_rows = (await session.execute(
            select(ConversationLog)
            .where(ConversationLog.phone_hash == phone_hash)
            .order_by(ConversationLog.created_at.desc())
            .limit(50)
        )).scalars().all()

    await log_access(actor="admin", action="read_lead", target_hash=phone_hash)

    return {
        "phone_hash": lead_row.phone_hash,
        "profile": lead_row.profile or {},
        "created_at": lead_row.created_at.isoformat(),
        "last_interaction_at": lead_row.last_interaction_at.isoformat(),
        "conversations": [
            {
                "id": str(c.id),
                "role": c.message_role,
                "content_masked": c.content_masked,
                "created_at": c.created_at.isoformat(),
            }
            for c in conv_rows
        ],
    }


# ── Catalog ───────────────────────────────────────────────────────────────────

@router.post("/catalog/resync", status_code=202)
async def resync_catalog(
    background_tasks: BackgroundTasks,
    _key: str = Depends(_require_admin),
) -> dict[str, str]:
    from app.rag.ingestion import sync_catalog

    async def _run() -> None:
        try:
            stats = await sync_catalog()
            logger.info("admin_resync done stats=%s", stats)
        except Exception as exc:
            logger.error("admin_resync failed: %s", exc)

    background_tasks.add_task(_run)
    await log_access(actor="admin", action="catalog_resync", target_hash="catalog")
    return {"status": "accepted", "detail": "Sync running in background"}


# ── Audit log ─────────────────────────────────────────────────────────────────

@router.get("/audit")
async def list_audit(
    actor: str | None = Query(default=None),
    action: str | None = Query(default=None),
    from_dt: datetime | None = Query(default=None, alias="from"),
    to_dt: datetime | None = Query(default=None, alias="to"),
    limit: int = Query(default=100, ge=1, le=500),
    _key: str = Depends(_require_admin),
) -> dict[str, Any]:
    from sqlalchemy import and_

    from app.storage.db import get_session
    from app.storage.models import AccessLog

    filters = []
    if actor:
        filters.append(AccessLog.actor == actor)
    if action:
        filters.append(AccessLog.action == action)
    if from_dt:
        filters.append(AccessLog.created_at >= from_dt)
    if to_dt:
        filters.append(AccessLog.created_at <= to_dt)

    async with get_session() as session:
        rows = (await session.execute(
            select(AccessLog)
            .where(and_(*filters) if filters else True)
            .order_by(AccessLog.created_at.desc())
            .limit(limit)
        )).scalars().all()

    await log_access(actor="admin", action="read_audit", target_hash="bulk")

    return {
        "entries": [
            {
                "id": str(r.id),
                "actor": r.actor,
                "action": r.action,
                "target_hash": r.target_hash,
                "created_at": r.created_at.isoformat(),
                "ip": r.ip,
                "metadata": r.metadata_,
            }
            for r in rows
        ],
        "count": len(rows),
    }
