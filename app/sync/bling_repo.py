"""Sprint 2.5 — DB helpers around the Bling tables.

Centralizes the SQLAlchemy queries the rest of the app needs so callers
don't have to know the column names. Keep the surface tiny and synchronous-
looking (await + async session) — the agent nodes use these via short
async with blocks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.storage.db import get_session
from app.storage.models import (
    BlingProduct,
    BlingSyncLog,
    BlingWebhookEvent,
)

logger = logging.getLogger(__name__)


def _row_to_dict(p: BlingProduct) -> dict[str, Any]:
    """Render a BlingProduct row as a plain dict shaped for ``_product_match``."""
    preco_value = p.preco
    if preco_value is None:
        price_cents = 0
    else:
        price_cents = int(round(float(preco_value) * 100))

    return {
        "id": p.id,
        "name": p.nome,
        "codigo": p.codigo,
        "price_cents": price_cents,
        "description": p.descricao_curta or p.descricao_complementar or "",
        "marca": p.marca,
        "modelo": p.modelo,
        "categoria_nome": p.categoria_nome,
        "is_raquete_praia": bool(p.is_raquete_praia),
        "weight_g": (
            int(round(float(p.peso_liquido) * 1000)) if p.peso_liquido else None
        ),
        "campos_customizados": p.campos_customizados or {},
        "atributos_parseados": p.atributos_parseados or {},
        "imagem_url": p.imagem_url,
        "situacao": p.situacao,
        "external_id": str(p.id),
    }


async def list_active_products(limit: int = 200) -> list[dict[str, Any]]:
    """Return every active product as plain dicts (cap at ``limit``).

    Used by the agent's match layer instead of the legacy semantic search
    when Bling integration is live. For thousands of products you'd want a
    real search index; in the pilot (~1240 products) a single SELECT is fast.
    """
    async with get_session() as session:
        result = await session.execute(
            select(BlingProduct)
            .where(BlingProduct.situacao == "A")
            .limit(limit)
        )
        rows = result.scalars().all()
    return [_row_to_dict(p) for p in rows]


async def fetch_product_by_id(produto_id: int) -> dict[str, Any] | None:
    async with get_session() as session:
        row = await session.get(BlingProduct, produto_id)
    return _row_to_dict(row) if row else None


async def fetch_product_by_name(name_substr: str) -> list[dict[str, Any]]:
    """Cheap ILIKE filter — used by the agent when narrowing by a noun phrase."""
    if not name_substr:
        return []
    pattern = f"%{name_substr}%"
    async with get_session() as session:
        result = await session.execute(
            select(BlingProduct)
            .where(BlingProduct.nome.ilike(pattern))
            .where(BlingProduct.situacao == "A")
            .limit(20)
        )
        rows = result.scalars().all()
    return [_row_to_dict(p) for p in rows]


# ── Upsert helpers used by the sync layer ───────────────────────────────

async def upsert_product(payload: dict[str, Any]) -> str:
    """UPSERT a Bling product. Returns 'inserted' or 'updated'."""
    async with get_session() as session:
        existed = await session.get(BlingProduct, payload["id"])
        stmt = pg_insert(BlingProduct).values(**payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[BlingProduct.id],
            set_={
                k: stmt.excluded[k] for k in payload.keys() if k != "id"
            } | {"updated_at": datetime.now(timezone.utc)},
        )
        await session.execute(stmt)
        await session.commit()
    return "updated" if existed else "inserted"


async def mark_product_inactive(produto_id: int) -> bool:
    """Set situacao='E' (excluído). Returns True if a row was touched."""
    async with get_session() as session:
        result = await session.execute(
            update(BlingProduct)
            .where(BlingProduct.id == produto_id)
            .values(situacao="E", updated_at=datetime.now(timezone.utc))
            .returning(BlingProduct.id)
        )
        touched = result.first() is not None
        await session.commit()
    return touched


# ── Sync-log + webhook-idempotency helpers ──────────────────────────────

async def open_sync_log(kind: str, metadata: dict | None = None) -> int:
    async with get_session() as session:
        row = BlingSyncLog(kind=kind, metadata_=metadata)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row.id


async def close_sync_log(log_id: int, **fields: Any) -> None:
    async with get_session() as session:
        await session.execute(
            update(BlingSyncLog)
            .where(BlingSyncLog.id == log_id)
            .values(finished_at=datetime.now(timezone.utc), **fields)
        )
        await session.commit()


async def record_webhook_event(
    product_id: int, event_kind: str, event_timestamp: datetime,
    payload: dict[str, Any] | None = None,
) -> bool:
    """Return True when this event is the newest one we have for ``product_id``.

    False means we already applied a newer event — caller should skip.
    """
    async with get_session() as session:
        result = await session.execute(
            select(BlingWebhookEvent.event_timestamp)
            .where(BlingWebhookEvent.product_id == product_id)
            .order_by(BlingWebhookEvent.event_timestamp.desc())
            .limit(1)
        )
        last_ts = result.scalar_one_or_none()
        if last_ts is not None:
            last = last_ts if last_ts.tzinfo else last_ts.replace(tzinfo=timezone.utc)
            incoming = event_timestamp if event_timestamp.tzinfo else event_timestamp.replace(tzinfo=timezone.utc)
            if incoming <= last:
                logger.info(
                    "bling_webhook_out_of_order product_id=%s incoming=%s last=%s",
                    product_id, incoming, last,
                )
                return False
        session.add(BlingWebhookEvent(
            product_id=product_id,
            event_kind=event_kind,
            event_timestamp=event_timestamp,
            payload=payload,
        ))
        await session.commit()
    return True
