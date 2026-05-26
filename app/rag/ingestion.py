"""Catalog ingestion: fetch → embed → upsert → soft-delete stale products."""
import logging
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.rag.embeddings import embed_batch
from app.storage.db import get_session
from app.storage.models import Product

logger = logging.getLogger(__name__)


def _embedding_text(p: dict[str, Any]) -> str:
    """Build the text we embed — concat non-empty semantic fields."""
    parts = [p.get("name"), p.get("description"), p.get("sport"), p.get("level")]
    return " ".join(str(x) for x in parts if x)


async def sync_catalog() -> dict[str, int]:
    """Fetch products from the configured source, embed and upsert them.

    Products present in the DB but absent from the latest source are soft-deleted
    (is_active=False) rather than removed.

    Returns a dict with keys inserted, updated, deactivated, total_source.
    """
    settings = get_settings()

    if settings.catalog_source == "api":
        from app.adapters.catalog.api_source import fetch_products_from_api

        products = await fetch_products_from_api()
    else:
        from app.adapters.catalog.file_source import fetch_products_from_file

        products = await fetch_products_from_file()

    if not products:
        logger.warning("catalog_sync source returned zero products — aborting")
        return {"inserted": 0, "updated": 0, "deactivated": 0, "total_source": 0}

    source_ids = {p["external_id"] for p in products}

    texts = [_embedding_text(p) for p in products]
    embeddings = await embed_batch(texts)

    async with get_session() as session:
        stats = await _upsert_all(session, products, embeddings, source_ids)
        await session.commit()

    logger.info(
        "catalog_sync done inserted=%d updated=%d deactivated=%d total_source=%d",
        stats["inserted"],
        stats["updated"],
        stats["deactivated"],
        stats["total_source"],
    )
    return stats


async def _upsert_all(
    session: AsyncSession,
    products: list[dict[str, Any]],
    embeddings: list[list[float]],
    source_ids: set[str],
) -> dict[str, int]:
    # Snapshot active external_ids before the upsert to compute inserted/updated counts
    result = await session.execute(
        select(Product.external_id).where(Product.is_active == True)  # noqa: E712
    )
    existing_ids: set[str] = {row[0] for row in result.all()}

    for product, embedding in zip(products, embeddings):
        stmt = (
            pg_insert(Product)
            .values(
                id=uuid.uuid4(),
                external_id=product["external_id"],
                name=product["name"],
                sport=product.get("sport"),
                level=product.get("level"),
                weight_g=product.get("weight_g"),
                balance=product.get("balance"),
                material=product.get("material"),
                price_cents=product.get("price_cents") or 0,
                stock=product.get("stock") or 0,
                description=product.get("description"),
                url=product.get("url"),
                image_url=product.get("image_url"),
                embedding=embedding,
                is_active=True,
            )
            .on_conflict_do_update(
                index_elements=["external_id"],
                set_={
                    "name": product["name"],
                    "sport": product.get("sport"),
                    "level": product.get("level"),
                    "weight_g": product.get("weight_g"),
                    "balance": product.get("balance"),
                    "material": product.get("material"),
                    "price_cents": product.get("price_cents") or 0,
                    "stock": product.get("stock") or 0,
                    "description": product.get("description"),
                    "url": product.get("url"),
                    "image_url": product.get("image_url"),
                    "embedding": embedding,
                    "is_active": True,
                },
            )
        )
        await session.execute(stmt)

    # Soft-delete products that disappeared from the source
    stale_ids = existing_ids - source_ids
    if stale_ids:
        await session.execute(
            update(Product)
            .where(Product.external_id.in_(stale_ids))
            .values(is_active=False)
        )

    return {
        "inserted": len(source_ids - existing_ids),
        "updated": len(source_ids & existing_ids),
        "deactivated": len(stale_ids),
        "total_source": len(source_ids),
    }
