"""Vector similarity search over products and knowledge_base tables."""
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import embed_text

logger = logging.getLogger(__name__)


def _vec(embedding: list[float]) -> str:
    """Format embedding as pgvector literal '[f1,f2,...]'."""
    return "[" + ",".join(str(round(x, 8)) for x in embedding) + "]"


def _coerce_row(row: dict) -> dict:
    """Convert asyncpg-specific types (UUID, Decimal) to plain Python primitives."""
    return {k: str(v) if hasattr(v, "hex") and not isinstance(v, float) else v for k, v in row.items()}


async def search_products(
    session: AsyncSession,
    query: str,
    filters: dict[str, Any] | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return up to k products ranked by cosine similarity to query.

    Supported filters:
      sport           — exact match (e.g. "beach_tennis", "padel")
      max_price_cents — upper bound on price_cents (inclusive)
      min_stock       — lower bound on stock (default 1)
      category        — Sprint 1.11: exact match against the new ``category``
                        column ("raquete" | "pala" | "bola" | "acessorio" |
                        "vestuario" | "calcado" | "bolsa" | "outros"). None
                        means "search all categories".
    """
    filters = filters or {}
    embedding = await embed_text(query)

    # \:\: escapes :: so SQLAlchemy doesn't parse :vector as a named param.
    # Sprint 1.11: search_products() now accepts p_category as the 6th arg.
    stmt = text(
        "SELECT * FROM search_products("
        "   :embedding\\:\\:vector,"
        "   :sport,"
        "   :max_price,"
        "   :min_stock,"
        "   :k,"
        "   :category"
        ")"
    )

    result = await session.execute(
        stmt,
        {
            "embedding": _vec(embedding),
            "sport": filters.get("sport"),
            "max_price": filters.get("max_price_cents"),
            "min_stock": filters.get("min_stock", 1),
            "k": k,
            "category": filters.get("category"),
        },
    )

    rows = result.mappings().all()
    logger.info("search_products query_len=%d k=%d results=%d", len(query), k, len(rows))
    # Coerce asyncpg UUID objects to str so the dicts are JSON-serialisable and
    # safe to store in LangGraph's MemorySaver checkpoint.
    return [_coerce_row(dict(row)) for row in rows]


async def search_knowledge_base(
    session: AsyncSession,
    query: str,
    category: str | None = None,
    k: int = 4,
) -> list[dict[str, Any]]:
    """Return up to k knowledge-base documents ranked by cosine similarity to query.

    Args:
        query    — user's question in natural language
        category — optional filter: 'faq' | 'shipping' | 'exchange' | 'warranty' |
                   'payment' | 'store' | 'general'
        k        — number of results to return (default 4)
    """
    embedding = await embed_text(query)

    stmt = text(
        "SELECT * FROM search_knowledge_base("
        "   :embedding\\:\\:vector,"
        "   :category,"
        "   :k"
        ")"
    )

    result = await session.execute(
        stmt,
        {
            "embedding": _vec(embedding),
            "category": category,
            "k": k,
        },
    )

    rows = result.mappings().all()
    logger.info(
        "search_knowledge_base query_len=%d category=%s k=%d results=%d",
        len(query), category, k, len(rows),
    )
    return [dict(row) for row in rows]
