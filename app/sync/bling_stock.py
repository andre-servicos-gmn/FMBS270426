"""Sprint 2.5 — real-time stock with Redis-backed cache.

We deliberately don't mirror stock into ``bling_products`` — the catalog
sync runs daily, but stock can change in minutes. Instead each
``get_stock(produto_id)`` call:

1. Checks Redis (5 min TTL by default; configurable via BLING_STOCK_CACHE_TTL).
2. On miss, hits ``GET /estoques/saldos`` on Bling.
3. Caches the result and returns it.

Any failure (timeout, 5xx, not authorized) returns ``None`` so the agent
can fall through to a friendly "tô confirmando aqui" reply instead of
blowing up the conversation.
"""
from __future__ import annotations

import logging
from typing import Any

from app.adapters.bling import BlingClient, BlingError
from app.config import get_settings

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "bling:stock:"


def _get_redis():
    # Sprint 2.6.5 — resilient client (keepalive + retry + health_check).
    from app.storage.redis_resilient import make_resilient_redis
    return make_resilient_redis(get_settings().redis_url)


def _extract_saldo(payload: dict[str, Any], produto_id: int) -> int | None:
    """Find the saldoFisico for ``produto_id`` in /estoques/saldos response."""
    items = payload.get("data") or []
    for item in items:
        if int(item.get("produto", {}).get("id", item.get("id") or 0)) == produto_id:
            saldo = (
                item.get("saldoFisicoTotal")
                or item.get("saldoFisico")
                or item.get("saldoVirtualTotal")
                or item.get("saldoVirtual")
            )
            if saldo is None:
                continue
            try:
                return int(float(saldo))
            except (TypeError, ValueError):
                return None
    return None


async def get_stock(produto_id: int) -> int | None:
    """Return the on-hand balance for ``produto_id`` or None on failure."""
    settings = get_settings()
    key = f"{_CACHE_PREFIX}{produto_id}"

    redis = _get_redis()
    try:
        cached = await redis.get(key)
        if cached is not None:
            try:
                value: int | None = int(cached)
            except ValueError:
                value = None
            logger.info("bling_stock cache_hit id=%s value=%s", produto_id, value)
            return value

        try:
            client = BlingClient()
            payload = await client.consultar_estoque(produto_id)
            saldo = _extract_saldo(payload, produto_id)
        except BlingError as exc:
            logger.warning("bling_stock api_error id=%s: %s", produto_id, exc)
            return None
        except Exception as exc:
            logger.warning("bling_stock unexpected_error id=%s: %s", produto_id, exc)
            return None

        # Only cache a KNOWN balance. Caching None as "0" turned "unknown"
        # (product absent from the Bling response) into a hard "esgotado" for
        # the whole TTL — the agent then denied stock it never actually read.
        if saldo is not None:
            await redis.setex(key, settings.bling_stock_cache_ttl, str(saldo))
        logger.info("bling_stock fetched id=%s value=%s", produto_id, saldo)
        return saldo
    finally:
        await redis.aclose()


async def get_stocks_bulk(
    produto_ids: list[int], client: Any | None = None
) -> dict[int, int | None]:
    """Fetch on-hand balances for many products in ONE /estoques/saldos call.

    Used by the catalog sync to mirror ``saldo`` into ``bling_products`` without
    paying a stock call per product. Returns ``{produto_id: saldo or None}`` —
    a product missing from the Bling response (or any failure) maps to None,
    i.e. "unknown", which the catalog filter keeps rather than hides.

    Bypasses the Redis cache on purpose (this is a sync-time bulk read, not the
    per-turn ``get_stock`` path). Fully guarded: any error returns ``{}``.
    """
    if not produto_ids:
        return {}
    own = client
    if own is None:
        from app.adapters.bling import BlingClient
        own = BlingClient()
    try:
        payload = await own.consultar_estoque(produto_ids)
    except Exception as exc:  # noqa: BLE001 — degrade to "unknown" for all
        logger.warning("bling_stock bulk_failed n=%d: %s", len(produto_ids), exc)
        return {}
    result = {pid: _extract_saldo(payload, pid) for pid in produto_ids}
    logger.info(
        "bling_stock bulk_fetched requested=%d resolved=%d",
        len(produto_ids), sum(1 for v in result.values() if v is not None),
    )
    return result


async def invalidate_stock(produto_id: int) -> None:
    """Drop the cached entry — called when the webhook signals an update."""
    redis = _get_redis()
    try:
        await redis.delete(f"{_CACHE_PREFIX}{produto_id}")
    finally:
        await redis.aclose()
