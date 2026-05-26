"""Fetch catalog products from an external HTTP API."""
import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


async def fetch_products_from_api() -> list[dict]:
    """GET CATALOG_API_URL, normalize and return list of canonical product dicts."""
    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.catalog_api_key:
        headers["Authorization"] = f"Bearer {settings.catalog_api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(settings.catalog_api_url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("products") or data.get("data") or data.get("items") or []
    else:
        raw = []

    logger.info("catalog_api fetched count=%d", len(raw))
    return [_normalize(p) for p in raw]


def _normalize(p: dict) -> dict:
    return {
        "external_id": str(p.get("id") or p.get("external_id") or ""),
        "name": p.get("name") or p.get("title") or "",
        "sport": p.get("sport"),
        "level": p.get("level") or p.get("player_level"),
        "weight_g": _to_int(p.get("weight_g") or p.get("weight")),
        "balance": p.get("balance"),
        "material": p.get("material"),
        "price_cents": _to_int(p.get("price_cents")) or _price_to_cents(p.get("price")),
        "stock": _to_int(p.get("stock") or p.get("quantity")) or 0,
        "description": p.get("description") or "",
        "url": p.get("url") or p.get("product_url"),
        "image_url": p.get("image_url") or p.get("image"),
    }


def _to_int(v: object) -> int | None:
    try:
        return int(v) if v is not None else None  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _price_to_cents(v: object) -> int:
    try:
        return round(float(v) * 100)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0
