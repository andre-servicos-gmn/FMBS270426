"""Fetch catalog products from a local CSV or JSON file."""
import csv
import json
import logging
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


async def fetch_products_from_file() -> list[dict]:
    """Read CATALOG_FILE_PATH (auto-detects .json or .csv), normalize and return products."""
    path = Path(get_settings().catalog_file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        raw = _load_json(path)
    elif suffix == ".csv":
        raw = _load_csv(path)
    else:
        raise ValueError(f"Unsupported catalog file format: {suffix!r} — expected .json or .csv")

    logger.info("catalog_file loaded count=%d path=%s", len(raw), path)
    return [_normalize(p) for p in raw]


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("products") or data.get("data") or data.get("items") or []


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
