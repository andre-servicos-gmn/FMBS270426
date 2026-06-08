"""Sprint 2.6.3 — in-memory catalog snapshot for the agent's match layer.

A single ``CatalogCache`` instance lives for the process lifetime. The
agent's recommend node fetches the full active-product list through this
cache instead of hitting Supabase on every inbound WhatsApp message — the
typical pilot catalog (~1240 products) takes a few hundred ms to load,
which would have added noticeable latency per turn without caching.

Design notes
------------
- TTL is configurable (``BLING_CATALOG_CACHE_TTL``, default 60 s).
- An ``asyncio.Lock`` guards the refresh path so concurrent inbound
  messages don't kick off N parallel refreshes.
- When a refresh fails AFTER we already have a snapshot, we serve the
  stale snapshot with a WARNING — conversations must never break because
  Supabase blipped. When the FIRST load fails, we propagate the exception
  so the caller can render a graceful fallback.
- ``invalidate()`` clears the snapshot so the next ``get_snapshot()`` call
  re-fetches. Wired into the Bling webhook handler so product edits in the
  ERP propagate without waiting for the TTL.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class CatalogCache:
    """Process-wide singleton cache for the active Bling products list."""

    _instance: "CatalogCache | None" = None

    def __init__(self) -> None:
        self._snapshot: list[dict[str, Any]] | None = None
        self._loaded_at: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    # ── Singleton accessor ─────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "CatalogCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the singleton (and any cached snapshot). Test-only."""
        cls._instance = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def get_snapshot(self) -> list[dict[str, Any]]:
        """Return the active-products list. Refresh transparently on TTL miss."""
        ttl = max(int(get_settings().bling_catalog_cache_ttl or 0), 0)
        now = time.monotonic()
        age = now - self._loaded_at if self._snapshot is not None else float("inf")

        if self._snapshot is not None and age <= ttl:
            return self._snapshot

        async with self._lock:
            # Double-check after acquiring — another coroutine may have refreshed.
            now = time.monotonic()
            age = now - self._loaded_at if self._snapshot is not None else float("inf")
            if self._snapshot is None or age > ttl:
                await self._refresh()
        return self._snapshot or []

    def invalidate(self) -> None:
        """Clear the snapshot so the next ``get_snapshot()`` refreshes."""
        self._snapshot = None
        self._loaded_at = 0.0
        logger.info("catalog_cache_invalidated")

    # ── Internal ───────────────────────────────────────────────────────────

    async def _refresh(self) -> None:
        from app.sync.bling_repo import list_active_products

        start = time.monotonic()
        try:
            snapshot = await list_active_products(limit=None)
        except Exception as exc:
            if self._snapshot is None:
                logger.error("catalog_cache_first_load_failed: %s", exc)
                raise
            age_s = time.monotonic() - self._loaded_at
            logger.warning(
                "catalog_cache_refresh_failed using_stale snapshot_age_s=%.1f err=%s",
                age_s, exc,
            )
            return

        self._snapshot = snapshot
        self._loaded_at = time.monotonic()
        elapsed_ms = (self._loaded_at - start) * 1000.0
        logger.info(
            "catalog_cache_refresh n=%d elapsed_ms=%.0f",
            len(snapshot), elapsed_ms,
        )


async def get_catalog_snapshot() -> list[dict[str, Any]]:
    """Module-level shorthand — the recommend node uses this."""
    return await CatalogCache.instance().get_snapshot()


def invalidate_catalog() -> None:
    """Module-level shorthand — webhook handlers call this."""
    CatalogCache.instance().invalidate()
