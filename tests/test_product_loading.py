"""Sprint 2.6.3 — catalog loading must cover the full active set.

The pre-2.6.3 ``list_active_products(limit=200)`` was silently truncating
the agent's match catalog to ~16% (200 of ~1240 products) in the pilot.
These tests pin the new behavior: the repo function returns everything by
default, the recommend node wires through the in-memory ``CatalogCache``,
and the cache survives Supabase blips by serving stale snapshots.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sync.bling_catalog_cache import CatalogCache


@asynccontextmanager
async def _mock_db_session():
    s = MagicMock()
    s.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    s.commit = AsyncMock()
    yield s


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a clean CatalogCache singleton."""
    CatalogCache.reset_for_tests()
    yield
    CatalogCache.reset_for_tests()


# ── list_active_products: no default LIMIT 200 ──────────────────────────────

@pytest.mark.asyncio
async def test_list_active_products_default_has_no_limit(monkeypatch):
    """Sprint 2.6.3 root-cause regression: the default must not cap at 200."""
    from app.sync import bling_repo

    captured: dict[str, object] = {}

    class _FakeStmt:
        def __init__(self, label="initial"):
            self.label = label
            captured.setdefault("first_label", label)
            self._limited = False

        def where(self, *a, **kw):
            return self

        def limit(self, n):
            self._limited = True
            captured["limit"] = n
            return self

    @asynccontextmanager
    async def _fake_session():
        session = MagicMock()
        session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )))
        yield session

    monkeypatch.setattr(bling_repo, "get_session", _fake_session)
    monkeypatch.setattr(bling_repo, "select", lambda *_args: _FakeStmt("initial"))

    await bling_repo.list_active_products()
    # With no explicit limit, .limit() must NOT have been called.
    assert "limit" not in captured, f"unexpected LIMIT applied: {captured}"


@pytest.mark.asyncio
async def test_list_active_products_explicit_limit_still_works():
    """Callers can still pass an explicit cap when they want one."""
    from app.sync import bling_repo

    captured: dict[str, object] = {}

    class _FakeStmt:
        def where(self, *a, **kw): return self
        def limit(self, n):
            captured["limit"] = n
            return self

    @asynccontextmanager
    async def _fake_session():
        session = MagicMock()
        session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )))
        yield session

    with patch.object(bling_repo, "get_session", _fake_session):
        with patch.object(bling_repo, "select", lambda *_args: _FakeStmt()):
            await bling_repo.list_active_products(limit=42)

    assert captured.get("limit") == 42


# ── recommend sees the full catalog via the cache ────────────────────────────

@pytest.mark.asyncio
async def test_match_sees_full_catalog_size(monkeypatch):
    """Recommend's match layer must see the FULL catalog size, not 200."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import _list_catalog_candidates

    fake_snapshot = [
        {"id": i, "name": f"Produto {i}", "price_cents": 1000, "is_raquete_praia": False}
        for i in range(1500)
    ]
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=fake_snapshot,
    ):
        candidates = await _list_catalog_candidates("qualquer coisa")

    assert len(candidates) == 1500, (
        f"recommend truncated to {len(candidates)}; "
        f"the LIMIT 200 bug from Sprint 2.5 is back"
    )


# ── Cache TTL behaviour ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_catalog_cache_refresh_loads_active_products(monkeypatch):
    monkeypatch.setenv("BLING_CATALOG_CACHE_TTL", "60")
    from app.config import get_settings
    get_settings.cache_clear()

    sample = [{"id": 1, "name": "Raquete A"}, {"id": 2, "name": "Raquete B"}]
    with patch(
        "app.sync.bling_repo.list_active_products",
        new_callable=AsyncMock,
        return_value=sample,
    ) as mocked:
        cache = CatalogCache.instance()
        snap1 = await cache.get_snapshot()
        snap2 = await cache.get_snapshot()  # within TTL → no extra load

    assert snap1 == sample
    assert snap2 == sample
    mocked.assert_awaited_once()  # second call served from cache


@pytest.mark.asyncio
async def test_catalog_cache_ttl_expires(monkeypatch):
    """Setting TTL=0 forces every call to refresh."""
    monkeypatch.setenv("BLING_CATALOG_CACHE_TTL", "0")
    from app.config import get_settings
    get_settings.cache_clear()

    sample = [{"id": 1, "name": "X"}]
    with patch(
        "app.sync.bling_repo.list_active_products",
        new_callable=AsyncMock,
        return_value=sample,
    ) as mocked:
        cache = CatalogCache.instance()
        await cache.get_snapshot()
        await cache.get_snapshot()

    assert mocked.await_count == 2  # TTL=0 → never hits cache


@pytest.mark.asyncio
async def test_catalog_cache_survives_supabase_failure_after_first_load():
    """When refresh fails AFTER an initial successful load, serve stale."""
    cache = CatalogCache.instance()

    sample = [{"id": 1, "name": "Raquete A"}]
    with patch(
        "app.sync.bling_repo.list_active_products",
        new_callable=AsyncMock,
        return_value=sample,
    ):
        await cache.get_snapshot()
    # Force TTL expiry to make the next call trigger refresh, then fail it.
    cache._loaded_at = 0.0

    with patch(
        "app.sync.bling_repo.list_active_products",
        new_callable=AsyncMock,
        side_effect=RuntimeError("supabase down"),
    ):
        stale = await cache.get_snapshot()

    assert stale == sample  # served from the previously-cached snapshot


@pytest.mark.asyncio
async def test_catalog_cache_first_load_failure_raises():
    """When the FIRST load fails, we propagate — caller renders fallback."""
    cache = CatalogCache.instance()
    with patch(
        "app.sync.bling_repo.list_active_products",
        new_callable=AsyncMock,
        side_effect=RuntimeError("supabase down"),
    ):
        with pytest.raises(RuntimeError):
            await cache.get_snapshot()


def test_catalog_cache_invalidate_clears_snapshot():
    cache = CatalogCache.instance()
    cache._snapshot = [{"id": 1}]
    cache._loaded_at = 999.0
    cache.invalidate()
    assert cache._snapshot is None
    assert cache._loaded_at == 0.0
