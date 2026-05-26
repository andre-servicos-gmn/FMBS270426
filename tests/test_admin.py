"""Tests for admin routes — GET /admin/leads, POST /admin/catalog/resync, GET /admin/audit."""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

_ADMIN_KEY = "test-admin-key-xyz"
_PHONE_HASH = "a" * 64


def _headers(key: str = _ADMIN_KEY) -> dict:
    return {"X-Admin-Key": key}


@pytest.fixture
def override_admin_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", _ADMIN_KEY)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@asynccontextmanager
async def _mock_session_cm(rows=None):
    """Generic DB session mock that returns rows from execute()."""
    rows = rows or []

    class _Result:
        def __init__(self, data):
            self._data = data

        def all(self):
            return self._data

        def scalars(self):
            return self

        def scalar_one_or_none(self):
            return self._data[0] if self._data else None

    session = MagicMock()
    session.execute = AsyncMock(return_value=_Result(rows))
    session.commit = AsyncMock()
    yield session


# ── Auth ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_no_key_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/leads")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_wrong_key_returns_401(override_admin_key):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/leads", headers={"X-Admin-Key": "wrong"})
    assert resp.status_code == 401


# ── GET /admin/leads ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_leads_returns_empty(override_admin_key):
    now = datetime.now(timezone.utc)

    @asynccontextmanager
    async def _empty_session():
        session = MagicMock()
        session.execute = AsyncMock(return_value=MagicMock(all=lambda: []))
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.api.admin.log_access", new=AsyncMock()),
        patch("app.storage.db.get_session", _empty_session),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/leads", headers=_headers())

    assert resp.status_code == 200
    body = resp.json()
    assert body["leads"] == []
    assert body["count"] == 0


@pytest.mark.asyncio
async def test_list_leads_returns_rows(override_admin_key):
    now = datetime.now(timezone.utc)
    fake_row = MagicMock()
    fake_row.phone_hash = _PHONE_HASH
    fake_row.profile = {"level": "iniciante"}
    fake_row.created_at = now
    fake_row.last_interaction_at = now

    @asynccontextmanager
    async def _session_with_rows():
        session = MagicMock()
        session.execute = AsyncMock(return_value=MagicMock(all=lambda: [fake_row]))
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.api.admin.log_access", new=AsyncMock()),
        patch("app.storage.db.get_session", _session_with_rows),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/leads", headers=_headers())

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["leads"][0]["phone_hash"] == _PHONE_HASH
    assert body["leads"][0]["profile"] == {"level": "iniciante"}


# ── GET /admin/leads/{phone_hash} ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_lead_not_found_returns_404(override_admin_key):
    @asynccontextmanager
    async def _empty():
        session = MagicMock()

        class _R:
            def scalar_one_or_none(self):
                return None

        session.execute = AsyncMock(return_value=_R())
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.api.admin.log_access", new=AsyncMock()),
        patch("app.storage.db.get_session", _empty),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/admin/leads/{_PHONE_HASH}", headers=_headers())

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_lead_returns_detail(override_admin_key):
    now = datetime.now(timezone.utc)

    fake_lead = MagicMock()
    fake_lead.phone_hash = _PHONE_HASH
    fake_lead.profile = {"sport": "beach_tennis"}
    fake_lead.created_at = now
    fake_lead.last_interaction_at = now
    fake_lead.deleted_at = None

    fake_conv = MagicMock()
    fake_conv.id = uuid.uuid4()
    fake_conv.message_role = "user"
    fake_conv.content_masked = "quero uma raquete"
    fake_conv.created_at = now

    call_count = 0

    @asynccontextmanager
    async def _session():
        nonlocal call_count

        class _LeadResult:
            def scalar_one_or_none(self):
                return fake_lead

        class _ConvResult:
            def scalars(self):
                return self

            def all(self):
                return [fake_conv]

        session = MagicMock()

        async def _execute(_stmt):
            nonlocal call_count
            call_count += 1
            return _LeadResult() if call_count == 1 else _ConvResult()

        session.execute = _execute
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.api.admin.log_access", new=AsyncMock()),
        patch("app.storage.db.get_session", _session),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/admin/leads/{_PHONE_HASH}", headers=_headers())

    assert resp.status_code == 200
    body = resp.json()
    assert body["phone_hash"] == _PHONE_HASH
    assert len(body["conversations"]) == 1
    assert body["conversations"][0]["role"] == "user"


# ── POST /admin/catalog/resync ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resync_returns_202(override_admin_key):
    with patch("app.api.admin.log_access", new=AsyncMock()):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/admin/catalog/resync", headers=_headers())

    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


# ── GET /admin/audit ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_no_filters_returns_entries(override_admin_key):
    now = datetime.now(timezone.utc)
    fake_entry = MagicMock()
    fake_entry.id = uuid.uuid4()
    fake_entry.actor = "webhook"
    fake_entry.action = "process_message"
    fake_entry.target_hash = _PHONE_HASH
    fake_entry.created_at = now
    fake_entry.ip = None
    fake_entry.metadata_ = None

    @asynccontextmanager
    async def _session():
        class _R:
            def scalars(self):
                return self

            def all(self):
                return [fake_entry]

        session = MagicMock()
        session.execute = AsyncMock(return_value=_R())
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.api.admin.log_access", new=AsyncMock()),
        patch("app.storage.db.get_session", _session),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/audit", headers=_headers())

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["entries"][0]["actor"] == "webhook"
    assert body["entries"][0]["action"] == "process_message"


@pytest.mark.asyncio
async def test_audit_with_actor_filter(override_admin_key):
    @asynccontextmanager
    async def _empty():
        class _R:
            def scalars(self):
                return self

            def all(self):
                return []

        session = MagicMock()
        session.execute = AsyncMock(return_value=_R())
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.api.admin.log_access", new=AsyncMock()),
        patch("app.storage.db.get_session", _empty),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(
                "/admin/audit",
                headers=_headers(),
                params={"actor": "webhook", "action": "process_message"},
            )

    assert resp.status_code == 200
    assert resp.json()["count"] == 0
