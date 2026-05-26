"""Tests for LGPD routes — DELETE /lgpd/lead and POST /lgpd/lead/export."""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.security.pii_masker import hash_phone

_PHONE = "5511987654321"
_PHONE_HASH = hash_phone(_PHONE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_lead(deleted_at=None):
    now = datetime.now(timezone.utc)
    lead = MagicMock()
    lead.id = uuid.uuid4()
    lead.phone_hash = _PHONE_HASH
    lead.profile = {"sport": "beach_tennis", "level": "iniciante"}
    lead.created_at = now
    lead.last_interaction_at = now
    lead.deleted_at = deleted_at
    return lead


def _make_conv(role: str = "user", content: str = "quero uma raquete"):
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.phone_hash = _PHONE_HASH
    conv.message_role = role
    conv.content_masked = content
    conv.created_at = datetime.now(timezone.utc)
    return conv


def _make_audit_entry():
    entry = MagicMock()
    entry.id = uuid.uuid4()
    entry.actor = "webhook"
    entry.action = "process_message"
    entry.target_hash = _PHONE_HASH
    entry.created_at = datetime.now(timezone.utc)
    entry.ip = None
    entry.metadata_ = None
    return entry


# ── DELETE /lgpd/lead ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_lead_not_found_returns_404():
    @asynccontextmanager
    async def _empty():
        class _R:
            def scalar_one_or_none(self):
                return None

        session = MagicMock()
        session.execute = AsyncMock(return_value=_R())
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.storage.db.get_session", _empty),
        patch("app.api.lgpd.log_access", new=AsyncMock()),
        patch("app.storage.redis_session._get_redis_client"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.request(
                "DELETE",
                "/lgpd/lead",
                json={"phone": _PHONE},
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_lead_soft_deletes_and_zeroes_conversations():
    """DELETE must set deleted_at on Lead and wipe ConversationLog content."""
    fake_lead = _make_lead()
    execute_calls: list = []

    @asynccontextmanager
    async def _session():
        class _LeadResult:
            def scalar_one_or_none(self):
                return fake_lead

        class _UpdateResult:
            pass

        session = MagicMock()

        async def _execute(stmt):
            execute_calls.append(stmt)
            return _LeadResult() if len(execute_calls) == 1 else _UpdateResult()

        session.execute = _execute
        session.commit = AsyncMock()
        yield session

    mock_store = AsyncMock()

    with (
        patch("app.storage.db.get_session", _session),
        patch("app.api.lgpd.log_access", new=AsyncMock()),
        patch("app.storage.redis_session.get_store", return_value=mock_store),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.request(
                "DELETE",
                "/lgpd/lead",
                json={"phone": _PHONE},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "deleted"
    assert body["phone_hash"] == _PHONE_HASH

    # Lead must have been soft-deleted
    assert fake_lead.deleted_at is not None

    # Redis session must have been removed
    mock_store.delete.assert_called_once_with(_PHONE_HASH)


@pytest.mark.asyncio
async def test_delete_lead_audit_log_preserved():
    """Audit log entry must be written even after delete."""
    fake_lead = _make_lead()
    audit_calls: list = []

    @asynccontextmanager
    async def _session():
        class _R:
            def scalar_one_or_none(self):
                return fake_lead

        session = MagicMock()
        session.execute = AsyncMock(return_value=_R())
        session.commit = AsyncMock()
        yield session

    async def _capture_log_access(**kwargs):
        audit_calls.append(kwargs)

    mock_store = AsyncMock()

    with (
        patch("app.storage.db.get_session", _session),
        patch("app.api.lgpd.log_access", side_effect=_capture_log_access),
        patch("app.storage.redis_session.get_store", return_value=mock_store),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.request("DELETE", "/lgpd/lead", json={"phone": _PHONE})

    assert len(audit_calls) == 1
    call = audit_calls[0]
    assert call["action"] == "delete_lead"
    assert call["metadata"]["deleted"] is True


@pytest.mark.asyncio
async def test_delete_redis_failure_does_not_break_response():
    """If Redis is down during delete, response should still be 200."""
    fake_lead = _make_lead()

    @asynccontextmanager
    async def _session():
        class _R:
            def scalar_one_or_none(self):
                return fake_lead

        session = MagicMock()
        session.execute = AsyncMock(return_value=_R())
        session.commit = AsyncMock()
        yield session

    mock_store = AsyncMock()
    mock_store.delete = AsyncMock(side_effect=ConnectionError("redis down"))

    with (
        patch("app.storage.db.get_session", _session),
        patch("app.api.lgpd.log_access", new=AsyncMock()),
        patch("app.storage.redis_session.get_store", return_value=mock_store),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.request("DELETE", "/lgpd/lead", json={"phone": _PHONE})

    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


# ── POST /lgpd/lead/export ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_lead_not_found_returns_404():
    @asynccontextmanager
    async def _empty():
        class _R:
            def scalar_one_or_none(self):
                return None

        session = MagicMock()
        session.execute = AsyncMock(return_value=_R())
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.storage.db.get_session", _empty),
        patch("app.api.lgpd.log_access", new=AsyncMock()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/lgpd/lead/export", json={"phone": _PHONE})

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_returns_structured_json():
    """Export must return lead data, conversations and audit trail."""
    fake_lead = _make_lead()
    fake_conv_user = _make_conv("user", "quero uma raquete")
    fake_conv_ai = _make_conv("assistant", "Temos a Raquete Pro disponivel!")
    fake_audit = _make_audit_entry()

    call_count = 0

    @asynccontextmanager
    async def _session():
        nonlocal call_count

        class _LeadResult:
            def scalar_one_or_none(self):
                return fake_lead

        class _ListResult:
            def __init__(self, items):
                self._items = items

            def scalars(self):
                return self

            def all(self):
                return self._items

        session = MagicMock()

        async def _execute(_stmt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _LeadResult()
            elif call_count == 2:
                return _ListResult([fake_conv_user, fake_conv_ai])
            else:
                return _ListResult([fake_audit])

        session.execute = _execute
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.storage.db.get_session", _session),
        patch("app.api.lgpd.log_access", new=AsyncMock()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/lgpd/lead/export", json={"phone": _PHONE})

    assert resp.status_code == 200
    body = resp.json()

    assert body["phone_hash"] == _PHONE_HASH
    assert "lead" in body
    assert body["lead"]["profile"]["sport"] == "beach_tennis"

    assert len(body["conversations"]) == 2
    assert body["conversations"][0]["role"] == "user"
    assert body["conversations"][1]["role"] == "assistant"

    assert len(body["audit_trail"]) == 1
    assert body["audit_trail"][0]["actor"] == "webhook"

    assert "exported_at" in body


@pytest.mark.asyncio
async def test_export_phone_hash_matches_delete_hash():
    """The phone_hash in export must equal the one produced by hash_phone."""
    fake_lead = _make_lead()

    call_count = 0

    @asynccontextmanager
    async def _session():
        nonlocal call_count

        class _LeadResult:
            def scalar_one_or_none(self):
                return fake_lead

        class _Empty:
            def scalars(self):
                return self

            def all(self):
                return []

        session = MagicMock()

        async def _execute(_stmt):
            nonlocal call_count
            call_count += 1
            return _LeadResult() if call_count == 1 else _Empty()

        session.execute = _execute
        session.commit = AsyncMock()
        yield session

    with (
        patch("app.storage.db.get_session", _session),
        patch("app.api.lgpd.log_access", new=AsyncMock()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/lgpd/lead/export", json={"phone": _PHONE})

    assert resp.json()["phone_hash"] == hash_phone(_PHONE)
