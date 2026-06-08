"""DEV_RESET_HOOK — tests for the /reset magic command.

⚠️ DEV/PILOT ONLY. Drop this file when removing the /reset feature; see
app/agent/reset.py for the full removal checklist.

Sprint 2.7 — /reset is now gated by ``RESET_ALLOWED_PHONES``. Tests cover
the allowed/denied paths plus the existing wipe + idempotency paths.
"""
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis as fakeredis_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.reset import is_reset_authorized, is_reset_command, reset_conversation
from app.main import app
from app.security.pii_masker import hash_phone


_TOKEN = "test-webhook-token-abc"
_PHONE = "5511987654321"
_UNAUTHORIZED_PHONE = "5511000000000"


# ── 1. is_reset_command — pure detection ─────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected",
    [
        ("/reset", True),
        ("/RESET", True),
        ("/Reset", True),
        (" /reset ", True),
        ("\t/reset\n", True),
        ("/reset agora", False),     # extra suffix
        ("preciso /reset", False),    # extra prefix
        ("reset", False),             # missing slash
        ("/resetar", False),          # not exact match
        ("", False),
        ("oi", False),
    ],
)
def test_is_reset_command_variants(text, expected):
    assert is_reset_command(text) is expected


# ── 2. reset_conversation — Redis cleanup ────────────────────────────────────

@pytest.mark.asyncio
async def test_reset_conversation_deletes_phone_hash_keys():
    """Every key whose name contains the phone_hash must be removed."""
    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)
    phone_hash = hash_phone(_PHONE)

    # Seed: 3 keys that should be wiped + 1 key that must survive.
    await fake.set(f"checkpoint:{phone_hash}:default:abc", "x")
    await fake.set(f"checkpoint_write:{phone_hash}:default:abc:0", "y")
    await fake.set(f"session:{phone_hash}", "z")
    await fake.set("processed_msg:UNRELATED-MSG-ID", "keep-me")

    with patch("app.agent.reset._get_redis_client", return_value=fake):
        deleted = await reset_conversation(phone_hash)

    assert deleted == 3
    assert await fake.get("processed_msg:UNRELATED-MSG-ID") == "keep-me"
    assert await fake.get(f"checkpoint:{phone_hash}:default:abc") is None
    assert await fake.get(f"session:{phone_hash}") is None


# ── 3. webhook — full path /reset → wipe + canned reply ──────────────────────

def _payload(text: str, message_id: str = "MSG-RESET-001") -> dict:
    return {
        "event": "messages.upsert",
        "instance": "test-instance",
        "data": {
            "key": {
                "remoteJid": f"{_PHONE}@s.whatsapp.net",
                "fromMe": False,
                "id": message_id,
            },
            "message": {"conversation": text},
            "messageType": "conversation",
            "messageTimestamp": 1700000000,
        },
    }


@pytest.fixture
def override_token(monkeypatch):
    """Sprint 2.7 — token + reset allowlist with _PHONE authorized. Existing
    tests that rely on /reset succeeding use this fixture."""
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", _TOKEN)
    monkeypatch.setenv("RESET_ALLOWED_PHONES", _PHONE)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def override_token_only(monkeypatch):
    """Webhook auth on, but RESET_ALLOWED_PHONES NOT set. Used by tests that
    verify /reset is denied for everyone in this state."""
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", _TOKEN)
    monkeypatch.delenv("RESET_ALLOWED_PHONES", raising=False)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def no_webhook_token(monkeypatch):
    """Webhook auth disabled (empty token). Mirrors the dev-pilot default."""
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", "")
    monkeypatch.setenv("RESET_ALLOWED_PHONES", _PHONE)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_webhook_reset_command_wipes_redis_and_replies(override_token):
    """End-to-end: /reset triggers reset_conversation + 'Conversa resetada ✅' reply,
    and DOES NOT invoke the agent graph."""
    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)
    phone_hash = hash_phone(_PHONE)

    # Seed a checkpoint-like key + a session key + an unrelated key.
    await fake.set(f"checkpoint:{phone_hash}:ns:01", "state")
    await fake.set(f"session:{phone_hash}", "envelope")
    await fake.set("processed_msg:OTHER", "keep")

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.agent.reset._get_redis_client", return_value=fake),
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        mock_evo_instance = AsyncMock()
        MockEvo.return_value = mock_evo_instance
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("/reset"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "reset": True}

    # Graph must NOT have run.
    mock_graph.ainvoke.assert_not_called()

    # Canned reply was sent via Evolution.
    mock_evo_instance.send_text.assert_called_once()
    call_args = mock_evo_instance.send_text.call_args
    assert call_args[0][0] == _PHONE
    assert call_args[0][1] == "Conversa resetada ✅"

    # Phone-hash keys gone, unrelated key intact.
    assert await fake.get(f"checkpoint:{phone_hash}:ns:01") is None
    assert await fake.get(f"session:{phone_hash}") is None
    assert await fake.get("processed_msg:OTHER") == "keep"


@pytest.mark.asyncio
async def test_webhook_non_reset_message_does_not_trigger_reset(override_token):
    """A regular message must NOT call reset_conversation."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "phone_hash": "x",
            "intent": "smalltalk",
            "player_profile": {},
            "recommended_products": [],
            "needs_handoff": False,
            "handoff_reason": None,
        }
    )
    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.reset_conversation") as mock_reset,
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi tudo bem", message_id="MSG-NORMAL-001"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json().get("reset") is not True
    mock_reset.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
# Sprint 2.7 — RESET_ALLOWED_PHONES allowlist + EVOLUTION_WEBHOOK_TOKEN auth
# ════════════════════════════════════════════════════════════════════════════


# ── is_reset_authorized ─────────────────────────────────────────────────────

def test_reset_authorized_empty_allowlist_denies_all(monkeypatch):
    monkeypatch.setenv("RESET_ALLOWED_PHONES", "")
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        assert is_reset_authorized(_PHONE) is False
        assert is_reset_authorized(_UNAUTHORIZED_PHONE) is False
        assert is_reset_authorized("") is False
    finally:
        get_settings.cache_clear()


def test_reset_authorized_matches_listed_phone(monkeypatch):
    monkeypatch.setenv("RESET_ALLOWED_PHONES", f"{_PHONE},{_UNAUTHORIZED_PHONE}")
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        assert is_reset_authorized(_PHONE) is True
        assert is_reset_authorized(_UNAUTHORIZED_PHONE) is True
        assert is_reset_authorized("5511555555555") is False
    finally:
        get_settings.cache_clear()


def test_reset_authorized_tolerates_cosmetic_formatting(monkeypatch):
    """Spaces, dashes, and a leading '+' in the env var must not break
    the digit-only comparison."""
    monkeypatch.setenv("RESET_ALLOWED_PHONES", "  +55 (11) 98765-4321 ")
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        assert is_reset_authorized("5511987654321") is True
    finally:
        get_settings.cache_clear()


# ── End-to-end /reset routing through the webhook ───────────────────────────

@pytest.mark.asyncio
async def test_reset_denied_for_unauthorized_phone(override_token_only, caplog):
    """RESET_ALLOWED_PHONES unset → ANY /reset is denied silently. The
    webhook acks 200 but no reset is dispatched and no client reply is sent."""
    import logging
    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock()

    caplog.set_level(logging.WARNING, logger="app.api.webhook")

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.reset_conversation") as mock_reset,
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("/reset", message_id="MSG-DENIED-001"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "reset": "denied"}, body

    # Critical: NOTHING was reset, NOTHING was replied.
    mock_reset.assert_not_called()
    MockEvo.return_value.send_text.assert_not_called()
    mock_graph.ainvoke.assert_not_called()

    # Diagnostic log present.
    assert any(
        "reset_denied" in rec.message and "unauthorized" in rec.message
        for rec in caplog.records
    ), "expected 'reset_denied ... unauthorized' warning"


@pytest.mark.asyncio
async def test_reset_allowed_for_authorized_phone(override_token):
    """Same /reset payload but with _PHONE in RESET_ALLOWED_PHONES → reset
    fires + canned reply (this is the existing happy-path, re-asserted)."""
    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.agent.reset._get_redis_client", return_value=fake),
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("/reset", message_id="MSG-ALLOWED-001"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "reset": True}
    MockEvo.return_value.send_text.assert_called_once()


# ── EVOLUTION_WEBHOOK_TOKEN auth ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_rejects_invalid_token(override_token, caplog):
    """Token configured + wrong header → 401 with 'token_mismatch' log."""
    import logging
    caplog.set_level(logging.WARNING, logger="app.api.webhook")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/webhook/whatsapp",
            json=_payload("oi", message_id="MSG-BAD-AUTH-001"),
            headers={"apikey": "wrong-token-xyz"},
        )

    assert resp.status_code == 401
    assert any(
        "webhook_auth_failed" in rec.message and "token_mismatch" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_webhook_rejects_missing_header(override_token, caplog):
    """Token configured + NO header → 401 with 'missing_header' log."""
    import logging
    caplog.set_level(logging.WARNING, logger="app.api.webhook")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/webhook/whatsapp",
            json=_payload("oi", message_id="MSG-NO-AUTH-001"),
            # no apikey header
        )

    assert resp.status_code == 401
    assert any(
        "webhook_auth_failed" in rec.message and "missing_header" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_webhook_accepts_valid_token(override_token, caplog):
    """Token configured + correct header → 200 + 'webhook_auth_ok' log."""
    import logging
    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "phone_hash": "x",
            "intent": "smalltalk",
            "player_profile": {},
            "recommended_products": [],
            "needs_handoff": False,
            "handoff_reason": None,
        }
    )

    caplog.set_level(logging.INFO, logger="app.api.webhook")

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="MSG-GOOD-AUTH-001"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert any("webhook_auth_ok" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_webhook_open_when_token_not_configured(no_webhook_token, caplog):
    """EVOLUTION_WEBHOOK_TOKEN empty → request accepted (dev-pilot mode),
    warning logged so the operator notices."""
    import logging
    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "phone_hash": "x",
            "intent": "smalltalk",
            "player_profile": {},
            "recommended_products": [],
            "needs_handoff": False,
            "handoff_reason": None,
        }
    )

    caplog.set_level(logging.WARNING, logger="app.api.webhook")

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="MSG-OPEN-001"),
                # no apikey header — allowed because token unset
            )

    assert resp.status_code == 200
    assert any("webhook_auth_disabled" in rec.message for rec in caplog.records)
