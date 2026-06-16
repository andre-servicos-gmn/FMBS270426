"""Tests for POST /webhook/whatsapp.

All external I/O (graph, Evolution API, DB, Redis) is mocked so tests run
without live services.  Background tasks execute synchronously inside the
ASGI test transport, so assertions after client.post() are safe.
"""
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis_aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.main import app
from app.storage.models import ConversationLog

# ── Constants ─────────────────────────────────────────────────────────────────

_TOKEN = "test-webhook-token-abc"
_PHONE = "5511987654321"
_JID = f"{_PHONE}@s.whatsapp.net"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _payload(
    text: str,
    message_id: str = "MSGID001",
    phone: str = _PHONE,
    from_me: bool = False,
    event: str = "messages.upsert",
) -> dict:
    return {
        "event": event,
        "instance": "test-instance",
        "data": {
            "key": {
                "remoteJid": f"{phone}@s.whatsapp.net",
                "fromMe": from_me,
                "id": message_id,
            },
            "message": {"conversation": text},
            "messageType": "conversation",
            "messageTimestamp": 1700000000,
        },
    }


def _headers(token: str = _TOKEN) -> dict:
    return {"apikey": token}


def _make_graph_result(ai_text: str = "Olá! Posso te ajudar.") -> dict:
    return {
        "messages": [
            HumanMessage(content="mensagem do usuário"),
            AIMessage(content=ai_text),
        ],
        "phone_hash": "fakehash",
        "intent": "faq",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_redis():
    """In-process Redis substitute — reset between tests."""
    return fakeredis_aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def override_token(monkeypatch):
    """Inject a known webhook token into settings.

    Sprint 2.7.2 — also configures the debounce buffer for cap=1 so every
    incoming text message flushes IMMEDIATELY (matching pre-2.7.2 behaviour
    that these tests assume). Tests that specifically exercise debounce
    grouping live in tests/test_debounce_buffer.py with their own fixture.
    """
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", _TOKEN)
    monkeypatch.setenv("MESSAGE_DEBOUNCE_CAP", "1")
    monkeypatch.setenv("MESSAGE_DEBOUNCE_MS", "10")
    monkeypatch.setenv("MESSAGE_DEBOUNCE_HARD_TTL_MS", "100")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.api.webhook import _reset_debounce_buffer
    _reset_debounce_buffer()
    yield
    get_settings.cache_clear()
    _reset_debounce_buffer()


# ── Shared mock context manager for DB session ────────────────────────────────

def _make_db_mock() -> tuple:
    """Returns (asynccontextmanager, list-of-added-objects) for DB patching."""
    added: list = []

    @asynccontextmanager
    async def _mock_get_session():
        session = MagicMock()
        session.add = lambda obj: added.append(obj)
        session.commit = AsyncMock()
        yield session

    return _mock_get_session, added


# ── 1. Auth modes (Sprint 1.4.1: token now optional in pilot) ────────────────

@pytest.fixture
def no_token(monkeypatch):
    """Force EVOLUTION_WEBHOOK_TOKEN to empty so auth bypass kicks in.

    Sprint 2.7.2 — same cap=1 trick as ``override_token`` so existing
    webhook tests bypass the debounce buffer.
    """
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", "")
    monkeypatch.setenv("MESSAGE_DEBOUNCE_CAP", "1")
    monkeypatch.setenv("MESSAGE_DEBOUNCE_MS", "10")
    monkeypatch.setenv("MESSAGE_DEBOUNCE_HARD_TTL_MS", "100")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.api.webhook import _reset_debounce_buffer
    _reset_debounce_buffer()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_webhook_accepts_when_no_token_configured(no_token, fake_redis, caplog):
    """With no token in settings, requests are accepted without an apikey header
    AND a structured WARNING is logged on every call."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())
    mock_session_cm, _ = _make_db_mock()

    caplog.set_level("WARNING", logger="app.api.webhook")

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", mock_session_cm),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Request has NO apikey header — should still succeed.
            resp = await client.post("/webhook/whatsapp", json=_payload("oi"))

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    # Sprint 2.7 renamed the log key to `webhook_auth_disabled` (snake_case
    # for grepability), aligned with `webhook_auth_ok` / `webhook_auth_failed`.
    assert any(
        "webhook_auth_disabled" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    ), "expected structured warning to be emitted on every bypass"


@pytest.mark.asyncio
async def test_webhook_rejects_when_token_required_and_header_absent(override_token):
    """With a token configured, a request without any apikey header is rejected (401)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/webhook/whatsapp", json=_payload("oi"))
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_webhook_rejects_when_token_mismatch(override_token):
    """With a token configured, a wrong apikey header is rejected with 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/webhook/whatsapp",
            json=_payload("oi"),
            headers={"apikey": "wrong-token"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_webhook_accepts_when_token_matches(override_token, fake_redis):
    """With a token configured and the request carrying the correct apikey, returns 200."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())
    mock_session_cm, _ = _make_db_mock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", mock_session_cm),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi"),
                headers=_headers(),
            )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── 2. Idempotency: same message_id processed only once ──────────────────────

@pytest.mark.asyncio
async def test_duplicate_message_not_processed_twice(override_token, fake_redis):
    """Second request with the same message_id must return duplicate=True
    and NOT invoke the graph a second time."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())
    mock_session_cm, _ = _make_db_mock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", mock_session_cm),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="SAME-ID-001"),
                headers=_headers(),
            )
            r2 = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="SAME-ID-001"),
                headers=_headers(),
            )

    assert r1.status_code == 200
    assert r1.json().get("duplicate") is not True  # first request processed

    assert r2.status_code == 200
    assert r2.json()["duplicate"] is True          # second request skipped

    # Graph must have been called exactly once
    assert mock_graph.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_different_message_ids_both_processed(override_token, fake_redis):
    """Two requests with different message_ids must both invoke the graph."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())
    mock_session_cm, _ = _make_db_mock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", mock_session_cm),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="MSG-A"),
                headers=_headers(),
            )
            await client.post(
                "/webhook/whatsapp",
                json=_payload("oi de novo", message_id="MSG-B"),
                headers=_headers(),
            )

    assert mock_graph.ainvoke.call_count == 2


# ── 3. PII masking in ConversationLog ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_cpf_masked_in_conversation_log(override_token, fake_redis):
    """User message containing a CPF must be stored with [CPF] placeholder,
    never with the raw digits."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result("Entendido, posso ajudar!"))
    mock_session_cm, added_objects = _make_db_mock()

    message_with_cpf = "meu cpf é 123.456.789-00 e quero uma raquete"

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", mock_session_cm),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
        # audit_log also writes to DB — redirect to the same mock
        patch("app.storage.db.get_session", mock_session_cm),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload(message_with_cpf, message_id="CPF-TEST-001"),
                headers=_headers(),
            )

    assert resp.status_code == 200

    # Background task should have completed before this line
    conv_logs = [o for o in added_objects if isinstance(o, ConversationLog)]
    assert len(conv_logs) >= 2, f"Expected ≥2 ConversationLog rows, got {len(conv_logs)}"

    user_log = next(
        (l for l in conv_logs if l.message_role == "user"), None
    )
    assert user_log is not None, "No user ConversationLog was saved"

    # Raw CPF must not appear in stored content
    assert "123.456.789-00" not in user_log.content_masked
    # Masked placeholder must be present
    assert "[CPF]" in user_log.content_masked


@pytest.mark.asyncio
async def test_ai_response_also_masked(override_token, fake_redis):
    """Even if the AI somehow echoes sensitive data, the stored response is masked."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value=_make_graph_result("Recebi seu CPF 123.456.789-00, mas não devo guardar isso.")
    )
    mock_session_cm, added_objects = _make_db_mock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", mock_session_cm),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        MockEvo.return_value.send_text = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload("teste", message_id="MASK-AI-001"),
                headers=_headers(),
            )

    conv_logs = [o for o in added_objects if isinstance(o, ConversationLog)]
    assistant_log = next((l for l in conv_logs if l.message_role == "assistant"), None)
    assert assistant_log is not None
    assert "123.456.789-00" not in assistant_log.content_masked
    assert "[CPF]" in assistant_log.content_masked


# ── 4. Ignored event types ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_upsert_event_ignored(override_token, fake_redis):
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", event="messages.update"),
                headers=_headers(),
            )

    assert resp.status_code == 200
    assert resp.json()["reason"] == "event_type"
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_from_me_message_ignored(override_token, fake_redis):
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result())

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", from_me=True),
                headers=_headers(),
            )

    assert resp.status_code == 200
    assert resp.json()["reason"] == "from_me"
    mock_graph.ainvoke.assert_not_called()


# ── 5. Evolution send called with correct phone ───────────────────────────────

@pytest.mark.asyncio
async def test_evolution_send_called_with_phone(override_token, fake_redis):
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=_make_graph_result("Resposta teste"))
    mock_session_cm, _ = _make_db_mock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", mock_session_cm),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        mock_evo_instance = AsyncMock()
        MockEvo.return_value = mock_evo_instance
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload("quero uma raquete", message_id="EVO-TEST-001"),
                headers=_headers(),
            )

    # Sprint 1.6: webhook delegates to send_text_blocks. For a short AI reply
    # the splitter returns a single block, but the API call is now send_text_blocks.
    mock_evo_instance.send_text_blocks.assert_called_once()
    call_args = mock_evo_instance.send_text_blocks.call_args
    assert call_args[0][0] == _PHONE                # correct phone number
    blocks_sent = call_args[0][1]
    assert isinstance(blocks_sent, list)
    assert "Resposta teste" in " ".join(blocks_sent)
