"""DEV_RESET_HOOK — tests for the /reset magic command.

⚠️ DEV/PILOT ONLY. Drop this file when removing the /reset feature; see
app/agent/reset.py for the full removal checklist.
"""
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis as fakeredis_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.reset import is_reset_command, reset_conversation
from app.main import app
from app.security.pii_masker import hash_phone


_TOKEN = "test-webhook-token-abc"
_PHONE = "5511987654321"


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
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", _TOKEN)
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
