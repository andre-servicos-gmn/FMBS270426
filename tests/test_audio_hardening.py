"""Sprint 3.10 — audio hardening: duration guard, size cap, rate limit, cache.

Complements tests/test_media.py (Sprint 1.12). Same webhook-integration style:
FastAPI app driven via httpx ASGITransport, Evolution/OpenAI/db mocked,
fakeredis behind app.storage.redis_session._get_redis_client.
"""
import hashlib
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis_aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.main import app

_TOKEN = "test-webhook-token-abc"
_PHONE = "5511987654321"
_JID = f"{_PHONE}@s.whatsapp.net"


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    session.add = lambda obj: None
    yield session


def _payload(message: dict, message_id: str = "MSG-HARD-001") -> dict:
    return {
        "event": "messages.upsert",
        "instance": "test-instance",
        "data": {
            "key": {"remoteJid": _JID, "fromMe": False, "id": message_id},
            "message": message,
            "messageTimestamp": 1700000000,
        },
    }


def _mock_graph() -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value={
        "messages": [HumanMessage(content="..."), AIMessage(content="Resposta do agente.")],
        "phone_hash": "x",
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
    })
    return graph


@pytest.fixture
def override_token(monkeypatch):
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", _TOKEN)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_redis():
    return fakeredis_aioredis.FakeRedis(decode_responses=True)


def _evo_mock(audio_bytes: bytes = b"fake-ogg-bytes") -> AsyncMock:
    evo = AsyncMock()
    evo.get_media_base64 = AsyncMock(return_value=(audio_bytes, "audio/ogg; codecs=opus"))
    evo.send_text = AsyncMock()
    evo.send_text_blocks = AsyncMock()
    return evo


# ── Guard: duration (pre-download, audioMessage.seconds) ─────────────────────

@pytest.mark.asyncio
async def test_audio_over_duration_rejected_before_download(override_token, fake_redis):
    """seconds > AUDIO_MAX_SECONDS → canned 'too long' reply; NO download,
    NO Whisper, NO graph."""
    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock()
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload(
                    {"audioMessage": {"mimetype": "audio/ogg", "seconds": 600}},
                    message_id="LONG-1",
                ),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json().get("rejected") == "too_long"
    evo.get_media_base64.assert_not_called()
    mock_whisper.assert_not_called()
    mock_graph.ainvoke.assert_not_called()
    evo.send_text.assert_called_once()
    assert "longo" in evo.send_text.call_args.args[1].lower()


@pytest.mark.asyncio
async def test_audio_within_duration_proceeds(override_token, fake_redis):
    """seconds under the cap → normal flow: download, Whisper, graph invoked."""
    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock()
        MockEvo.return_value = evo
        mock_whisper.return_value = "quero uma raquete intermediaria"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload(
                    {"audioMessage": {"mimetype": "audio/ogg", "seconds": 30}},
                    message_id="OK-30S",
                ),
                headers={"apikey": _TOKEN},
            )

    assert resp.json()["kind"] == "audio"
    evo.get_media_base64.assert_called_once()
    mock_whisper.assert_called_once()
    mock_graph.ainvoke.assert_called_once()


# ── Guard: size cap (post-download bytes) ────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_oversized_bytes_rejected(override_token, fake_redis, monkeypatch):
    """Downloaded blob above AUDIO_MAX_BYTES → canned 'too long' reply,
    Whisper NOT called."""
    from app.config import get_settings
    monkeypatch.setenv("AUDIO_MAX_BYTES", "10")
    get_settings.cache_clear()

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock(audio_bytes=b"x" * 100)  # 100 bytes > cap of 10
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {"mimetype": "audio/ogg"}}, message_id="BIG-1"),
                headers={"apikey": _TOKEN},
            )

    evo.get_media_base64.assert_called_once()
    mock_whisper.assert_not_called()
    mock_graph.ainvoke.assert_not_called()
    evo.send_text.assert_called_once()
    assert "longo" in evo.send_text.call_args.args[1].lower()


# ── Guard: rate limit per phone_hash ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_rate_limit_exceeded(override_token, fake_redis):
    """Counter already at the limit → canned rate-limit reply; NO download,
    NO Whisper."""
    from app.security.pii_masker import hash_phone

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        # Default limit is 10/h — seed the fixed-window counter at the cap so
        # this audio becomes the 11th.
        phone_hash = hash_phone(_PHONE)
        await fake_redis.set(f"audio_rate:{phone_hash}", "10")

        evo = _evo_mock()
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {"mimetype": "audio/ogg"}}, message_id="RATE-11"),
                headers={"apikey": _TOKEN},
            )

    evo.get_media_base64.assert_not_called()
    mock_whisper.assert_not_called()
    mock_graph.ainvoke.assert_not_called()
    evo.send_text.assert_called_once()
    assert "texto" in evo.send_text.call_args.args[1].lower()


@pytest.mark.asyncio
async def test_audio_rate_limit_fail_open_on_redis_error(override_token, fake_redis):
    """Redis down on the rate-limit check → audio still processed (fail-open)."""
    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch(
            "app.api.webhook.count_audio_message",
            new_callable=AsyncMock,
            side_effect=RuntimeError("redis down"),
        ),
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock()
        MockEvo.return_value = evo
        mock_whisper.return_value = "oi tudo bem"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {"mimetype": "audio/ogg"}}, message_id="FAILOPEN-1"),
                headers={"apikey": _TOKEN},
            )

    mock_whisper.assert_called_once()
    mock_graph.ainvoke.assert_called_once()


# ── Guard: transcript cache by content hash ──────────────────────────────────

@pytest.mark.asyncio
async def test_transcript_cache_hit_skips_whisper(override_token, fake_redis):
    """Cached transcript for identical bytes → Whisper NOT called, graph gets
    the cached text."""
    audio_bytes = b"identical-voice-note"
    sha = hashlib.sha256(audio_bytes).hexdigest()

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        await fake_redis.set(f"audio_transcript:{sha}", "quero ver bolas de beach tennis")

        evo = _evo_mock(audio_bytes=audio_bytes)
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {"mimetype": "audio/ogg"}}, message_id="CACHE-HIT-1"),
                headers={"apikey": _TOKEN},
            )

    mock_whisper.assert_not_called()
    mock_graph.ainvoke.assert_called_once()
    invoked_state = mock_graph.ainvoke.call_args.args[0]
    user_msgs = [m for m in invoked_state["messages"] if isinstance(m, HumanMessage)]
    assert any("bolas de beach tennis" in m.content for m in user_msgs)


@pytest.mark.asyncio
async def test_transcript_cache_written_on_miss(override_token, fake_redis):
    """Cache miss → Whisper called once and the transcript is stored with TTL."""
    audio_bytes = b"fresh-voice-note"
    sha = hashlib.sha256(audio_bytes).hexdigest()

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock(audio_bytes=audio_bytes)
        MockEvo.return_value = evo
        mock_whisper.return_value = "tem raquete de padel?"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {"mimetype": "audio/ogg"}}, message_id="CACHE-MISS-1"),
                headers={"apikey": _TOKEN},
            )

        mock_whisper.assert_called_once()
        cached = await fake_redis.get(f"audio_transcript:{sha}")
        assert cached == "tem raquete de padel?"
        ttl = await fake_redis.ttl(f"audio_transcript:{sha}")
        assert ttl > 0
