"""Sprint 1.12 — audio (Whisper) + media kinds dispatching.

Covers _classify_message, transcribe_audio (mocked OpenAI), and the webhook
integration that ties audioMessage payloads to the existing text flow.
"""
import base64
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.media_processor import _mime_to_extension, transcribe_audio
from app.api.webhook import _classify_message
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


# ── _classify_message — pure dispatch logic ──────────────────────────────────

def test_extract_text_message_type():
    result = _classify_message({"conversation": "oi"})
    assert result == {"kind": "text", "text": "oi"}


def test_extract_extended_text_message_type():
    result = _classify_message({"extendedTextMessage": {"text": "olá novamente"}})
    assert result == {"kind": "text", "text": "olá novamente"}


def test_extract_audio_message_type():
    payload = {"audioMessage": {"url": "https://...", "mimetype": "audio/ogg"}}
    result = _classify_message(payload)
    assert result == {"kind": "audio", "text": None}


def test_extract_image_message_type():
    payload = {"imageMessage": {"url": "https://...", "mimetype": "image/jpeg"}}
    result = _classify_message(payload)
    assert result == {"kind": "image", "text": None}


def test_extract_document_message_type():
    payload = {"documentMessage": {"fileName": "spec.pdf"}}
    result = _classify_message(payload)
    assert result == {"kind": "document", "text": None}


def test_extract_sticker_message_type():
    payload = {"stickerMessage": {"url": "https://..."}}
    result = _classify_message(payload)
    assert result == {"kind": "sticker", "text": None}


def test_extract_video_message_type():
    payload = {"videoMessage": {"url": "https://...", "mimetype": "video/mp4"}}
    result = _classify_message(payload)
    assert result == {"kind": "video", "text": None}


def test_extract_unknown_message_type():
    payload = {"reactionMessage": {"emoji": "👍"}}
    result = _classify_message(payload)
    assert result == {"kind": "unknown", "text": None}


# ── mime → extension mapping ─────────────────────────────────────────────────

def test_mime_to_extension_strips_codec_parameter():
    """WhatsApp sends 'audio/ogg; codecs=opus' — the codec suffix must be stripped."""
    assert _mime_to_extension("audio/ogg; codecs=opus") == "ogg"


def test_mime_to_extension_handles_common_formats():
    assert _mime_to_extension("audio/mpeg") == "mp3"
    assert _mime_to_extension("audio/mp4") == "m4a"
    assert _mime_to_extension("audio/wav") == "wav"
    assert _mime_to_extension("audio/webm") == "webm"


def test_mime_to_extension_defaults_to_ogg_for_unknown():
    assert _mime_to_extension("audio/weird-codec") == "ogg"
    assert _mime_to_extension("") == "ogg"


# ── transcribe_audio — mocked OpenAI ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_transcribe_audio_returns_text():
    """Happy path: Whisper returns a transcription object; we return .text trimmed."""
    fake_response = MagicMock()
    fake_response.text = "  oi quero uma raquete de beach tennis  "

    with patch("app.adapters.media_processor.AsyncOpenAI") as MockClient:
        instance = MagicMock()
        instance.audio.transcriptions.create = AsyncMock(return_value=fake_response)
        MockClient.return_value = instance

        text = await transcribe_audio(b"fake-audio-bytes", "audio/ogg; codecs=opus")

    assert text == "oi quero uma raquete de beach tennis"
    # Check the SDK was called with PT language + the right model.
    create_call = instance.audio.transcriptions.create.call_args
    kwargs = create_call.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["language"] == "pt"
    # file tuple: (filename, bytes, mimetype)
    filename, content, mimetype = kwargs["file"]
    assert filename == "audio.ogg"
    assert content == b"fake-audio-bytes"
    assert mimetype == "audio/ogg; codecs=opus"


@pytest.mark.asyncio
async def test_transcribe_audio_empty_returns_empty_string():
    """Silent/inaudible audio → Whisper returns empty .text → we return ''."""
    fake_response = MagicMock()
    fake_response.text = ""

    with patch("app.adapters.media_processor.AsyncOpenAI") as MockClient:
        instance = MagicMock()
        instance.audio.transcriptions.create = AsyncMock(return_value=fake_response)
        MockClient.return_value = instance

        text = await transcribe_audio(b"...", "audio/ogg")

    assert text == ""


# ── Webhook integration ──────────────────────────────────────────────────────

def _payload(message: dict, message_id: str = "MSG-AUDIO-001") -> dict:
    return {
        "event": "messages.upsert",
        "instance": "test-instance",
        "data": {
            "key": {"remoteJid": _JID, "fromMe": False, "id": message_id},
            "message": message,
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


@pytest.fixture
def fake_redis():
    return fakeredis_aioredis.FakeRedis(decode_responses=True)


@pytest.mark.asyncio
async def test_webhook_audio_full_flow(override_token, fake_redis):
    """End-to-end: audio payload → download → Whisper → graph invoked with text."""
    from langchain_core.messages import AIMessage, HumanMessage

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "messages": [HumanMessage(content="..."), AIMessage(content="Resposta do agente.")],
        "phone_hash": "x",
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
    })

    audio_bytes = b"\x00\x01\x02 fake ogg opus"

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo_instance = AsyncMock()
        evo_instance.get_media_base64 = AsyncMock(return_value=(audio_bytes, "audio/ogg; codecs=opus"))
        evo_instance.send_text = AsyncMock()
        evo_instance.send_text_blocks = AsyncMock()
        MockEvo.return_value = evo_instance

        mock_whisper.return_value = "oi quero uma raquete de beach tennis"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {"mimetype": "audio/ogg; codecs=opus"}}),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json()["kind"] == "audio"

    # The transcribed text must have reached the graph as the user message.
    evo_instance.get_media_base64.assert_called_once()
    mock_whisper.assert_called_once_with(audio_bytes, "audio/ogg; codecs=opus")
    mock_graph.ainvoke.assert_called_once()
    invoked_state = mock_graph.ainvoke.call_args.args[0]
    user_msgs = [m for m in invoked_state["messages"] if isinstance(m, HumanMessage)]
    assert any("raquete" in m.content.lower() for m in user_msgs)


@pytest.mark.asyncio
async def test_transcribe_audio_empty_triggers_fallback_at_webhook(override_token, fake_redis):
    """Empty Whisper transcription → 'Não consegui entender o áudio' canned reply,
    graph NOT invoked."""
    mock_graph = AsyncMock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.transcribe_audio", new_callable=AsyncMock) as mock_whisper,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo_instance = AsyncMock()
        evo_instance.get_media_base64 = AsyncMock(return_value=(b"silent", "audio/ogg"))
        evo_instance.send_text = AsyncMock()
        evo_instance.send_text_blocks = AsyncMock()
        MockEvo.return_value = evo_instance
        mock_whisper.return_value = ""  # silent audio

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {"mimetype": "audio/ogg"}}, message_id="EMPTY-1"),
                headers={"apikey": _TOKEN},
            )

    evo_instance.send_text.assert_called_once()
    canned = evo_instance.send_text.call_args.args[1]
    assert "Não consegui entender" in canned
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_image_not_racket_returns_canned_response(override_token, fake_redis):
    """Sprint 3.11 — image without a racket → canned 'não identifiquei raquete'
    reply, graph NOT invoked. (The full vision flow lives in
    tests/test_image_identification.py.)"""
    mock_graph = AsyncMock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch(
            "app.api.webhook.identify_racket_image",
            new_callable=AsyncMock,
            return_value={"is_racket": False, "brand": None, "model": None, "confidence": None},
        ),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo_instance = AsyncMock()
        evo_instance.get_media_base64 = AsyncMock(return_value=(b"jpeg-bytes", "image/jpeg"))
        evo_instance.send_text = AsyncMock()
        MockEvo.return_value = evo_instance

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload({"imageMessage": {"mimetype": "image/jpeg"}}, message_id="IMG-1"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json()["kind"] == "image"
    evo_instance.send_text.assert_called_once()
    msg = evo_instance.send_text.call_args.args[1]
    assert "raquete" in msg.lower()
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_document_returns_canned_response(override_token, fake_redis):
    """Document payload → canned 'can't open docs' reply, graph NOT invoked."""
    mock_graph = AsyncMock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo_instance = AsyncMock()
        evo_instance.send_text = AsyncMock()
        MockEvo.return_value = evo_instance

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload({"documentMessage": {"fileName": "manual.pdf"}}, message_id="DOC-1"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json()["kind"] == "document"
    evo_instance.send_text.assert_called_once()
    msg = evo_instance.send_text.call_args.args[1]
    assert "documento" in msg.lower()
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_sticker_silently_ignored(override_token, fake_redis):
    """Sticker → 200 but no reply and no graph invocation."""
    mock_graph = AsyncMock()
    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo_instance = AsyncMock()
        evo_instance.send_text = AsyncMock()
        MockEvo.return_value = evo_instance

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload({"stickerMessage": {"url": "..."}}, message_id="STK-1"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json()["reason"] == "sticker"
    evo_instance.send_text.assert_not_called()
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_audio_download_failure_falls_back(override_token, fake_redis):
    """If Evolution getMedia fails, customer gets an audio-failure canned reply."""
    mock_graph = AsyncMock()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo_instance = AsyncMock()
        evo_instance.get_media_base64 = AsyncMock(side_effect=RuntimeError("boom"))
        evo_instance.send_text = AsyncMock()
        MockEvo.return_value = evo_instance

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload({"audioMessage": {}}, message_id="BOOM-1"),
                headers={"apikey": _TOKEN},
            )

    evo_instance.send_text.assert_called_once()
    canned = evo_instance.send_text.call_args.args[1]
    assert "problema" in canned.lower() or "texto" in canned.lower()
    mock_graph.ainvoke.assert_not_called()


# ── Evolution.get_media_base64 — small unit check on payload parsing ─────────

@pytest.mark.asyncio
async def test_get_media_base64_decodes_response():
    """The adapter returns (decoded_bytes, mimetype) from the Evolution JSON."""
    from app.adapters.evolution import EvolutionClient

    raw = b"hello-binary"
    encoded = base64.b64encode(raw).decode()
    fake_json = {"base64": encoded, "mimetype": "audio/ogg; codecs=opus"}

    with patch("app.config.get_settings") as gs:
        gs.return_value = MagicMock(
            evolution_api_url="http://localhost:8080",
            evolution_api_key="k",
            evolution_instance="inst",
        )
        client = EvolutionClient()

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json = MagicMock(return_value=fake_json)
    fake_resp.raise_for_status = MagicMock()

    fake_httpx = AsyncMock()
    fake_httpx.__aenter__.return_value.post = AsyncMock(return_value=fake_resp)

    with patch("httpx.AsyncClient", return_value=fake_httpx):
        out_bytes, mime = await client.get_media_base64({"id": "MID", "remoteJid": "x@s"})

    assert out_bytes == raw
    assert mime == "audio/ogg; codecs=opus"
