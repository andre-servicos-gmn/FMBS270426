"""Sprint 3.11 — racket identification from customer photos (GPT-4o vision).

Covers:
- identify_racket_image (mocked OpenAI): JSON parsing, defaults, caption masking.
- Webhook integration: happy path (photo → vision → graph as product inquiry),
  not-a-racket / unidentified canned replies, rate limit, size cap,
  identification cache hit/miss, vision failure fallback.
- Triage short-circuit for image_product_query.
- recommend_node photo-aware wording.

Same webhook-integration style as tests/test_audio_hardening.py: FastAPI app
driven via httpx ASGITransport, Evolution/OpenAI/db mocked, fakeredis behind
app.storage.redis_session._get_redis_client.
"""
import hashlib
import json
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

_RACKET_ID = {
    "is_racket": True,
    "brand": "Shark",
    "model": "Attack 2024",
    "confidence": "high",
}


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    session.add = lambda obj: None
    yield session


def _payload(message: dict, message_id: str = "MSG-IMG-001") -> dict:
    return {
        "event": "messages.upsert",
        "instance": "test-instance",
        "data": {
            "key": {"remoteJid": _JID, "fromMe": False, "id": message_id},
            "message": message,
            "messageTimestamp": 1700000000,
        },
    }


def _image_payload(message_id: str = "MSG-IMG-001", caption: str | None = None) -> dict:
    image_msg: dict = {"mimetype": "image/jpeg"}
    if caption is not None:
        image_msg["caption"] = caption
    return _payload({"imageMessage": image_msg}, message_id=message_id)


def _mock_graph() -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value={
        "messages": [HumanMessage(content="..."), AIMessage(content="Resposta do agente.")],
        "phone_hash": "x",
        "intent": "product_inquiry",
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


def _evo_mock(image_bytes: bytes = b"fake-jpeg-bytes") -> AsyncMock:
    evo = AsyncMock()
    evo.get_media_base64 = AsyncMock(return_value=(image_bytes, "image/jpeg"))
    evo.send_text = AsyncMock()
    evo.send_text_blocks = AsyncMock()
    return evo


# ── identify_racket_image — mocked OpenAI ────────────────────────────────────

def _vision_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    return resp


@pytest.mark.asyncio
async def test_identify_racket_image_parses_model_output():
    from app.adapters.media_processor import identify_racket_image

    with patch("app.adapters.media_processor.AsyncOpenAI") as MockClient:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_vision_response(_RACKET_ID)
        )
        MockClient.return_value = instance

        result = await identify_racket_image(b"jpeg", "image/jpeg")

    assert result == _RACKET_ID
    kwargs = instance.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    # The image must go as a base64 data URL.
    user_content = kwargs["messages"][1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_identify_racket_image_masks_caption_pii():
    """Caption travels to the vision API PII-masked."""
    from app.adapters.media_processor import identify_racket_image

    with patch("app.adapters.media_processor.AsyncOpenAI") as MockClient:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_vision_response(_RACKET_ID)
        )
        MockClient.return_value = instance

        await identify_racket_image(
            b"jpeg", "image/jpeg", caption="me chama no (11) 98765-4321, tem essa?"
        )

    user_content = instance.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    caption_part = next(p for p in user_content if p.get("type") == "text")
    assert "98765" not in caption_part["text"]
    assert "tem essa?" in caption_part["text"]


@pytest.mark.asyncio
async def test_identify_racket_image_defaults_missing_fields():
    """Model omitting fields → negative-case defaults, no KeyError."""
    from app.adapters.media_processor import identify_racket_image

    with patch("app.adapters.media_processor.AsyncOpenAI") as MockClient:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(
            return_value=_vision_response({"is_racket": True})
        )
        MockClient.return_value = instance

        result = await identify_racket_image(b"jpeg", "image/jpeg")

    assert result == {"is_racket": True, "brand": None, "model": None, "confidence": None}


@pytest.mark.asyncio
async def test_identify_racket_image_raises_on_non_json():
    from app.adapters.media_processor import identify_racket_image

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "desculpe, não consigo analisar"

    with patch("app.adapters.media_processor.AsyncOpenAI") as MockClient:
        instance = MagicMock()
        instance.chat.completions.create = AsyncMock(return_value=resp)
        MockClient.return_value = instance

        with pytest.raises(ValueError):
            await identify_racket_image(b"jpeg", "image/jpeg")


# ── Webhook integration — happy path ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_image_identified_flows_to_graph(override_token, fake_redis):
    """Photo of a known racket → graph invoked with the synthetic 'brand
    model' query AND image_product_query=True."""
    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch(
            "app.api.webhook.identify_racket_image",
            new_callable=AsyncMock,
            return_value=dict(_RACKET_ID),
        ) as mock_vision,
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock()
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_image_payload(caption="tem essa?"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json()["kind"] == "image"
    evo.get_media_base64.assert_called_once()
    mock_vision.assert_called_once()
    # Caption reaches the vision call.
    assert mock_vision.call_args.args[2] == "tem essa?"

    mock_graph.ainvoke.assert_called_once()
    invoked_state = mock_graph.ainvoke.call_args.args[0]
    assert invoked_state["image_product_query"] is True
    user_msgs = [m for m in invoked_state["messages"] if isinstance(m, HumanMessage)]
    assert any("Shark Attack 2024" in m.content for m in user_msgs)


@pytest.mark.asyncio
async def test_webhook_image_racket_unidentified_canned(override_token, fake_redis):
    """is_racket=True but no brand/model readable → 'me diz a marca' canned,
    graph NOT invoked."""
    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch(
            "app.api.webhook.identify_racket_image",
            new_callable=AsyncMock,
            return_value={"is_racket": True, "brand": None, "model": None, "confidence": "low"},
        ),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock()
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_image_payload(message_id="UNID-1"),
                headers={"apikey": _TOKEN},
            )

    mock_graph.ainvoke.assert_not_called()
    evo.send_text.assert_called_once()
    assert "marca" in evo.send_text.call_args.args[1].lower()


@pytest.mark.asyncio
async def test_webhook_image_vision_failure_canned(override_token, fake_redis):
    """Vision call raising → polite failure canned reply, graph NOT invoked."""
    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch(
            "app.api.webhook.identify_racket_image",
            new_callable=AsyncMock,
            side_effect=ValueError("vision model returned non-JSON output"),
        ),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock()
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_image_payload(message_id="VISFAIL-1"),
                headers={"apikey": _TOKEN},
            )

    mock_graph.ainvoke.assert_not_called()
    evo.send_text.assert_called_once()
    assert "problema" in evo.send_text.call_args.args[1].lower()


# ── Guards: rate limit / size cap / cache ────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_image_rate_limit_exceeded(override_token, fake_redis):
    """Counter above the limit → canned rate-limit reply; NO download, NO vision."""
    from app.security.pii_masker import hash_phone

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.identify_racket_image", new_callable=AsyncMock) as mock_vision,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        phone_hash = hash_phone(_PHONE)
        await fake_redis.set(f"image_rate:{phone_hash}", "10")

        evo = _evo_mock()
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_image_payload(message_id="RATE-IMG-11"),
                headers={"apikey": _TOKEN},
            )

    evo.get_media_base64.assert_not_called()
    mock_vision.assert_not_called()
    mock_graph.ainvoke.assert_not_called()
    evo.send_text.assert_called_once()
    assert "texto" in evo.send_text.call_args.args[1].lower()


@pytest.mark.asyncio
async def test_webhook_image_oversized_rejected(override_token, fake_redis, monkeypatch):
    """Downloaded blob above IMAGE_MAX_BYTES → canned reply, vision NOT called."""
    from app.config import get_settings
    monkeypatch.setenv("IMAGE_MAX_BYTES", "10")
    get_settings.cache_clear()

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.identify_racket_image", new_callable=AsyncMock) as mock_vision,
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock(image_bytes=b"x" * 100)  # 100 bytes > cap of 10
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_image_payload(message_id="BIG-IMG-1"),
                headers={"apikey": _TOKEN},
            )

    evo.get_media_base64.assert_called_once()
    mock_vision.assert_not_called()
    mock_graph.ainvoke.assert_not_called()
    evo.send_text.assert_called_once()


@pytest.mark.asyncio
async def test_image_id_cache_hit_skips_vision(override_token, fake_redis):
    """Cached identification for identical bytes → vision NOT called, graph
    still gets the product query."""
    image_bytes = b"identical-racket-photo"
    sha = hashlib.sha256(image_bytes).hexdigest()

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.api.webhook.identify_racket_image", new_callable=AsyncMock) as mock_vision,
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        await fake_redis.set(f"image_id:{sha}", json.dumps(_RACKET_ID))

        evo = _evo_mock(image_bytes=image_bytes)
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_image_payload(message_id="CACHE-IMG-HIT"),
                headers={"apikey": _TOKEN},
            )

    mock_vision.assert_not_called()
    mock_graph.ainvoke.assert_called_once()
    invoked_state = mock_graph.ainvoke.call_args.args[0]
    user_msgs = [m for m in invoked_state["messages"] if isinstance(m, HumanMessage)]
    assert any("Shark Attack 2024" in m.content for m in user_msgs)


@pytest.mark.asyncio
async def test_image_id_cache_written_on_miss(override_token, fake_redis):
    """Cache miss → vision called once and the identification stored with TTL."""
    image_bytes = b"fresh-racket-photo"
    sha = hashlib.sha256(image_bytes).hexdigest()

    mock_graph = _mock_graph()

    with (
        patch("app.api.webhook._get_graph", return_value=mock_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch(
            "app.api.webhook.identify_racket_image",
            new_callable=AsyncMock,
            return_value=dict(_RACKET_ID),
        ) as mock_vision,
        patch("app.storage.db.get_session", _mock_db_session),
        patch("app.storage.redis_session._get_redis_client", return_value=fake_redis),
    ):
        evo = _evo_mock(image_bytes=image_bytes)
        MockEvo.return_value = evo

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_image_payload(message_id="CACHE-IMG-MISS"),
                headers={"apikey": _TOKEN},
            )

        mock_vision.assert_called_once()
        cached = await fake_redis.get(f"image_id:{sha}")
        assert json.loads(cached) == _RACKET_ID
        ttl = await fake_redis.ttl(f"image_id:{sha}")
        assert ttl > 0


# ── Triage short-circuit ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_triage_image_product_query_short_circuits_to_product_inquiry():
    from app.agent.nodes.triage import triage_node

    state = {
        "messages": [HumanMessage(content="Shark Attack 2024")],
        "phone_hash": "abc",
        "image_product_query": True,
    }
    with patch("app.agent.nodes.triage.OpenAIClient") as MockClient:
        result = await triage_node(state)

    assert result["intent"] == "product_inquiry"
    MockClient.assert_not_called()


# ── recommend_node — photo-aware wording ─────────────────────────────────────

_CATALOG_PRODUCT = {"id": "p1", "name": "Raquete Shark Attack 2024", "price_cents": 129900}


@pytest.mark.asyncio
async def test_recommend_from_image_uses_photo_wording_and_clears_flag():
    from app.agent.nodes.recommend import recommend_node

    state = {
        "messages": [HumanMessage(content="Shark Attack 2024")],
        "phone_hash": "abc",
        "image_product_query": True,
    }
    with patch(
        "app.agent.nodes.recommend._list_catalog_candidates",
        new_callable=AsyncMock,
        return_value=[_CATALOG_PRODUCT],
    ):
        result = await recommend_node(state)

    text = result["response_blocks"][0]
    assert "Pela foto" in text
    assert "Shark Attack 2024" in text
    # Follow-up contract preserved (detail-choice short-circuit) + flag consumed.
    assert result["awaiting_detail_choice"] is True
    assert result["image_product_query"] is False
    assert result["recommended_products"] == [_CATALOG_PRODUCT]


@pytest.mark.asyncio
async def test_recommend_from_text_keeps_legacy_wording():
    from app.agent.nodes.recommend import recommend_node

    state = {
        "messages": [HumanMessage(content="tem a shark attack 2024?")],
        "phone_hash": "abc",
    }
    with patch(
        "app.agent.nodes.recommend._list_catalog_candidates",
        new_callable=AsyncMock,
        return_value=[_CATALOG_PRODUCT],
    ):
        result = await recommend_node(state)

    text = result["response_blocks"][0]
    assert "Pela foto" not in text
    assert text.startswith("Sim, temos a")


@pytest.mark.asyncio
async def test_recommend_from_image_not_found_confirms_identification():
    from app.agent.nodes.recommend import recommend_node

    state = {
        "messages": [HumanMessage(content="Marca Inexistente XYZ")],
        "phone_hash": "abc",
        "image_product_query": True,
    }
    with patch(
        "app.agent.nodes.recommend._list_catalog_candidates",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await recommend_node(state)

    text = result["response_blocks"][0]
    assert "Pela foto" in text
    assert "não encontrei" in text
    assert result["image_product_query"] is False
