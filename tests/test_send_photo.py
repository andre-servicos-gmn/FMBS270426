"""Agent sends product photos: enviar_foto_produto tool + webhook delivery."""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.tools_v2 import FOTO_TOOL_NAME, enviar_foto_produto, photos_from_turn
from app.sync.bling_sync import _extract_first_image_link

_DETAIL = {
    "midia": {
        "imagens": {
            "externas": [],
            "internas": [{"link": "https://bling.example/img/123.jpg?sig=abc"}],
        }
    }
}
_PRODUCT = {"id": 123, "name": "Raquete Beach Tennis Drop Shot Pentax 3.0", "imagem_url": None}


def _photo_msg(url: str, legenda: str = "X", call_id: str = "c1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"status": "ok", "url": url, "legenda": legenda}),
        name=FOTO_TOOL_NAME,
        tool_call_id=call_id,
    )


# ── link extraction ──────────────────────────────────────────────────────────

def test_sync_extraction_stays_external_only():
    assert _extract_first_image_link(_DETAIL) is None


def test_live_extraction_falls_back_to_internal():
    link = _extract_first_image_link(_DETAIL, kinds=("externas", "internas"))
    assert link == "https://bling.example/img/123.jpg?sig=abc"


def test_external_wins_over_internal():
    detail = {"midia": {"imagens": {
        "externas": [{"link": "https://cdn.example/a.png"}],
        "internas": [{"link": "https://bling.example/b.jpg"}],
    }}}
    assert _extract_first_image_link(detail, kinds=("externas", "internas")) == "https://cdn.example/a.png"


# ── tool ─────────────────────────────────────────────────────────────────────

async def _run_tool(bling_resp=None, bling_exc=None, product=_PRODUCT) -> dict:
    client = MagicMock()
    client.consultar_produto = AsyncMock(return_value=bling_resp, side_effect=bling_exc)
    with (
        patch("app.adapters.bling.BlingClient", return_value=client),
        patch("app.sync.bling_repo.fetch_product_by_id", AsyncMock(return_value=product)),
    ):
        raw = await enviar_foto_produto.ainvoke({"produto_id": "123"})
    return json.loads(raw)


@pytest.mark.asyncio
async def test_tool_queues_live_bling_photo_with_human_caption():
    out = await _run_tool(bling_resp={"data": _DETAIL})
    assert out["url"] == "https://bling.example/img/123.jpg?sig=abc"
    assert out["legenda"] == "Drop Shot Pentax 3.0"


@pytest.mark.asyncio
async def test_tool_falls_back_to_mirror_when_bling_down():
    product = dict(_PRODUCT, imagem_url="https://cdn.example/mirror.jpg")
    out = await _run_tool(bling_exc=RuntimeError("refresh token expired"), product=product)
    assert out["url"] == "https://cdn.example/mirror.jpg"


@pytest.mark.asyncio
async def test_tool_reports_missing_photo():
    out = await _run_tool(bling_resp={"data": {"midia": {"imagens": {"externas": []}}}})
    assert "erro" in out and "url" not in out


@pytest.mark.asyncio
async def test_tool_rejects_bad_id():
    raw = await enviar_foto_produto.ainvoke({"produto_id": "abc"})
    assert "erro" in json.loads(raw)


# ── photos_from_turn ─────────────────────────────────────────────────────────

def test_photos_only_from_current_turn_deduped_and_capped():
    msgs = [
        HumanMessage(content="oi"),
        _photo_msg("https://x/old.jpg", call_id="old"),
        AIMessage(content="..."),
        HumanMessage(content="manda foto"),
        _photo_msg("https://x/1.jpg", call_id="a"),
        _photo_msg("https://x/1.jpg", call_id="b"),
        _photo_msg("https://x/2.jpg", call_id="c"),
        _photo_msg("https://x/3.jpg", call_id="d"),
        _photo_msg("https://x/4.jpg", call_id="e"),
        ToolMessage(content=json.dumps({"erro": "produto sem foto cadastrada"}),
                    name=FOTO_TOOL_NAME, tool_call_id="f"),
        AIMessage(content="Essa é a Pentax."),
    ]
    urls = [p["url"] for p in photos_from_turn(msgs)]
    assert urls == ["https://x/1.jpg", "https://x/2.jpg", "https://x/3.jpg"]


# ── sanitize drops a pasted photo link ───────────────────────────────────────

def test_sanitize_removes_photo_link_from_text():
    from app.agent.supervisor import sanitize_node

    url = "https://bling.example/img/123.jpg?sig=abc"
    last = AIMessage(content=f"Essa é a Pentax 3.0 {url}")
    out = sanitize_node({"messages": [HumanMessage(content="foto?"), _photo_msg(url), last]})
    assert url not in out["messages"][0].content


# ── Evolution payload ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_image_payload(monkeypatch):
    monkeypatch.setenv("EVOLUTION_API_URL", "http://localhost:9999")
    monkeypatch.setenv("EVOLUTION_API_KEY", "test-key")
    monkeypatch.setenv("EVOLUTION_INSTANCE", "inst")
    from app.adapters.evolution import EvolutionClient
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        evo = EvolutionClient()
        with patch.object(evo, "_post_with_retry", new_callable=AsyncMock) as post:
            await evo.send_image("5511999999999", "https://x/a.png?sig=1", "Pentax 3.0")
        url, payload = post.call_args.args[0], post.call_args.args[1]
        assert url.endswith("/message/sendMedia/inst")
        assert payload["mediatype"] == "image"
        assert payload["mimetype"] == "image/png"
        assert payload["media"] == "https://x/a.png?sig=1"
        assert payload["caption"] == "Pentax 3.0"
    finally:
        get_settings.cache_clear()


# ── webhook: photo before text, photo failure doesn't block text ─────────────

@asynccontextmanager
async def _db():
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    yield session


async def _process(send_image_side_effect=None) -> list[str]:
    from app.api.webhook import _process_message

    calls: list[str] = []
    result = {
        "messages": [
            HumanMessage(content="manda foto da pentax"),
            _photo_msg("https://x/pentax.jpg", "Drop Shot Pentax 3.0"),
            AIMessage(content="Essa é a Pentax 3.0."),
        ]
    }
    evo = MagicMock()

    async def _img(phone, url, caption):
        calls.append(f"image:{url}")
        if send_image_side_effect:
            raise send_image_side_effect

    async def _txt(phone, blocks):
        calls.append("text")

    evo.send_image = _img
    evo.send_text_blocks = _txt
    with (
        patch("app.api.webhook._ainvoke_with_retry", AsyncMock(return_value=result)),
        patch("app.api.webhook.EvolutionClient", return_value=evo),
        patch("app.storage.db.get_session", _db),
    ):
        await _process_message(raw_phone="5511999999999", phone_hash="h" * 64, message_text="manda foto")
    return calls


@pytest.mark.asyncio
async def test_webhook_sends_photo_before_text():
    assert await _process() == ["image:https://x/pentax.jpg", "text"]


@pytest.mark.asyncio
async def test_webhook_photo_failure_still_sends_text():
    assert await _process(RuntimeError("evolution 400")) == ["image:https://x/pentax.jpg", "text"]
