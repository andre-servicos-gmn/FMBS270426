"""Sprint 2.5 — Bling integration tests.

Coverage:
- OAuth: authorize-URL build, code exchange, refresh, auto-refresh on 401,
  429 backoff.
- Sync: pagination, category filter, description parser, raquete detection
  via custom field AND via category fallback, single-product upsert,
  delete-via-mark-inactive.
- Webhook: HMAC validation (valid / invalid), 200 OK, idempotency
  (out-of-order events), product.updated triggers sync.
- Stock cache: hit / miss / API error → None.
- Agent integration: recommend uses bling_products, raquete_praia gates
  the Consultoria pitch, price_inquiry surfaces stock check.

All external I/O is mocked. No live Bling, no Postgres, no Redis required.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import HumanMessage

from app.adapters.bling import (
    BlingClient,
    BlingError,
    BlingNotAuthorizedError,
)
from app.agent.state import AgentState
from app.sync.bling_sync import (
    BlingSync,
    is_raquete_de_praia,
    parse_attributes_from_description,
    detail_to_row,
)


# ── Helpers ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def _fake_db_session(creds_row=None, scalar_one_or_none_chain=None):
    """An async session mock used by adapter / repo tests."""
    session = MagicMock()
    result = MagicMock()
    if scalar_one_or_none_chain is not None:
        result.scalar_one_or_none = MagicMock(side_effect=scalar_one_or_none_chain)
    else:
        result.scalar_one_or_none = MagicMock(return_value=creds_row)
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.get = AsyncMock(return_value=creds_row)
    yield session


def _fake_creds_row(expires_at=None):
    row = MagicMock()
    row.access_token = "tok-abc"
    row.refresh_token = "ref-xyz"
    row.expires_at = expires_at or (datetime.now(timezone.utc) + timedelta(hours=2))
    row.scope = "produtos categorias depositos"
    return row


def _make_mock_response(status_code: int, json_payload=None, headers=None):
    """Build a real-enough httpx.Response stand-in."""
    request = httpx.Request("GET", "https://api.bling.com.br/Api/v3/dummy")
    resp = httpx.Response(
        status_code=status_code,
        request=request,
        headers=headers or {},
        content=json.dumps(json_payload or {}).encode("utf-8")
        if json_payload is not None else b"",
    )
    return resp


def _patch_async_client_with_responses(responses: list[httpx.Response]):
    """Return a patch-target for ``httpx.AsyncClient`` that yields the
    given responses in order across .request/.post calls."""
    iter_resp = iter(responses)

    class _FakeClient:
        def __init__(self, *_, **__): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False
        async def request(self, *args, **kwargs):
            return next(iter_resp)
        async def post(self, *args, **kwargs):
            return next(iter_resp)
    return _FakeClient


# ════════════════════════════════════════════════════════════════════════════
# OAUTH
# ════════════════════════════════════════════════════════════════════════════

def test_get_authorize_url(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid-123")
    monkeypatch.setenv("BLING_REDIRECT_URI", "https://example.com/cb")
    from app.config import get_settings
    get_settings.cache_clear()
    url = BlingClient().get_authorize_url(state="random-state")
    assert "client_id=cid-123" in url
    assert "state=random-state" in url
    assert "response_type=code" in url
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcb" in url


@pytest.mark.asyncio
async def test_exchange_code_for_token_success(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    monkeypatch.setenv("BLING_CLIENT_SECRET", "secret")
    from app.config import get_settings
    get_settings.cache_clear()
    payload = {
        "access_token": "AAA", "refresh_token": "RRR",
        "expires_in": 3600, "scope": "produtos",
    }
    resp = _make_mock_response(200, payload)
    with patch("httpx.AsyncClient", new=_patch_async_client_with_responses([resp])):
        with patch(
            "app.adapters.bling.BlingClient._save_credentials",
            new_callable=AsyncMock,
        ) as save:
            data = await BlingClient().exchange_code_for_token("the-code")
    assert data["access_token"] == "AAA"
    save.assert_awaited_once()


@pytest.mark.asyncio
async def test_exchange_code_for_token_invalid_code(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    monkeypatch.setenv("BLING_CLIENT_SECRET", "secret")
    from app.config import get_settings
    get_settings.cache_clear()
    resp = _make_mock_response(400, {"error": "invalid_grant"})
    with patch("httpx.AsyncClient", new=_patch_async_client_with_responses([resp])):
        with pytest.raises(BlingError):
            await BlingClient().exchange_code_for_token("bad-code")


@pytest.mark.asyncio
async def test_refresh_access_token_success(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    monkeypatch.setenv("BLING_CLIENT_SECRET", "secret")
    from app.config import get_settings
    get_settings.cache_clear()
    payload = {"access_token": "NEW", "refresh_token": "NEW_REF", "expires_in": 3600}
    resp = _make_mock_response(200, payload)
    with patch("httpx.AsyncClient", new=_patch_async_client_with_responses([resp])):
        with patch(
            "app.adapters.bling.BlingClient._load_credentials",
            new_callable=AsyncMock,
            return_value=_fake_creds_row(),
        ):
            with patch(
                "app.adapters.bling.BlingClient._save_credentials",
                new_callable=AsyncMock,
            ) as save:
                data = await BlingClient().refresh_access_token()
    assert data["access_token"] == "NEW"
    save.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_raises_when_no_credentials(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    with patch(
        "app.adapters.bling.BlingClient._load_credentials",
        new_callable=AsyncMock, return_value=None,
    ):
        with pytest.raises(BlingNotAuthorizedError):
            await BlingClient().refresh_access_token()


@pytest.mark.asyncio
async def test_request_auto_refresh_on_401(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    monkeypatch.setenv("BLING_CLIENT_SECRET", "s")
    from app.config import get_settings
    get_settings.cache_clear()
    # First request → 401, then refresh response (handled by direct patch on
    # refresh_access_token), then a successful retry.
    resp_401 = _make_mock_response(401, {"error": "expired"})
    resp_ok = _make_mock_response(200, {"data": [{"id": 1}]})
    with patch("httpx.AsyncClient", new=_patch_async_client_with_responses([resp_401, resp_ok])):
        with patch(
            "app.adapters.bling.BlingClient.ensure_authorized",
            new_callable=AsyncMock,
            return_value=_fake_creds_row(),
        ):
            with patch(
                "app.adapters.bling.BlingClient.refresh_access_token",
                new_callable=AsyncMock,
            ) as refresh:
                data = await BlingClient()._request("GET", "/produtos")
    refresh.assert_awaited_once()
    assert data["data"][0]["id"] == 1


@pytest.mark.asyncio
async def test_request_backoff_on_429(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    resp_429 = _make_mock_response(429, {}, headers={"Retry-After": "0"})
    resp_ok = _make_mock_response(200, {"data": []})
    with patch("httpx.AsyncClient", new=_patch_async_client_with_responses([resp_429, resp_ok])):
        with patch(
            "app.adapters.bling.BlingClient.ensure_authorized",
            new_callable=AsyncMock, return_value=_fake_creds_row(),
        ):
            with patch("app.adapters.bling.asyncio.sleep", new_callable=AsyncMock):
                data = await BlingClient()._request("GET", "/produtos")
    assert data == {"data": []}


# ════════════════════════════════════════════════════════════════════════════
# SYNC — parser + raquete detection
# ════════════════════════════════════════════════════════════════════════════

def test_parse_description_extracts_target_attributes_only():
    """Sprint 2.6.7 — the parser now whitelist-filters to the 5 target
    attributes (peso/equilibrio/composicao/espessura/comprimento). Labels
    like "Perfil" and "Detalhamento" used to land in atributos_parseados
    too, but they were marketing/free-form noise — dropped now."""
    html = """
    <p>Detalhes da raquete:</p>
    <ul>
    <li>- Perfil: 22mm</li>
    <li>- Composição: Fibra de carbono 12k</li>
    <li>- Detalhamento: ideal pra intermediário</li>
    </ul>
    """
    parsed = parse_attributes_from_description(html)
    # Composição (a target) is captured.
    assert parsed.get("composicao") == "Fibra de carbono 12k"
    # Perfil + Detalhamento are NOT targets — they no longer appear.
    assert "perfil" not in parsed
    assert "detalhamento" not in parsed


def test_parse_description_empty_html():
    assert parse_attributes_from_description("") == {}
    assert parse_attributes_from_description(None) == {}


def test_is_raquete_praia_via_custom_field_true():
    detail = {"id": 1, "categoria": {"descricao": "Outras"}}
    custom = {"Es raquete de praia": True}
    assert is_raquete_de_praia(detail, custom) is True


def test_is_raquete_praia_via_custom_field_str_sim():
    detail = {"id": 1, "categoria": {"descricao": "Outras"}}
    assert is_raquete_de_praia(detail, {"Es raquete de praia": "Sim"}) is True
    assert is_raquete_de_praia(detail, {"Es raquete de praia": "Não"}) is False


def test_is_raquete_praia_via_category_fallback():
    detail = {"id": 1, "categoria": {"descricao": "Raquetes de Praia"}}
    custom = {}  # no toggle
    assert is_raquete_de_praia(detail, custom) is True


def test_is_raquete_praia_neither_returns_false():
    detail = {"id": 1, "categoria": {"descricao": "Bola Beach TEnnis"}}
    assert is_raquete_de_praia(detail, {}) is False


# ── Sprint 2.5.1 — defensive shape tests ─────────────────────────────────

def test_parse_product_with_empty_custom_fields():
    """Empty camposCustomizados list / None / dict must not raise."""
    base = {"id": 1, "nome": "P1", "situacao": "A"}
    for variant in (
        {**base, "camposCustomizados": []},
        {**base, "camposCustomizados": None},
        {**base, "camposCustomizados": {}},
        {**base, "campos_customizados": []},
        base,  # key absent entirely
    ):
        row = detail_to_row(variant)
        assert row["campos_customizados"] == {}
        assert row["is_raquete_praia"] is False


def test_parse_product_with_missing_category():
    """``categoria`` absent / None / string / list-of-dicts all work."""
    base = {"id": 2, "nome": "P2", "situacao": "A"}

    # Absent
    row = detail_to_row(base)
    assert row["categoria_nome"] is None
    assert row["categoria_id"] is None

    # None
    row = detail_to_row({**base, "categoria": None})
    assert row["categoria_nome"] is None

    # String (some Bling builds return just the name)
    row = detail_to_row({**base, "categoria": "Raquetes de Praia"})
    assert row["categoria_nome"] == "Raquetes de Praia"
    assert row["is_raquete_praia"] is True  # category-fallback hits

    # List shape (different Bling build variant)
    row = detail_to_row({
        **base,
        "categoria": [{"id": 9, "descricao": "GRIPS"}],
    })
    assert row["categoria_nome"] == "GRIPS"
    assert row["categoria_id"] == 9


def test_parse_product_with_minimal_data():
    """Bare-bones response (id + nothing else) must produce a usable row."""
    row = detail_to_row({"id": 99})
    assert row["id"] == 99
    assert row["nome"] == "Produto 99"  # synthetic fallback
    assert row["situacao"] == "A"
    assert row["campos_customizados"] == {}
    assert row["atributos_parseados"] == {}
    assert row["categoria_nome"] is None
    assert row["marca"] is None
    assert row["imagem_url"] is None
    assert row["preco"] is None
    assert row["is_raquete_praia"] is False


def test_parse_product_with_empty_externas_does_not_raise():
    """Sprint 2.5.1 root-cause regression: midia.imagens.externas: [] used
    to raise IndexError. Must now return None imagem_url cleanly."""
    detail = {
        "id": 100, "nome": "P100", "situacao": "A",
        "midia": {"imagens": {"externas": []}},
    }
    row = detail_to_row(detail)
    assert row["imagem_url"] is None


def test_parse_product_with_externas_none_does_not_raise():
    detail = {
        "id": 101, "nome": "P101", "situacao": "A",
        "midia": {"imagens": {"externas": None}},
    }
    row = detail_to_row(detail)
    assert row["imagem_url"] is None


def test_parse_product_picks_first_image_link_when_present():
    detail = {
        "id": 102, "nome": "P102", "situacao": "A",
        "midia": {"imagens": {"externas": [
            {"link": "https://cdn.bling.com.br/img1.jpg"},
            {"link": "https://cdn.bling.com.br/img2.jpg"},
        ]}},
    }
    row = detail_to_row(detail)
    assert row["imagem_url"] == "https://cdn.bling.com.br/img1.jpg"


def test_parse_product_with_marca_as_string():
    detail = {"id": 103, "nome": "P103", "situacao": "A", "marca": "BeachPro"}
    row = detail_to_row(detail)
    assert row["marca"] == "BeachPro"


def test_parse_product_with_marca_none():
    detail = {"id": 104, "nome": "P104", "situacao": "A", "marca": None}
    row = detail_to_row(detail)
    assert row["marca"] is None


def test_parse_product_with_dimensoes_none():
    detail = {"id": 105, "nome": "P105", "situacao": "A", "dimensoes": None}
    row = detail_to_row(detail)
    assert row["largura"] is None
    assert row["altura"] is None


def test_parse_product_invalid_id_raises():
    """We tolerate everything EXCEPT a missing/non-numeric id (we can't
    UPSERT without a primary key)."""
    import pytest
    with pytest.raises(ValueError):
        detail_to_row({"nome": "sem id"})
    with pytest.raises(ValueError):
        detail_to_row({"id": "not-a-number"})


def test_is_raquete_praia_never_raises_on_garbage():
    """Defensive: empty / None / wrong-type inputs must return False, not throw."""
    assert is_raquete_de_praia({}, {}) is False
    assert is_raquete_de_praia({"categoria": None}, {}) is False
    assert is_raquete_de_praia({"categoria": 42}, {}) is False  # type: ignore[arg-type]
    assert is_raquete_de_praia({}, {"weird key": object()}) is False  # value not str/bool


# ── Sprint 2.5.2 — real-shape tests against the synthesized Mormaii fixture ─

import os
from pathlib import Path

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bling_raquete_real.json"


def _load_real_fixture() -> dict:
    with open(_FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _real_maps() -> tuple[dict[int, str], dict[int, str]]:
    """ID→name maps that mirror what BlingSync._ensure_maps would load."""
    category_map = {
        14821094: "Raquetes de Praia",
        20: "Bola Beach TEnnis",
    }
    field_map = {
        7001: "Marca",
        7002: "Modelo",
        7003: "Es raquete de praia",
        7004: "Composição",
        7005: "Materiais do exterior",
        7006: "Espessura do perfil",
    }
    return category_map, field_map


def test_parse_categoria_from_real_payload():
    """Bling V3 returns categoria as {id: N} only; sync must resolve via map."""
    from app.sync.bling_sync import detail_to_row

    fixture = _load_real_fixture()
    category_map, field_map = _real_maps()
    row = detail_to_row(
        fixture["data"], category_map=category_map, field_map=field_map,
    )
    assert row["categoria_id"] == 14821094
    assert row["categoria_nome"] == "Raquetes de Praia"


def test_detect_is_raquete_praia_from_real_payload():
    """Custom field "Es raquete de praia" = "Ativado" → True."""
    from app.sync.bling_sync import detail_to_row

    fixture = _load_real_fixture()
    category_map, field_map = _real_maps()
    row = detail_to_row(
        fixture["data"], category_map=category_map, field_map=field_map,
    )
    assert row["is_raquete_praia"] is True


def test_extract_marca_from_real_payload():
    """marca either as top-level string or via custom field — fixture has both."""
    from app.sync.bling_sync import detail_to_row

    fixture = _load_real_fixture()
    category_map, field_map = _real_maps()
    row = detail_to_row(
        fixture["data"], category_map=category_map, field_map=field_map,
    )
    assert row["marca"] == "Mormaii"


def test_extract_modelo_from_real_payload():
    """modelo is in camposCustomizados — sync resolves via the field_map."""
    from app.sync.bling_sync import detail_to_row

    fixture = _load_real_fixture()
    category_map, field_map = _real_maps()
    row = detail_to_row(
        fixture["data"], category_map=category_map, field_map=field_map,
    )
    assert row["modelo"] == "Sunset Plus"


def test_parse_atributos_from_html_description():
    """Sprint 2.6.7 — the parser is now restricted to the 5 target slugs
    (peso / equilibrio / composicao / espessura / comprimento). The
    Sprint 2.5.2 expectations for "perfil" and "detalhamento" were
    dropped — those labels are no longer captured (they were marketing /
    free-form noise contaminating ``atributos_parseados``).
    """
    from app.sync.bling_sync import detail_to_row

    fixture = _load_real_fixture()
    category_map, field_map = _real_maps()
    row = detail_to_row(
        fixture["data"], category_map=category_map, field_map=field_map,
    )
    parsed = row["atributos_parseados"]
    # Target attributes that survived the whitelist.
    assert "composicao" in parsed
    assert "carbono" in (parsed.get("composicao") or "").lower()
    assert parsed.get("comprimento") == "50cm"
    # Restricted-out labels.
    assert "perfil" not in parsed
    assert "detalhamento" not in parsed


def test_strip_html_before_regex_parses():
    """Sprint 2.6.7 — the parser strips HTML and handles mixed dash chars.
    "Perfil" is no longer captured (restricted), so we only assert the
    targets that survive the whitelist (composicao + peso)."""
    from app.sync.bling_sync import parse_attributes_from_description

    html = """<p style="font-family: Tailwind"><strong>Características:</strong></p>
              <p>- Perfil: Iniciante;</p>
              <p>- Composição: Fibra de vidro;</p>
              <p>– Peso: 320g;</p>"""  # mixed dashes
    parsed = parse_attributes_from_description(html)
    assert parsed.get("composicao") == "Fibra de vidro"
    assert parsed.get("peso") == "320g"
    # Perfil is restricted-out now.
    assert "perfil" not in parsed


def test_custom_fields_id_only_resolves_via_map():
    """Sprint 2.5.2 root-cause regression: camposCustomizados with only
    idCampoCustomizado + valor must be resolved via field_map."""
    from app.sync.bling_sync import detail_to_row

    detail = {
        "id": 999, "nome": "P999", "situacao": "A",
        "camposCustomizados": [
            {"idCampoCustomizado": 100, "valor": "Mormaii"},
            {"idCampoCustomizado": 101, "valor": "Sunset"},
        ],
    }
    row = detail_to_row(detail, field_map={100: "Marca", 101: "Modelo"})
    assert row["campos_customizados"] == {"Marca": "Mormaii", "Modelo": "Sunset"}


def test_custom_fields_id_only_without_map_falls_back_to_synthetic_key():
    """No field_map → data still captured under campo_<id> keys."""
    from app.sync.bling_sync import detail_to_row

    detail = {
        "id": 1000, "nome": "P1000", "situacao": "A",
        "camposCustomizados": [
            {"idCampoCustomizado": 100, "valor": "Mormaii"},
        ],
    }
    row = detail_to_row(detail)
    assert row["campos_customizados"] == {"campo_100": "Mormaii"}


def test_is_raquete_praia_truthy_string_ativado():
    """The toggle string "Ativado" is now accepted as truthy."""
    detail = {"id": 1, "categoria": {"id": 2}}  # category alone wouldn't match
    custom = {"Es raquete de praia": "Ativado"}
    assert is_raquete_de_praia(detail, custom) is True


def test_categoria_resolves_via_map_when_only_id_in_payload():
    """Sprint 2.5.2: categoria: {id: N} → name resolved via category_map."""
    from app.sync.bling_sync import detail_to_row

    detail = {
        "id": 2001, "nome": "P2001", "situacao": "A",
        "categoria": {"id": 555},
    }
    row = detail_to_row(detail, category_map={555: "GRIPS"})
    assert row["categoria_id"] == 555
    assert row["categoria_nome"] == "GRIPS"


@pytest.mark.asyncio
async def test_blingsync_ensure_maps_loads_both_caches():
    """BlingSync pre-loads categoria + custom-field maps once per sync."""
    from app.sync.bling_sync import BlingSync

    client = MagicMock()
    client.listar_categorias = AsyncMock(return_value=[
        {"id": 1, "descricao": "Raquetes de Praia"},
        {"id": 2, "descricao": "GRIPS"},
    ])
    client.listar_campos_customizados = AsyncMock(return_value=[
        {"id": 100, "nome": "Marca"},
        {"id": 101, "nome": "Modelo"},
        {"id": 102, "nome": "Es raquete de praia"},
    ])
    sync = BlingSync(client=client)
    await sync._ensure_maps()
    assert sync._category_map == {1: "Raquetes de Praia", 2: "GRIPS"}
    assert sync._field_map == {100: "Marca", 101: "Modelo", 102: "Es raquete de praia"}
    # Idempotent: second call doesn't re-fetch.
    await sync._ensure_maps()
    client.listar_categorias.assert_awaited_once()
    client.listar_campos_customizados.assert_awaited_once()


@pytest.mark.asyncio
async def test_blingsync_survives_field_map_endpoint_missing():
    """If listar_campos_customizados raises, sync continues with empty map."""
    from app.sync.bling_sync import BlingSync

    client = MagicMock()
    client.listar_categorias = AsyncMock(return_value=[])
    client.listar_campos_customizados = AsyncMock(side_effect=BlingError("404"))
    sync = BlingSync(client=client)
    await sync._ensure_maps()
    assert sync._field_map == {}
    assert sync._maps_loaded is True


@pytest.mark.asyncio
async def test_full_sync_uses_maps_for_detail_to_row():
    """End-to-end: full_sync resolves categoria + custom fields via maps."""
    from app.sync.bling_sync import BlingSync

    client = MagicMock()
    client.listar_categorias = AsyncMock(return_value=[
        {"id": 999, "descricao": "Raquetes de Praia"}
    ])
    client.listar_campos_customizados = AsyncMock(return_value=[
        {"id": 7003, "nome": "Es raquete de praia"},
    ])
    client.listar_produtos = AsyncMock(side_effect=[
        {"data": [{"id": 1, "categoria": {"id": 999}}]},
        {"data": []},
    ])
    client.consultar_produto = AsyncMock(return_value={
        "data": {
            "id": 1, "nome": "P1", "situacao": "A",
            "categoria": {"id": 999},
            "camposCustomizados": [
                {"idCampoCustomizado": 7003, "valor": "Ativado"},
            ],
        }
    })

    captured = {}
    async def _capture(row):
        captured["row"] = row
        return "inserted"
    with patch("app.sync.bling_sync.open_sync_log", new_callable=AsyncMock, return_value=1):
        with patch("app.sync.bling_sync.close_sync_log", new_callable=AsyncMock):
            with patch("app.sync.bling_sync.upsert_product", side_effect=_capture):
                stats = await BlingSync(client=client).full_sync()

    assert stats["inserted"] == 1
    row = captured["row"]
    assert row["categoria_nome"] == "Raquetes de Praia"
    assert row["campos_customizados"].get("Es raquete de praia") == "Ativado"
    assert row["is_raquete_praia"] is True


def test_detail_to_row_extracts_all_fields():
    detail = {
        "id": 42, "nome": "Raquete X", "codigo": "SKU-X",
        "preco": "899.00", "situacao": "A",
        "descricaoCurta": "<p>linda</p>",
        "descricaoComplementar": "<p>- Perfil: 22mm\n- Composição: Carbono</p>",
        "marca": {"nome": "BeachPro"},
        "modelo": "Carbon X5",
        "categoria": {"id": 7, "descricao": "Raquetes de Praia"},
        "pesoLiquido": 0.350,
        "pesoBruto": 0.4,
        "dimensoes": {"largura": 28, "altura": 50, "profundidade": 4},
        "gtin": "7891234567890",
        "camposCustomizados": [
            {"nome": "Es raquete de praia", "valor": True},
            {"nome": "Indicação", "valor": "intermediário"},
        ],
    }
    row = detail_to_row(detail)
    assert row["id"] == 42
    assert row["nome"] == "Raquete X"
    assert row["is_raquete_praia"] is True
    assert row["campos_customizados"]["Indicação"] == "intermediário"
    # Sprint 2.6.7 — only the 5 target attributes are captured from the
    # description; "Perfil" is restricted-out. Composição still goes through.
    assert "perfil" not in row["atributos_parseados"]
    assert row["atributos_parseados"].get("composicao") == "Carbono"
    assert row["marca"] == "BeachPro"
    assert row["categoria_nome"] == "Raquetes de Praia"
    assert float(row["preco"]) == 899.0
    assert int(float(row["peso_liquido"]) * 1000) == 350


@pytest.mark.asyncio
async def test_full_sync_paginates_correctly():
    """3 pages of products, full_sync visits all + opens/closes log."""
    page_a = {"data": [{"id": 1, "categoria": {"descricao": "Raquetes de Praia"}}]}
    page_b = {"data": [{"id": 2, "categoria": {"descricao": "Raquetes de Praia"}}]}
    page_empty: dict = {"data": []}

    client = MagicMock()
    client.listar_produtos = AsyncMock(side_effect=[page_a, page_b, page_empty])
    client.consultar_produto = AsyncMock(side_effect=[
        {"data": {"id": 1, "nome": "P1", "categoria": {"descricao": "Raquetes de Praia"}, "situacao": "A"}},
        {"data": {"id": 2, "nome": "P2", "categoria": {"descricao": "Raquetes de Praia"}, "situacao": "A"}},
    ])

    sync = BlingSync(client=client)
    with patch("app.sync.bling_sync.open_sync_log", new_callable=AsyncMock, return_value=1):
        with patch("app.sync.bling_sync.close_sync_log", new_callable=AsyncMock):
            with patch("app.sync.bling_sync.upsert_product",
                       new_callable=AsyncMock, return_value="inserted"):
                stats = await sync.full_sync(only_active=True)
    assert stats["total_processed"] == 2
    assert stats["inserted"] == 2
    assert client.listar_produtos.await_count == 3
    assert client.consultar_produto.await_count == 2


@pytest.mark.asyncio
async def test_full_sync_filters_by_categories(monkeypatch):
    monkeypatch.setenv("BLING_SYNC_CATEGORIES", "Raquetes de Praia")
    from app.config import get_settings
    get_settings.cache_clear()

    page = {"data": [
        {"id": 1, "categoria": {"descricao": "Raquetes de Praia"}},
        {"id": 2, "categoria": {"descricao": "Outras"}},   # excluded by filter
    ]}
    empty: dict = {"data": []}
    client = MagicMock()
    client.listar_produtos = AsyncMock(side_effect=[page, empty])
    client.consultar_produto = AsyncMock(return_value={
        "data": {"id": 1, "nome": "P1", "categoria": {"descricao": "Raquetes de Praia"}, "situacao": "A"},
    })
    with patch("app.sync.bling_sync.open_sync_log", new_callable=AsyncMock, return_value=1):
        with patch("app.sync.bling_sync.close_sync_log", new_callable=AsyncMock):
            with patch("app.sync.bling_sync.upsert_product",
                       new_callable=AsyncMock, return_value="inserted"):
                stats = await BlingSync(client=client).full_sync()
    assert stats["total_processed"] == 2
    assert stats["inserted"] == 1
    assert stats["skipped"] == 1


@pytest.mark.asyncio
async def test_sync_single_product_upsert():
    detail = {"id": 5, "nome": "P5", "categoria": {"descricao": "X"}, "situacao": "A"}
    client = MagicMock()
    client.consultar_produto = AsyncMock(return_value={"data": detail})
    with patch("app.sync.bling_sync.upsert_product",
               new_callable=AsyncMock, return_value="updated") as up:
        outcome = await BlingSync(client=client).sync_single_product(5)
    assert outcome == "updated"
    up.assert_awaited_once()
    args = up.await_args.args[0]
    assert args["id"] == 5
    assert args["nome"] == "P5"


@pytest.mark.asyncio
async def test_delete_product_marks_inactive():
    client = MagicMock()
    with patch("app.sync.bling_sync.mark_product_inactive",
               new_callable=AsyncMock, return_value=True) as mp:
        ok = await BlingSync(client=client).delete_product(99)
    assert ok is True
    mp.assert_awaited_once_with(99)


# ════════════════════════════════════════════════════════════════════════════
# WEBHOOK
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def webhook_app(monkeypatch):
    """Tiny FastAPI app exposing /bling/webhook for TestClient."""
    monkeypatch.setenv("BLING_WEBHOOK_SECRET", "topsecret")
    from app.config import get_settings
    get_settings.cache_clear()

    from fastapi import FastAPI
    from app.api.bling import router as bling_router
    app = FastAPI()
    app.include_router(bling_router)
    return app


def _sign(body: bytes, secret: str = "topsecret") -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def test_webhook_valid_signature(webhook_app):
    from fastapi.testclient import TestClient
    payload = {"event": "product.updated", "data": {"id": 1}, "timestamp": "2026-05-25T10:00:00Z"}
    body = json.dumps(payload).encode("utf-8")
    sig = _sign(body)

    with patch("app.api.bling._process_webhook_event", new_callable=AsyncMock):
        client = TestClient(webhook_app)
        resp = client.post(
            "/bling/webhook",
            content=body,
            headers={"X-Bling-Signature": sig, "Content-Type": "application/json"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"status": "accepted"}


def test_webhook_invalid_signature_returns_401(webhook_app):
    from fastapi.testclient import TestClient
    body = b'{"event":"product.updated","data":{"id":1}}'
    client = TestClient(webhook_app)
    resp = client.post(
        "/bling/webhook",
        content=body,
        headers={"X-Bling-Signature": "deadbeef", "Content-Type": "application/json"},
    )
    assert resp.status_code == 401


def test_webhook_responds_200_immediately(webhook_app):
    """Body processing is queued via BackgroundTasks — endpoint returns 200 fast."""
    from fastapi.testclient import TestClient
    body = json.dumps({"event": "product.created", "data": {"id": 7}}).encode("utf-8")
    sig = _sign(body)

    with patch("app.api.bling._process_webhook_event", new_callable=AsyncMock) as bg:
        client = TestClient(webhook_app)
        resp = client.post(
            "/bling/webhook", content=body,
            headers={"X-Bling-Signature": sig, "Content-Type": "application/json"},
        )
    assert resp.status_code == 200
    # BackgroundTasks ran the coroutine; assert it was scheduled.
    bg.assert_awaited()


@pytest.mark.asyncio
async def test_process_webhook_product_updated_triggers_sync():
    """A product.updated event triggers sync_single_product."""
    from app.api.bling import _process_webhook_event
    payload = {
        "event": "product.updated",
        "data": {"id": 42},
        "timestamp": "2026-05-25T10:00:00Z",
    }
    with patch(
        "app.api.bling.record_webhook_event",
        new_callable=AsyncMock, return_value=True,
    ):
        with patch("app.api.bling.BlingSync") as Sync:
            instance = Sync.return_value
            instance.sync_single_product = AsyncMock(return_value="updated")
            instance.delete_product = AsyncMock()
            await _process_webhook_event(payload)
    instance.sync_single_product.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_process_webhook_out_of_order_event_ignored():
    from app.api.bling import _process_webhook_event
    payload = {
        "event": "product.updated",
        "data": {"id": 7},
        "timestamp": "2026-05-25T09:00:00Z",
    }
    with patch(
        "app.api.bling.record_webhook_event",
        new_callable=AsyncMock, return_value=False,  # older than what we already applied
    ):
        with patch("app.api.bling.BlingSync") as Sync:
            instance = Sync.return_value
            instance.sync_single_product = AsyncMock()
            await _process_webhook_event(payload)
    instance.sync_single_product.assert_not_called()


@pytest.mark.asyncio
async def test_process_webhook_deleted_triggers_inactive():
    from app.api.bling import _process_webhook_event
    payload = {
        "event": "product.deleted",
        "data": {"id": 13},
        "timestamp": "2026-05-25T10:00:00Z",
    }
    with patch(
        "app.api.bling.record_webhook_event",
        new_callable=AsyncMock, return_value=True,
    ):
        with patch("app.api.bling.BlingSync") as Sync:
            instance = Sync.return_value
            instance.delete_product = AsyncMock(return_value=True)
            instance.sync_single_product = AsyncMock()
            await _process_webhook_event(payload)
    instance.delete_product.assert_awaited_once_with(13)
    instance.sync_single_product.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
# STOCK CACHE
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_stock_cache_hit():
    from app.sync.bling_stock import get_stock

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value="7")
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()
    with patch("app.sync.bling_stock._get_redis", return_value=fake_redis):
        with patch("app.adapters.bling.BlingClient.consultar_estoque", new_callable=AsyncMock) as api:
            value = await get_stock(42)
    assert value == 7
    api.assert_not_called()


@pytest.mark.asyncio
async def test_get_stock_cache_miss_calls_api():
    from app.sync.bling_stock import get_stock

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()
    with patch("app.sync.bling_stock._get_redis", return_value=fake_redis):
        with patch(
            "app.adapters.bling.BlingClient.consultar_estoque",
            new_callable=AsyncMock,
            return_value={"data": [{"produto": {"id": 42}, "saldoFisicoTotal": 5}]},
        ):
            value = await get_stock(42)
    assert value == 5
    fake_redis.setex.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_stock_returns_none_on_api_error():
    from app.sync.bling_stock import get_stock

    fake_redis = MagicMock()
    fake_redis.get = AsyncMock(return_value=None)
    fake_redis.setex = AsyncMock()
    fake_redis.aclose = AsyncMock()
    with patch("app.sync.bling_stock._get_redis", return_value=fake_redis):
        with patch(
            "app.adapters.bling.BlingClient.consultar_estoque",
            new_callable=AsyncMock,
            side_effect=BlingError("upstream timeout"),
        ):
            value = await get_stock(42)
    assert value is None


# ════════════════════════════════════════════════════════════════════════════
# AGENT INTEGRATION
# ════════════════════════════════════════════════════════════════════════════

def _base_state(**overrides) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quanto custa?")],
        "phone_hash": "bling25" * 9,
        "intent": "price_inquiry",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "customer_intent_path": "determined",
        "determined_question_count": 0,
        "consultoria_mentioned_count": 0,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _bling_row(name: str, *, is_raquete: bool, price_cents: int = 89900) -> dict:
    return {
        "id": 1, "name": name, "price_cents": price_cents,
        "is_raquete_praia": is_raquete,
        "description": "",
        "marca": "X", "modelo": name,
        "categoria_nome": "Raquetes de Praia" if is_raquete else "Bola Beach TEnnis",
        "external_id": "1",
    }


@pytest.mark.asyncio
async def test_recommend_uses_bling_products_when_enabled(monkeypatch):
    """Sprint 2.6.3 — recommend's catalog candidates come from
    ``get_catalog_snapshot()`` (the in-memory cache), no longer via the
    per-message ``fetch_product_by_name``. Mocking the snapshot fully
    controls the candidate list the matcher sees."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.nodes.recommend import _list_catalog_candidates
    bling_row = _bling_row("Raquete BeachPro Carbon X5", is_raquete=True)
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock, return_value=[bling_row],
    ):
        candidates = await _list_catalog_candidates("Raquete BeachPro Carbon X5")
    assert candidates
    assert candidates[0]["name"] == "Raquete BeachPro Carbon X5"


@pytest.mark.asyncio
async def test_raquete_praia_triggers_consultoria_pitch():
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _base_state(
        recommended_products=[_bling_row("Raquete X", is_raquete=True, price_cents=80000)],
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    # PRICE preset's distinctive opener (raquete path).
    assert "investimento numa raquete" in full.lower()


@pytest.mark.asyncio
async def test_non_raquete_does_NOT_trigger_consultoria():
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _base_state(
        recommended_products=[_bling_row("Bola Beach 3-Pack", is_raquete=False, price_cents=4900)],
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    assert "Consultoria" not in full
    assert result.get("consultoria_mentioned_count") is None


@pytest.mark.asyncio
async def test_price_inquiry_includes_stock_check_when_bling_active(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _base_state(
        recommended_products=[_bling_row("Raquete X", is_raquete=True)],
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    with patch(
        "app.sync.bling_stock.get_stock",
        new_callable=AsyncMock, return_value=5,
    ) as stock_call:
        await price_inquiry_node(state)
    stock_call.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_out_of_stock_shows_friendly_message(monkeypatch):
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _base_state(
        recommended_products=[_bling_row("Raquete X", is_raquete=True)],
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    with patch(
        "app.sync.bling_stock.get_stock",
        new_callable=AsyncMock, return_value=0,
    ):
        result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    assert "em falta" in full.lower() or "avisar quando voltar" in full.lower()


@pytest.mark.asyncio
async def test_product_detail_non_raquete_no_pitch():
    """Sprint 2.5 — product_detail must suppress the subtle pitch for non-rackets."""
    from app.agent.nodes.product_detail import product_detail_node

    state = _base_state(
        intent="product_detail",
        messages=[HumanMessage(content="qual o material?")],
        recommended_products=[_bling_row(
            "Top Treino", is_raquete=False, price_cents=12900,
        )],
        determined_question_count=1,  # would normally fire DELAYED pitch
    )
    # Override description-search path defensively
    result = await product_detail_node(state)
    full = " ".join(result["response_blocks"])
    # Sprint 2.6.9 — the micro-tag at product_detail.py also says
    # "Consultoria Base Sports" (brand cleanup) so we detect the PITCH
    # by its unique signature: price + abatido + CTA.
    assert "R$ 350" not in full
    assert "abatido" not in full.lower()
    # We don't increment the cap counter when no real pitch happened.
    assert result.get("consultoria_mentioned_count") is None
