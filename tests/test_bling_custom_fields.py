"""Sprint 2.7.4 — Bling V3 custom-fields endpoint discovery + pagination.

The Sprint 2.5.2 code guessed two non-existent paths
(``/produtos/campos-customizados``, ``/campos-customizados``). Both return
404. The real endpoint pattern is:

    1. GET /campos-customizados/modulos           → list modules
       Find the item whose nome == "Produtos", extract id.
    2. GET /campos-customizados/modulos/{idModulo}?pagina=N&limite=100
       → list custom fields for the module, paginated.

These tests mock ``BlingClient._request`` directly so we exercise the new
discovery + pagination logic without hitting the network.
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.adapters.bling import BlingClient, BlingError


# ── Fixtures ────────────────────────────────────────────────────────────────

def _modules_response(produtos_id: int = 42) -> dict:
    """Mimics ``GET /campos-customizados/modulos``."""
    return {
        "data": [
            {"id": 1, "nome": "Pedidos", "modulo": "VENDAS", "agrupador": "ERP", "permissoes": []},
            {"id": produtos_id, "nome": "Produtos", "modulo": "CATALOG", "agrupador": "ERP", "permissoes": []},
            {"id": 99, "nome": "Contatos", "modulo": "CRM", "agrupador": "ERP", "permissoes": []},
        ]
    }


def _fields_page(items: list[dict]) -> dict:
    """Mimics one page of ``GET /campos-customizados/modulos/{id}``."""
    return {"data": items}


# ════════════════════════════════════════════════════════════════════════════
# DISCOVERY — _find_produtos_module_id
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_finds_produtos_module_by_name():
    client = BlingClient()
    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value=_modules_response(produtos_id=42),
    ):
        module_id = await client._find_produtos_module_id()
    assert module_id == 42
    # Cached on instance — second call doesn't re-hit.
    assert client._produtos_module_id == 42


@pytest.mark.asyncio
async def test_module_match_is_case_and_accent_insensitive():
    """Defensive: even if Bling shifts to 'produtos' or accented variants,
    the case+accent-insensitive matcher catches it."""
    client = BlingClient()
    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value={"data": [{"id": 77, "nome": "PRODUTOS"}]},
    ):
        assert await client._find_produtos_module_id() == 77

    client2 = BlingClient()
    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value={"data": [{"id": 88, "nome": "produtos"}]},
    ):
        assert await client2._find_produtos_module_id() == 88


@pytest.mark.asyncio
async def test_module_not_found_returns_none(caplog):
    """If the modules list comes back without 'Produtos', return None and
    log it."""
    import logging
    caplog.set_level(logging.INFO, logger="app.adapters.bling")

    client = BlingClient()
    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value={"data": [{"id": 1, "nome": "Pedidos"}]},
    ):
        result = await client._find_produtos_module_id()

    assert result is None
    assert any(
        "bling_custom_fields_module_not_found" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_discovery_endpoint_error_returns_none(caplog):
    """If the modules endpoint itself errors (404, 401, transport), the
    method swallows and returns None so the sync can degrade gracefully."""
    import logging
    caplog.set_level(logging.INFO, logger="app.adapters.bling")

    client = BlingClient()
    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        side_effect=BlingError("404 not found"),
    ):
        result = await client._find_produtos_module_id()

    assert result is None
    assert any(
        "bling_custom_fields_modules_failed" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_discovery_caches_module_id():
    """Second call uses the cached id — only ONE call to _request."""
    client = BlingClient()
    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value=_modules_response(produtos_id=42),
    ) as mock_req:
        await client._find_produtos_module_id()
        await client._find_produtos_module_id()
        await client._find_produtos_module_id()

    assert mock_req.await_count == 1
    assert client._produtos_module_id == 42


# ════════════════════════════════════════════════════════════════════════════
# LIST + FILTER + PAGINATION — listar_campos_customizados
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_returns_active_fields_only():
    """Sprint 2.7.4: situacao != 'A' is filtered out."""
    client = BlingClient()
    client._produtos_module_id = 42  # skip discovery

    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value=_fields_page([
            {"id": 100, "nome": "Peso", "situacao": "A"},
            {"id": 101, "nome": "Composição", "situacao": "A"},
            {"id": 102, "nome": "Campo Descontinuado", "situacao": "I"},  # filtered
            {"id": 103, "nome": "Equilíbrio", "situacao": "A"},
        ]),
    ):
        result = await client.listar_campos_customizados()

    names = [item["nome"] for item in result]
    assert names == ["Peso", "Composição", "Equilíbrio"]
    assert all(item.get("situacao", "A") == "A" for item in result)


@pytest.mark.asyncio
async def test_list_field_without_situacao_kept_defensively():
    """If a field has no 'situacao' key (defensive — older builds), we keep
    it instead of dropping silently."""
    client = BlingClient()
    client._produtos_module_id = 42

    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value=_fields_page([
            {"id": 200, "nome": "Atributo Legacy"},   # no situacao
            {"id": 201, "nome": "Outro", "situacao": "A"},
        ]),
    ):
        result = await client.listar_campos_customizados()

    names = [item["nome"] for item in result]
    assert "Atributo Legacy" in names
    assert "Outro" in names


@pytest.mark.asyncio
async def test_list_paginates_when_first_page_is_full():
    """A full page (100 items) triggers fetching the next page; a short
    page terminates the loop."""
    client = BlingClient()
    client._produtos_module_id = 42

    page1 = [{"id": i, "nome": f"Campo {i}", "situacao": "A"} for i in range(1, 101)]
    page2 = [{"id": 101, "nome": "Campo 101", "situacao": "A"}]
    pages = [_fields_page(page1), _fields_page(page2)]

    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        side_effect=pages,
    ) as mock_req:
        result = await client.listar_campos_customizados()

    # Two HTTP calls: pagina=1 (full), pagina=2 (short → stop).
    assert mock_req.await_count == 2
    # Verify query params on each call.
    call1 = mock_req.await_args_list[0]
    call2 = mock_req.await_args_list[1]
    assert call1.kwargs["params"] == {"pagina": 1, "limite": 100}
    assert call2.kwargs["params"] == {"pagina": 2, "limite": 100}
    # All 101 active items collected.
    assert len(result) == 101


@pytest.mark.asyncio
async def test_list_single_short_page_terminates_quickly():
    """A short first page → exactly one HTTP call."""
    client = BlingClient()
    client._produtos_module_id = 42

    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        return_value=_fields_page([
            {"id": 1, "nome": "Peso", "situacao": "A"},
            {"id": 2, "nome": "Marca", "situacao": "A"},
        ]),
    ) as mock_req:
        result = await client.listar_campos_customizados()

    assert mock_req.await_count == 1
    assert len(result) == 2


@pytest.mark.asyncio
async def test_list_endpoint_error_returns_empty(caplog):
    """If the per-module list endpoint errors, return [] (graceful
    degradation → ``campo_<id>`` synthetic keys downstream)."""
    import logging
    caplog.set_level(logging.INFO, logger="app.adapters.bling")

    client = BlingClient()
    client._produtos_module_id = 42

    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        side_effect=BlingError("503 service unavailable"),
    ):
        result = await client.listar_campos_customizados()

    assert result == []
    assert any(
        "bling_custom_fields_list_failed" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_list_module_discovery_failure_returns_empty():
    """Discovery failed → list returns [] without ever calling the
    per-module endpoint."""
    client = BlingClient()
    # discovery returns None → don't even try the second endpoint
    with patch.object(
        BlingClient, "_find_produtos_module_id", new_callable=AsyncMock,
        return_value=None,
    ):
        with patch.object(
            BlingClient, "_request", new_callable=AsyncMock,
        ) as mock_req:
            result = await client.listar_campos_customizados()

    assert result == []
    mock_req.assert_not_called()


@pytest.mark.asyncio
async def test_list_full_flow_discovery_plus_fields():
    """End-to-end: no cached module id; discovery happens, then fields are
    listed and filtered."""
    client = BlingClient()
    assert client._produtos_module_id is None

    call_sequence = [
        # 1. modules discovery
        _modules_response(produtos_id=42),
        # 2. fields for module 42 (single short page)
        _fields_page([
            {"id": 1979924, "nome": "Marca", "situacao": "A"},
            {"id": 3474356, "nome": "Composição", "situacao": "A"},
            {"id": 9999999, "nome": "Inativo", "situacao": "I"},
        ]),
    ]

    with patch.object(
        BlingClient, "_request", new_callable=AsyncMock,
        side_effect=call_sequence,
    ) as mock_req:
        result = await client.listar_campos_customizados()

    assert mock_req.await_count == 2
    # Module id was discovered and cached.
    assert client._produtos_module_id == 42
    # Inactive field filtered.
    names = [item["nome"] for item in result]
    assert names == ["Marca", "Composição"]
