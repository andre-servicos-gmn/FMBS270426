"""Sprint 2.7.5 — map customer attribute questions to the REAL Bling
custom-field keys that the Sprint 2.7.4 fix now lands in
``atributos_parseados``.

Real Felipe catalog (confirmed in Supabase):
    marca:                  "Mormaii"
    modelo:                 "Sunset Plus"
    materiais_do_exterior:  "Carbono 3k"   ← material/composição
    espessura_do_perfil_mm: "22"           ← needs to render as "22mm"
    material_am:            "car"          ← BLOCKED (internal abbreviation)
    es_raquete_de_praia:    "true"         ← BLOCKED (internal flag)
    baterias_sao_necessarias: "false"      ← BLOCKED (marketplace noise)

Coverage:
  - marca / modelo (new slugs added)
  - composicao → resolves via ``materiais_do_exterior`` (Bling key)
  - espessura → resolves via ``espessura_do_perfil_mm`` + formats "22"→"22mm"
  - peso/equilibrio/comprimento absent → honest-missing (NOT invented)
  - blocked keys NEVER leak even via full-scan or broad-detail listing
  - boolean-string values ("true"/"false") never leak
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.attribute_inquiry import (
    _BLOCKED_KEYS,
    _BOOLEAN_OR_NULL_VALUES,
    _format_value,
    _is_meaningful_value,
    _list_available_attributes,
    _read_attribute,
    attribute_inquiry_node,
)
from app.agent.state import AgentState


# ── Fixtures ────────────────────────────────────────────────────────────────

def _mormaii_felipe() -> dict:
    """Exact attributes shape from the Felipe production DB (post 2.7.4 sync)."""
    return {
        "id": 18472103881,
        "name": "Raquete Mormaii Sunset Plus 2026",
        "price_cents": 89900,
        "is_raquete_praia": True,
        "external_id": "mormaii-sunset-plus",
        "atributos_parseados": {
            "marca": "Mormaii",
            "modelo": "Sunset Plus",
            "materiais_do_exterior": "Carbono 3k",
            "espessura_do_perfil_mm": "22",
            "material_am": "car",
            "es_raquete_de_praia": "true",
            "baterias_sao_necessarias": "false",
        },
    }


def _legacy_only_composicao() -> dict:
    """A product without the Bling custom field but with the old
    description-parsed 'composicao' key — used to confirm the slug
    candidate chain still resolves legacy data."""
    return {
        "id": 999,
        "name": "Raquete Legacy",
        "price_cents": 30000,
        "is_raquete_praia": True,
        "external_id": "legacy",
        "atributos_parseados": {
            "composicao": "Carbono 3K (legacy)",
        },
    }


def _bare_peso_numeric() -> dict:
    """A product with peso as a bare number string — confirms the unit
    formatter appends 'g'."""
    return {
        "id": 555,
        "name": "Raquete Test Bare Peso",
        "price_cents": 30000,
        "is_raquete_praia": True,
        "external_id": "bare-peso",
        "atributos_parseados": {"peso": "320"},
    }


def _state(message: str, products: list[dict]) -> AgentState:
    return {  # type: ignore[return-value]
        "messages": [HumanMessage(content=message)],
        "phone_hash": "felipe" * 10,
        "intent": "attribute_inquiry",
        "player_profile": {},
        "recommended_products": products,
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }


# ════════════════════════════════════════════════════════════════════════════
# UNIT — _read_attribute on Felipe's REAL data
# ════════════════════════════════════════════════════════════════════════════

def test_read_marca_returns_value():
    product = _mormaii_felipe()
    assert _read_attribute(product, "marca") == "Mormaii"


def test_read_modelo_returns_value():
    product = _mormaii_felipe()
    assert _read_attribute(product, "modelo") == "Sunset Plus"


def test_read_composicao_resolves_via_materiais_do_exterior():
    """Material question → composicao slug → chain looks up
    materiais_do_exterior FIRST (Bling V3 field name)."""
    product = _mormaii_felipe()
    assert _read_attribute(product, "composicao") == "Carbono 3k"


def test_read_espessura_appends_mm_unit():
    """Espessura comes as bare '22' from the Bling sync; we render '22mm'."""
    product = _mormaii_felipe()
    assert _read_attribute(product, "espessura") == "22mm"


def test_read_peso_missing_returns_none_honest():
    """Felipe doesn't cadastrate peso — the matcher returns None (NOT
    inventing a value). The honest-missing branch in the node then fires."""
    product = _mormaii_felipe()
    assert _read_attribute(product, "peso") is None
    assert _read_attribute(product, "equilibrio") is None
    assert _read_attribute(product, "comprimento") is None


def test_read_legacy_composicao_key_still_works():
    """Backward compat: products without the Bling key but with the
    description-parsed 'composicao' (Sprint 2.6.7) still resolve."""
    product = _legacy_only_composicao()
    assert _read_attribute(product, "composicao") == "Carbono 3K (legacy)"


def test_read_bare_peso_appends_g_unit():
    product = _bare_peso_numeric()
    assert _read_attribute(product, "peso") == "320g"


def test_read_already_formatted_value_preserved():
    """If the sync already stored a value with the unit ('320g'), don't
    double-suffix it ('320gg' would be a bug)."""
    product = {"atributos_parseados": {"peso": "320g"}}
    assert _read_attribute(product, "peso") == "320g"

    product2 = {"atributos_parseados": {"espessura": "22mm"}}
    assert _read_attribute(product2, "espessura") == "22mm"


# ════════════════════════════════════════════════════════════════════════════
# BLOCKLIST — keys & values
# ════════════════════════════════════════════════════════════════════════════

def test_blocklist_keys_never_returned_via_candidates():
    """Even if a future slug somehow maps to material_am, the blocklist
    drops it before returning the lixo value 'car'."""
    product = _mormaii_felipe()
    # Synthesize a slug→list_with_blocked_key situation via direct read.
    # _read_attribute should NEVER return 'car' for any meaningful slug.
    # (material_am IS in the dict, but no canonical slug points at it.)
    for slug in ("marca", "modelo", "composicao", "espessura",
                 "peso", "equilibrio", "comprimento"):
        result = _read_attribute(product, slug)
        assert result != "car", (
            f"slug={slug} leaked the material_am lixo value 'car'"
        )


def test_blocklist_keys_excluded_from_broad_detail_listing():
    """``_list_available_attributes`` (used by 'detalhes' broad query)
    must NOT include the blocked keys in the rendered ficha."""
    product = _mormaii_felipe()
    available = _list_available_attributes(product)

    # The 4 GOOD attributes appear.
    assert available.get("marca") == "Mormaii"
    assert available.get("modelo") == "Sunset Plus"
    assert available.get("composicao") == "Carbono 3k"
    assert available.get("espessura") == "22mm"

    # The 3 BAD ones never appear, regardless of slug.
    for slug, value in available.items():
        assert value not in ("car", "true", "false", "True", "False")
        # And the source keys themselves aren't in the slug→value map.
        assert slug not in _BLOCKED_KEYS


def test_boolean_string_value_not_returned():
    """User addition: even if a key is NOT in the blocklist, a value like
    'true'/'false' (boolean-stringified) is never returned as ficha."""
    # Synthesize a product where 'marca' contains "true" (degenerate case)
    product = {
        "atributos_parseados": {
            "marca": "true",
            "modelo": "False",
            "composicao": "  ",   # whitespace-only
        }
    }
    assert _read_attribute(product, "marca") is None
    assert _read_attribute(product, "modelo") is None
    assert _read_attribute(product, "composicao") is None


def test_blocklist_constant_includes_critical_keys():
    """Smoke-check the blocklist content so future refactors don't accidentally drop entries."""
    assert "material_am" in _BLOCKED_KEYS
    assert "es_raquete_de_praia" in _BLOCKED_KEYS
    assert "baterias_sao_necessarias" in _BLOCKED_KEYS


def test_boolean_or_null_blocklist_lowercased():
    """Sanity: comparison is case-insensitive — 'TRUE', 'True', 'true' all blocked."""
    assert not _is_meaningful_value("true")
    assert not _is_meaningful_value("TRUE")
    assert not _is_meaningful_value("True")
    assert not _is_meaningful_value("false")
    assert not _is_meaningful_value("None")
    assert not _is_meaningful_value("null")
    # Real values pass through.
    assert _is_meaningful_value("Carbono 3k")
    assert _is_meaningful_value("22")
    assert _is_meaningful_value("Mormaii")


# ════════════════════════════════════════════════════════════════════════════
# UNIT — _format_value
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("slug,raw,expected", [
    ("espessura", "22",     "22mm"),
    ("espessura", "22mm",   "22mm"),
    ("espessura", "16,5",   "16,5mm"),
    ("peso",      "320",    "320g"),
    ("peso",      "320g",   "320g"),
    ("peso",      "220-240g", "220-240g"),   # range with unit preserved
    ("comprimento", "50",   "50cm"),
    ("comprimento", "50cm", "50cm"),
    ("marca",     "Mormaii", "Mormaii"),     # no unit
    ("modelo",    "Sunset Plus", "Sunset Plus"),
    ("composicao", "Carbono 3k", "Carbono 3k"),
])
def test_format_value(slug, raw, expected):
    assert _format_value(slug, raw) == expected


# ════════════════════════════════════════════════════════════════════════════
# END-TO-END — attribute_inquiry_node + Felipe-pattern questions
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_node_marca_question_returns_mormaii():
    state = _state("qual a marca?", [_mormaii_felipe()])
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "Mormaii" in text
    # No leaked lixo.
    assert "car" not in text
    assert "true" not in text.lower()


@pytest.mark.asyncio
async def test_node_modelo_question_returns_sunset_plus():
    state = _state("qual o modelo?", [_mormaii_felipe()])
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "Sunset Plus" in text


@pytest.mark.asyncio
async def test_node_material_question_returns_carbono_3k():
    state = _state("qual o material?", [_mormaii_felipe()])
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "Carbono 3k" in text
    # NOT the abbreviation "car" from material_am.
    # Check the response doesn't quote "car" as a value.
    assert ": car." not in text
    assert " car." not in text


@pytest.mark.asyncio
async def test_node_espessura_question_returns_22mm():
    state = _state("qual a espessura?", [_mormaii_felipe()])
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "22mm" in text


@pytest.mark.asyncio
async def test_node_peso_missing_fires_honest_path():
    """Peso doesn't exist in the Felipe catalog. The node must take the
    honest-missing path (not invent a value), and fire the internal alert."""
    state = _state("qual o peso?", [_mormaii_felipe()])
    with patch(
        "app.agent.nodes.attribute_inquiry._send_missing_attr_alert",
        new_callable=AsyncMock,
    ) as alert_mock:
        result = await attribute_inquiry_node(state)

    text = result["response_blocks"][0]
    assert "não consta" in text.lower() or "vou confirmar" in text.lower()
    alert_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_node_broad_detail_lists_real_attributes_no_lixo():
    """'detalhes' on the Felipe product → lists marca, modelo, composicao
    (resolved), espessura (formatted). Never includes material_am /
    es_raquete_de_praia / baterias."""
    state = _state("me passa os detalhes", [_mormaii_felipe()])
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]

    assert "Mormaii" in text
    assert "Sunset Plus" in text
    assert "Carbono 3k" in text
    assert "22mm" in text

    # Lixo blocked:
    assert ": car" not in text
    assert "baterias" not in text.lower()
    assert "es_raquete" not in text.lower()
    assert "material_am" not in text


@pytest.mark.asyncio
async def test_node_fabricante_synonym_works():
    """Felipe's customer might say 'fabricante' instead of 'marca'."""
    state = _state("qual o fabricante?", [_mormaii_felipe()])
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "Mormaii" in text
