"""Sprint 2.6.6 — attribute_inquiry tests.

The pre-2.6.6 bug: "qual o peso?" was classified as ``product_inquiry`` and
the token matcher found Lead Tape / Overgrip (products whose NAMES contain
"peso"). Sprint 2.6.6 introduces the ``attribute_inquiry`` intent + node so:

- "qual o peso dela?" reads the structured attribute from the ACTIVE product
- a missing attribute triggers an honest "vou confirmar e te retorno" reply
- the promise is backed by a real internal alert to Andre
- anti-spam prevents repeat alerts for the same product+attribute
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState


def _product(
    name: str,
    *,
    atributos: dict | None = None,
    is_raquete: bool = True,
    price_cents: int = 100000,
    pid: int | None = None,
) -> dict:
    return {
        "id": pid if pid is not None else abs(hash(name)) & 0xFFFFFFFF,
        "name": name,
        "price_cents": price_cents,
        "is_raquete_praia": is_raquete,
        "description": "",
        "external_id": name,
        "atributos_parseados": atributos or {},
    }


def _state(message: str, **overrides) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content=message)],
        "phone_hash": "attrnode" * 8,
        "intent": "attribute_inquiry",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ── Intent classification ───────────────────────────────────────────────────

def test_attribute_inquiry_in_valid_intent_set():
    from app.agent.nodes.triage import _VALID_INTENTS
    assert "attribute_inquiry" in _VALID_INTENTS


def test_attribute_inquiry_in_intent_to_node_map():
    from app.agent.graph import _INTENT_TO_NODE
    assert _INTENT_TO_NODE["attribute_inquiry"] == "attribute_inquiry"


def test_system_triage_prompt_documents_attribute_inquiry():
    from app.agent.prompts import SYSTEM_TRIAGE
    assert "attribute_inquiry" in SYSTEM_TRIAGE
    # The prompt must teach the LLM the distinction from product_inquiry.
    assert "característica técnica" in SYSTEM_TRIAGE.lower() or (
        "caracteristica tecnica" in SYSTEM_TRIAGE.lower()
    )


# ── detect_requested_attributes + synonyms ──────────────────────────────────

def test_detect_attribute_peso():
    from app.agent.nodes.attribute_inquiry import detect_requested_attributes
    for q in ("qual o peso?", "quanto pesa?", "ela é pesada?"):
        assert "peso" in detect_requested_attributes(q), q


def test_detect_attribute_equilibrio_via_balance_synonym():
    from app.agent.nodes.attribute_inquiry import detect_requested_attributes
    for q in (
        "qual o balance?",
        "qual o balanço?",
        "qual o equilíbrio dela?",
        "qual o ponto de impacto?",
    ):
        assert "equilibrio" in detect_requested_attributes(q), q


def test_detect_attribute_composicao_via_material_synonym():
    from app.agent.nodes.attribute_inquiry import detect_requested_attributes
    for q in ("qual o material?", "de que é feita?", "composição?"):
        assert "composicao" in detect_requested_attributes(q), q


def test_detect_multiple_attributes_in_one_question():
    from app.agent.nodes.attribute_inquiry import detect_requested_attributes
    found = detect_requested_attributes("qual o peso e balance?")
    assert "peso" in found and "equilibrio" in found


def test_full_spec_request_detected():
    from app.agent.nodes.attribute_inquiry import is_full_spec_request
    assert is_full_spec_request("me fala a ficha técnica")
    assert is_full_spec_request("quais as specs?")
    assert not is_full_spec_request("qual o peso?")


# ── get_active_product helper ──────────────────────────────────────────────

def test_get_active_product_returns_single_when_only_one():
    from app.agent.nodes.attribute_inquiry import get_active_product
    state = _state("...", recommended_products=[_product("Raquete X")])
    assert get_active_product(state)["name"] == "Raquete X"


def test_get_active_product_returns_none_when_zero():
    from app.agent.nodes.attribute_inquiry import get_active_product
    assert get_active_product(_state("...", recommended_products=[])) is None


def test_get_active_product_returns_none_when_multiple():
    """Multiple products → ambiguous, no single active product."""
    from app.agent.nodes.attribute_inquiry import get_active_product
    state = _state(
        "...",
        recommended_products=[_product("A"), _product("B")],
    )
    assert get_active_product(state) is None


# ── Found attribute → direct answer ────────────────────────────────────────

@pytest.mark.asyncio
async def test_attribute_found_returns_value():
    """Mormaii Sunset has peso='320g' → "A *X* pesa 320g."."""
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product(
        "Raquete Mormaii Sunset Plus",
        atributos={
            "peso": "320g (+/- 10g)",
            "equilibrio": "Aproximadamente 27,5cm",
        },
    )
    state = _state(
        "qual o peso dela?",
        recommended_products=[product],
    )
    result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "Mormaii Sunset Plus" in reply
    assert "320g" in reply


@pytest.mark.asyncio
async def test_attribute_balance_uses_equilibrio_key():
    """The customer says "balance"; the node reads "equilibrio" key."""
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product(
        "Raquete Mormaii Sunset Plus",
        atributos={
            "peso": "320g",
            "equilibrio": "27,5cm",
        },
    )
    state = _state("qual o balance dela?", recommended_products=[product])
    result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "27,5cm" in reply


@pytest.mark.asyncio
async def test_attribute_multiple_in_one_turn():
    """qual o peso e balance? → answers both in one bullet list."""
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product(
        "Raquete Sunset",
        atributos={"peso": "320g", "equilibrio": "27,5cm"},
    )
    state = _state("qual o peso e balance?", recommended_products=[product])
    result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "320g" in reply
    assert "27,5cm" in reply


# ── Partial: some attributes found, others missing → honest mix ────────────

@pytest.mark.asyncio
async def test_attribute_partial_tells_what_is_known_and_promises_rest(monkeypatch):
    """Product has composicao but no peso. Customer asks for both → mention
    composicao + honest promise for peso + alert fired."""
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product(
        "Raquete Parcial",
        atributos={"composicao": "Carbono 3K"},
    )
    state = _state(
        "qual o peso e composição?",
        recommended_products=[product],
    )
    with patch(
        "app.agent.nodes.attribute_inquiry._send_missing_attr_alert",
        new_callable=AsyncMock,
    ) as alert:
        result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "Carbono 3K" in reply
    assert "não consta" in reply.lower()
    assert "confirmar com a equipe" in reply.lower()
    alert.assert_awaited_once()


# ── Totally missing → honest + alert ───────────────────────────────────────

@pytest.mark.asyncio
async def test_attribute_totally_missing_triggers_alert(monkeypatch):
    """Furia Attack has nothing in atributos_parseados → honest reply + alert."""
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product("Raquete Drop Shot Furia Attack PK", atributos={})
    state = _state("qual o peso dela?", recommended_products=[product])
    with patch(
        "app.agent.nodes.attribute_inquiry._send_missing_attr_alert",
        new_callable=AsyncMock,
    ) as alert:
        result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "não consta" in reply.lower() or "vou confirmar" in reply.lower()
    assert "Furia Attack" in reply
    alert.assert_awaited_once()


# ── Regression: matcher is NOT invoked for "qual o peso?" alone ────────────

@pytest.mark.asyncio
async def test_attribute_inquiry_does_not_call_product_matcher_for_attr_only():
    """The bug from production: "qual o peso dela?" must NEVER fetch
    products named "Peso Adesivo" or "Fita Lead Tape"."""
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product(
        "Raquete Drop Shot Furia Attack PK",
        atributos={"peso": "335g"},
    )
    state = _state("qual o peso dela?", recommended_products=[product])

    # Snapshot deliberately includes the "garbage" products that bit us
    # in production. The node must NOT promote them.
    bad_snapshot = [
        product,
        {"id": 9001, "name": "Lead Tape Peso Adesivo Chumbo Raquete",
         "price_cents": 2900, "atributos_parseados": {}, "external_id": "9001"},
        {"id": 9002, "name": "Overgrip Heroes Pacote", "price_cents": 1900,
         "atributos_parseados": {}, "external_id": "9002"},
    ]
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock, return_value=bad_snapshot,
    ):
        result = await attribute_inquiry_node(state)

    reply = result["response_blocks"][0]
    # The active product wins.
    assert "Furia Attack" in reply
    assert "335g" in reply
    # The bad results must NOT appear.
    assert "Lead Tape" not in reply
    assert "Overgrip" not in reply


# ── No active product → ask which product ──────────────────────────────────

@pytest.mark.asyncio
async def test_attribute_inquiry_no_active_product_asks_which():
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    state = _state("qual o peso?", recommended_products=[])
    # Catalog snapshot returns empty so no in-sentence match either.
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock, return_value=[],
    ):
        result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "qual produto" in reply.lower() or "qual raquete" in reply.lower()


# ── Customer NAMES a product in the same sentence ──────────────────────────

@pytest.mark.asyncio
async def test_attribute_named_product_in_same_sentence():
    """qual o peso da Mormaii Sunset? — even without an active product,
    the node resolves Mormaii Sunset from the catalog and answers."""
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    sunset = _product(
        "Raquete Mormaii Sunset Plus",
        atributos={"peso": "320g"},
        pid=42,
    )
    state = _state(
        "qual o peso da Mormaii Sunset?",
        recommended_products=[],   # no active product
    )
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock, return_value=[sunset],
    ):
        result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "Mormaii Sunset Plus" in reply
    assert "320g" in reply


# ── Anti-spam: same product+attr alert fires only once ─────────────────────

@pytest.mark.asyncio
async def test_alert_not_duplicated_same_attribute(monkeypatch):
    """Second turn asking the SAME missing attr must NOT trigger a second alert."""
    monkeypatch.setenv("DOSSIER_RECIPIENT_PHONE", "5511999999999")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.nodes.attribute_inquiry import _send_missing_attr_alert

    product = _product("Raquete X", atributos={}, pid=777)
    state: AgentState = _state(
        "...",
        recommended_products=[product],
        alerted_missing_attrs=["777:peso"],   # already alerted
    )

    sends: list[tuple[str, str]] = []

    async def _fake_send(self, phone, text):
        sends.append((phone, text))

    monkeypatch.setattr(
        "app.adapters.evolution.EvolutionClient.send_text", _fake_send
    )
    await _send_missing_attr_alert(state, product, ["peso"])
    assert sends == []   # suppressed by anti-spam marker


@pytest.mark.asyncio
async def test_alert_fires_first_time_then_marker_added(monkeypatch):
    """First time we promise → alert sent + marker added to state."""
    monkeypatch.setenv("DOSSIER_RECIPIENT_PHONE", "5511999999999")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product("Raquete X", atributos={}, pid=555)
    state = _state("qual o peso?", recommended_products=[product])

    sends: list[str] = []

    async def _fake_send(self, phone, text):
        sends.append(text)

    monkeypatch.setattr(
        "app.adapters.evolution.EvolutionClient.send_text", _fake_send
    )
    result = await attribute_inquiry_node(state)
    assert len(sends) == 1
    # Marker is now in the returned state update.
    markers = result.get("alerted_missing_attrs") or []
    assert any("555" in m and "peso" in m for m in markers)


# ── Full-spec request lists everything available ───────────────────────────

@pytest.mark.asyncio
async def test_generic_specs_request_lists_all():
    from app.agent.nodes.attribute_inquiry import attribute_inquiry_node

    product = _product(
        "Raquete Sunset",
        atributos={
            "peso": "320g",
            "equilibrio": "27,5cm",
            "composicao": "Carbono 3K",
        },
    )
    state = _state(
        "me fala a ficha técnica dela",
        recommended_products=[product],
    )
    result = await attribute_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "320g" in reply
    assert "27,5cm" in reply
    assert "Carbono 3K" in reply
