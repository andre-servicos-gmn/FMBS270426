"""Sprint 1.10 — post-recommendation follow-up flows.

Covers the 5 new intents (price_inquiry, product_selection, re_recommendation,
product_detail, out_of_scope) and the triage gating that only activates them
when ``recommended_products`` + ``last_recommendation_at`` are both set.
"""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState


# ── Helpers ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _product(name: str, *, price_cents: int = 70000, description: str = "", **extra) -> dict:
    base = {
        "id": f"id-{name}",
        "name": name,
        "sport": "beach_tennis",
        "level": "intermediário",
        "price_cents": price_cents,
        "stock": 5,
        "description": description or f"Descrição genérica de {name}",
        "similarity": 0.9,
        "external_id": name.replace(" ", "-"),
        "url": None,
        "image_url": None,
        "updated_at": None,
        "is_active": True,
        "weight_g": 350,
        "balance": "médio",
        "material": "carbono",
    }
    base.update(extra)
    return base


def _post_rec_state(
    *,
    user_msg: str,
    products: list[dict],
    selected: dict | None = None,
) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=user_msg)],
        phone_hash="followup1" * 7,
        intent="recommend",
        player_profile={
            "nivel_jogo": "intermediário",
            "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma",
            "esporte_raquete_previo": "nao_aplicavel",
            "modelo_desejado": "nenhum",
        },
        recommended_products=products,
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
        selected_product=selected,
    )


# ════════════════════════════════════════════════════════════════════════════
# PRICE INQUIRY
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_price_inquiry_specific_product():
    """Customer names a specific product → response carries that product's price."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    products = [
        _product("Raquete BeachPro Carbon X5", price_cents=89900),
        _product("Raquete AirBlast Carbon Pro", price_cents=129900),
    ]
    state = _post_rec_state(
        user_msg="Quanto custa a AirBlast Carbon Pro?",
        products=products,
    )
    result = await price_inquiry_node(state)
    blocks = result["response_blocks"]
    full = " ".join(blocks)
    assert "AirBlast Carbon Pro" in full
    assert "1.299" in full  # R$ 1.299 (cents/100 = 1299, formatted with dot)
    assert "899" not in full  # the OTHER product's price must not appear


@pytest.mark.asyncio
async def test_price_inquiry_ambiguous():
    """Vague 'quanto custa?' → list every recommended product with its price."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    products = [
        _product("Raquete A", price_cents=50000),
        _product("Raquete B", price_cents=80000),
    ]
    state = _post_rec_state(user_msg="quanto custa?", products=products)
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    assert "Raquete A" in full and "Raquete B" in full
    assert "500" in full and "800" in full  # both prices shown


# ════════════════════════════════════════════════════════════════════════════
# PRODUCT SELECTION
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_product_selection_routes_to_close():
    """product_selection node stores selected_product and routes to close.

    Here we just verify the node populates ``selected_product``. The graph
    edge sends control to ``close`` — covered by test_product_selection_with_store_info.
    """
    from app.agent.nodes.product_selection import product_selection_node

    products = [
        _product("Raquete BeachPro Carbon X5"),
        _product("Raquete AirBlast Carbon Pro"),
    ]
    state = _post_rec_state(
        user_msg="prefiro a AirBlast",
        products=products,
    )
    result = await product_selection_node(state)
    assert "selected_product" in result
    assert result["selected_product"]["name"] == "Raquete AirBlast Carbon Pro"


# ════════════════════════════════════════════════════════════════════════════
# RE-RECOMMENDATION — Sprint 2.0 pivots to Consultoria offer
# ════════════════════════════════════════════════════════════════════════════
# The pre-2.0 cheaper / more_advanced / different flavour logic was removed.
# Re-recommendation now delegates to consultoria_offer_node; see
# ``tests/test_pivot_no_recommendation.py::test_re_recommendation_pivots_to_consultoria``
# for the new behaviour.


# ════════════════════════════════════════════════════════════════════════════
# PRODUCT DETAIL
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_product_detail_known_attribute():
    """Asking about weight returns the product's weight_g field."""
    from app.agent.nodes.product_detail import product_detail_node

    products = [_product("Raquete X", weight_g=360)]
    state = _post_rec_state(user_msg="qual o peso da Raquete X?", products=products)
    result = await product_detail_node(state)
    full = " ".join(result["response_blocks"])
    assert "360" in full
    assert "peso" in full.lower()


@pytest.mark.asyncio
async def test_product_detail_unknown_attribute_fallback():
    """Asking about an attribute we don't have → honest fallback + Consultoria pointer."""
    from app.agent.nodes.product_detail import product_detail_node

    # Description has no mention of antivibration anywhere.
    products = [_product("Raquete X", description="Raquete leve com bom controle", weight_g=350)]
    state = _post_rec_state(
        user_msg="a Raquete X tem antivibração?",
        products=products,
    )
    result = await product_detail_node(state)
    full = " ".join(result["response_blocks"]).lower()
    assert "não tenho" in full or "não" in full
    assert "consultoria" in full


# ════════════════════════════════════════════════════════════════════════════
# OUT OF SCOPE
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_out_of_scope_sets_handoff_flag():
    from app.agent.nodes.out_of_scope_handoff import out_of_scope_handoff_node

    state = _post_rec_state(
        user_msg="vocês entregam em casa?",
        products=[_product("Raquete X")],
    )
    with patch("app.storage.db.get_session", _mock_db_session):
        result = await out_of_scope_handoff_node(state)

    assert result["needs_handoff"] is True
    assert result["handoff_reason"] == "out_of_scope"


@pytest.mark.asyncio
async def test_out_of_scope_returns_canned_response():
    from app.agent.nodes.out_of_scope_handoff import out_of_scope_handoff_node

    state = _post_rec_state(
        user_msg="aceita pix?",
        products=[_product("Raquete X")],
    )
    with patch("app.storage.db.get_session", _mock_db_session):
        result = await out_of_scope_handoff_node(state)

    msg = result["response_blocks"][0]
    assert "atendimento humano" in msg
    assert "equipe" in msg


# ════════════════════════════════════════════════════════════════════════════
# TRIAGE GATING
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_triage_classifies_price_after_recommendation():
    """Router maps intent=price_inquiry → 'price_inquiry' when post-rec is true."""
    from app.agent.graph import _triage_router

    state = AgentState(
        messages=[HumanMessage(content="quanto custa?")],
        phone_hash="x" * 64,
        intent="price_inquiry",
        player_profile={},
        recommended_products=[_product("Raquete X")],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    assert _triage_router(state) == "price_inquiry"
