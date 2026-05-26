"""Sprint 1.14 — scheduling_inquiry tests.

Coverage:
    - Triage prompt distinguishes consultoria (pitch) vs scheduling_inquiry (handoff)
    - Node sets needs_handoff with the right reason
    - Canned response is the scheduling one
    - Triage gating: scheduling_inquiry is "classical-tier" (works before or
      after a recommendation — the customer may arrive already knowing the
      product)
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.scheduling_inquiry import scheduling_inquiry_node
from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _state(*, post_rec: bool = False) -> AgentState:
    s: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="como agendo a consultoria?")],
        "phone_hash": "sched" * 13,
        "intent": "scheduling_inquiry",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    if post_rec:
        s["recommended_products"] = [{"name": "Raquete X", "price_cents": 70000}]
        s["last_recommendation_at"] = datetime.now(timezone.utc).isoformat()
    return s


# ── Triage prompt content (disambiguation) ───────────────────────────────────

def test_consultoria_vs_scheduling_disambiguation_in_prompt():
    """SYSTEM_TRIAGE must list both intents AND show explicit disambiguation."""
    from app.agent.prompts import SYSTEM_TRIAGE
    s = SYSTEM_TRIAGE.lower()
    assert "scheduling_inquiry" in s
    assert "consultoria" in s
    # The "EXEMPLOS DE DESAMBIGUAÇÃO" block must be present.
    assert "desambiguação" in s or "desambiguacao" in s
    # Specific pairs that must be in the prompt for the LLM to learn the split.
    assert "como agendo" in s
    assert "como funciona a consultoria" in s


def test_scheduling_inquiry_classified_after_pitch():
    """Triage router accepts scheduling_inquiry regardless of post_rec state."""
    from app.agent.graph import _triage_router

    state = _state(post_rec=True)
    assert _triage_router(state) == "scheduling_inquiry"


def test_scheduling_inquiry_NOT_blocked_before_pitch():
    """Even pre-recommendation, scheduling_inquiry routes to its node.

    This is intentional: the customer may walk in already knowing what the
    Consultoria is (referral, ad, etc.) and ask to book. We don't gate
    scheduling_inquiry on having pitched first.
    """
    from app.agent.graph import _triage_router

    state = _state(post_rec=False)
    assert _triage_router(state) == "scheduling_inquiry"


# ── Node behaviour ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduling_inquiry_sets_handoff_flag():
    state = _state()
    with patch("app.storage.db.get_session", _mock_db_session):
        result = await scheduling_inquiry_node(state)

    assert result["needs_handoff"] is True
    assert result["handoff_reason"] == "scheduling"
    assert result["consultoria_interest"] is True


@pytest.mark.asyncio
async def test_scheduling_inquiry_returns_canned_response():
    state = _state()
    with patch("app.storage.db.get_session", _mock_db_session):
        result = await scheduling_inquiry_node(state)

    msg = result["response_blocks"][0]
    assert "agendar" in msg.lower() or "atendimento humano" in msg.lower()
    assert "equipe" in msg.lower() or "contato" in msg.lower()


# ── Override determinístico: cliente nomeia raquete já mostrada ──────────────

def test_product_selection_override_when_naming_recommended():
    """If triage classifies as recommend but the customer's last message names
    a product already in recommended_products, _triage_router OVERRIDES to
    product_selection — this is the deterministic fix for the rerun-cego bug."""
    from app.agent.graph import _triage_router

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="gostei da BeachPro Carbon X5")],
        "phone_hash": "override" * 8,
        "intent": "recommend",  # LLM mistakenly emitted this
        "player_profile": {},
        "recommended_products": [
            {"name": "Raquete BeachPro Carbon X5", "price_cents": 89900},
            {"name": "Raquete AirBlast Pro", "price_cents": 119900},
        ],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "last_recommendation_at": datetime.now(timezone.utc).isoformat(),
    }
    # Override should activate and route to product_selection.
    assert _triage_router(state) == "product_selection"


def test_product_selection_no_override_when_no_match():
    """If the message doesn't name any recommended product, no override —
    keep the legacy recommend_rerun routing."""
    from app.agent.graph import _triage_router

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quero ver mais opções")],
        "phone_hash": "noover" * 10,
        "intent": "recommend",
        "player_profile": {},
        "recommended_products": [
            {"name": "Raquete BeachPro Carbon X5", "price_cents": 89900},
        ],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "last_recommendation_at": datetime.now(timezone.utc).isoformat(),
    }
    assert _triage_router(state) == "recommend_rerun"


# Sprint 2.0 — the exhibited-only filter in recommend was removed: PROFILE
# delegates to consultoria_offer (no shortlist), REFERENCE-SIM keeps the
# single matched product, REFERENCE-NÃO clears the shortlist. The pre-2.0
# test_recommended_products_reflects_exhibited_only was dropped.


@pytest.mark.asyncio
async def test_price_inquiry_after_recommend_uses_only_exhibited():
    """End-to-end check: price_inquiry consumes ``recommended_products`` —
    after Sprint 1.14 that list is exhibited-only, so price_inquiry shows
    only the prices the customer actually saw."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quanto custa?")],
        "phone_hash": "price" * 13,
        "intent": "price_inquiry",
        "player_profile": {},
        "recommended_products": [
            {"name": "Raquete A", "price_cents": 60000},
            {"name": "Raquete C", "price_cents": 100000},
        ],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "last_recommendation_at": datetime.now(timezone.utc).isoformat(),
    }
    result = await price_inquiry_node(state)
    blob = " ".join(result["response_blocks"])
    # Both exhibited products show up
    assert "Raquete A" in blob
    assert "Raquete C" in blob
    # No phantom third product
    assert "Raquete B" not in blob
