"""Sprint 2.6.4 — multi-product context preserved across turns.

When recommend returns ``ambiguous``, we stash the candidates in
``state.last_product_candidates``. The next turn:
- Triage detects multi-product references ("as duas", "ambas", "todas")
  and routes directly to price_inquiry.
- price_inquiry quotes the price of EVERY candidate.
- price_inquiry can still resolve a single candidate by name when the
  customer disambiguates (covered by the existing matcher tests).
"""
from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.price_inquiry import (
    is_multi_product_reference,
    price_inquiry_node,
)
from app.agent.state import AgentState


def _bling_row(name: str, *, price_cents: int = 100000, is_raquete: bool = True) -> dict:
    return {
        "id": abs(hash(name)) & 0xFFFFFFFF,
        "name": name,
        "price_cents": price_cents,
        "is_raquete_praia": is_raquete,
        "description": "",
        "external_id": name,
    }


def _state(message: str, **overrides) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content=message)],
        "phone_hash": "multictx" * 8,
        "intent": "price_inquiry",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "last_recommendation_at": datetime.now(timezone.utc).isoformat(),
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ── is_multi_product_reference detector ──────────────────────────────────────

@pytest.mark.parametrize("text", [
    "qual o preço das duas?",
    "preço dos dois",
    "as duas opções",
    "ambas",
    "ambos",
    "todas elas",
    "todos eles",
    "as 2 raquetes",
    "o preço dos 3",
    "as três",
])
def test_multi_product_reference_detector_positive(text):
    assert is_multi_product_reference(text), f"{text!r} should be detected"


@pytest.mark.parametrize("text", [
    "qual o preço dessa?",
    "quanto custa a primeira?",
    "essa mesma",
    "a Carbon X5",
    "vou levar a outra",
])
def test_multi_product_reference_detector_negative(text):
    assert not is_multi_product_reference(text), f"{text!r} should NOT be detected"


# ── price_inquiry multi-candidate path ───────────────────────────────────────

@pytest.mark.asyncio
async def test_price_inquiry_handles_multiple_candidates():
    """Customer says 'as duas' after a Gaivota-ambiguous turn → both quoted."""
    candidates = [
        _bling_row(
            "Raquete Gaivota Original Beach Tennis 12k Fibra Carbono",
            price_cents=149900,
        ),
        _bling_row(
            "Raquete Beach Tennis Gaivota Original 12k Fibra de Carbono",
            price_cents=189900,
        ),
    ]
    state = _state(
        "qual o preço das duas Gaivotas?",
        last_product_candidates=candidates,
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    # Both prices must appear; both names too.
    assert "Raquete Gaivota Original Beach Tennis 12k Fibra Carbono" in full
    assert "Raquete Beach Tennis Gaivota Original 12k Fibra de Carbono" in full
    assert "R$ 1.499" in full
    assert "R$ 1.899" in full


@pytest.mark.asyncio
async def test_price_inquiry_uses_last_candidates_when_no_active_product():
    """Single-candidate fallback when recommended_products is empty."""
    candidates = [_bling_row("Raquete Gaivota Original", price_cents=149900)]
    state = _state(
        "qual o preço?",
        last_product_candidates=candidates,
        recommended_products=[],
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    assert "Raquete Gaivota Original" in full
    assert "R$ 1.499" in full


@pytest.mark.asyncio
async def test_price_inquiry_no_candidates_no_products_gives_neutral_reply():
    """Empty state → neutral ask-for-detail reply."""
    state = _state("qual o preço?", last_product_candidates=None)
    result = await price_inquiry_node(state)
    reply = result["response_blocks"][0]
    assert "produto" in reply.lower() or "preço" in reply.lower()
    # No "essa raquete" leak even when we have nothing on the table.
    assert "essa raquete" not in reply.lower()


# ── triage routes 'as duas' to price_inquiry ─────────────────────────────────

@pytest.mark.asyncio
async def test_triage_detects_multi_product_reference_and_routes_to_price():
    """When state has candidates AND user says 'as duas', triage shortcuts
    to price_inquiry without an LLM call."""
    from unittest.mock import AsyncMock, patch

    from app.agent.nodes.triage import triage_node

    candidates = [
        _bling_row("Raquete Gaivota A"),
        _bling_row("Raquete Gaivota B"),
    ]
    state = _state(
        "qual o preço das duas?",
        last_product_candidates=candidates,
    )

    with patch(
        "app.adapters.openai_client.OpenAIClient.chat",
        new_callable=AsyncMock,
    ) as llm:
        result = await triage_node(state)

    llm.assert_not_called()  # short-circuit: no LLM hit
    assert result["intent"] == "price_inquiry"


@pytest.mark.asyncio
async def test_triage_ignores_multi_ref_when_no_candidates():
    """Without last_product_candidates, multi-ref keyword goes through LLM."""
    import json
    from unittest.mock import AsyncMock, patch

    from app.agent.nodes.triage import triage_node

    state = _state("qual o preço das duas?", last_product_candidates=None)

    with patch(
        "app.adapters.openai_client.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value=json.dumps({"intent": "price_inquiry"}),
    ) as llm:
        result = await triage_node(state)

    llm.assert_awaited_once()
    assert result["intent"] == "price_inquiry"
