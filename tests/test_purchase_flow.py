"""Sprint 2.4 — purchase flow + brand greeting test suite.

Covers:
- ``product_selection`` emits a short pickup invite for existing products,
  with no handoff and no dossier.
- 4 randomized pickup variations; each ends with the "porta aberta" phrase.
- REFERENCE-NÃO determined now asks about alternatives + sets the
  ``awaiting_alternatives_decision`` flag; triage flips to exploring on
  "sim" and emits a canned goodbye on "não".
- STOCK pitch removed from the REFERENCE-SIM determined branch (regression
  ensures the post-stock follow-up turns still emit pitch for PRICE /
  technical questions).
- Brand-aware greeting on the first interaction.
- Regression: other handoff types (user_requested / scheduling /
  out_of_scope) still drive the dossier pipeline.
"""
from contextlib import asynccontextmanager
import json
import random
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.product_selection import (
    _PICKUP_VARIATIONS,
    get_pickup_message,
    product_selection_node,
)
from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    s = MagicMock()
    s.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    s.commit = AsyncMock()
    yield s


def _product(name: str = "Raquete BeachPro Carbon X5", price: int = 89900) -> dict:
    return {
        "id": f"id-{name}", "name": name, "sport": "beach_tennis",
        "level": "intermediário", "price_cents": price, "stock": 5,
        "description": f"desc {name}", "similarity": 0.9,
        "external_id": name.replace(" ", "-"), "url": None, "image_url": None,
        "updated_at": None, "is_active": True, "weight_g": 350,
        "balance": "médio", "material": "carbono", "category": "raquete",
    }


def _post_rec(**overrides) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quero comprar")],
        "phone_hash": "purchase4" * 7,
        "intent": "product_selection",
        "player_profile": {"nivel_jogo": "intermediário"},
        "recommended_products": [_product()],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "customer_name": "Marcelo",
        "customer_intent_path": "determined",
        "last_recommendation_at": datetime.now(timezone.utc).isoformat(),
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ════════════════════════════════════════════════════════════════════════════
# PURCHASE EXISTING (pickup invite, no handoff, no dossier)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_purchase_existing_product_invites_to_store_short():
    result = await product_selection_node(_post_rec())
    invite = result["response_blocks"][0]
    # Short — single block, fewer than ~300 chars.
    assert len(result["response_blocks"]) == 1
    assert len(invite) < 300
    assert "Raquete BeachPro Carbon X5" in invite


@pytest.mark.asyncio
async def test_purchase_existing_product_does_NOT_trigger_handoff():
    result = await product_selection_node(_post_rec())
    assert result.get("needs_handoff") is not True
    assert result.get("handoff_reason") is None


@pytest.mark.asyncio
async def test_purchase_existing_product_does_NOT_send_dossier():
    """No persist_dossier / send_dossier_to_recipient / summarize calls."""
    with patch(
        "app.agent.dossier.persist_dossier", new_callable=AsyncMock,
    ) as persist, patch(
        "app.agent.dossier.send_dossier_to_recipient", new_callable=AsyncMock,
    ) as send, patch(
        "app.agent.dossier.summarize_conversation", new_callable=AsyncMock,
    ) as summ:
        await product_selection_node(_post_rec())
    persist.assert_not_called()
    send.assert_not_called()
    summ.assert_not_called()


def test_purchase_existing_uses_one_of_4_variations():
    seen: set[str] = set()
    for _ in range(50):
        seen.add(get_pickup_message("Raquete Test"))
    # We can't force exact randomness in 50 draws, but each call must come
    # from the 4 templates (each contains its first-word marker).
    starts = ("Show!", "Bora!", "Demais!", "Fechou!")
    for msg in seen:
        assert msg.startswith(starts), f"unexpected variation start: {msg!r}"
    # And with 50 random draws we expect to see at least 2 distinct ones.
    assert len(seen) >= 2


def test_purchase_existing_includes_product_name_in_bold():
    msg = get_pickup_message("Raquete Carbon X5")
    assert "*Raquete Carbon X5*" in msg


def test_purchase_existing_ends_with_door_open_phrase():
    """Every variation MUST end with the porta-aberta phrase."""
    for template in _PICKUP_VARIATIONS:
        rendered = template.format(nome="Raquete X")
        assert "qualquer dúvida" in rendered.lower() or "qualquer coisa" in rendered.lower()


# ════════════════════════════════════════════════════════════════════════════
# PURCHASE NON-EXISTING (offers alternatives, yes/no follow-up)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.skip(reason="awaiting_alternatives flow removed in Sprint 2.6")

@pytest.mark.asyncio
async def test_purchase_nonexistent_offers_alternative_diagnose():
    from app.agent.nodes.recommend import recommend_node

    state = _post_rec(
        messages=[HumanMessage(content="quero comprar a Wilson Pro Staff")],
        intent="recommend",
        recommended_products=[],
        last_recommendation_at=None,
        player_profile={"modelo_desejado": "Wilson Pro Staff"},
        customer_intent_path="determined",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock):
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []  # nothing matches → REFERENCE-NÃO
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    reply = result["response_blocks"][0]
    assert "Wilson Pro Staff" in reply
    assert "outras opções" in reply or "outras opcoes" in reply
    assert result.get("awaiting_alternatives_decision") is True
@pytest.mark.skip(reason="exploring path removed in Sprint 2.6")


@pytest.mark.asyncio
async def test_purchase_nonexistent_user_accepts_enters_exploring_path():
    """Triage flips determined → exploring on a "sim" reply after the offer."""
    from app.agent.nodes.triage import triage_node

    state = _post_rec(
        messages=[HumanMessage(content="sim, pode")],
        intent="diagnose",  # any classical intent the LLM might emit
        awaiting_alternatives_decision=True,
        recommended_products=[],
        player_profile={"modelo_desejado": "Wilson Pro Staff"},
    )
    result = await triage_node(state)
    assert result["customer_intent_path"] == "exploring"
    assert result["intent"] == "bare_recommendation_request"
    assert result["player_profile"].get("modelo_desejado") == "nenhum"
    assert result.get("awaiting_alternatives_decision") is False
@pytest.mark.skip(reason="awaiting_alternatives flow removed in Sprint 2.6")


@pytest.mark.asyncio
async def test_purchase_nonexistent_user_declines_ends_gracefully():
    """Triage routes to smalltalk with ``goodbye_pending`` for a "não" reply."""
    from app.agent.graph import _smalltalk_node
    from app.agent.nodes.triage import triage_node

    state = _post_rec(
        messages=[HumanMessage(content="não obrigado")],
        intent="diagnose",
        awaiting_alternatives_decision=True,
    )
    triage_result = await triage_node(state)
    assert triage_result["intent"] == "smalltalk"
    assert triage_result.get("goodbye_pending") is True
    assert triage_result.get("awaiting_alternatives_decision") is False

    # Now the smalltalk node should emit the canned goodbye.
    state_with_flag = {**state, **triage_result}
    sm_result = await _smalltalk_node(state_with_flag)
    assert "mudar de ideia" in sm_result["response_blocks"][0].lower()
    assert sm_result.get("goodbye_pending") is False


# ════════════════════════════════════════════════════════════════════════════
# PITCH TIMING
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_stock_confirmation_no_longer_emits_pitch():
    from app.agent.nodes.recommend import recommend_node

    state = _post_rec(
        messages=[HumanMessage(content="vocês têm a Raquete X?")],
        intent="recommend",
        recommended_products=[],
        last_recommendation_at=None,
        player_profile={"modelo_desejado": "Raquete X"},
        customer_intent_path="determined",
        determined_question_count=0,
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock):
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = [_product("Raquete X")]
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    blocks = result["response_blocks"]
    assert len(blocks) == 1
    assert "Consultoria" not in blocks[0]


@pytest.mark.asyncio
async def test_pitch_still_works_for_price_inquiry():
    """After confirming stock, a PRICE question still triggers the immediate pitch."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _post_rec(
        messages=[HumanMessage(content="quanto custa?")],
        intent="price_inquiry",
        recommended_products=[_product("Raquete X", price=80000)],
        determined_question_count=0,
        consultoria_mentioned_count=0,
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    assert "investimento numa raquete" in full.lower()  # PRICE preset opener
    assert result.get("consultoria_mentioned_count") == 1


@pytest.mark.asyncio
async def test_pitch_still_works_for_product_detail_second_question():
    """DELAYED type (WEIGHT) still fires on the 2nd determined question."""
    from app.agent.nodes.product_detail import product_detail_node

    state = _post_rec(
        messages=[HumanMessage(content="qual o peso?")],
        intent="product_detail",
        recommended_products=[_product("Raquete X")],
        determined_question_count=1,  # 2nd determined question
        consultoria_mentioned_count=0,
    )
    result = await product_detail_node(state)
    full = " ".join(result["response_blocks"])
    assert "Consultoria Base Sports" in full
    assert result.get("consultoria_mentioned_count") == 1


@pytest.mark.asyncio
async def test_direct_purchase_after_stock_emits_no_pitch():
    """Stock → 'quero comprar' (product_selection) → no pitch anywhere."""
    state = _post_rec(
        messages=[HumanMessage(content="quero comprar")],
        determined_question_count=0,
        consultoria_mentioned_count=0,
    )
    result = await product_selection_node(state)
    full = " ".join(result["response_blocks"])
    assert "Consultoria" not in full


# ════════════════════════════════════════════════════════════════════════════
# BRAND GREETING
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_welcome_message_mentions_base_sports():
    from app.agent.graph import _smalltalk_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="oi")],
        "phone_hash": "brand123" * 8,
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        result = await _smalltalk_node(state)
    # No LLM call — canned brand greeting.
    llm.assert_not_called()
    text = result["response_blocks"][0]
    assert "Base Sports" in text
    assert "Bem-vindo" in text


@pytest.mark.asyncio
async def test_welcome_message_format_emoji():
    from app.agent.graph import _smalltalk_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="olá")],
        "phone_hash": "brand456" * 8,
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    result = await _smalltalk_node(state)
    text = result["response_blocks"][0]
    assert "👋" in text  # waving-hand emoji as per brand guidance


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION — other handoffs unaffected
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_user_requested_handoff_still_works():
    from app.agent.nodes.handoff import handoff_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quero falar com humano")],
        "phone_hash": "reqhand" * 9,
        "intent": "handoff",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    with patch(
        "app.agent.nodes.handoff.handoff_dossier_pipeline", new_callable=AsyncMock
    ) as pipeline:
        result = await handoff_node(state)

    assert result["needs_handoff"] is True
    assert result["handoff_reason"] == "user_requested"
    pipeline.assert_called_once()


@pytest.mark.asyncio
async def test_scheduling_handoff_still_works():
    from app.agent.nodes.scheduling_inquiry import scheduling_inquiry_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="como agendo?")],
        "phone_hash": "sched12" * 9,
        "intent": "scheduling_inquiry",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    with patch(
        "app.agent.nodes.scheduling_inquiry.handoff_dossier_pipeline",
        new_callable=AsyncMock,
    ) as pipeline:
        result = await scheduling_inquiry_node(state)

    assert result["needs_handoff"] is True
    assert result["handoff_reason"] == "scheduling"
    pipeline.assert_called_once()


@pytest.mark.asyncio
async def test_out_of_scope_handoff_still_works():
    from app.agent.nodes.out_of_scope_handoff import out_of_scope_handoff_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="vocês entregam em casa?")],
        "phone_hash": "oos1234" * 9,
        "intent": "out_of_scope",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    with patch(
        "app.agent.nodes.out_of_scope_handoff.handoff_dossier_pipeline",
        new_callable=AsyncMock,
    ) as pipeline:
        result = await out_of_scope_handoff_node(state)

    assert result["needs_handoff"] is True
    assert result["handoff_reason"] == "out_of_scope"
    pipeline.assert_called_once()
@pytest.mark.skip(reason="exploring path removed in Sprint 2.6")


@pytest.mark.asyncio
async def test_exploring_path_still_offers_consultoria_at_end():
    """Regression: exploring → diagnose → recommend (PROFILE) → consultoria_offer."""
    from app.agent.nodes.recommend import recommend_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quero indicação")],
        "phone_hash": "explor1" * 9,
        "intent": "recommend",
        "player_profile": {
            "nivel_jogo": "intermediário",
            "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma",
            "esporte_raquete_previo": "nao_aplicavel",
            "modelo_desejado": "nenhum",
        },
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "customer_intent_path": "exploring",
    }
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": [
            "Pelo seu perfil…",
            "*Consultoria Base Sports* — R$ 350, 100% abatido.",
            "Quer saber como funciona?",
        ]})
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock):
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    full = " ".join(result["response_blocks"])
    assert "Consultoria Base Sports" in full
    assert result.get("consultoria_interest") is True
