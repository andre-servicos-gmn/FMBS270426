"""Sprint 2.6.1 — recommend_node closing-question tone.

The pre-2.6.1 confirmation ended with "ou já quer fechar?" which felt
pushy in WhatsApp for high-ticket items. The replacement keeps the
consultive tone: offer details OR an in-store visit. Vocabulary like
"fechar", "comprar", "pedido", "concluir" is reserved for explicit
purchase intent — these two tests pin the new shape.

The legacy Sprint 2.0 file body (REFERENCE-SIM/NÃO assertions) was
removed when diagnose was deprecated in Sprint 2.6.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    s = MagicMock()
    s.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    s.commit = AsyncMock()
    yield s


def _bling_match(name: str = "Raquete BeachPro Carbon X5") -> list[dict]:
    return [{
        "id": 1, "name": name, "price_cents": 179900,
        "is_raquete_praia": True, "description": "",
        "marca": "BeachPro", "modelo": "Carbon X5",
        "categoria_nome": "Raquetes de Praia",
        "external_id": "1", "weight_g": 350,
    }]


def _state(query: str = "vocês têm a BeachPro Carbon X5?") -> AgentState:
    return AgentState(  # type: ignore[typeddict-item]
        messages=[HumanMessage(content=query)],
        phone_hash="recnode2" * 8,
        intent="product_inquiry",
        player_profile={},
        recommended_products=[],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
    )


@pytest.mark.asyncio
async def test_recommend_closing_question_does_not_mention_fechar(monkeypatch):
    """The match-found confirmation must NOT use purchase vocabulary."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=_bling_match(),
    ):
        result = await recommend_node(_state())

    reply = result["response_blocks"][0].lower()
    assert "fechar" not in reply, f"reply must not say 'fechar': {reply!r}"
    assert "comprar" not in reply, f"reply must not say 'comprar': {reply!r}"
    assert "pedido" not in reply, f"reply must not say 'pedido': {reply!r}"
    assert "concluir" not in reply, f"reply must not say 'concluir': {reply!r}"


@pytest.mark.asyncio
async def test_recommend_closing_question_offers_details_and_store(monkeypatch):
    """The consultive replacement must offer details AND mention the loja."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=_bling_match(),
    ):
        result = await recommend_node(_state())

    reply = result["response_blocks"][0].lower()
    assert "detalhes" in reply, f"reply must mention 'detalhes': {reply!r}"
    assert "loja" in reply, f"reply must mention 'loja': {reply!r}"


# ── Sprint 2.6.2 — typo suggestion + confirmation flow ──────────────────────

@pytest.mark.asyncio
async def test_recommend_offers_typo_suggestion(monkeypatch):
    """Levenshtein distance ~2 → 'Você quis dizer X?' + flag set."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    # Query has a heavier typo: "mormai sunsex" vs "Mormaii Sunset".
    # Sprint 2.6.3 — catalog comes via get_catalog_snapshot now.
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=[_bling_match("Raquete Mormaii Sunset")[0]],
    ):
        result = await recommend_node(_state("mormai sunsex"))

    reply = result["response_blocks"][0]
    if "Você quis dizer" in reply:
        # Triggered the fuzzy_low suggestion path.
        assert "Mormaii Sunset" in reply
        assert result.get("awaiting_match_confirmation") is not None
        # The "not yet confirmed" flow does NOT pre-populate recommended_products.
        assert not result.get("recommended_products")


@pytest.mark.asyncio
async def test_recommend_handles_match_confirmation_yes(monkeypatch):
    """When state has awaiting_match_confirmation, recommend promotes it."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    candidate = _bling_match("Raquete Emit Hammer")[0]
    state = _state("sim, é essa mesma")
    state["awaiting_match_confirmation"] = candidate  # type: ignore[typeddict-unknown-key]

    result = await recommend_node(state)
    reply = result["response_blocks"][0]
    assert "Raquete Emit Hammer" in reply
    assert "Posso te passar mais detalhes" in reply
    assert result["recommended_products"][0]["name"] == "Raquete Emit Hammer"
    assert result.get("awaiting_match_confirmation") is None


@pytest.mark.asyncio
async def test_triage_routes_match_confirmation_yes_to_product_inquiry():
    """Triage sees the yes reply + flag → routes to product_inquiry (recommend)."""
    from langchain_core.messages import HumanMessage
    from app.agent.nodes.triage import triage_node

    candidate = {"id": 1, "name": "Raquete Emit Hammer", "price_cents": 100000}
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="sim")],
        "phone_hash": "confyes" * 9,
        "intent": None,
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "awaiting_match_confirmation": candidate,
    }

    result = await triage_node(state)
    assert result["intent"] == "product_inquiry"


@pytest.mark.asyncio
async def test_triage_routes_match_confirmation_no_to_smalltalk():
    """Triage sees the no reply + flag → smalltalk with match_decline_pending."""
    from langchain_core.messages import HumanMessage
    from app.agent.nodes.triage import triage_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="não")],
        "phone_hash": "confno" * 10,
        "intent": None,
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "awaiting_match_confirmation": {"name": "Foo"},
    }

    result = await triage_node(state)
    assert result["intent"] == "smalltalk"
    assert result.get("match_decline_pending") is True
    assert result.get("awaiting_match_confirmation") is None


@pytest.mark.asyncio
async def test_smalltalk_emits_canned_reply_on_match_decline():
    """Smalltalk node handles match_decline_pending with a canned reply."""
    from langchain_core.messages import HumanMessage
    from app.agent.graph import _smalltalk_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="não")],
        "phone_hash": "decline" * 9,
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "match_decline_pending": True,
        "customer_name": "Andre",
    }

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        result = await _smalltalk_node(state)

    llm.assert_not_called()  # canned reply, no LLM call
    reply = result["response_blocks"][0]
    assert "Sem problemas" in reply
    assert result.get("match_decline_pending") is False


# ── Sprint 2.6.4 — neutral "produto" wording in not-found path ──────────────

@pytest.mark.asyncio
async def test_recommend_not_found_uses_neutral_word_produto(monkeypatch):
    """The agent must say 'produto', not 'raquete', when nothing matched.

    A customer asking about a sock or manguito should NOT see "essa
    raquete" — that breaks the illusion of a real attendant.
    """
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=[],   # empty catalog → guaranteed not-found
    ):
        result = await recommend_node(_state("vocês têm meias?"))

    reply = result["response_blocks"][0].lower()
    assert "essa raquete" not in reply
    assert "outra" not in reply or "outro" in reply
    assert "produto" in reply or "item" in reply


@pytest.mark.asyncio
async def test_recommend_not_found_for_clothing_works(monkeypatch):
    """Customer asked about a clothing item not in catalog → neutral reply."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    # Catalog with non-clothing items only → match returns none.
    catalog = [
        {
            "id": 1, "name": "Raquete BeachPro Carbon X5",
            "price_cents": 89900, "is_raquete_praia": True,
            "description": "", "external_id": "1",
        },
    ]
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock, return_value=catalog,
    ):
        result = await recommend_node(_state("tem boné azul?"))

    reply = result["response_blocks"][0].lower()
    assert "essa raquete" not in reply
    # Either a neutral "não encontrei esse produto" OR a top-3 fallback that
    # at least doesn't claim it's a raquete.
    assert (
        "produto" in reply
        or "item" in reply
        or "modelos parecidos" in reply
    )
