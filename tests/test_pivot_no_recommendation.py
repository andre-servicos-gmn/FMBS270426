"""Sprint 2.0 — strategic pivot test suite.

These tests verify the qualifier-mode behaviour of the agent:

1. **Name capture**: smalltalk asks for the name once, persists it, uses it
   sparingly on subsequent turns.
2. **bare_recommendation_request** intent: routes through diagnose, ends in
   consultoria_offer (never in active recommendation).
3. **REFERENCE-SIM**: when the customer names a racket that EXISTS, the node
   confirms stock and asks for the next step; NO alternatives listed.
4. **REFERENCE-NÃO**: when the racket does NOT exist, the node briefly says
   so and offers the Consultoria; NO alternatives listed.
5. **PROFILE**: when the diagnose ends without a model, recommend delegates
   to consultoria_offer (never names a racket).
6. **Dossier**: build, render, persist on handoff.

All external I/O (OpenAI, DB, retriever) is mocked.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.dossier import build_dossier, format_dossier_for_whatsapp
from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _product(name: str, *, price_cents: int = 70000) -> dict:
    return {
        "id": f"id-{name}",
        "name": name,
        "sport": "beach_tennis",
        "level": "intermediário",
        "price_cents": price_cents,
        "stock": 5,
        "description": f"desc {name}",
        "similarity": 0.9,
        "external_id": name.replace(" ", "-"),
        "url": None,
        "image_url": None,
        "updated_at": None,
        "is_active": True,
        "weight_g": 350,
        "balance": "médio",
        "material": "carbono",
        "category": "raquete",
    }


def _profile_state(
    *, modelo: str = "nenhum", customer_name: str | None = None
) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="me indica uma raquete")],
        "phone_hash": "pivot20" * 9,
        "intent": "recommend",
        "player_profile": {
            "nivel_jogo": "intermediário",
            "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma",
            "esporte_raquete_previo": "nao_aplicavel",
            "modelo_desejado": modelo,
        },
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    if customer_name:
        state["customer_name"] = customer_name
    return state


# ════════════════════════════════════════════════════════════════════════════
# 1. NAME CAPTURE
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_smalltalk_asks_name_on_first_interaction():
    """Phase 2: first 'oi' with no name → SYSTEM_NAME_ASK reply, name_asked=True."""
    from app.agent.graph import _smalltalk_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="oi")],
        "phone_hash": "nametest" * 8,
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        result = await _smalltalk_node(state)

    assert result.get("name_asked") is True
    text = result["response_blocks"][0]
    assert "nome" in text.lower()
    # Sprint 2.4 — canned brand greeting, no LLM call on first ask.
    assert "Base Sports" in text
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_smalltalk_extracts_name_when_asked_last_turn():
    """Phase 1: name_asked=True, customer responds with a name → captured."""
    from app.agent.graph import _smalltalk_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="Andre")],
        "phone_hash": "nameext" * 8,
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "name_asked": True,
    }

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"extracted_name": "Andre"}',  # NAME_EXTRACT
            "Show, Andre! Em que posso te ajudar?",  # normal smalltalk reply
        ]
        result = await _smalltalk_node(state)

    assert result.get("customer_name") == "Andre"
    assert result.get("name_asked") is False
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_smalltalk_skips_name_ask_when_already_captured():
    """Phase 3: customer_name present → straight to normal smalltalk reply."""
    from app.agent.graph import _smalltalk_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="oi de novo")],
        "phone_hash": "namepres" * 8,
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "customer_name": "Maria",
    }

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = "Oi, Maria! Que bom te ver de novo."
        result = await _smalltalk_node(state)

    assert result.get("name_asked") is None or result.get("name_asked") is False
    assert llm.call_count == 1
    # The user block sent to the LLM must carry the name context.
    user_msg = llm.call_args.kwargs["messages"][-1]["content"]
    assert "Maria" in user_msg


# ════════════════════════════════════════════════════════════════════════════
# 2. BARE RECOMMENDATION REQUEST INTENT
# ════════════════════════════════════════════════════════════════════════════

def test_bare_recommendation_request_in_triage_intent_set():
    """The new intent must be in _VALID_INTENTS so triage doesn't demote it."""
    from app.agent.nodes.triage import _VALID_INTENTS
    assert "bare_recommendation_request" in _VALID_INTENTS


def test_triage_prompt_lists_bare_recommendation_request():
    """The prompt must teach the LLM the new category."""
    from app.agent.prompts import SYSTEM_TRIAGE
    assert "bare_recommendation_request" in SYSTEM_TRIAGE


def test_triage_router_remaps_bare_recommendation_to_recommend():
    """The router transforms bare_recommendation_request → recommend path."""
    from app.agent.graph import _triage_router

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="qual você indica?")],
        "phone_hash": "bareroute" * 7,
        "intent": "bare_recommendation_request",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    # Classical state (no products) → "recommend" → mapped to "diagnose" in edges.
    assert _triage_router(state) == "recommend"


# ════════════════════════════════════════════════════════════════════════════
# 3. REFERENCE-SIM — racket exists
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_reference_sim_only_keeps_matched_product_in_shortlist():
    """REFERENCE-SIM never lists alternatives — only the matched racket survives."""
    from app.agent.nodes.recommend import recommend_node

    candidates = [
        _product("Raquete BeachPro Carbon X5"),  # the matched one
        _product("Raquete A"),
        _product("Raquete B"),
    ]
    state = _profile_state(modelo="BeachPro Carbon X5")

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({
            "messages": [
                "Sim, temos a *Raquete BeachPro Carbon X5* aqui!",
                "Quer saber preço, peso, indicação, ou já fechamos?",
            ]
        })
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    shortlist = result["recommended_products"]
    assert len(shortlist) == 1
    assert shortlist[0]["name"] == "Raquete BeachPro Carbon X5"
    # produto_pesquisado is recorded for dossier rendering.
    assert result["produto_pesquisado"] == "BeachPro Carbon X5"


@pytest.mark.asyncio
async def test_reference_sim_context_signals_have_stock():
    """The user-message context must declare 'Modo: REFERENCE-SIM' + 'TEMOS NO ESTOQUE'."""
    from app.agent.nodes.recommend import recommend_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _profile_state(modelo="BeachPro Carbon X5")

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": ["Sim, temos!"]})
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                await recommend_node(state)

    user_ctx = llm.call_args.kwargs["messages"][-1]["content"]
    assert "Modo: REFERENCE-SIM" in user_ctx
    assert "TEMOS NO ESTOQUE" in user_ctx


# ════════════════════════════════════════════════════════════════════════════
# 4. REFERENCE-NÃO — racket missing from catalog
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_reference_nao_does_not_keep_alternatives_in_shortlist():
    """REFERENCE-NÃO clears recommended_products — no alternatives offered."""
    from app.agent.nodes.recommend import recommend_node

    # Retriever returns other products, but none match Wilson Pro Staff.
    candidates = [_product("Raquete A"), _product("Raquete B")]
    state = _profile_state(modelo="Wilson Pro Staff")

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": [
            "A *Wilson Pro Staff* específica a gente não tem.",
            "Pra encontrar a raquete certa, oferecemos a *Consultoria Base Sports* (R$350).",
            "Quer saber como funciona?",
        ]})
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    assert result["recommended_products"] == []
    assert result["consultoria_interest"] is True
    assert result["produto_pesquisado"] == "Wilson Pro Staff"


@pytest.mark.asyncio
async def test_reference_nao_context_signals_missing_and_has_consultoria_price():
    """REFERENCE-NÃO context tells the LLM: not in catalog + consultoria price."""
    from app.agent.nodes.recommend import recommend_node

    state = _profile_state(modelo="Babolat Pure")
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": ["nao tem", "consultoria"]})
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []  # nothing matches
            with patch("app.storage.db.get_session", _mock_db_session):
                await recommend_node(state)

    user_ctx = llm.call_args.kwargs["messages"][-1]["content"]
    assert "Modo: REFERENCE-NÃO" in user_ctx
    assert "NÃO TEMOS NO CATÁLOGO" in user_ctx
    assert "R$" in user_ctx  # the consultoria investment line


# ════════════════════════════════════════════════════════════════════════════
# 5. PROFILE — no model → consultoria_offer
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_profile_mode_delegates_to_consultoria_offer():
    """No modelo_desejado → recommend never lists rackets; it offers the Consultoria."""
    from app.agent.nodes.recommend import recommend_node

    state = _profile_state(modelo="nenhum")
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": [
            "Pelo seu perfil, vale fazermos isso com calma.",
            "Temos a *Consultoria Base Sports* (*R$350*, 100% abatido).",
            "Quer saber como funciona ou já agendar?",
        ]})
        # Retriever should NOT be called in PROFILE mode.
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    search.assert_not_called()
    assert result["consultoria_interest"] is True
    full = " ".join(result["response_blocks"])
    # The pitch must carry the price + abatimento signal.
    assert "350" in full
    assert "abatido" in full.lower() or "abate" in full.lower()


@pytest.mark.asyncio
async def test_profile_mode_uses_customer_name_when_available():
    """When customer_name is set, the consultoria_offer user context carries it."""
    from app.agent.nodes.recommend import recommend_node

    state = _profile_state(modelo="nenhum", customer_name="Andre")
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": ["Andre, pelo seu perfil…", "consultoria", "agendar?"]})
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock):
            with patch("app.storage.db.get_session", _mock_db_session):
                await recommend_node(state)

    user_ctx = llm.call_args.kwargs["messages"][-1]["content"]
    assert "Andre" in user_ctx


@pytest.mark.asyncio
async def test_re_recommendation_pivots_to_consultoria():
    """Sprint 2.0 — re_recommendation no longer fetches another shortlist."""
    from app.agent.nodes.re_recommendation import re_recommendation_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="tem alguma mais barata?")],
        "phone_hash": "reretest" * 8,
        "intent": "re_recommendation",
        "player_profile": {
            "nivel_jogo": "intermediário",
            "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma",
            "esporte_raquete_previo": "nao_aplicavel",
            "modelo_desejado": "nenhum",
        },
        "recommended_products": [_product("Raquete X")],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "last_recommendation_at": "2026-01-01T00:00:00+00:00",
    }
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": ["pivot", "consultoria 350", "agendar?"]})
        # The retriever must NOT be called in the new flow.
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await re_recommendation_node(state)

    search.assert_not_called()
    # Shortlist is cleared so the next turn isn't routed as post-rec.
    assert result.get("recommended_products") == []
    assert result.get("last_recommendation_at") is None
    assert result.get("consultoria_interest") is True


# ════════════════════════════════════════════════════════════════════════════
# 6. DOSSIER
# ════════════════════════════════════════════════════════════════════════════

def test_build_dossier_includes_all_essential_fields():
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="oi")],
        "phone_hash": "doss" + "x" * 60,
        "intent": "handoff",
        "player_profile": {
            "nivel_jogo": "intermediário",
            "lesoes": "tendinite",
            "regiao_lesao": "cotovelo",
            "esporte_raquete_previo": "tênis",
            "modelo_desejado": "Carbon X5",
        },
        "recommended_products": [],
        "needs_handoff": True,
        "handoff_reason": "user_requested",
        "consultoria_interest": True,
        "customer_name": "Andre",
        "produto_pesquisado": "Carbon X5",
    }
    dossier = build_dossier(state)

    assert dossier["nome"] == "Andre"
    assert dossier["telefone_hash"].startswith("doss")
    assert dossier["nivel"] == "intermediário"
    assert dossier["lesoes"] == "tendinite"
    assert dossier["regiao_lesao"] == "cotovelo"
    assert dossier["esporte_raquete_previo"] == "tênis"
    assert dossier["modelo_desejado"] == "Carbon X5"
    assert dossier["produto_pesquisado"] == "Carbon X5"
    assert dossier["consultoria_interesse"] is True
    assert dossier["needs_handoff_reason"] == "user_requested"
    assert dossier["timestamp"]


def test_format_dossier_for_whatsapp_renders_visible_headings():
    dossier = {
        "nome": "Andre",
        "telefone_hash": "abc12345def67890",
        "nivel": "iniciante",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "nenhum",
        "modelo_desejado": "nenhum",
        "produto_pesquisado": None,
        "consultoria_interesse": True,
        "needs_handoff_reason": "purchase_closing",
        "transcricao_resumo": "Conversa com 12 mensagens.",
        "timestamp": "2026-05-20T14:30:00+00:00",
    }
    text = format_dossier_for_whatsapp(dossier)
    assert "NOVO LEAD" in text
    assert "Andre" in text
    assert "iniciante" in text
    # Sprint 2.2 — handoff_reason is rendered as a PT-BR label, not raw.
    assert "Quer comprar raquete" in text
    # ISO timestamp is humanized.
    assert "20/05/2026" in text and "14:30" in text


@pytest.mark.asyncio
async def test_product_selection_invites_to_store_without_handoff():
    """Sprint 2.4 — product_selection now emits a short pickup invite and
    does NOT trigger handoff / persist dossier (cliente já sabe onde é a
    loja porque chegou pelo WhatsApp dela)."""
    from app.agent.nodes.product_selection import product_selection_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="vou de Carbon X5")],
        "phone_hash": "selpurch" * 8,
        "intent": "product_selection",
        "player_profile": {"nivel_jogo": "intermediário", "lesoes": "nenhuma"},
        "recommended_products": [_product("Raquete BeachPro Carbon X5")],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "customer_name": "Andre",
    }

    result = await product_selection_node(state)

    assert "needs_handoff" not in result or result["needs_handoff"] is False
    assert result.get("handoff_reason") is None
    assert result["selected_product"]["name"] == "Raquete BeachPro Carbon X5"
    invite = result["response_blocks"][0]
    assert "Raquete BeachPro Carbon X5" in invite
    # 4 random variations: 3 use "qualquer dúvida", 1 uses "qualquer coisa".
    assert "qualquer dúvida" in invite.lower() or "qualquer coisa" in invite.lower()


@pytest.mark.asyncio
async def test_handoff_node_persists_dossier():
    """Generic handoff routes through the Sprint 2.2 dossier pipeline."""
    from app.agent.nodes.handoff import handoff_node

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quero falar com humano")],
        "phone_hash": "handpers" * 8,
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
    # Pipeline receives the state with the right handoff_reason stamped.
    passed_state = pipeline.call_args.args[0]
    assert passed_state["handoff_reason"] == "user_requested"


# ════════════════════════════════════════════════════════════════════════════
# 7. REGRESSION — supreme rule + consultoria pitch parity
# ════════════════════════════════════════════════════════════════════════════

def test_system_recommend_prompt_carries_supreme_rule():
    """The prompt body must contain the Sprint 2.0 'REGRA SUPREMA' guard."""
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND
    assert "REGRA SUPREMA" in s
    assert "REFERENCE-SIM" in s and "REFERENCE-NÃO" in s and "PROFILE" in s


def test_system_pitch_consultoria_mentions_price_and_abatimento():
    """The pitch must explicitly tell the LLM about R$<preco> + abatimento."""
    from app.agent.prompts import SYSTEM_PITCH_CONSULTORIA_TEMPLATE
    s = SYSTEM_PITCH_CONSULTORIA_TEMPLATE
    assert "{consultoria_preco}" in s
    assert "100% abatido" in s


def test_consultoria_offer_prompt_carries_price_and_abatimento():
    """Sprint 2.0 consultoria_offer prompt mentions investment + abatimento."""
    from app.agent.prompts import build_consultoria_offer_prompt

    class _S:
        consultoria_preco = 350

    s = build_consultoria_offer_prompt(_S())
    assert "350" in s
    assert "abatido" in s
