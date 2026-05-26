"""Sprint 2.0 — recommend node tests.

Pre-2.0 this file had ~22 tests asserting alternatives lists, fallback
blocks and an alternatives-injection guardrail. The strategic pivot
removes all of those code paths, so this file is now scoped to:

- Pure helpers (``_has_model_reference``, ``_find_name_match``).
- Mode detection on the user-message context handed to the LLM
  (REFERENCE-SIM, REFERENCE-NÃO, PROFILE delegation).
- Prompt-body invariants (``REGRA SUPREMA``, 3 modes).

The full SIM/NÃO/PROFILE behavioural surface is covered by
``tests/test_pivot_no_recommendation.py``.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.recommend import (
    _find_name_match,
    _has_model_reference,
    recommend_node,
)
from app.agent.state import AgentState


# ── Fixtures and helpers ─────────────────────────────────────────────────────

@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _product(name: str, *, level: str = "intermediário", price: int = 70000) -> dict:
    return {
        "id": f"id-{name}",
        "name": name,
        "sport": "beach_tennis",
        "level": level,
        "price_cents": price,
        "stock": 5,
        "description": f"Descrição de {name}",
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


def _state(*, modelo: str | None = None, lesion: str | None = None) -> AgentState:
    profile: dict = {
        "nivel_jogo": "intermediário",
        "lesoes": "nenhuma" if not lesion else lesion,
        "regiao_lesao": "nenhuma" if not lesion else lesion,
        "esporte_raquete_previo": "nao_aplicavel",
    }
    if modelo is not None:
        profile["modelo_desejado"] = modelo
    return AgentState(
        messages=[HumanMessage(content="me indica uma raquete")],
        phone_hash="recsprint" * 8,
        intent="recommend",
        player_profile=profile,
        recommended_products=[],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
    )


def _last_user_context(mock) -> str:
    last_call = mock.call_args_list[-1]
    user_messages = [m for m in last_call.kwargs["messages"] if m["role"] == "user"]
    return user_messages[-1]["content"] if user_messages else ""


_DUMMY_REC = json.dumps({"messages": ["bloco 1", "bloco 2"]})


# ── Unit tests on the routing helpers ────────────────────────────────────────

def test_has_model_reference_detects_named_model():
    assert _has_model_reference({"modelo_desejado": "BeachPro X5"}) is True
    assert _has_model_reference({"modelo_desejado": "Wilson Pro Staff"}) is True


@pytest.mark.parametrize(
    "value",
    ["", "nenhum", "Nenhum", "NENHUM", "nenhuma", "  nenhum  ", "não", "nao"],
)
def test_has_model_reference_treats_no_preference_values_as_false(value):
    assert _has_model_reference({"modelo_desejado": value}) is False


def test_find_name_match_returns_product_when_name_overlaps():
    products = [
        _product("Raquete BeachPro Carbon X5"),
        _product("Raquete BeachPro Foam Series 300"),
        _product("Wilson Burn"),
    ]
    match = _find_name_match(products, "BeachPro Carbon X5")
    assert match is not None
    assert match["name"] == "Raquete BeachPro Carbon X5"


def test_find_name_match_returns_none_when_no_overlap():
    products = [
        _product("Raquete BeachPro Carbon X5"),
        _product("Raquete BeachPro Foam Series 300"),
    ]
    assert _find_name_match(products, "Wilson Pro Staff") is None


def test_find_name_match_is_accent_and_case_insensitive():
    products = [_product("Raquete Avançada Pro")]
    assert _find_name_match(products, "avancada pro") is not None


# ── Reference mode — context block signaling ─────────────────────────────────

@pytest.mark.asyncio
async def test_reference_sim_context_declares_have_stock():
    products = [_product("Raquete BeachPro Carbon X5")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = _DUMMY_REC
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = products
            with patch("app.storage.db.get_session", _mock_db_session):
                await recommend_node(_state(modelo="BeachPro Carbon X5"))

    ctx = _last_user_context(llm)
    assert "Modo: REFERENCE-SIM" in ctx
    assert "BeachPro Carbon X5" in ctx
    assert "TEMOS NO ESTOQUE" in ctx


@pytest.mark.asyncio
async def test_reference_nao_context_declares_missing_from_catalog():
    products = [
        _product("Raquete BeachPro Carbon X5"),
        _product("Raquete BeachPro Foam Series 300"),
    ]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = _DUMMY_REC
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = products
            with patch("app.storage.db.get_session", _mock_db_session):
                await recommend_node(_state(modelo="Wilson Pro Staff"))

    ctx = _last_user_context(llm)
    assert "Modo: REFERENCE-NÃO" in ctx
    assert "NÃO TEMOS NO CATÁLOGO" in ctx
    assert "Wilson Pro Staff" in ctx


@pytest.mark.asyncio
async def test_reference_sim_shortlist_is_only_the_matched_product():
    """SIM mode never offers alternatives."""
    products = [
        _product("Raquete BeachPro Carbon X5"),
        _product("Raquete BeachPro Foam Series 300"),
        _product("Raquete C"),
    ]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = _DUMMY_REC
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = products
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(_state(modelo="BeachPro Carbon X5"))

    assert [p["name"] for p in result["recommended_products"]] == ["Raquete BeachPro Carbon X5"]


@pytest.mark.asyncio
async def test_reference_nao_clears_shortlist_and_flags_consultoria_interest():
    products = [_product("Raquete A"), _product("Raquete B")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = _DUMMY_REC
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = products
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(_state(modelo="Inexistente"))

    assert result["recommended_products"] == []
    assert result["consultoria_interest"] is True
    assert result["produto_pesquisado"] == "Inexistente"


# ── Profile mode → consultoria_offer delegation ─────────────────────────────

@pytest.mark.asyncio
async def test_profile_mode_does_not_call_search_products():
    """No modelo_desejado → recommend delegates to consultoria_offer (no retriever)."""
    state = _state(modelo=None)
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = _DUMMY_REC
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    search.assert_not_called()
    assert result["consultoria_interest"] is True


@pytest.mark.asyncio
async def test_profile_mode_treats_nenhum_as_no_reference():
    state = _state(modelo="nenhum")
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = _DUMMY_REC
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    search.assert_not_called()
    assert result["consultoria_interest"] is True


# ── Prompt-body invariants ──────────────────────────────────────────────────

def test_system_recommend_has_supreme_rule_and_three_modes():
    """The Sprint 2.0 prompt must declare the qualifier guard + all 3 modes."""
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND
    assert "REGRA SUPREMA" in s
    assert "REFERENCE-SIM" in s
    assert "REFERENCE-NÃO" in s
    assert "MODO PROFILE" in s or "PROFILE" in s


def test_system_recommend_prohibits_listing_alternatives_in_reference_nao():
    """The REFERENCE-NÃO mode body forbids alternative-listing."""
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND.lower()
    # The mode body says NÃO sugerir alternativas concretas.
    assert "não sugerir" in s or "proibido listar" in s
