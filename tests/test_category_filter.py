"""Sprint 1.11 — products.category filter end-to-end.

Five tests cover the deterministic filtering logic so the bug from Sprint
1.10 Cenário C (Kit Bolas returned as a "raquete mais barata") can never
silently regress.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.recommend import _build_filters, recommend_node
from app.agent.nodes.re_recommendation import re_recommendation_node
from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _product(name, *, price=70000, category="raquete", external_id=None) -> dict:
    return {
        "id": f"id-{name}",
        "name": name,
        "sport": "padel" if category == "pala" else "beach_tennis",
        "level": "intermediário",
        "price_cents": price,
        "stock": 5,
        "description": f"desc {name}",
        "similarity": 0.9,
        "external_id": external_id or name.replace(" ", "-"),
        "url": None,
        "image_url": None,
        "updated_at": None,
        "is_active": True,
        "weight_g": 350,
        "balance": "médio",
        "material": "carbono",
        "category": category,
    }


def _profile_state(
    *, sport: str = "beach tennis", products=None, modelo: str = "Carbon X5"
) -> AgentState:
    """Sprint 2.0: by default we pass modelo_desejado so the test exercises
    REFERENCE mode (which still calls search_products with the category
    filter). PROFILE mode now delegates to consultoria_offer and skips the
    retriever entirely, so it can't be used to assert filter wiring."""
    return AgentState(
        messages=[HumanMessage(content="me indica")],
        phone_hash="catfilter" * 7,
        intent="recommend",
        player_profile={
            "nivel_jogo": "intermediário",
            "esporte_praticado": sport,
            "esporte_raquete_previo": "nao_aplicavel",
            "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma",
            "modelo_desejado": modelo,
        },
        recommended_products=products or [],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
    )


# ── Unit: _build_filters wiring ──────────────────────────────────────────────

def test_search_products_filters_by_category():
    """_build_filters always pins a category for beach/padel customers."""
    f_beach = _build_filters({"esporte_praticado": "beach tennis", "nivel_jogo": "iniciante"})
    f_padel = _build_filters({"esporte_praticado": "padel", "nivel_jogo": "iniciante"})
    f_none = _build_filters({"nivel_jogo": "iniciante"})  # no sport at all

    assert f_beach["category"] == "raquete"
    assert f_padel["category"] == "pala"
    # When sport isn't declared, we default to raquete (beach tennis is the
    # business default — see Sprint 1.5 strategy).
    assert f_none["category"] == "raquete"


# ── Integration: recommend forwards category to retriever ────────────────────

@pytest.mark.asyncio
async def test_recommend_uses_raquete_category_by_default():
    """For a beach-tennis customer, the retriever call carries category='raquete'."""
    products = [_product("Raquete X", category="raquete")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": ["*X*", "Posso?"]})
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = products
            with patch("app.storage.db.get_session", _mock_db_session):
                await recommend_node(_profile_state(sport="beach tennis"))

    # search_products is called with (session, query, filters, k=...)
    call = search.call_args
    filters = call.args[2] if len(call.args) >= 3 else call.kwargs.get("filters", {})
    assert filters.get("category") == "raquete"


@pytest.mark.asyncio
async def test_recommend_uses_pala_category_when_padel():
    """Padel customer → retriever called with category='pala'."""
    products = [_product("Pala X", category="pala")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({"messages": ["*X*", "Posso?"]})
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = products
            with patch("app.storage.db.get_session", _mock_db_session):
                await recommend_node(_profile_state(sport="padel"))

    call = search.call_args
    filters = call.args[2] if len(call.args) >= 3 else call.kwargs.get("filters", {})
    assert filters.get("category") == "pala"


# Sprint 2.0 — re_recommendation no longer fetches rackets (it pivots to
# the Consultoria), so there's no retriever call to assert filters on. The
# old test_re_recommendation_excludes_non_raquete_categories and
# test_re_recommendation_for_padel_uses_pala_category tests were dropped
# along with the cheaper/premium/different flavour logic.
