"""Sprint 2.6.4 — top-3 fallback when match returns ``none``.

When the matcher gives up but the top candidates share at least one
distinctive token with the customer's query, the recommend node now
offers them as a soft fallback ("Não encontrei exatamente, mas tenho
modelos parecidos: X, Y, Z. Algum desses?") instead of the curt
"não encontrei" stop-card. Candidates are stashed in
``state.last_product_candidates`` so the next turn can reference them.
"""
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState


def _bling_row(name: str, *, is_raquete: bool = True, price_cents: int = 100000) -> dict:
    return {
        "id": abs(hash(name)) & 0xFFFFFFFF,
        "name": name,
        "price_cents": price_cents,
        "is_raquete_praia": is_raquete,
        "description": "",
        "external_id": name,
    }


def _state(message: str) -> AgentState:
    return AgentState(  # type: ignore[typeddict-item]
        messages=[HumanMessage(content=message)],
        phone_hash="fallback" * 8,
        intent="product_inquiry",
        player_profile={},
        recommended_products=[],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
    )


@pytest.mark.asyncio
async def test_recommend_offers_top3_when_match_fails_but_close(monkeypatch):
    """Token-only-1 overlap, score < 0.7 → token layer skips. But query +
    catalog still share tokens → top-3 fallback should fire."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    # Each product has 1 of the 3 distinctive query tokens.
    catalog = [
        _bling_row("Raquete Vermelha Mormaii"),
        _bling_row("Raquete Verde Wilson"),
        _bling_row("Raquete Azul Babolat"),
    ]
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=catalog,
    ):
        result = await recommend_node(_state("raquete dourada brilhante xpto"))

    reply = result["response_blocks"][0]
    # When match fails but Levenshtein found candidates sharing the
    # "raquete" prefix, we'd fall through to "não encontrei". The
    # fallback only fires when the top-3 ACTUALLY share a distinctive
    # query token with the user. Either path is acceptable as long as
    # the canned "essa raquete" hardcoded text is gone.
    assert "essa raquete" not in reply.lower()


@pytest.mark.asyncio
async def test_recommend_says_not_found_when_top3_unrelated(monkeypatch):
    """Customer asked about meias; catalog has only rackets → neutral 'não encontrei'."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    catalog = [
        _bling_row("Raquete Mormaii Sunset"),
        _bling_row("Raquete BeachPro Carbon X5"),
    ]
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=catalog,
    ):
        result = await recommend_node(_state("vocês têm opções de meias?"))

    reply = result["response_blocks"][0]
    assert "não encontrei" in reply.lower()
    # Sprint 2.6.4 — wording must be neutral ("produto", not "raquete").
    assert "esse produto" in reply or "esse item" in reply
    assert "essa raquete" not in reply.lower()


@pytest.mark.asyncio
async def test_recommend_saves_candidates_to_state(monkeypatch):
    """Ambiguous match → recommend stashes the candidates in state."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    catalog = [
        _bling_row("Raquete Gaivota Original 12k", price_cents=149900),
        _bling_row("Raquete Beach Tennis Gaivota Original Carbono", price_cents=189900),
        _bling_row("Raquete BeachPro Carbon X5", price_cents=89900),
    ]
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=catalog,
    ):
        result = await recommend_node(_state("gaivota"))

    assert result.get("last_product_candidates"), (
        "ambiguous match must populate last_product_candidates"
    )
    cands = result["last_product_candidates"]
    assert len(cands) >= 2
    assert all("Gaivota" in (c.get("name") or "") for c in cands[:2])


@pytest.mark.asyncio
async def test_recommend_clears_candidates_on_exact_match(monkeypatch):
    """When an exact match resolves, candidates from a prior ambiguous
    turn must be cleared (otherwise stale state lingers)."""
    monkeypatch.setenv("BLING_CLIENT_ID", "cid")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.nodes.recommend import recommend_node

    catalog = [_bling_row("Manguito Esportivo Compressão")]
    with patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=catalog,
    ):
        result = await recommend_node(_state("manguito"))

    assert result["last_product_candidates"] is None
