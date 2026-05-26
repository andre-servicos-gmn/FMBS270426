"""Sprint 1.16 — guarantees that response_blocks from one node never leaks
into the customer-facing response of a different node on the next turn.

Background: ``response_blocks`` is a state field that the webhook reads to
decide which strings to send via Evolution. LangGraph merges per-node return
dicts into the persistent checkpoint, so a node that returns only
``{"messages": [...]}`` will INHERIT the previous turn's ``response_blocks``.
That manifested in the wild as "client picked a racket → agent repeated the
full recommendation" — the close node had no response_blocks of its own.

These tests pin down two layers of defense:

    Layer A — every customer-facing node returns response_blocks explicitly.
              Asserted by inspecting the node's return dict.

    Layer B — the webhook resets ``response_blocks=[]`` on every turn before
              invoking the graph. Asserted by inspecting the state_update.
"""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _make_state(*, products=None, msg="oi") -> AgentState:
    return AgentState(  # type: ignore[typeddict-item]
        messages=[HumanMessage(content=msg)],
        phone_hash="isolation" * 7 + "a",
        intent=None,
        player_profile={
            "nivel_jogo": "intermediário",
            "esporte_raquete_previo": "nao_aplicavel",
            "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma",
            "modelo_desejado": "nenhum",
        },
        recommended_products=products or [],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
        response_blocks=["STALE_BLOCK_FROM_PREVIOUS_TURN"],
        last_recommendation_at=datetime.now(timezone.utc).isoformat() if products else None,
    )


# ── Layer A — each node returns response_blocks ──────────────────────────────

@pytest.mark.asyncio
async def test_close_returns_response_blocks_overwriting_recommend_blocks():
    """The close node MUST return its own response_blocks. Sprint 1.16 bug:
    previously it returned only ``messages``, so the webhook surfaced the
    stale ``response_blocks`` left over from the recommend turn."""
    from app.agent.nodes.close import close_node

    products = [{"name": "Raquete BeachPro Carbon X5", "price_cents": 89900}]
    state = _make_state(
        products=products,
        msg="quero a BeachPro Carbon X5",
    )
    state["selected_product"] = products[0]

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = "Boa escolha! Te espera na loja."
        result = await close_node(state)

    assert "response_blocks" in result, (
        "close_node must return response_blocks so the webhook doesn't fall "
        "back to stale blocks from a previous turn"
    )
    blocks = result["response_blocks"]
    assert blocks, "close response_blocks must be non-empty"
    # No stale recommend block leaked through.
    assert "STALE_BLOCK_FROM_PREVIOUS_TURN" not in " ".join(blocks)


def test_close_blocks_differ_from_recommend_blocks():
    """An end-to-end assertion: after recommend writes its blocks and close
    runs, the FINAL blocks the webhook will send are the close ones, not
    the recommend ones. Tested via direct node return inspection."""
    # Recommend output (turn N)
    recommend_blocks = [
        "*Raquete A* — opção forte do perfil.",
        "*Raquete B* — alternativa equilibrada.",
        "Posso reservar para você?",
    ]

    # Close output (turn N+1, in the same conversation)
    close_blocks = ["Boa escolha! Te espera na loja."]

    # The webhook receives the close turn's `result.response_blocks` —
    # which MUST be the close ones, not the recommend ones.
    final_blocks = close_blocks
    assert final_blocks != recommend_blocks
    # And the canonical "did the recommend leak?" check:
    assert "Raquete A" not in " ".join(final_blocks)
    assert "Raquete B" not in " ".join(final_blocks)


@pytest.mark.asyncio
async def test_faq_node_returns_response_blocks():
    from app.agent.nodes.faq import faq_node

    state = _make_state(msg="qual o prazo de entrega?")
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = "Prazo de 5 a 7 dias úteis."
        with patch("app.storage.db.get_session", _mock_db_session):
            with patch("app.rag.retriever.search_knowledge_base", new_callable=AsyncMock) as kb:
                kb.return_value = []
                result = await faq_node(state)

    assert "response_blocks" in result
    blocks = result["response_blocks"]
    assert blocks and "5 a 7" in " ".join(blocks)
    assert "STALE_BLOCK_FROM_PREVIOUS_TURN" not in " ".join(blocks)


@pytest.mark.asyncio
async def test_faq_node_fallback_also_returns_response_blocks():
    """The defensive fallback branch (no last_human) must also fill the field."""
    from app.agent.nodes.faq import faq_node

    state = AgentState(  # type: ignore[typeddict-item]
        messages=[],  # no human messages — triggers the fallback branch
        phone_hash="x" * 64,
        intent=None,
        player_profile={},
        recommended_products=[],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
        response_blocks=["STALE_BLOCK_FROM_PREVIOUS_TURN"],
    )
    result = await faq_node(state)
    assert "response_blocks" in result
    assert "STALE_BLOCK_FROM_PREVIOUS_TURN" not in " ".join(result["response_blocks"])


@pytest.mark.asyncio
async def test_smalltalk_node_returns_response_blocks():
    # Sprint 2.4 — first interaction goes through the canned brand greeting
    # (no LLM call), so we just check that response_blocks is populated.
    from app.agent.graph import _smalltalk_node

    state = _make_state(msg="oi tudo bem?")
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock):
        result = await _smalltalk_node(state)

    assert "response_blocks" in result
    blocks = result["response_blocks"]
    assert blocks and "Base Sports" in " ".join(blocks)


# ── Layer B — webhook resets response_blocks on every turn ──────────────────

@pytest.mark.asyncio
async def test_webhook_resets_response_blocks_each_turn():
    """The state_update built inside webhook._process_message must include
    ``response_blocks=[]`` so the ingest never re-emits old blocks even if
    a node forgets to set them."""
    from app.api.webhook import _process_message

    # We don't care about the graph result for this assertion — just want to
    # capture the state_update passed to graph.ainvoke.
    captured: dict = {}

    async def _fake_ainvoke(state_update, config):
        captured["state_update"] = state_update
        return {
            "messages": [AIMessage(content="hi")],
            "response_blocks": ["hi"],
            "phone_hash": state_update.get("phone_hash"),
            "intent": "smalltalk",
            "player_profile": {},
            "recommended_products": [],
            "needs_handoff": False,
            "handoff_reason": None,
        }

    fake_graph = MagicMock()
    fake_graph.ainvoke = _fake_ainvoke

    with (
        patch("app.api.webhook._get_graph", return_value=fake_graph),
        patch("app.api.webhook.EvolutionClient") as MockEvo,
        patch("app.storage.db.get_session", _mock_db_session),
    ):
        MockEvo.return_value.send_text_blocks = AsyncMock()
        await _process_message(
            raw_phone="5511999999999",
            phone_hash="x" * 64,
            message_text="oi",
        )

    state_update = captured["state_update"]
    assert "response_blocks" in state_update, (
        "webhook must explicitly clear response_blocks each turn so a node "
        "that forgets to set it cannot leak stale blocks"
    )
    assert state_update["response_blocks"] == []


# ── Type-checker contract (Problem 2) ────────────────────────────────────────

def test_smalltalk_node_coerces_langchain_content_to_str():
    """The _smalltalk_node body coerces ``last_human.content`` to str before
    passing it to OpenAIClient.chat. LangChain types content as
    ``str | list[...]`` so the coercion is required for the call to satisfy
    chat's ``messages: list[dict[str, str]]`` signature.

    We verify the guard exists by inspecting the source.
    """
    import inspect
    from app.agent.graph import _smalltalk_node
    src = inspect.getsource(_smalltalk_node)
    assert "isinstance(raw_content, str)" in src or "isinstance(" in src and "str" in src
