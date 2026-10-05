"""Fixes from the 2026-10-05 WhatsApp test conversation.

1. A bare racket browse (newest first) must keep the cheapest racket visible,
   or the agent tells a beginner "temos de R$ 2.000 a R$ 3.600" when the
   floor is R$ 449.
2. After the agent has pitched the Consultoria, every later turn carries a
   note telling it not to pitch again.
"""
import json
from datetime import datetime
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.supervisor import _conversation_note, _build_openai_request
from app.agent.tools_v2 import buscar_catalogo


def _racket(pid: int, price: float, created: datetime) -> dict:
    return {"id": pid, "name": f"Raquete Beach Tennis Modelo {pid}", "price_cents": int(price * 100),
            "is_raquete_praia": True, "categoria_nome": "Raquetes de Praia", "stock": 5,
            "created_at": created}


# 10 rackets: the 9 newest are expensive, the cheapest is the oldest.
_CATALOG = [_racket(i, 2000 + i * 100, datetime(2026, 1, i + 1)) for i in range(1, 10)]
_CATALOG.append(_racket(99, 449, datetime(2024, 1, 1)))


async def _snapshot():
    return list(_CATALOG)


@pytest.mark.asyncio
async def test_browse_keeps_cheapest_racket_visible():
    with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", _snapshot):
        raw = await buscar_catalogo.ainvoke({"consulta": "", "categoria": "beach tennis"})
    prices = [p["preco"] for p in json.loads(raw)]
    assert len(prices) == 8
    assert "R$ 449,00" in prices, prices
    # Still mostly the newest arrivals (Sprint 3.9 recency).
    assert prices[0] == "R$ 2.900,00", prices


def test_no_note_before_consultoria_is_mentioned():
    msgs = [HumanMessage(content="oi"), AIMessage(content="Oi! O que você procura?")]
    assert _conversation_note(msgs) == ""


def test_note_after_consultoria_pitch():
    msgs = [
        HumanMessage(content="sou iniciante, qual levo?"),
        AIMessage(content="Isso a gente crava na Consultoria, R$ 350 abatidos na compra."),
        HumanMessage(content="e essa aqui é boa?"),
    ]
    note = _conversation_note(msgs)
    assert "1 vez" in note and "NÃO ofereça de novo" in note


def test_tool_call_messages_are_not_counted():
    msgs = [AIMessage(content="", tool_calls=[{"name": "buscar_conhecimento",
                                               "args": {"consulta": "consultoria"}, "id": "t1"}])]
    assert _conversation_note(msgs) == ""


def test_note_lands_at_end_of_system_prompt():
    from app.config import get_settings

    state = {"messages": [
        HumanMessage(content="oi"),
        AIMessage(content="Temos a Consultoria por R$ 350."),
        HumanMessage(content="legal"),
    ]}
    req = _build_openai_request(state, get_settings(), force_search=False)
    system = req["messages"][0]["content"]
    assert "CONTEXTO DESTA CONVERSA" in system.split("\n\n")[-1]


def test_note_after_brand_question():
    msgs = [
        HumanMessage(content="quero uma raquete"),
        AIMessage(content="Tem alguma marca em mente? Trabalhamos com Drop Shot e Head."),
        HumanMessage(content="queria uma mais barata"),
    ]
    note = _conversation_note(msgs)
    assert "NÃO pergunte de marca de novo" in note
    assert "Consultoria" not in note


def test_brand_mention_without_question_is_not_counted():
    msgs = [AIMessage(content="A marca Drop Shot tem ótimas opções de entrada.")]
    assert _conversation_note(msgs) == ""
