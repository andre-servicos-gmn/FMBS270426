"""Sprint 2.1 — smoke test of the 3 scenarios.

A: Determined customer (the conversation that broke pre-2.1).
B: Exploring customer (regression — full diagnose must still run).
C: Determined → exploring transition (opinion-seek flips path).

External I/O (OpenAI, DB, retriever) is mocked.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db():
    s = MagicMock()
    s.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    s.commit = AsyncMock()
    yield s


def _product(name: str, price: int = 89900) -> dict:
    return {
        "id": f"id-{name}", "name": name, "sport": "beach_tennis",
        "level": "intermediário", "price_cents": price, "stock": 5,
        "description": f"desc {name}", "similarity": 0.9,
        "external_id": name.replace(" ", "-"), "url": None, "image_url": None,
        "updated_at": None, "is_active": True, "weight_g": 350,
        "balance": "médio", "material": "carbono", "category": "raquete",
    }


def show(label: str, result: dict) -> None:
    last = next(
        (m for m in reversed(result.get("messages") or []) if isinstance(m, AIMessage)),
        None,
    )
    text = last.content if last else "(no message)"
    print(f"\n=== {label} ===")
    print(
        f"intent={result.get('intent')} "
        f"path={result.get('customer_intent_path')!r} "
        f"consultoria_mentioned_count={result.get('consultoria_mentioned_count')}"
    )
    print(f"products_on_table={[p['name'] for p in result.get('recommended_products') or []]}")
    print(f"reply:\n{text}")


async def main() -> None:
    graph = build_graph(checkpointer=MemorySaver())

    # ── SCENARIO A — cliente determinado (the regression we're fixing) ────
    print("\n##### SCENARIO A: cliente determinado #####")
    init = AgentState(
        messages=[HumanMessage(content="oi")],
        phone_hash="scenarioA" * 7,
        intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False,
    )
    # Turn 1: "oi" → smalltalk asks for name
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "smalltalk"}',
            "Oi! Tudo certo? Antes de mais nada, qual seu nome? 😊",
        ]
        with patch("app.storage.db.get_session", _mock_db):
            ra1 = await graph.ainvoke(init, {"configurable": {"thread_id": "tA"}})
    show("A.1 (oi → ask name)", ra1)

    # Turn 2: "Marcelo" → name captured
    s2 = dict(ra1); s2["messages"] = [HumanMessage(content="Marcelo")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "smalltalk"}',
            '{"extracted_name": "Marcelo"}',
            "Show, Marcelo! Em que posso te ajudar?",
        ]
        with patch("app.storage.db.get_session", _mock_db):
            ra2 = await graph.ainvoke(s2, {"configurable": {"thread_id": "tA"}})
    show("A.2 (Marcelo → name captured)", ra2)

    # Turn 3: "vocês têm a beach pro carbon x5?" → determined → REFERENCE-SIM
    s3 = dict(ra2); s3["messages"] = [
        HumanMessage(content="vocês têm a beach pro carbon x5?")
    ]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        # Only 1 LLM call: triage (determined branch is LLM-free).
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch(
            "app.rag.retriever.search_products", new_callable=AsyncMock
        ) as search:
            search.return_value = [_product("Raquete BeachPro Carbon X5")]
            with patch("app.storage.db.get_session", _mock_db):
                ra3 = await graph.ainvoke(s3, {"configurable": {"thread_id": "tA"}})
    show("A.3 (carbon x5 → REFERENCE-SIM determined)", ra3)
    print(f"LLM call count for A.3: {llm.call_count} (expected: 1 — triage only)")

    # Turn 4: "quanto custa?" → price_inquiry natural + subtle pitch (1st mention)
    s4 = dict(ra3); s4["messages"] = [HumanMessage(content="quanto custa?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "price_inquiry"}']
        with patch("app.storage.db.get_session", _mock_db):
            ra4 = await graph.ainvoke(s4, {"configurable": {"thread_id": "tA"}})
    show("A.4 (quanto custa? → price + subtle pitch)", ra4)

    # Turn 5: "tem antivibração?" → product_detail; pitch already mentioned → NOT repeated
    s5 = dict(ra4); s5["messages"] = [HumanMessage(content="tem antivibração?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_detail"}']
        with patch("app.storage.db.get_session", _mock_db):
            ra5 = await graph.ainvoke(s5, {"configurable": {"thread_id": "tA"}})
    show("A.5 (tem antivibração? → detail, no repeat pitch)", ra5)
    if "Consultoria Base Sports" in ra5["messages"][-1].content:
        print("⚠️  subtle pitch repeated — count=", ra5.get("consultoria_mentioned_count"))
    else:
        print("✓ subtle pitch NOT repeated (consultoria_mentioned_count="
              f"{ra5.get('consultoria_mentioned_count')})")

    # ── SCENARIO B — cliente explorador (regression) ──────────────────────
    print("\n\n##### SCENARIO B: cliente explorador (regression) #####")
    initB = AgentState(
        messages=[HumanMessage(content="quero uma raquete")],
        phone_hash="scenarioB" * 7,
        intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Maria",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "diagnose"}',          # triage; catalog has nothing to match → not determined
            json.dumps({"extracted_slots": {}}),
            "Qual seu nível de jogo, Maria? Iniciante, intermediário ou avançado?",
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []  # no match → not determined
            with patch("app.storage.db.get_session", _mock_db):
                rB1 = await graph.ainvoke(initB, {"configurable": {"thread_id": "tB"}})
    show("B.1 (quero raquete → diagnose)", rB1)

    # Turn 2: "não sei qual" → diagnose extracts modelo_desejado=nenhum → next slot
    sB2 = dict(rB1); sB2["messages"] = [HumanMessage(content="não sei qual")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        # state.intent == "diagnose" → _start_router shortcuts to diagnose
        # (skips triage). Only diagnose's 2 LLM calls (extract + phrase).
        llm.side_effect = [
            json.dumps({"extracted_slots": {"modelo_desejado": "nenhum"}}),
            "Anotado. E qual seu nível de jogo?",
        ]
        with patch("app.storage.db.get_session", _mock_db):
            rB2 = await graph.ainvoke(sB2, {"configurable": {"thread_id": "tB"}})
    show("B.2 (não sei qual → diagnose continues)", rB2)

    # ── SCENARIO C — determined → exploring transition ────────────────────
    print("\n\n##### SCENARIO C: determined → exploring transition #####")
    initC = AgentState(
        messages=[HumanMessage(content="vocês têm a beach pro carbon x5?")],
        phone_hash="scenarioC" * 7,
        intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Carlos",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = [_product("Raquete BeachPro Carbon X5")]
            with patch("app.storage.db.get_session", _mock_db):
                rC1 = await graph.ainvoke(initC, {"configurable": {"thread_id": "tC"}})
    show("C.1 (determined → REFERENCE-SIM)", rC1)

    # Turn 2: "você acha que ela serve mesmo pra mim?" → opinion-seek → exploring → diagnose
    sC2 = dict(rC1); sC2["messages"] = [
        HumanMessage(content="você acha que ela serve mesmo pra mim?")
    ]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "product_detail"}',      # triage; opinion-seek overrides
            json.dumps({"extracted_slots": {}}), # diagnose extract
            "Pra eu te indicar com mais segurança, qual seu nível de jogo, Carlos?",
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock):
            with patch("app.storage.db.get_session", _mock_db):
                rC2 = await graph.ainvoke(sC2, {"configurable": {"thread_id": "tC"}})
    show("C.2 (você acha? → diagnose runs again)", rC2)


if __name__ == "__main__":
    asyncio.run(main())
