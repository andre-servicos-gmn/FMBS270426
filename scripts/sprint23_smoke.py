"""Sprint 2.3 — smoke test for the 4 contextual-pitch scenarios.

A. Pitch IMEDIATO no STOCK (1ª pergunta) — confirma estoque + pitch STOCK.
B. Pitch DELAYED no WEIGHT (1ª pergunta) — só info, sem pitch.
   Depois MATERIAL (2ª pergunta) — pitch DEFAULT fires.
C. Cap respeitado: PRICE na 1ª, depois WEIGHT/COMFORT sem repetir pitch.
D. Cliente EXPLORER nunca recebe pitch sutil (regressão).
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


def _product(name: str = "Raquete BeachPro Carbon X5", price: int = 89900) -> dict:
    return {
        "id": f"id-{name}", "name": name, "sport": "beach_tennis",
        "level": "intermediário", "price_cents": price, "stock": 5,
        "description": "Raquete com fibra de carbono e bom controle, peso médio.",
        "similarity": 0.9, "external_id": name.replace(" ", "-"),
        "url": None, "image_url": None, "updated_at": None, "is_active": True,
        "weight_g": 350, "balance": "médio", "material": "carbono",
        "category": "raquete",
    }


def show(label: str, result: dict) -> None:
    last = next(
        (m for m in reversed(result.get("messages") or []) if isinstance(m, AIMessage)),
        None,
    )
    text = last.content if last else "(no message)"
    print(f"\n=== {label} ===")
    print(
        f"path={result.get('customer_intent_path')!r} "
        f"determined_qcount={result.get('determined_question_count')} "
        f"consultoria_mentioned_count={result.get('consultoria_mentioned_count')}"
    )
    print(f"REPLY:\n{text}")


async def _scenario_a(graph) -> None:
    print("\n\n##### CENÁRIO A — pitch IMEDIATO (STOCK) #####")
    init = AgentState(
        messages=[HumanMessage(content="vocês têm a beach pro carbon x5?")],
        phone_hash="SA" + "x" * 62, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Marcelo",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as s:
            s.return_value = [_product()]
            with patch("app.storage.db.get_session", _mock_db):
                r = await graph.ainvoke(init, {"configurable": {"thread_id": "tA"}})
    show("A.1 — vocês têm a beach pro carbon x5? (STOCK, immediate)", r)


async def _scenario_b(graph) -> None:
    print("\n\n##### CENÁRIO B — pitch DELAYED (WEIGHT → MATERIAL) #####")
    init = AgentState(
        messages=[HumanMessage(content="qual o peso da beach pro carbon x5?")],
        phone_hash="SB" + "x" * 62, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Marcelo",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        # triage only (recommend determined → no LLM).
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as s:
            s.return_value = [_product()]
            with patch("app.storage.db.get_session", _mock_db):
                r1 = await graph.ainvoke(init, {"configurable": {"thread_id": "tB"}})
    show("B.1 — qual o peso? (WEIGHT not classified yet — STOCK fires because triage tagged determined via match)", r1)
    # B.1 actually goes through recommend REFERENCE-SIM (because triage marks
    # determined + the message also contains the product name). Pitch is
    # STOCK (immediate). So the truer "delayed" scenario is B.2: after stock
    # confirmation in B.1, ask about peso — that goes through product_detail.

    # B.2: after the stock confirmation, ask about peso → product_detail with
    # WEIGHT (delayed). consultoria_mentioned_count should already be 1 from B.1.
    s2 = dict(r1)
    s2["messages"] = [HumanMessage(content="qual o peso?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_detail"}']
        with patch("app.storage.db.get_session", _mock_db):
            r2 = await graph.ainvoke(s2, {"configurable": {"thread_id": "tB"}})
    show("B.2 — qual o peso? (WEIGHT delayed; cap full from B.1)", r2)

    # B.3: more techy questions (material) — cap still full.
    s3 = dict(r2)
    s3["messages"] = [HumanMessage(content="qual o material?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_detail"}']
        with patch("app.storage.db.get_session", _mock_db):
            r3 = await graph.ainvoke(s3, {"configurable": {"thread_id": "tB"}})
    show("B.3 — qual o material? (MATERIAL delayed; cap full from B.1)", r3)


async def _scenario_c(graph) -> None:
    print("\n\n##### CENÁRIO C — cap respeitado (PRICE 1ª → outras sem pitch) #####")
    init = AgentState(
        messages=[HumanMessage(content="vocês têm a beach pro carbon x5?")],
        phone_hash="SC" + "x" * 62, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Marcelo",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as s:
            s.return_value = [_product()]
            with patch("app.storage.db.get_session", _mock_db):
                r1 = await graph.ainvoke(init, {"configurable": {"thread_id": "tC"}})
    show("C.1 — vocês têm? (STOCK fires; cap=1)", r1)

    s2 = dict(r1); s2["messages"] = [HumanMessage(content="quanto pesa?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_detail"}']
        with patch("app.storage.db.get_session", _mock_db):
            r2 = await graph.ainvoke(s2, {"configurable": {"thread_id": "tC"}})
    show("C.2 — quanto pesa? (cap cheio; sem pitch)", r2)

    s3 = dict(r2); s3["messages"] = [HumanMessage(content="tem antivibração?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_detail"}']
        with patch("app.storage.db.get_session", _mock_db):
            r3 = await graph.ainvoke(s3, {"configurable": {"thread_id": "tC"}})
    show("C.3 — tem antivibração? (cap cheio; sem pitch)", r3)


async def _scenario_d(graph) -> None:
    print("\n\n##### CENÁRIO D — explorer não recebe pitch sutil (regressão) #####")
    init = AgentState(
        messages=[HumanMessage(content="quero uma raquete")],
        phone_hash="SD" + "x" * 62, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Maria",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        # triage → diagnose extract → diagnose phrase (catalog has nothing relevant
        # → loose match doesn't fire → not determined).
        llm.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": {}}),
            "Qual seu nível de jogo, Maria?",
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as s:
            s.return_value = []  # no catalog match → exploring path
            with patch("app.storage.db.get_session", _mock_db):
                r = await graph.ainvoke(init, {"configurable": {"thread_id": "tD"}})
    show("D — quero uma raquete (explorer; sem pitch sutil)", r)


async def main() -> None:
    graph = build_graph(checkpointer=MemorySaver())
    await _scenario_a(graph)
    await _scenario_b(graph)
    await _scenario_c(graph)
    await _scenario_d(graph)


if __name__ == "__main__":
    asyncio.run(main())
