"""Sprint 2.4 — smoke for the 4 scenarios pre-call Felipe.

A: Determined + raquete EXISTE → confirmação limpa + pickup invite (curto, no humano).
B: Determined + raquete NÃO EXISTE → oferta de alternativas + cliente aceita.
C: Determined + pergunta técnica (PRICE) entre stock e compra → pitch fires PRICE.
D: Brand greeting na saudação.
"""
import asyncio
import random
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
        f"path={result.get('customer_intent_path')!r} "
        f"needs_handoff={result.get('needs_handoff')!r} "
        f"awaiting_alts={result.get('awaiting_alternatives_decision')!r} "
        f"goodbye_pending={result.get('goodbye_pending')!r} "
        f"consultoria_mentioned={result.get('consultoria_mentioned_count')!r}"
    )
    print(f"REPLY:\n{text}")


async def _scenario_a() -> None:
    print("\n\n##### CENÁRIO A — Determined + EXISTE + compra direta #####")
    random.seed(42)  # deterministic variation for the report
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "tA"}}

    # 1) oi
    init = AgentState(
        messages=[HumanMessage(content="oi")],
        phone_hash="A" * 64, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False,
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}']
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(init, cfg)
    show("A.1 oi (brand greeting)", r)

    # 2) Marcelo
    s = dict(r); s["messages"] = [HumanMessage(content="Marcelo")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}', '{"extracted_name": "Marcelo"}',
                           "Show, Marcelo! Em que posso te ajudar?"]
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(s, cfg)
    show("A.2 Marcelo (name captured)", r)

    # 3) vocês têm a Carbon X5?
    s = dict(r); s["messages"] = [HumanMessage(content="vocês têm a Carbon X5?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as srch:
            srch.return_value = [_product("Raquete BeachPro Carbon X5")]
            with patch("app.storage.db.get_session", _mock_db):
                r = await graph.ainvoke(s, cfg)
    show("A.3 vocês têm a Carbon X5? (stock confirm, NO pitch)", r)

    # 4) quero comprar
    s = dict(r); s["messages"] = [HumanMessage(content="quero comprar")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_selection"}']
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(s, cfg)
    show("A.4 quero comprar (pickup invite, NO handoff)", r)


async def _scenario_b() -> None:
    print("\n\n##### CENÁRIO B — Determined + NÃO EXISTE + cliente aceita alternativas #####")
    random.seed(7)
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "tB"}}

    init = AgentState(
        messages=[HumanMessage(content="oi")],
        phone_hash="B" * 64, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False,
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}']
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(init, cfg)
    show("B.1 oi", r)

    s = dict(r); s["messages"] = [HumanMessage(content="Marcelo")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}', '{"extracted_name": "Marcelo"}',
                           "Show, Marcelo!"]
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(s, cfg)
    show("B.2 Marcelo", r)

    s = dict(r); s["messages"] = [HumanMessage(content="quero comprar a Wilson Pro Staff")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as srch:
            srch.return_value = []  # not in catalog
            with patch("app.storage.db.get_session", _mock_db):
                r = await graph.ainvoke(s, cfg)
    show("B.3 quero comprar a Wilson Pro Staff (NÃO existe, oferece alternativas)", r)

    s = dict(r); s["messages"] = [HumanMessage(content="sim")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        # triage early-returns on awaiting_alternatives_decision; then diagnose
        # extract + phrase fire.
        import json as _json
        llm.side_effect = [_json.dumps({"extracted_slots": {}}),
                           "Pra te indicar com mais segurança, qual seu nível de jogo, Marcelo?"]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock):
            with patch("app.storage.db.get_session", _mock_db):
                r = await graph.ainvoke(s, cfg)
    show("B.4 sim (transição → exploring → diagnose)", r)


async def _scenario_c() -> None:
    print("\n\n##### CENÁRIO C — Determined + pergunta PRICE entre stock e compra #####")
    random.seed(1)
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "tC"}}

    init = AgentState(
        messages=[HumanMessage(content="oi")],
        phone_hash="C" * 64, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False,
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}']
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(init, cfg)
    show("C.1 oi", r)

    s = dict(r); s["messages"] = [HumanMessage(content="Marcelo")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}', '{"extracted_name": "Marcelo"}',
                           "Show, Marcelo!"]
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(s, cfg)
    show("C.2 Marcelo", r)

    s = dict(r); s["messages"] = [HumanMessage(content="vocês têm a Carbon X5?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as srch:
            srch.return_value = [_product("Raquete BeachPro Carbon X5")]
            with patch("app.storage.db.get_session", _mock_db):
                r = await graph.ainvoke(s, cfg)
    show("C.3 vocês têm a Carbon X5? (stock confirm, NO pitch)", r)

    s = dict(r); s["messages"] = [HumanMessage(content="quanto custa?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "price_inquiry"}']
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(s, cfg)
    show("C.4 quanto custa? (PRICE preset fires)", r)

    s = dict(r); s["messages"] = [HumanMessage(content="quero comprar")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_selection"}']
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(s, cfg)
    show("C.5 quero comprar (pickup, NO pitch repetido)", r)


async def _scenario_d() -> None:
    print("\n\n##### CENÁRIO D — Brand greeting na saudação #####")
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "tD"}}

    init = AgentState(
        messages=[HumanMessage(content="oi")],
        phone_hash="D" * 64, intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False,
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}']
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(init, cfg)
    show("D.1 oi (brand)", r)

    s = dict(r); s["messages"] = [HumanMessage(content="Marcelo")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "smalltalk"}', '{"extracted_name": "Marcelo"}',
                           "Show, Marcelo! Em que posso te ajudar?"]
        with patch("app.storage.db.get_session", _mock_db):
            r = await graph.ainvoke(s, cfg)
    show("D.2 Marcelo (cumprimento com nome)", r)


async def main() -> None:
    await _scenario_a()
    await _scenario_b()
    await _scenario_c()
    await _scenario_d()


if __name__ == "__main__":
    asyncio.run(main())
