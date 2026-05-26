"""Sprint 2.0 — smoke test of the 4 strategic scenarios.

Mocks OpenAI + DB + retriever so it runs offline. Exercises the real
LangGraph topology to verify end-to-end wiring of:
  A. Name capture (first contact + name turn).
  B. bare_recommendation_request → diagnose → consultoria_offer (NOT recommend).
  C. REFERENCE-SIM (specific racket in catalog) → confirm + ask.
  D. product_selection → purchase_closing handoff.
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


def _product(name: str, price: int = 80000) -> dict:
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
        f"needs_handoff={result.get('needs_handoff')} "
        f"handoff_reason={result.get('handoff_reason')}"
    )
    print(
        f"customer_name={result.get('customer_name')!r} "
        f"name_asked={result.get('name_asked')!r}"
    )
    print(f"reply: {text[:240]}")


async def main() -> None:
    graph = build_graph(checkpointer=MemorySaver())

    # ── SCENARIO A: First contact → ask name → extract name ──────────────
    print("\n##### SCENARIO A: name capture #####")
    init = AgentState(
        messages=[HumanMessage(content="oi tudo bem?")],
        phone_hash="scenarioa" * 7,
        intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False,
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "smalltalk"}',
            "Oi! Tudo certo? Antes de mais nada, qual seu nome?",
        ]
        with patch("app.storage.db.get_session", _mock_db):
            r1 = await graph.ainvoke(init, {"configurable": {"thread_id": "t-a"}})
    show("A.1 (greeting)", r1)

    state2 = dict(r1)
    state2["messages"] = [HumanMessage(content="Andre")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "smalltalk"}',
            '{"extracted_name": "Andre"}',
            "Show, Andre! Em que posso te ajudar?",
        ]
        with patch("app.storage.db.get_session", _mock_db):
            r2 = await graph.ainvoke(state2, {"configurable": {"thread_id": "t-a"}})
    show("A.2 (name)", r2)

    # ── SCENARIO B: bare_recommendation_request → consultoria offer ──────
    print("\n##### SCENARIO B: bare_recommendation_request -> consultoria #####")
    state_b = AgentState(
        messages=[HumanMessage(content="qual raquete vocês indicam?")],
        phone_hash="scenariob" * 7, intent=None,
        player_profile={
            "nivel_jogo": "intermediário", "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma", "esporte_raquete_previo": "nao_aplicavel",
            "modelo_desejado": "nenhum",
        },
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Andre",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "bare_recommendation_request"}',
            json.dumps({"extracted_slots": {}}),
            json.dumps({"messages": [
                "Andre, pelo perfil que você me passou…",
                "A gente prefere fazer com a *Consultoria Base Sports* — análise + "
                "teste em quadra. Investimento R$350, 100% abatido.",
                "Quer saber como funciona ou já agendar?",
            ]}),
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock):
            with patch("app.storage.db.get_session", _mock_db):
                r_b = await graph.ainvoke(state_b, {"configurable": {"thread_id": "t-b"}})
    show("B", r_b)
    print(f"consultoria_interest={r_b.get('consultoria_interest')}")
    print(f"recommended_products={[p['name'] for p in r_b.get('recommended_products') or []]}")

    # ── SCENARIO C: REFERENCE-SIM ────────────────────────────────────────
    print("\n##### SCENARIO C: REFERENCE-SIM (specific racket exists) #####")
    state_c = AgentState(
        messages=[HumanMessage(content="vocês têm a Raquete BeachPro Carbon X5?")],
        phone_hash="scenarioc" * 7, intent=None,
        player_profile={
            "nivel_jogo": "intermediário", "lesoes": "nenhuma",
            "regiao_lesao": "nenhuma", "esporte_raquete_previo": "nao_aplicavel",
            "modelo_desejado": "BeachPro Carbon X5",
        },
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Andre",
    )
    candidates = [_product("Raquete BeachPro Carbon X5")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": {}}),
            json.dumps({"messages": [
                "Sim, temos a *Raquete BeachPro Carbon X5* aqui, Andre!",
                "Quer saber preço, peso e indicação, ou já fechamos?",
            ]}),
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as s:
            s.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db):
                r_c = await graph.ainvoke(state_c, {"configurable": {"thread_id": "t-c"}})
    show("C", r_c)
    print(f"recommended_products={[p['name'] for p in r_c.get('recommended_products') or []]}")
    print(f"produto_pesquisado={r_c.get('produto_pesquisado')!r}")
    print(f"last_recommendation_at set: {bool(r_c.get('last_recommendation_at'))}")

    # ── SCENARIO D: product_selection → purchase_closing handoff ─────────
    print("\n##### SCENARIO D: product_selection -> purchase_closing handoff #####")
    state_d = dict(r_c)
    state_d["messages"] = [HumanMessage(content="quero fechar com a Carbon X5")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "product_selection"}']
        with patch(
            "app.agent.nodes.product_selection.persist_dossier",
            new_callable=AsyncMock,
        ) as persist:
            r_d = await graph.ainvoke(state_d, {"configurable": {"thread_id": "t-c"}})
    show("D", r_d)
    sp = r_d.get("selected_product") or {}
    print(f"selected_product={sp.get('name')!r}")
    print(f"persist_dossier called: {persist.called}")
    if persist.called:
        dossier = persist.call_args.args[1]
        print(
            f"dossier.needs_handoff_reason={dossier.get('needs_handoff_reason')!r} "
            f"dossier.produto_escolhido={dossier.get('produto_escolhido')!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
