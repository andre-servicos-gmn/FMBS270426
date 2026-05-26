"""Sprint 2.1.1 — smoke test for the 3 message variations.

Each variation should:
- log ``determined_check intent=diagnose matched=True``
- log ``intent rewrite: diagnose → recommend``
- produce a single-block REFERENCE-SIM confirmation (no diagnose questions)
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.agent.state import AgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(message)s",
)
for noisy in ("httpcore", "httpx", "openai", "langgraph"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


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


async def _run(label: str, message: str) -> None:
    print(f"\n##### {label} #####")
    graph = build_graph(checkpointer=MemorySaver())
    init = AgentState(
        messages=[HumanMessage(content=message)],
        phone_hash="smoke211" * 8,
        intent=None, player_profile={},
        recommended_products=[], needs_handoff=False, handoff_reason=None,
        consultoria_interest=False, customer_name="Marcelo",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']  # only triage runs LLM
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = [_product("Raquete BeachPro Carbon X5")]
            with patch("app.storage.db.get_session", _mock_db):
                r = await graph.ainvoke(init, {"configurable": {"thread_id": label}})

    last = next(
        (m for m in reversed(r.get("messages") or []) if isinstance(m, AIMessage)),
        None,
    )
    text = last.content if last else "(no message)"
    print(
        f"\nFINAL: intent={r.get('intent')} "
        f"path={r.get('customer_intent_path')!r} "
        f"products={[p['name'] for p in r.get('recommended_products') or []]}"
    )
    print(f"REPLY:\n{text}")
    print(f"LLM calls: {llm.call_count} (expected 1)")
    asks_level = "nível" in text.lower() or "nivel" in text.lower()
    print(f"asks level? {'YES (bug!)' if asks_level else 'no (good)'}")


async def main() -> None:
    await _run("ORIGINAL: vocês tem a beach pro carbon x5?", "vocês tem a beach pro carbon x5?")
    await _run("VAR 1: queria a beachpro carbon x5", "queria a beachpro carbon x5")
    await _run("VAR 2: oi, vocês têm a Carbon X5?", "oi, vocês têm a Carbon X5?")


if __name__ == "__main__":
    asyncio.run(main())
