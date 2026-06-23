"""Sprint 3.3 replay — the price/category turns that failed in production.

Replays, through the REAL V2 supervisor (live OpenAI loop) against the LIVE
catalog (Supabase/Bling), the exact turns Felipe reported as broken:

    1. "vocês tem raquetes de beach tennis abaixo de mil reais?"
    2. "quero as mais baratas"
    3. "tem até 1500?"
    4. comparison still works: "compara a mormaii sunset e a macaw"

Prints, per turn: the tool calls (so we can see categoria/preco_max/ordenacao
were passed), the tool results, and the final agent answer. Also asserts the
final answers never end with the banned fixed closing line.

Run:
    .venv/Scripts/python scripts/replay_preco_categoria.py
"""
import asyncio
import os

os.environ.setdefault("PYTHONUTF8", "1")

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

from app.agent.supervisor import build_supervisor_graph  # noqa: E402

THREAD_ID = "replay-preco-categoria-1"
PHONE_HASH = "preco_categoria_test_hash"

TURNS = [
    # The EXACT production turns that failed (agent answered "não tem" with
    # tool_calls=0, hallucinating unavailability).
    "tem raquetes até 1k?",
    "e até 2 mil reais?",
    # Original price/category turns.
    "quero as mais baratas",
    "compara a mormaii sunset e a macaw",
]

# The fixed closing line that must NEVER appear (Felipe's complaint).
BANNED = "se precisar de mais informações ou ajuda, é só avisar"


def _short(content, n=700):
    s = content if isinstance(content, str) else str(content)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


async def main() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_supervisor_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": THREAD_ID}}

    banned_hits: list[int] = []

    for i, user_msg in enumerate(TURNS, start=1):
        print("\n" + "=" * 78)
        print(f"TURN {i}  CLIENTE: {user_msg}")
        print("-" * 78)

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_msg)],
                "phone_hash": PHONE_HASH,
                "thread_id": THREAD_ID,
            },
            config=cfg,
        )

        msgs = result["messages"]
        start = 0
        for idx in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[idx], HumanMessage):
                start = idx
                break

        final_text = ""
        for m in msgs[start:]:
            if isinstance(m, AIMessage):
                for tc in (getattr(m, "tool_calls", None) or []):
                    print(f"  [tool_call] {tc['name']}({tc.get('args')})")
                if m.content:
                    final_text = m.content if isinstance(m.content, str) else str(m.content)
            elif isinstance(m, ToolMessage):
                print(f"  [tool_result] {getattr(m, 'name', '?')}: {_short(m.content, 320)}")

        print("-" * 78)
        print(f"  AGENTE: {_short(final_text, 900)}")
        if BANNED in final_text.lower():
            banned_hits.append(i)

    print("\n" + "=" * 78)
    if banned_hits:
        print(f"FALHA: frase de fechamento fixa apareceu nos turnos {banned_hits}")
    else:
        print("OK: nenhuma resposta terminou com a frase de fechamento fixa")


if __name__ == "__main__":
    asyncio.run(main())
