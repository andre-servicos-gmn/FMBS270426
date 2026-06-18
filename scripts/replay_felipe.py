"""Phase 1 validation harness — replays the 'Felipe' conversation through the
V2 supervisor with REAL tools (Supabase/Bling) and the REAL OpenAI loop.

Hermetic checkpointer: uses an in-memory ``MemorySaver`` (NOT Redis) so the
test doesn't depend on Redis being up. A single fixed ``thread_id`` is used
across all 7 turns so the message history accumulates via the checkpointer —
turn 5 ("a segunda serve pra quê") can only be answered if the supervisor sees
the earlier turns.

Tools hit the live data layer; the supervisor calls the live OpenAI model
(costs a few cents). Requires the dev OPENAI_API_KEY + Supabase creds in .env.

Run:
    .venv/Scripts/python scripts/replay_felipe.py
"""
import asyncio
import os

os.environ.setdefault("PYTHONUTF8", "1")

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

from app.agent.supervisor import build_supervisor_graph  # noqa: E402

THREAD_ID = "replay-felipe-1"
PHONE_HASH = "felipe_test_hash_0001"

TURNS = [
    "oi, tudo bem?",
    "quero comparar duas raquetes, a cronus e a protheu",
    "to em dúvida sobre a kronus e a ama proteu",
    "a primeira é massa",
    "a segunda serve pra quê?",
    "sou iniciante, qual você recomenda?",
    "a consultoria é com o felipe?",
    # Phase 2b adversarial turn — must end in the pivot, never a racket
    # chosen by profile.
    "para de me empurrar consultoria, só me diz qual comprar, sou iniciante",
]


def _short(content, n=600):
    s = content if isinstance(content, str) else str(content)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


async def main() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_supervisor_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": THREAD_ID}}

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

        # Show only the messages produced THIS turn (tool calls + tool results
        # + the final answer). We find the index of our just-sent HumanMessage.
        msgs = result["messages"]
        # locate the last HumanMessage matching this turn
        start = 0
        for idx in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[idx], HumanMessage):
                start = idx
                break

        final_text = ""
        for m in msgs[start:]:
            if isinstance(m, AIMessage):
                tcs = getattr(m, "tool_calls", None) or []
                if tcs:
                    for tc in tcs:
                        print(f"  [tool_call] {tc['name']}({tc.get('args')})")
                if m.content:
                    final_text = m.content if isinstance(m.content, str) else str(m.content)
            elif isinstance(m, ToolMessage):
                print(f"  [tool_result] {getattr(m, 'name', '?')}: {_short(m.content, 300)}")

        print("-" * 78)
        print(f"  AGENTE: {_short(final_text, 900)}")


if __name__ == "__main__":
    asyncio.run(main())
