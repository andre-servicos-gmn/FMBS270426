"""Phase 0 smoke test for the V2 supervisor loop.

Invokes the supervisor graph with one hardcoded message and prints the full
message trace, so you can SEE the loop: HumanMessage → AIMessage(tool_calls)
→ ToolMessage(s) with the mock → final AIMessage(text).

Run:
    .venv/Scripts/python scripts/smoke_supervisor.py

This does NOT go through the webhook and does NOT touch the legacy graph. It
needs a real OpenAI key (reads OPENAI_API_KEY from .env) because the loop's
routing decision comes from the model's tool-calling. The tools themselves are
stubs returning mock data.

Checkpointer: tries the project's AsyncRedisSaver first (same as production);
if Redis is unreachable it falls back to an in-memory MemorySaver so the smoke
still exercises the loop. The fallback is printed so it's never silent.
"""
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from app.agent.supervisor import build_supervisor_graph  # noqa: E402


async def _make_checkpointer():
    """Prefer the real AsyncRedisSaver; fall back to MemorySaver for the smoke."""
    try:
        from app.agent.checkpointer import init_checkpointer
        saver = await init_checkpointer()
        print("[smoke] checkpointer = AsyncRedisSaver (Redis reachable)")
        return saver, "redis"
    except Exception as exc:  # noqa: BLE001 — smoke convenience
        from langgraph.checkpoint.memory import MemorySaver
        print(f"[smoke] Redis unavailable ({exc!r}); using MemorySaver fallback")
        return MemorySaver(), "memory"


async def main() -> None:
    checkpointer, kind = await _make_checkpointer()
    graph = build_supervisor_graph(checkpointer)

    cfg = {"configurable": {"thread_id": "smoke-1"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="qual a diferença entre a Kronos e a Proteo?")]},
        config=cfg,
    )

    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(
            type(m).__name__,
            "|",
            getattr(m, "content", ""),
            "|",
            getattr(m, "tool_calls", None),
        )

    if kind == "redis":
        from app.agent.checkpointer import close_checkpointer
        await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
