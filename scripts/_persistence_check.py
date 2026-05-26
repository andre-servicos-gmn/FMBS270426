"""One-shot graph invocation for testing checkpoint persistence across restarts.

Usage:
    python scripts/_persistence_check.py <phone> <message>

Each run is a fresh Python process — if state survives across calls with the
same <phone>, it proves the checkpointer is persisting to Redis (not memory).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_graph
from app.security.pii_masker import hash_phone


async def main() -> None:
    phone = sys.argv[1]
    msg = sys.argv[2]
    phone_hash = hash_phone(phone)

    await init_checkpointer()
    graph = build_graph()
    config = {"configurable": {"thread_id": phone_hash}}

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=msg)],
            "phone_hash": phone_hash,
            "needs_handoff": False,
            "handoff_reason": None,
        },
        config=config,
    )

    print(f"--- PID={__import__('os').getpid()} thread_id={phone_hash[:12]} ---")
    print(f"INTENT      : {result.get('intent')}")
    print(f"PROFILE     : {result.get('player_profile')}")
    print(f"MSGS_TOTAL  : {len(result.get('messages') or [])}")
    for m in reversed(result.get("messages") or []):
        if isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            print(f"AI_REPLY    : {content}")
            break

    # Sprint 1.6 — surface response_blocks + simulated delays when present.
    blocks = result.get("response_blocks") or []
    if blocks:
        from app.agent.message_splitter import compute_typing_delay
        print(f"BLOCKS_COUNT: {len(blocks)}")
        for i, b in enumerate(blocks):
            delay = compute_typing_delay(b) if i > 0 else 0.0
            print(f"  block[{i}] len={len(b):4d}  delay_before={delay:.2f}s  text={b[:120]}{'…' if len(b)>120 else ''}")

    await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
