"""Seed the post-recommendation state and fire one follow-up message.

Walks the agent through diagnose + recommend first so the checkpoint has real
recommended_products + last_recommendation_at, then sends a single follow-up
message and prints the result.

Usage:
    python scripts/_post_rec_check.py <phone> "<follow_up_message>"
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
from app.agent.message_splitter import compute_typing_delay
from app.security.pii_masker import hash_phone


async def _run_turn(graph, phone_hash, msg):
    return await graph.ainvoke(
        {
            "messages": [HumanMessage(content=msg)],
            "phone_hash": phone_hash,
            "needs_handoff": False,
            "handoff_reason": None,
        },
        config={"configurable": {"thread_id": phone_hash}},
    )


async def main() -> None:
    phone = sys.argv[1]
    follow_up = sys.argv[2]
    phone_hash = hash_phone(phone)

    await init_checkpointer()
    graph = build_graph()

    # Walk through a full diagnose so the checkpoint reaches post-recommendation state.
    print("== Seeding diagnose + recommend ==")
    seed_turns = [
        "quero uma raquete",
        "intermediário",
        "não, sem lesão",
        "não tenho",
    ]
    for s in seed_turns:
        await _run_turn(graph, phone_hash, s)

    # Fire the follow-up message.
    print(f"\n== Follow-up: {follow_up!r} ==\n")
    result = await _run_turn(graph, phone_hash, follow_up)

    print(f"INTENT      : {result.get('intent')}")
    print(f"PROFILE     : {result.get('player_profile')}")
    print(f"NEEDS_HANDOFF: {result.get('needs_handoff')}")
    if result.get("selected_product"):
        print(f"SELECTED    : {result['selected_product']['name']}")
    if result.get("recommended_products"):
        names = [p.get("name") for p in result["recommended_products"]]
        print(f"REC_PRODUCTS: {names}")

    for m in reversed(result.get("messages") or []):
        if isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            print(f"\nAI_REPLY    :\n{content}")
            break

    blocks = result.get("response_blocks") or []
    if blocks:
        print(f"\nBLOCKS_COUNT: {len(blocks)}")
        for i, b in enumerate(blocks):
            delay = compute_typing_delay(b) if i > 0 else 0.0
            print(f"  block[{i}] len={len(b):4d}  delay={delay:.2f}s")

    await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
