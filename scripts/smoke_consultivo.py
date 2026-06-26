"""Smoke test (non-interactive) for the consultative-presentation change.

Drives the V2 supervisor graph through the exact conversation Andre flagged,
to verify the agent now ASKS a qualifying question (brand/model/budget) on a
broad racket request instead of dumping a list — without crossing the
Consultoria line (never asks about level/injury/playing time).

Runs against the REAL OpenAI API (needs OPENAI_API_KEY) but uses an in-memory
checkpointer (no Redis) and never touches Postgres. It does NOT send WhatsApp.

Usage (project root, venv active):
    python scripts/smoke_consultivo.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import app.agent.supervisor as sup
from app.agent.supervisor import build_supervisor_graph


def _patch_model(model: str) -> None:
    """Override the model the supervisor uses, adapting params for gpt-5*.

    gpt-5* are reasoning models: they reject custom ``temperature`` and use
    ``max_completion_tokens`` instead of ``max_tokens``. We wrap the request
    builder so the smoke can target gpt-5-mini WITHOUT touching production code.
    """
    orig_build = sup._build_openai_request

    def patched(state, settings, *, force_search):
        req = orig_build(state, settings, force_search=force_search)
        req["model"] = model
        if model.startswith("gpt-5"):
            req.pop("temperature", None)  # only default (1) allowed
            if "max_tokens" in req:
                req["max_completion_tokens"] = max(req.pop("max_tokens"), 4000)
        return req

    sup._build_openai_request = patched
    # The fence classifier also calls OpenAI with its own model read from
    # settings; leave it as-is (gpt-4o-mini) — we're testing the SUPERVISOR loop.

# The exact turns from the flagged conversation, in order.
TURNS = [
    "Quero ver umas raquetes até 2 mil reais",   # budget given → ASK about brand
    "pode ser Drop Shot",                         # brand given → show, SPREAD the range
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="override supervisor model (e.g. gpt-5-mini)")
    args = parser.parse_args()
    if args.model:
        _patch_model(args.model)
        print(f"(modelo do supervisor: {args.model})")

    graph = build_supervisor_graph(MemorySaver())
    config = {"configurable": {"thread_id": "smoke-consultivo"}}

    for i, text in enumerate(TURNS, 1):
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config=config,
        )
        reply = ""
        for m in reversed(result.get("messages") or []):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                reply = m.content
                break
        # Show every tool call the model made this turn (debug the args).
        calls = []
        for m in result.get("messages") or []:
            for tc in (getattr(m, "tool_calls", None) or []):
                calls.append(f"{tc['name']}({tc.get('args')})")
        print(f"\n[{i}] CLIENTE: {text}")
        for c in calls:
            print(f"    TOOL   : {c}")
        print(f"    AGENTE : {reply}")

    print("\n--- fim do smoke ---")


if __name__ == "__main__":
    asyncio.run(main())
