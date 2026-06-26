"""Multi-scenario smoke for the V2 supervisor, runnable on any model.

Runs several INDEPENDENT conversations (each in its own thread, so context
doesn't bleed) covering the project's critical behaviors, and prints every
agent reply + tool call. Use it to compare gpt-4o-mini vs gpt-5-mini on the
SAME prompt before deciding to migrate.

Usage (project root, venv active):
    python scripts/smoke_matrix.py --model gpt-4o-mini
    python scripts/smoke_matrix.py --model gpt-5-mini

Real OpenAI API; in-memory checkpointer (no Redis); no WhatsApp; no Postgres.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import app.agent.supervisor as sup
from app.agent.supervisor import build_supervisor_graph

# Each scenario is (name, [client turns]). Each runs in its own thread.
SCENARIOS = [
    ("consultivo_orcamento", [
        "Quero ver umas raquetes até 2 mil reais",
        "pode ser Drop Shot",
    ]),
    ("pedido_especifico", [
        "vc tem a Drop Shot Excalibur Pro?",
    ]),
    ("comparacao", [
        "qual a diferença entre a Excalibur Pro e a Sexy Sirf?",
    ]),
    ("nivel_de_jogo", [
        "sou avançado, qual a melhor raquete pra mim?",
    ]),
    ("faixa_vazia", [
        "tem raquete de beach tennis abaixo de 300 reais?",
    ]),
    ("faq_institucional", [
        "qual o endereço e horário da loja?",
    ]),
    ("mais_opcoes", [
        "me mostra raquetes de beach tennis",
        "tem mais opções?",
    ]),
]


def _patch_model(model: str) -> None:
    orig_build = sup._build_openai_request

    def patched(state, settings, *, force_search):
        req = orig_build(state, settings, force_search=force_search)
        req["model"] = model
        if model.startswith("gpt-5"):
            req.pop("temperature", None)
            if "max_tokens" in req:
                req["max_completion_tokens"] = max(req.pop("max_tokens"), 4000)
        return req

    sup._build_openai_request = patched


async def run_scenario(name: str, turns: list[str]) -> None:
    graph = build_supervisor_graph(MemorySaver())
    config = {"configurable": {"thread_id": f"matrix-{name}"}}
    print(f"\n{'='*70}\nCENÁRIO: {name}\n{'='*70}")
    for text in turns:
        result = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
        reply = ""
        for m in reversed(result.get("messages") or []):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                reply = m.content
                break
        calls = [
            f"{tc['name']}({tc.get('args')})"
            for m in (result.get("messages") or [])
            for tc in (getattr(m, "tool_calls", None) or [])
        ]
        print(f"\nCLIENTE: {text}")
        for c in calls:
            print(f"  TOOL : {c}")
        print(f"AGENTE : {reply}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--only", default=None, help="run only scenarios whose name contains this")
    args = parser.parse_args()
    if args.model:
        _patch_model(args.model)
        print(f"(modelo do supervisor: {args.model})")
    for name, turns in SCENARIOS:
        if args.only and args.only not in name:
            continue
        try:
            await run_scenario(name, turns)
        except Exception as exc:
            print(f"\nCENÁRIO {name} FALHOU: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
