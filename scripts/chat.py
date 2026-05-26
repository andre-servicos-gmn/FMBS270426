"""Interactive terminal chat — test the agent without WhatsApp or curl.

Usage (from project root, with .venv activated):
    python scripts/chat.py
    python scripts/chat.py --phone 5511999990001  # custom phone number
"""
import asyncio
import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import configure_logging, get_settings
from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_graph
from app.security.audit_log import log_access
from app.security.pii_masker import hash_phone, mask_pii
from langchain_core.messages import AIMessage, HumanMessage

configure_logging(get_settings())
logger = logging.getLogger(__name__)


async def _upsert_lead(phone_hash: str) -> None:
    from sqlalchemy import func
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.storage.db import get_session
    from app.storage.models import Lead

    async with get_session() as session:
        stmt = (
            pg_insert(Lead)
            .values(id=uuid.uuid4(), phone_hash=phone_hash, profile={})
            .on_conflict_do_update(
                index_elements=["phone_hash"],
                set_={"last_interaction_at": func.now()},
            )
        )
        await session.execute(stmt)
        await session.commit()


async def _save_logs(phone_hash: str, user_text: str, ai_text: str) -> None:
    from app.storage.db import get_session
    from app.storage.models import ConversationLog

    async with get_session() as session:
        session.add(ConversationLog(
            id=uuid.uuid4(),
            phone_hash=phone_hash,
            message_role="user",
            content_masked=mask_pii(user_text),
        ))
        session.add(ConversationLog(
            id=uuid.uuid4(),
            phone_hash=phone_hash,
            message_role="assistant",
            content_masked=mask_pii(ai_text),
        ))
        await session.commit()


async def chat(phone: str) -> None:
    phone_hash = hash_phone(phone)
    await init_checkpointer()
    graph = build_graph()
    config = {"configurable": {"thread_id": phone_hash}}

    try:
        await _upsert_lead(phone_hash)
    except Exception as exc:
        logger.warning("lead_upsert_failed: %s", exc)

    print(f"\n{'='*55}")
    print("  Agente Beach Tennis / Padel — modo de teste local")
    print(f"{'='*55}")
    print("  Digite sua mensagem e pressione Enter.")
    print("  Comandos: 'sair' ou Ctrl+C para encerrar.")
    print(f"{'='*55}\n")

    try:
        while True:
            try:
                user_input = input("Voce: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEncerrando chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                print("Encerrando chat.")
                break

            state_update = {
                "messages": [HumanMessage(content=user_input)],
                "phone_hash": phone_hash,
                "needs_handoff": False,
                "handoff_reason": None,
            }

            print("Agente: ", end="", flush=True)
            try:
                result = await graph.ainvoke(state_update, config=config)
            except Exception as exc:
                print(f"[ERRO ao invocar o agente: {exc}]")
                continue

            ai_response = ""
            for m in reversed(result.get("messages") or []):
                if isinstance(m, AIMessage):
                    ai_response = m.content
                    break

            print(ai_response or "[sem resposta]")
            print()

            try:
                await _save_logs(phone_hash, user_input, ai_response)
                await log_access(actor="chat_script", action="process_message", target_hash=phone_hash)
            except Exception as exc:
                logger.warning("persistence_failed: %s", exc)
    finally:
        await close_checkpointer()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone", default="5511999990001", help="Numero de telefone simulado")
    args = parser.parse_args()
    asyncio.run(chat(args.phone))


if __name__ == "__main__":
    main()
