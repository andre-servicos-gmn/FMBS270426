"""pitch_consultoria_node — presents the Consultoria Base Esportes to the customer.

Triggered when triage classifies the message as ``consultoria`` (customer asked
about price, how it works, or expressed interest in booking). The node uses an
LLM call with SYSTEM_PITCH_CONSULTORIA — the price is injected from settings —
and flips ``consultoria_interest=True`` in state so downstream steps can
remember the lead without storing booking details (booking is out of scope).

When ``consultoria_enabled`` is False the node short-circuits with a brief
"not offered here" reply and does NOT flip the interest flag.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.anti_rerun import (
    fallback_message_for,
    should_block_rerun,
    stamp_node_execution,
)
from app.agent.message_splitter import parse_messages
from app.agent.prompts import build_pitch_consultoria_prompt
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


async def pitch_consultoria_node(state: AgentState) -> dict:
    settings = get_settings()
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    last_msg = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    if not settings.consultoria_enabled:
        logger.info("pitch_consultoria skipped — consultoria_enabled=False")
        canned = "No momento esta unidade não está com a Consultoria disponível, mas posso te ajudar com indicação por aqui mesmo. 😉"
        return {
            "messages": [AIMessage(content=canned)],
            "response_blocks": [canned],
            **stamp_node_execution("pitch_consultoria"),
        }

    # Sprint 1.14 — anti-rerun guard. If we just pitched the Consultoria and
    # the customer's next message is short / vague, fallback contextual reply
    # instead of repeating the same 3-block pitch.
    if should_block_rerun(state, "pitch_consultoria", last_msg):
        fallback = fallback_message_for("pitch_consultoria")
        logger.info(
            "pitch_consultoria_blocked_by_anti_rerun phone_hash=%.8s",
            (state.get("phone_hash") or "")[:8],
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "response_blocks": [fallback],
            **stamp_node_execution("pitch_consultoria"),
        }

    system = build_pitch_consultoria_prompt(settings)
    user_text = last_msg or "Conte sobre a consultoria."

    customer_name = state.get("customer_name")
    user_block = (
        f"Nome do cliente: {customer_name}\n\n{user_text}"
        if customer_name else user_text
    )

    client = OpenAIClient()
    response = await client.chat(
        messages=[{"role": "user", "content": user_block}],
        system=system,
        max_tokens=600,
        temperature=0.6,
        json_mode=True,
    )

    blocks = parse_messages(response)
    if not blocks:
        # Defensive — never let the customer get an empty reply.
        blocks = [
            "Temos uma consultoria personalizada que te ajuda a escolher a "
            "raquete certa, com teste em quadra. Quer agendar?"
        ]

    joined = "\n\n".join(blocks)

    logger.info(
        "pitch_consultoria done preco=%d blocks=%d",
        settings.consultoria_preco, len(blocks),
    )

    return {
        "messages": [AIMessage(content=joined)],
        "response_blocks": blocks,
        "consultoria_interest": True,
        **stamp_node_execution("pitch_consultoria"),
    }
