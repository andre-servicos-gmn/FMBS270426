import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.message_splitter import parse_messages
from app.agent.prompts import build_faq_prompt
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_HANDOFF_MARKER = "[HANDOFF]"


async def faq_node(state: AgentState) -> dict:
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_human:
        # Sprint 1.16 — even on fallback, populate response_blocks so the
        # webhook never inherits stale blocks from a previous turn.
        fallback = "Como posso te ajudar?"
        return {
            "messages": [AIMessage(content=fallback)],
            "response_blocks": [fallback],
            "needs_handoff": False,
        }

    # RAG: retrieve relevant KB documents before answering
    kb_docs: list[dict] = []
    try:
        from app.rag.retriever import search_knowledge_base
        from app.storage.db import get_session

        async with get_session() as session:
            kb_docs = await search_knowledge_base(session, last_human.content, k=4)
        logger.info("faq_rag retrieved=%d", len(kb_docs))
    except Exception as exc:
        logger.warning("faq_rag_failed (proceeding without context): %s", exc)

    system = build_faq_prompt(kb_docs)

    customer_name = state.get("customer_name")
    raw_msg = last_human.content if isinstance(last_human.content, str) else str(last_human.content)
    user_block = (
        f"Nome do cliente: {customer_name}\n\n{raw_msg}"
        if customer_name else raw_msg
    )

    client = OpenAIClient()
    response = await client.chat(
        messages=[{"role": "user", "content": user_block}],
        system=system,
        max_tokens=512,
        temperature=0.3,
    )

    needs_handoff = _HANDOFF_MARKER in response
    clean = response.replace(_HANDOFF_MARKER, "").strip()

    # Sprint 1.16 — populate response_blocks to prevent stale-block leakage.
    blocks = parse_messages(clean) or [clean]

    result: dict = {
        "messages": [AIMessage(content=clean)],
        "response_blocks": blocks,
        "needs_handoff": needs_handoff,
    }
    if needs_handoff:
        result["handoff_reason"] = "faq_escalation"
        logger.info("faq_handoff_triggered")

    return result
