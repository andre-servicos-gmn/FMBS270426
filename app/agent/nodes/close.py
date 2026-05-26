import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.message_splitter import parse_messages
from app.agent.prompts import build_close_prompt
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


async def close_node(state: AgentState) -> dict:
    messages = state.get("messages") or []
    products = state.get("recommended_products") or []
    selected = state.get("selected_product")
    customer_name = state.get("customer_name")
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    products_ctx = "\n".join(
        f"- {p['name']}: R${int(p['price_cents']) / 100:.0f}"
        for p in products
    ) or "(produto não identificado)"

    last_msg = last_human.content if last_human else ""

    # Sprint 1.10: if product_selection ran and identified the chosen product,
    # surface it explicitly to the LLM so it doesn't have to re-infer.
    selected_line = ""
    if selected and isinstance(selected, dict) and selected.get("name"):
        selected_line = f"\nProduto escolhido pelo cliente: {selected['name']}"

    name_line = f"Nome do cliente: {customer_name}\n" if customer_name else ""
    context = (
        f"{name_line}"
        f"Produtos que foram apresentados ao cliente:\n{products_ctx}"
        f"{selected_line}\n\n"
        f"Última mensagem do cliente: {last_msg}"
    )

    system = build_close_prompt(get_settings())

    client = OpenAIClient()
    response = await client.chat(
        messages=[{"role": "user", "content": context}],
        system=system,
        max_tokens=200,
        temperature=0.6,
    )

    # Sprint 1.16 — return response_blocks explicitly. Without this, the
    # webhook would fall back to a prior turn's blocks (e.g. recommend) that
    # LangGraph's state merge had carried over from the checkpoint.
    blocks = parse_messages(response) or [response.strip() or ""]

    logger.info("close_node done blocks=%d", len(blocks))

    return {
        "messages": [AIMessage(content=response)],
        "response_blocks": blocks,
    }
