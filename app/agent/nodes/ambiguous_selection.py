"""Sprint 1.15 — disambiguation node.

Fires when the customer made a pronominal reference ("gostei dessa") but
multiple products are on the table. We ask them to clarify, listing the
options as bullets. NO LLM call — deterministic so the canned phrasing
stays consistent.

The node leaves ``recommended_products`` untouched and does NOT mark a
``selected_product``. The next customer turn will go through triage again
and ideally hit product_selection with a clearer reference.
"""
import logging

from langchain_core.messages import AIMessage

from app.agent.state import AgentState

logger = logging.getLogger(__name__)


_PROMPT = (
    "Qual delas? Você pode me dizer o nome ou só a posição — "
    "'a primeira' ou 'a segunda', por exemplo."
)


async def ambiguous_selection_node(state: AgentState) -> dict:
    products = state.get("recommended_products") or []
    options = "\n".join(f"• *{p.get('name', '?')}*" for p in products) or "(sem opções)"
    msg = f"{_PROMPT}\n\n{options}"

    logger.info(
        "ambiguous_selection phone_hash=%.8s options=%d",
        (state.get("phone_hash") or "")[:8], len(products),
    )

    return {
        "messages": [AIMessage(content=msg)],
        "response_blocks": [msg],
    }
