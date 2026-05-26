"""Sprint 1.10 / 2.4 / 2.5 — price inquiry node.

Sprint 2.5 additions:
- When the active product comes from the Bling catalog, we attach
  ``is_raquete`` so the subtle Consultoria pitch only fires for racket
  products (non-rackets receive a clean price answer).
- We also check the real-time stock via the cached helper; if the product
  is out of stock, we replace the prompt-for-next-step with a friendly
  "tô confirmando, mas no momento parece estar em falta" message.

The node remains LLM-free.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes._pitch_classification import QuestionType
from app.agent.nodes._product_match import format_price_brl, match_product_in_text
from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


def _is_raquete(product: dict | None) -> bool:
    """Default True for legacy / local-catalog products (no Bling field)."""
    if not product:
        return True
    if "is_raquete_praia" in product:
        return bool(product["is_raquete_praia"])
    return True


async def _maybe_stock_note(product: dict | None) -> str | None:
    """Sprint 2.5 — when Bling is live, check real-time stock."""
    settings = get_settings()
    if not (settings.bling_client_id and product):
        return None
    produto_id = product.get("id")
    if not isinstance(produto_id, int):
        return None
    try:
        from app.sync.bling_stock import get_stock
        saldo = await get_stock(produto_id)
    except Exception as exc:
        logger.warning("price_inquiry stock_check_failed id=%s: %s", produto_id, exc)
        return None
    if saldo is not None and saldo <= 0:
        return (
            "Só uma observação: tá em falta no momento. Se quiser, "
            "posso te avisar quando voltar."
        )
    return None


async def price_inquiry_node(state: AgentState) -> dict:
    products = state.get("recommended_products") or []
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    matched = match_product_in_text(user_text, products)
    target = matched or (products[0] if len(products) == 1 else None)

    pitch_update: dict = {}

    if matched is not None or (products and len(products) == 1):
        name = target.get("name", "essa raquete")
        price = format_price_brl(target.get("price_cents"))
        blocks = [f"A *{name}* sai por {price}.", "Posso tirar mais alguma dúvida?"]

        stock_note = await _maybe_stock_note(target)
        if stock_note:
            # When out of stock, swap the prompt-for-next-step for the heads-up.
            blocks = [f"A *{name}* sai por {price}.", stock_note]

        blocks, pitch_update = maybe_add_subtle_consultoria_offer(
            state, blocks, QuestionType.PRICE, is_raquete=_is_raquete(target),
        )
        logger.info(
            "price_inquiry matched product=%s is_raquete=%s",
            name, _is_raquete(target),
        )
    elif products:
        lines = [
            f"• *{p.get('name')}* — {format_price_brl(p.get('price_cents'))}"
            for p in products
        ]
        blocks = [
            "Da seleção que te passei:\n" + "\n".join(lines),
            "Posso tirar mais alguma dúvida?",
        ]
        logger.info("price_inquiry ambiguous listed=%d", len(products))
    else:
        blocks = [
            "Posso te indicar os preços assim que te apresentar as opções. "
            "Quer que eu te ajude a encontrar uma raquete?"
        ]
        logger.warning("price_inquiry called with empty recommended_products")

    joined = "\n\n".join(blocks)
    return {
        "messages": [AIMessage(content=joined)],
        "response_blocks": blocks,
        **pitch_update,
    }
