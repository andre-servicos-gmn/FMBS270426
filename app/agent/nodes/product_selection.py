"""Sprint 2.4 — product_selection emits a short pickup invite (no handoff).

The Sprint 2.0 design treated explicit purchase intent as a HANDOFF event
because we assumed the human attendant always had to close the sale. Real
usage on the pilot showed this is unnecessary friction for customers who
already know the racket they want: the customer arrived from the store's
WhatsApp number, so they already know where the store is. A short invite
("vem retirar, te esperamos, qualquer dúvida me chama") closes the loop
without forcing a human handoff.

Behaviour:
- Resolve which displayed product the customer chose (Sprint 1.15
  tolerant-name / positional / pronominal resolution — unchanged).
- Emit one of 4 randomized pickup-invite variations.
- Do NOT trigger handoff, do NOT persist or send a dossier.
- The customer stays in the conversation; if they later ask for a human,
  ``handoff_node`` (via the explicit ``handoff`` intent) handles it.
"""
import logging
import random

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes._positional_reference import detect_positional_reference
from app.agent.nodes._product_match import match_product_tolerant
from app.agent.nodes._pronominal_reference import detect_pronominal_reference
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# Each variation MUST end with the "porta aberta" phrase ("qualquer dúvida me
# chama") so the customer never feels closed off. ``test_purchase_existing_
# ends_with_door_open_phrase`` enforces this invariant.
_PICKUP_VARIATIONS: tuple[str, ...] = (
    "Show! Pode passar aqui pra retirar a sua *{nome}*. Te esperamos! "
    "Qualquer dúvida, me chama.",
    "Bora! A *{nome}* tá separada esperando você. Quando vier, é só "
    "chegar. Qualquer dúvida, me chama.",
    "Demais! A *{nome}* tá te aguardando aqui. Quando for passar, "
    "qualquer coisa me chama.",
    "Fechou! Pode vir buscar sua *{nome}*. Te esperamos! Qualquer "
    "dúvida, me chama.",
)


def get_pickup_message(product_name: str) -> str:
    """Return a randomly chosen pickup-invite text for ``product_name``."""
    template = random.choice(_PICKUP_VARIATIONS)
    return template.format(nome=product_name)


def _resolve_selected(products: list[dict], user_text: str) -> dict | None:
    """Return the chosen product (Sprint 1.15 resolution order)."""
    match = match_product_tolerant(user_text, products)
    if match.product is not None:
        logger.info(
            "product_selection resolved=name matched=%s method=%s confidence=%s",
            match.product.get("name"), match.method, match.confidence,
        )
        return match.product

    idx = detect_positional_reference(user_text, len(products))
    if idx is not None:
        logger.info(
            "product_selection resolved=positional idx=%d name=%s",
            idx, products[idx].get("name"),
        )
        return products[idx]

    if len(products) == 1 and detect_pronominal_reference(user_text):
        logger.info(
            "product_selection resolved=pronominal_single name=%s",
            products[0].get("name"),
        )
        return products[0]

    logger.info("product_selection no_explicit_match — defaulting to first product")
    return products[0] if products else None


async def product_selection_node(state: AgentState) -> dict:
    products = state.get("recommended_products") or []
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    selected = _resolve_selected(products, user_text)

    # Defensive: if there's literally nothing to pick (post-rec state lost
    # the products somehow), bail with a friendly reset.
    if selected is None:
        fallback = (
            "Pra fechar a compra preciso saber qual raquete você quer. "
            "Me passa o nome de novo?"
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "response_blocks": [fallback],
            "selected_product": None,
        }

    product_name = selected.get("name", "essa raquete")
    pickup_text = get_pickup_message(product_name)

    logger.info("product_selection pickup_invite product=%s", product_name)
    return {
        "messages": [AIMessage(content=pickup_text)],
        "response_blocks": [pickup_text],
        "selected_product": selected,
    }
