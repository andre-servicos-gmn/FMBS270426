"""Sprint 1.10 / 2.4 / 2.5 / 2.6.4 — price inquiry node.

Sprint 2.6.4 additions:
- When ``state.last_product_candidates`` is populated (set by recommend
  on an ambiguous match), the node detects multi-product references like
  "as duas", "ambas", "os três" and quotes the price of EVERY candidate.
- Otherwise it falls back to the Sprint 2.5 single-product path with
  stock check + subtle Consultoria pitch.

The node remains LLM-free.
"""
import logging
import re
import unicodedata

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes._pitch_classification import QuestionType
from app.agent.nodes._product_match import format_price_brl, match_product_in_text
from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


# Sprint 2.6.4 — pronouns the customer uses to reference a whole group
# (e.g. "os dois", "as duas", "ambas", "todas"). Detected accent-insensitive.
_MULTI_REF_PATTERNS = (
    "as duas", "os dois", "as tres", "os tres", "as três", "os três",
    "ambas", "ambos", "todas", "todos",
    "os 2", "as 2", "os 3", "as 3",
    "todas elas", "todos eles", "as opcoes", "as opções",
)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def is_multi_product_reference(text: str) -> bool:
    """True when the customer references multiple products at once."""
    if not text:
        return False
    norm = _strip_accents(text).lower()
    return any(p in norm for p in _MULTI_REF_PATTERNS)


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
    candidates = state.get("last_product_candidates") or []
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    # ── Sprint 2.6.4 — multi-product reference path ────────────────────────
    # Customer said "as duas" / "ambas" / "todas" AND we have candidates
    # from the previous ambiguous turn → quote price of every candidate.
    if candidates and len(candidates) >= 2 and is_multi_product_reference(user_text):
        lines = [
            f"A *{c.get('name')}* sai por {format_price_brl(c.get('price_cents'))}."
            for c in candidates
        ]
        blocks = ["\n".join(lines), "Posso tirar mais alguma dúvida?"]
        logger.info(
            "price_inquiry multi_candidates_reference n=%d",
            len(candidates),
        )
        joined = "\n\n".join(blocks)
        return {
            "messages": [AIMessage(content=joined)],
            "response_blocks": blocks,
        }

    matched = match_product_in_text(user_text, products)
    target = matched or (products[0] if len(products) == 1 else None)

    pitch_update: dict = {}

    if matched is not None or (products and len(products) == 1):
        name = target.get("name", "esse produto")
        price = format_price_brl(target.get("price_cents"))
        blocks = [f"A *{name}* sai por {price}.", "Posso tirar mais alguma dúvida?"]

        stock_note = await _maybe_stock_note(target)
        if stock_note:
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
        # Sprint 2.6.4 — fall back to candidates when no shortlist confirmed.
        if candidates:
            target = candidates[0]
            name = target.get("name", "esse produto")
            price = format_price_brl(target.get("price_cents"))
            blocks = [
                f"A *{name}* sai por {price}.",
                "Posso tirar mais alguma dúvida?",
            ]
            logger.info("price_inquiry single_candidate_fallback name=%s", name)
        else:
            blocks = [
                "Pode me dizer qual produto te interessa? Aí te passo o preço."
            ]
            logger.warning("price_inquiry called with empty recommended_products + candidates")

    joined = "\n\n".join(blocks)
    return {
        "messages": [AIMessage(content=joined)],
        "response_blocks": blocks,
        **pitch_update,
    }
