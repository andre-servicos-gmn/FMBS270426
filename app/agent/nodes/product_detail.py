"""Sprint 1.10 — answer technical questions about already-recommended products.

The node tries to identify (a) WHICH product the customer is asking about
and (b) WHICH attribute. Both are best-effort heuristics; when ambiguous we
fall back gracefully.

When we can answer: emit a 1-2 block response with the technical info pulled
from the product's structured fields + description, plus a 1-sentence
contextual nudge.
When we cannot: honest fallback that points to the in-store Consultoria.

No LLM call — keeps the answer deterministic and prevents the LLM from
inventing specs that aren't in the catalog.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes._pitch_classification import classify_question
from app.agent.nodes._product_match import match_product_in_text, normalize
from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# (attribute_label, list_of_synonyms_or_substrings, field_path_or_None_for_description_search)
_ATTRIBUTE_QUERIES: list[tuple[str, list[str], str | None]] = [
    ("peso", ["peso", "pesa", "gramas", "leve", "pesada", "pesado"], "weight_g"),
    ("material", ["material", "feito", "feita", "composicao", "composição"], "material"),
    ("balance", ["balance", "balanco", "balanço", "ponto de equilibrio"], "balance"),
    ("nivel", ["nivel", "level", "iniciante", "intermediario", "avancado"], "level"),
    # Description-only attributes (no structured field — search in description text)
    ("antivibracao", ["antivibracao", "antivibração", "vibracao", "vibração"], None),
    ("flexibilidade", ["flexibilidade", "flexivel", "flexível", "rigidez", "rigida"], None),
    ("formato", ["formato", "diamante", "lagrima", "lágrima", "redondo", "shape"], None),
]


def _format_attribute_value(label: str, value) -> str:
    """Render a structured attribute value into a customer-friendly string."""
    if value is None or value == "":
        return ""
    if label == "peso":
        try:
            grams = int(value)
            return f"peso aproximado de {grams}g"
        except (TypeError, ValueError):
            return f"peso {value}"
    if label == "material":
        return f"material: {value}"
    if label == "balance":
        return f"balance {value}"
    if label == "nivel":
        return f"voltada para perfil {value}"
    return f"{label}: {value}"


def _find_description_snippet(description: str, keywords: list[str]) -> str | None:
    """Return the sentence in description that mentions any of the keywords."""
    if not description:
        return None
    desc_norm = normalize(description)
    for kw in keywords:
        if normalize(kw) in desc_norm:
            # Find the original sentence containing this keyword (rough)
            for sentence in description.replace("\n", ". ").split("."):
                if normalize(sentence).find(normalize(kw)) >= 0:
                    return sentence.strip()
    return None


def _identify_attribute(text: str) -> tuple[str, list[str], str | None] | None:
    """Return (label, keywords, field) for the first attribute matched in text."""
    text_norm = normalize(text)
    for label, keywords, field in _ATTRIBUTE_QUERIES:
        for kw in keywords:
            if normalize(kw) in text_norm:
                return label, keywords, field
    return None


async def product_detail_node(state: AgentState) -> dict:
    products = state.get("recommended_products") or []
    selected = state.get("selected_product")
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    # Pick the product: explicit name match wins; otherwise fall back to
    # the most recently selected product, or the first recommended.
    target = match_product_in_text(user_text, products)
    if target is None and selected:
        target = selected
    if target is None and products:
        target = products[0]

    attribute = _identify_attribute(user_text)

    if target is None:
        blocks = [
            "Posso te passar detalhes técnicos das raquetes que te indiquei. "
            "Qual delas você quer saber mais?"
        ]
        logger.info("product_detail no_target_and_no_products")
    elif attribute is None:
        # Customer asked something vague about the product. Pull a short summary.
        name = target.get("name", "essa raquete")
        desc = (target.get("description") or "").strip()
        if desc:
            snippet = desc[:240]
            blocks = [f"*{name}*\n\n{snippet}"]
        else:
            blocks = [
                f"Sobre a *{name}* eu não tenho um descritivo completo aqui. "
                "Vale conferir direto na loja ou na *Consultoria Base Sports*."
            ]
        logger.info("product_detail vague target=%s", name)
    else:
        label, keywords, field = attribute
        name = target.get("name", "essa raquete")
        value_part: str | None = None

        if field is not None:
            field_value = target.get(field)
            if field_value not in (None, ""):
                value_part = _format_attribute_value(label, field_value)

        if value_part is None:
            # Try the description text as a fallback.
            snippet = _find_description_snippet(target.get("description") or "", keywords)
            if snippet:
                value_part = snippet

        if value_part:
            blocks = [
                f"*{name}* — {value_part}.",
                "_Vale conferir mais detalhes em quadra na Consultoria Base Sports._",
            ]
            logger.info("product_detail answered target=%s attr=%s", name, label)
        else:
            blocks = [
                f"Sobre o {label} da *{name}* eu não tenho esse detalhe específico aqui.",
                "Vale conferir direto na loja ou na *Consultoria Base Sports*, que aprofunda essas características pra ti.",
            ]
            logger.info("product_detail missing target=%s attr=%s", name, label)

    # Sprint 2.1/2.3 — subtle Consultoria pitch for determined customers,
    # capped at one mention per conversation. The preset + timing depend on
    # the question's type — WEIGHT/MATERIAL/BALANCE/OTHER are DELAYED (only
    # fire from the 2nd determined question on); COMFORT fires immediately.
    # Sprint 2.5 — pitch is suppressed for non-racket products (e.g. balls,
    # apparel) via the ``is_raquete`` flag derived from the active product.
    question_type = classify_question(user_text)
    is_raquete = True
    if target is not None and "is_raquete_praia" in target:
        is_raquete = bool(target["is_raquete_praia"])
    blocks, pitch_update = maybe_add_subtle_consultoria_offer(
        state, blocks, question_type, is_raquete=is_raquete,
    )

    joined = "\n\n".join(blocks)
    return {
        "messages": [AIMessage(content=joined)],
        "response_blocks": blocks,
        **pitch_update,
    }
