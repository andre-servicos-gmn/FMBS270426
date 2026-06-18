"""Sprint 2.6.6 — attribute_inquiry_node.

When the customer asks "qual o peso?", "qual o balance dela?", "de que
material é feita?" — they're asking about a CHARACTERISTIC of the active
product, not searching for a product whose name contains "peso". The
pre-2.6.6 code routed these to product_inquiry → matcher → garbage
results (Lead Tape, Overgrip) because the token scorer found products
whose NAME contained "peso".

This node:

1. Identifies the active product (``recommended_products[0]`` when there's
   exactly one — the same convention price_inquiry already uses), OR uses
   a product the customer named in the same sentence ("qual o peso da
   Mormaii Sunset?").
2. Maps PT-BR synonyms to the structured ``atributos_parseados`` keys
   stored by the Bling sync.
3. Answers honestly:
   - found → "A *X* pesa Y."
   - not found but other attributes exist → "tenho A e B, mas o peso não
     consta — confirmo com a equipe e te retorno"
   - nothing technical at all → same honest reply
4. When the agent promises "vou confirmar e te retorno", an INTERNAL alert
   is fired to DOSSIER_RECIPIENT_PHONE so the promise translates into a
   real action (and so missing-attribute reports build a real backlog
   over time). Anti-spam: same (product_id, attribute) only fires once
   per conversation.
"""
from __future__ import annotations

import logging
import unicodedata
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes._product_match import (
    _distinctive_tokens,
    match_product_tolerant,
)
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Synonyms: customer keyword → ``atributos_parseados`` key ────────────────
#
# Each entry maps an attribute slug (the key we expect in the structured
# data) to the list of PT-BR triggers that mean "the customer wants this".
# Order in ``_ATTRIBUTE_TRIGGERS`` matters because we iterate in order to
# resolve which attribute the customer asked about (peso before composicao
# so "qual o peso?" doesn't accidentally match a "composicao" trigger).

_ATTRIBUTE_TRIGGERS: list[tuple[str, tuple[str, ...]]] = [
    # Sprint 2.7.5 — marca/modelo added because the Felipe production
    # catalog stores them as Bling custom fields, and the agent had no
    # slug for either. Listed FIRST so "qual a marca?" can't be hijacked
    # by a later "material" trigger.
    ("marca", (
        "marca", "qual marca", "qual a marca", "fabricante",
        "fabricado por",
    )),
    ("modelo", (
        "modelo", "qual modelo", "qual o modelo", "que modelo",
    )),
    ("peso", (
        "peso", "pesa", "pesada", "pesado", "gramas", "kilo", "quilos",
        "quao pesado", "quao pesada",
    )),
    ("equilibrio", (
        "equilibrio", "equilíbrio", "balance", "balanco", "balanço",
        "ponto de equilibrio", "ponto de equilíbrio",
        "centro de impacto", "ponto de impacto",
    )),
    ("composicao", (
        "composicao", "composição", "material", "feita de", "feito de",
        "do que e feita", "do que é feita", "do que e feito",
        "de que e feita", "de que é feita", "de que e feito",
        "fibra", "carbono",
        # Sprint 2.6.10 — "quantos k", "quantos K", "12k", "18k". The
        # "k" in racket composition refers to carbon weave count (3K /
        # 12K / 18K). Standalone "k" is too noisy so we require either
        # "quantos k" or "k de carbono"; "12k" survives as a numeric
        # token elsewhere.
        "quantos k", "k de carbono", "k de fibra",
    )),
    ("espessura", (
        "espessura", "grossura", "perfil grosso", "fino", "perfil",
    )),
    ("comprimento", (
        "comprimento", "cumprimento", "tamanho", "altura",
    )),
    ("detalhamento", (
        "detalhamento", "detalhe tecnico", "detalhe técnico", "especificacao",
    )),
]

# Sprint 2.7.5 — mapeamento slug-canônico → lista de chaves REAIS que valem
# como esse atributo nos produtos. Substitui o ``[slug, slug_aproximado,
# slug_total]`` hardcoded da Sprint 2.6.6, que falhava quando o Bling
# custom field tinha um nome diferente do slug (caso real: campo
# ``materiais_do_exterior`` carregando o material da raquete, e o
# matcher só procurava por ``composicao``).
#
# Ordem importa: primeira chave que retornar valor "meaningful" vence.
# Coloca o nome do Bling V3 PRIMEIRO (mais específico/recente), depois
# slugs legados como fallback de compatibilidade.
_SLUG_FIELD_CANDIDATES: dict[str, list[str]] = {
    "marca":        ["marca"],
    "modelo":       ["modelo"],
    "peso":         ["peso", "peso_aproximado", "peso_total", "peso_g", "peso_gramas"],
    "equilibrio":   ["equilibrio", "balance", "balanco"],
    "composicao":   ["materiais_do_exterior", "composicao", "material", "materiais"],
    "espessura":    ["espessura_do_perfil_mm", "espessura", "espessura_perfil"],
    "comprimento":  ["comprimento", "comprimento_cm", "tamanho_cm"],
    "detalhamento": ["detalhamento"],
}

# Sprint 2.7.5 — chaves que NUNCA podem virar resposta de ficha técnica.
# São controle interno (es_raquete_de_praia, já consumido em
# is_raquete_praia bool), abreviação interna sem sentido pro cliente
# (material_am = "car"), ou ruído de marketplace (baterias_sao_necessarias).
# Filtradas em ``_read_attribute`` E em ``_list_available_attributes``
# como defesa em profundidade.
_BLOCKED_KEYS: frozenset[str] = frozenset({
    "material_am",
    "es_raquete_de_praia",
    "baterias_sao_necessarias",
    "baterias_necessarias",
    "baterias",
})

# Sprint 2.7.5 — valores que NUNCA são apresentáveis como ficha técnica
# mesmo quando a chave passa pela blocklist. Cobre booleanos como string
# (Bling/marketplaces enviam "true"/"false") e nulos serializados.
_BOOLEAN_OR_NULL_VALUES: frozenset[str] = frozenset({
    "true", "false", "none", "null", "nan",
})

# Generic "give me the full spec" patterns — list all available attributes.
_FULL_SPEC_TRIGGERS = (
    "ficha tecnica", "ficha técnica", "specs", "especificacoes",
    "especificações", "detalhes tecnicos", "detalhes técnicos",
    "informacoes tecnicas", "informações técnicas",
    "me fala tudo", "todas as caracteristicas", "todas as características",
)

# Sprint 2.6.10 — BROAD-DETAIL triggers. Customer wants ANY info about
# the active product, not a specific attribute. Felipe's "Detalhes por
# favor" / "detalhes da raquete proteo" land here. The response cascades
# atributos_parseados → descricao → preço/estoque honest. NEVER falls
# back to the "X não consta — vou confirmar" path on its own — that
# response is only for SPECIFIC attribute misses.
_BROAD_DETAIL_TRIGGERS = (
    "detalhes", "detalhe",
    "me conta", "me conta sobre", "me conta dessa", "me conta dela",
    "fala mais", "fala mais dessa", "fala mais dela",
    "mais informacao", "mais informação", "mais informacoes", "mais informações",
    "mais detalhes", "quero detalhes", "quero saber mais",
    "ficha completa", "me passa tudo", "passa tudo",
    "me explica", "explica",
)


def is_broad_detail_request(text: str) -> bool:
    """True if the customer wants generic info about the active product
    (Sprint 2.6.10), not a specific attribute."""
    norm = _norm(text)
    return any(t in norm for t in _BROAD_DETAIL_TRIGGERS)

# Human labels used in the rendered answer.
_HUMAN_LABELS: dict[str, str] = {
    "marca": "Marca",
    "modelo": "Modelo",
    "peso": "Peso",
    "equilibrio": "Equilíbrio",
    "composicao": "Composição",
    "espessura": "Espessura",
    "comprimento": "Comprimento",
    "detalhamento": "Detalhamento",
}

# Sprint 2.6.10 — gender of the article that precedes the label in
# customer-facing sentences ("A composição não consta..." vs "O peso
# não consta..."). Felipe's print showed "O ficha técnica" — clearly
# broken concordance — because we used a hardcoded "O" everywhere.
_ARTICLE: dict[str, str] = {
    "marca": "A",
    "modelo": "O",
    "peso": "O",
    "equilibrio": "O",
    "composicao": "A",
    "espessura": "A",
    "comprimento": "O",
    "detalhamento": "O",
    "ficha": "A",
    "ficha tecnica": "A",
    "ficha técnica": "A",
}


def _article_for(label: str) -> str:
    """Return 'A' or 'O' for the given attribute label. Defaults to 'O'
    (masculine) when unknown, but the map should cover every label we emit."""
    key = _norm(label).split()[0] if label else ""
    return _ARTICLE.get(key, "O")


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def _norm(text: str) -> str:
    return _strip_accents((text or "").lower())


def _strip_attribute_triggers_from_query(text: str) -> str:
    """Remove attribute-question phrasing so the matcher sees only the
    product-name portion (e.g. "qual o peso da Mormaii Sunset" → "Mormaii Sunset").

    Sprint 2.6.6 — without this, the residual tokens "peso", "qual" etc.
    dilute the token-score and the catalog match comes back as ``none``,
    even though the product name is right there.
    """
    norm = _norm(text)
    # Strip the question-stem; both "qual o X" and "qual a X" variants.
    for stem in (
        "qual o ", "qual a ", "qual e o ", "qual e a ",
        "qual é o ", "qual é a ", "qual ",
        "quanto ",
        "me fala ", "me diz ",
    ):
        if norm.startswith(stem):
            norm = norm[len(stem):]
            break
    # Strip every known attribute trigger we can identify in the query.
    for _, triggers in _ATTRIBUTE_TRIGGERS:
        for t in triggers:
            t_norm = _norm(t)
            if t_norm in norm:
                norm = norm.replace(t_norm, " ")
    # Drop common possessive/articles we don't want as match tokens.
    for filler in (" dela ", " dele ", " dessa ", " desse ", " da ", " do ",
                   " de ", " e ", " ou "):
        norm = norm.replace(filler, " ")
    return " ".join(norm.split())


def detect_requested_attributes(text: str) -> list[str]:
    """Return the attribute slugs the customer asked about, in canonical
    order. Empty list when no specific attribute matched (likely a
    full-spec request — caller should check ``is_full_spec_request``)."""
    norm = _norm(text)
    found: list[str] = []
    for slug, triggers in _ATTRIBUTE_TRIGGERS:
        if any(_norm(t) in norm for t in triggers):
            if slug not in found:
                found.append(slug)
    return found


def is_full_spec_request(text: str) -> bool:
    norm = _norm(text)
    return any(t in norm for t in _FULL_SPEC_TRIGGERS)


def get_active_product(state: AgentState) -> dict | None:
    """Return the single product on the table, or None.

    Mirrors the convention price_inquiry already follows: when
    ``recommended_products`` has exactly one item, that's the active
    product. With zero or multiple items, the caller must ask the
    customer to disambiguate.
    """
    products = state.get("recommended_products") or []
    return products[0] if len(products) == 1 else None


def _is_meaningful_value(v: Any) -> bool:
    """Sprint 2.7.5 — does this raw value qualify as user-facing ficha?

    Drops:
      - None
      - empty / whitespace-only strings
      - boolean-stringified values ("true"/"false") that leak from
        marketplace integrations and internal flags
      - serialized null markers ("none"/"null"/"nan")

    Defense in depth: even if a key escapes ``_BLOCKED_KEYS``, this
    value gate prevents nonsense ending up in front of the customer.
    """
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    if s.lower() in _BOOLEAN_OR_NULL_VALUES:
        return False
    return True


def _format_value(slug: str, raw: str) -> str:
    """Sprint 2.7.5 — append the canonical unit to numeric values when the
    Bling sync stored them bare.

    The Felipe catalog stores ``espessura_do_perfil_mm: "22"`` (a bare
    integer, with the unit baked into the column NAME). The customer
    expects to see "22mm" not "22". This helper appends "mm"/"g"/"cm"
    only when the raw value is pure-numeric — values that already carry
    a letter (like "320g" from the description parser) are preserved
    untouched.
    """
    s = str(raw).strip()
    if not s:
        return s
    # If the value already has any letter, assume the unit is embedded.
    if any(c.isalpha() for c in s):
        return s
    # Pure numeric (digits, comma, dot, minus, plus, parens, spaces) →
    # append the canonical unit for the slug.
    _UNIT_BY_SLUG: dict[str, str] = {
        "espessura": "mm",
        "peso": "g",
        "comprimento": "cm",
    }
    unit = _UNIT_BY_SLUG.get(slug)
    return f"{s}{unit}" if unit else s


def _read_attribute(product: dict, slug: str) -> str | None:
    """Read ``slug`` from the product's ``atributos_parseados`` dict.

    Sprint 2.7.5 — uses ``_SLUG_FIELD_CANDIDATES`` to iterate the list of
    real keys that count as this attribute. Skips ``_BLOCKED_KEYS``
    entirely. Returns the FORMATTED value (with unit appended for bare
    numerics) or None when nothing meaningful is found.
    """
    attrs = product.get("atributos_parseados") or {}
    if not isinstance(attrs, dict):
        return None

    # 1. Slug-specific candidate chain (the curated list per slug). First
    # meaningful value wins, ordered by specificity.
    candidates = _SLUG_FIELD_CANDIDATES.get(slug, [slug])
    for key in candidates:
        if key in _BLOCKED_KEYS:
            continue
        v = attrs.get(key)
        if _is_meaningful_value(v):
            return _format_value(slug, str(v))

    # 2. Full-scan fallback — same accent-insensitive contract as 2.6.6,
    # but now respecting both the key blocklist and the value gate.
    target = _norm(slug)
    for key, value in attrs.items():
        if key in _BLOCKED_KEYS:
            continue
        if _norm(str(key)) == target and _is_meaningful_value(value):
            return _format_value(slug, str(value))

    return None


def _list_available_attributes(product: dict) -> dict[str, str]:
    """Return the subset of known attributes that have a value.

    Sprint 2.7.5 — the per-slug read goes through ``_read_attribute``,
    which already enforces both the key blocklist and the value gate,
    so ``material_am`` / ``es_raquete_de_praia`` / ``true``/``false``
    NEVER leak into a broad-detail listing.
    """
    out: dict[str, str] = {}
    for slug, _ in _ATTRIBUTE_TRIGGERS:
        v = _read_attribute(product, slug)
        if v:
            out[slug] = v
    return out


# ── Alert helper (Sprint 2.6.5 send_text reuse) ──────────────────────────────

async def _send_missing_attr_alert(
    state: AgentState,
    product: dict,
    requested: list[str],
) -> None:
    """Send Andre an internal alert when we promised "vou confirmar e te retorno".

    Honors the anti-spam list in state: same (product_id, attr) combo only
    triggers once per conversation. Best-effort: Evolution failures are
    logged and swallowed.
    """
    settings = get_settings()
    recipient = (settings.dossier_recipient_phone or "").strip()
    if not recipient:
        logger.info("attr_alert_skipped reason=no_recipient_configured")
        return

    product_id = product.get("id") or product.get("external_id") or "?"
    requested_key = ",".join(sorted(requested)) or "any"
    marker = f"{product_id}:{requested_key}"

    already: list[str] = list(state.get("alerted_missing_attrs") or [])
    if marker in already:
        logger.info("attr_alert_suppressed marker=%s", marker)
        return

    name = product.get("name", "?")
    customer_name = state.get("customer_name") or "(sem nome)"
    phone_hash = (state.get("phone_hash") or "")[:12]
    pretty = ", ".join(_HUMAN_LABELS.get(s, s) for s in requested) or "spec geral"

    alert = (
        f"📩 PEDIDO DE INFORMAÇÃO — Base Sports\n"
        f"Cliente: {customer_name}\n"
        f"Hash: {phone_hash}…\n"
        f"Produto: {name}\n"
        f"Pergunta: {pretty}\n"
        f"Status: dado não consta no cadastro.\n"
        f"Ação: responder ao cliente / cadastrar no Bling.\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    try:
        from app.adapters.evolution import EvolutionClient
        await EvolutionClient().send_text(recipient, alert)
        logger.info("attr_alert_sent marker=%s", marker)
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.error("attr_alert_send_failed marker=%s: %s", marker, exc)

    # Mark it sent regardless of Evolution success — we don't want to spam
    # Andre on transient send failures either.
    state.get("alerted_missing_attrs") or []  # ensure state field shape
    already.append(marker)


# ── Renderers ────────────────────────────────────────────────────────────────

def _render_found(product: dict, slug: str, value: str) -> str:
    name = product.get("name", "esse produto")
    if slug == "peso":
        return f"A *{name}* pesa {value}."
    if slug == "equilibrio":
        return f"O equilíbrio da *{name}* é {value}."
    if slug == "composicao":
        return f"A *{name}* é feita de {value}."
    # Sprint 2.7.5 — marca/modelo natural phrasing.
    if slug == "marca":
        return f"A marca da *{name}* é *{value}*."
    if slug == "modelo":
        return f"O modelo é *{value}*."
    label = _HUMAN_LABELS.get(slug, slug.capitalize())
    return f"{label} da *{name}*: {value}."


def _render_multiple(product: dict, found_pairs: list[tuple[str, str]]) -> str:
    name = product.get("name", "esse produto")
    lines = []
    for slug, value in found_pairs:
        label = _HUMAN_LABELS.get(slug, slug.capitalize())
        lines.append(f"• {label}: {value}")
    return f"Sobre a *{name}*:\n" + "\n".join(lines)


def _render_partial_then_promise(
    product: dict, found_pairs: list[tuple[str, str]], missing_labels: list[str]
) -> str:
    name = product.get("name", "esse produto")
    available = "\n".join(
        f"• {_HUMAN_LABELS.get(s, s.capitalize())}: {v}" for s, v in found_pairs
    )
    missing_text = " e ".join(missing_labels) if len(missing_labels) <= 2 else (
        ", ".join(missing_labels[:-1]) + " e " + missing_labels[-1]
    )
    # Sprint 2.6.10 — gender-aware article. The first missing label
    # decides between "O" / "A" so "A composição e a espessura não
    # constam" reads correctly. Multi-label cases default to plural
    # "As" for feminine head, "Os" for masculine.
    first_article = _article_for(missing_labels[0]) if missing_labels else "O"
    plural_article = (
        "As" if (first_article == "A" and len(missing_labels) > 1)
        else "Os" if (first_article == "O" and len(missing_labels) > 1)
        else first_article
    )
    article = plural_article if len(missing_labels) > 1 else first_article
    return (
        f"Sobre a *{name}*, o que tenho aqui é:\n{available}\n\n"
        f"{article} {missing_text} não consta no meu cadastro — posso "
        f"confirmar com a equipe e te retorno. Quer?"
    )


def _render_honest_missing(product: dict, requested_labels: list[str]) -> str:
    name = product.get("name", "esse produto")
    if requested_labels:
        detail = ", ".join(requested_labels)
        article = _article_for(requested_labels[0])
    else:
        detail = "esse detalhe"
        article = "Esse"
    return (
        f"Boa pergunta! {article} {detail} da *{name}* não consta aqui no "
        f"sistema. Vou confirmar com a equipe e já te retorno, tá?"
    )


# ── Sprint 2.6.10 — broad-detail cascade ────────────────────────────────────

import re as _re

_HTML_TAG_RE = _re.compile(r"<[^>]+>")
_WHITESPACE_RE = _re.compile(r"\s+")
_DESCRIPTION_MAX_CHARS = 450


def _clean_description_for_whatsapp(raw: str) -> str:
    """Strip HTML, collapse whitespace, truncate at a sentence boundary
    so the description fits a WhatsApp message without cutting mid-word.

    Returns empty string when input is empty/None.
    """
    if not raw:
        return ""
    text = _HTML_TAG_RE.sub(" ", raw)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) <= _DESCRIPTION_MAX_CHARS:
        return text
    # Try to cut at the last sentence boundary within the cap.
    head = text[:_DESCRIPTION_MAX_CHARS]
    for sep in (". ", "! ", "? ", "; "):
        cut = head.rfind(sep)
        if cut >= _DESCRIPTION_MAX_CHARS // 2:
            return head[: cut + 1].strip()
    # Fallback: hard cut + ellipsis.
    return head.rstrip() + "…"


def _format_price(product: dict) -> str | None:
    """Return 'R$ XXX' from price_cents, or None if unavailable."""
    cents = product.get("price_cents")
    if not isinstance(cents, (int, float)) or cents <= 0:
        return None
    reais = int(cents) // 100
    return f"R$ {reais:,}".replace(",", ".")


def _render_broad_details(product: dict) -> str:
    """Sprint 2.6.10 — cascade response for "quero detalhes" / "me conta".

    Order of preference:
      (a) atributos_parseados → bullet list of every known attribute that has
          a value, optionally followed by the price.
      (b) descricao_curta / description → cleaned commercial blurb + price.
      (c) nothing technical → name + price + honest "specs detalhadas não
          constam no sistema" (no follow-up promise; this is broad detail,
          not a specific-attribute miss).

    Never returns the "X não consta — vou confirmar" honest-missing line as
    the SOLE response; that path is reserved for specific attribute misses
    where the customer asked something targeted.
    """
    name = product.get("name", "esse produto")
    price = _format_price(product)

    available = _list_available_attributes(product)
    if available:
        lines = [
            f"• {_HUMAN_LABELS.get(slug, slug.capitalize())}: {value}"
            for slug, value in available.items()
        ]
        body = "\n".join(lines)
        text = f"Sobre a *{name}*:\n{body}"
        if price:
            text += f"\n\nPreço: {price}."
        return text

    description = product.get("description") or product.get("descricao_curta") or ""
    clean = _clean_description_for_whatsapp(description)
    if clean:
        text = f"Sobre a *{name}*:\n\n{clean}"
        if price:
            text += f"\n\nPreço: {price}."
        return text

    # Bare-bones fallback — name + price + honest about the rest. No "vou
    # confirmar" because we don't know which attribute the customer cares
    # about; this is broad-detail context.
    pieces = [f"Sobre a *{name}*, o que tenho aqui é o nome"]
    if price:
        pieces.append(f"e o preço ({price})")
    pieces_text = " ".join(pieces) + "."
    return (
        f"{pieces_text} As specs técnicas mais detalhadas não constam no "
        f"meu cadastro — se quiser saber algo específico, é só me dizer."
    )


# ── Main node ────────────────────────────────────────────────────────────────

async def attribute_inquiry_node(state: AgentState) -> dict:
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    # Try the active product first (the cheap path).
    product = get_active_product(state)

    # If the customer also NAMED a product in the same sentence (e.g.
    # "qual o peso da Mormaii Sunset?"), use that — it overrides any
    # stale active product. We strip attribute keywords FIRST so the
    # matcher sees only the product-name portion of the query.
    if user_text:
        try:
            stripped = _strip_attribute_triggers_from_query(user_text)
            from app.sync.bling_catalog_cache import get_catalog_snapshot
            snapshot = await get_catalog_snapshot()
            distinctive = _distinctive_tokens(stripped)
            # Only try when the stripped query still has ≥2 distinctive
            # tokens — single token (just "carbono" or "x5") is too weak.
            if len(distinctive) >= 2:
                m = match_product_tolerant(stripped, snapshot)
                if m.status in ("exact", "fuzzy_high") and m.product is not None:
                    product = m.product
                    logger.info(
                        "attribute_inquiry product_named_in_sentence name=%s "
                        "(stripped_query=%r)",
                        product.get("name"), stripped,
                    )
        except Exception as exc:
            logger.warning("attribute_inquiry catalog_match_failed: %s", exc)

    if product is None:
        text = (
            "Sobre qual produto você quer saber? Me diz o nome que eu "
            "te falo."
        )
        logger.info("attribute_inquiry no_active_product")
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
        }

    # What did the customer ask for?
    requested = detect_requested_attributes(user_text)
    wants_full = is_full_spec_request(user_text)
    wants_broad = is_broad_detail_request(user_text)

    # Sprint 2.6.10 — BROAD detail request ("detalhes", "me conta dela"):
    # use the cascade (atributos → descricao → fallback). This path NEVER
    # falls into the "X não consta — vou confirmar" honest-missing branch
    # used for specific attribute misses. Specific attribute paths below
    # are unchanged from Sprint 2.6.6.
    if wants_broad and not requested:
        text = _render_broad_details(product)
        logger.info(
            "attribute_inquiry broad_details product=%s has_attrs=%s "
            "has_desc=%s",
            product.get("name"),
            bool(_list_available_attributes(product)),
            bool(product.get("description") or product.get("descricao_curta")),
        )
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            # Broad-detail path doesn't keep the choice flag — customer
            # already accepted; future turns flow through normal triage.
            "awaiting_detail_choice": False,
        }

    if wants_full or not requested:
        # Customer asked broadly ("me fala a ficha técnica") OR we couldn't
        # pinpoint a specific attribute — list everything we have.
        available = _list_available_attributes(product)
        if available:
            text = _render_multiple(
                product, [(slug, available[slug]) for slug in available]
            )
            logger.info(
                "attribute_inquiry full_spec_listed product=%s n_attrs=%d",
                product.get("name"), len(available),
            )
            return {
                "messages": [AIMessage(content=text)],
                "response_blocks": [text],
            }
        # No attributes at all → honest + alert.
        await _send_missing_attr_alert(state, product, requested or ["ficha"])
        text = _render_honest_missing(product, ["ficha técnica"])
        marker = f"{product.get('id') or product.get('external_id')}:{','.join(sorted(requested or ['ficha']))}"
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            "alerted_missing_attrs": list(state.get("alerted_missing_attrs") or []) + [marker],
        }

    # Customer asked for specific attributes.
    found_pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for slug in requested:
        v = _read_attribute(product, slug)
        if v:
            found_pairs.append((slug, v))
        else:
            missing.append(slug)

    if found_pairs and not missing:
        # Single attribute found and that's all they asked for.
        if len(found_pairs) == 1:
            slug, value = found_pairs[0]
            text = _render_found(product, slug, value)
        else:
            text = _render_multiple(product, found_pairs)
        logger.info(
            "attribute_inquiry resolved product=%s attrs=%s",
            product.get("name"), [s for s, _ in found_pairs],
        )
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
        }

    if found_pairs and missing:
        # Partial: tell what we have, be honest about the rest, alert internally.
        missing_labels = [_HUMAN_LABELS.get(s, s) for s in missing]
        text = _render_partial_then_promise(product, found_pairs, missing_labels)
        await _send_missing_attr_alert(state, product, missing)
        marker = f"{product.get('id') or product.get('external_id')}:{','.join(sorted(missing))}"
        logger.info(
            "attribute_inquiry partial product=%s found=%s missing=%s",
            product.get("name"), [s for s, _ in found_pairs], missing,
        )
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            "alerted_missing_attrs": list(state.get("alerted_missing_attrs") or []) + [marker],
        }

    # Nothing found at all → honest + alert.
    requested_labels = [_HUMAN_LABELS.get(s, s).lower() for s in requested]
    text = _render_honest_missing(product, requested_labels)
    await _send_missing_attr_alert(state, product, requested)
    marker = f"{product.get('id') or product.get('external_id')}:{','.join(sorted(requested))}"
    logger.info(
        "attribute_inquiry honest_missing product=%s requested=%s",
        product.get("name"), requested,
    )
    return {
        "messages": [AIMessage(content=text)],
        "response_blocks": [text],
        "alerted_missing_attrs": list(state.get("alerted_missing_attrs") or []) + [marker],
    }
