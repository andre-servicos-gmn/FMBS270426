"""Sprint 2.0 — qualifier-mode recommend.

The Sprint 1.9 two-layer architecture (REFERENCE vs PROFILE) is preserved as
mode detection, but the SEMANTICS flipped: this node no longer "recommends"
in the active sense. The agent is now a qualifier whose job is to either:

- REFERENCE-SIM: confirm the named racket is in stock and ask whether the
  customer wants details or to close. NO alternatives — even ones that "would
  fit better". Customer already chose; respect that.
- REFERENCE-NÃO: briefly say we don't carry that exact model and offer the
  Consultoria Base Sports as the path to find the right one for the profile.
  NO alternatives — listing them undermines the Consultoria pitch.
- PROFILE: delegate to ``consultoria_offer_node``. The agent never picks a
  racket for the customer; the Consultoria does (with on-court testing).

``last_recommendation_at`` and ``recommended_products`` are still set in
REFERENCE-SIM so the follow-up triage state machine (price_inquiry,
product_selection) can resolve the single product the customer named.
"""
import json
import logging
import unicodedata
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.anti_rerun import (
    fallback_message_for,
    should_block_rerun,
    stamp_node_execution,
)
from app.agent.message_splitter import parse_messages
from app.agent.nodes._product_match import match_product_tolerant
from app.agent.nodes.consultoria_offer import consultoria_offer_node
from app.agent.prompts import build_recommend_prompt
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)

_HANDOFF_MARKER = "[HANDOFF]"

# Casual-Brazilian "no preference / no reference" values — in sync with the
# mapping table in SYSTEM_DIAGNOSE_EXTRACT.
_EMPTY_MODEL_VALUES = {"", "nenhum", "nenhuma", "none", "null", "nao", "não"}


# ── Helpers (re-exported; used by re_recommendation too) ─────────────────────

def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _normalize_name(s: str) -> str:
    return _strip_accents((s or "").lower().strip())


def _sport_value(profile: dict) -> str | None:
    raw = profile.get("esporte_praticado") or profile.get("sport")
    if not raw:
        return None
    raw = str(raw).strip().lower()
    if raw in ("padel", "pala", "pala de padel"):
        return "padel"
    return "beach_tennis"


def _build_filters(profile: dict) -> dict:
    """Hard DB filters — always pin category to raquete/pala by sport."""
    filters: dict = {"min_stock": 1}
    sport = _sport_value(profile)
    if sport:
        filters["sport"] = sport
    filters["category"] = "pala" if sport == "padel" else "raquete"
    return filters


def _build_query(profile: dict) -> str:
    parts: list[str] = []

    sport = _sport_value(profile)
    if sport == "padel":
        parts.append("pala de padel")
    else:
        parts.append("raquete de beach tennis")

    level = profile.get("nivel_jogo") or profile.get("level")
    if level:
        parts.append(f"nível {level}")

    prev = profile.get("esporte_raquete_previo")
    if prev and str(prev).lower() not in ("nenhum", "nao_aplicavel", "nao", "não", ""):
        parts.append(f"compatível com jogador de {prev}")

    return " ".join(parts)


def _has_model_reference(profile: dict) -> bool:
    raw = profile.get("modelo_desejado")
    if not raw:
        return False
    return _strip_accents(str(raw).lower().strip()) not in _EMPTY_MODEL_VALUES


def _find_name_match(products: list[dict], requested: str) -> dict | None:
    """Delegates to match_product_tolerant (Sprint 1.15)."""
    if not requested or not products:
        return None
    return match_product_tolerant(requested, products).product


def _get_lesion(profile: dict) -> str | None:
    region = profile.get("regiao_lesao")
    if not region:
        return None
    norm = _strip_accents(str(region).lower().strip())
    if norm in ("nenhuma", "nenhum", "none", ""):
        return None
    return norm


def _format_products_for_context(products: list[dict]) -> str:
    if not products:
        return "(nenhum produto encontrado no momento)"
    return "\n".join(
        f"- {p['name']}: R${int(p['price_cents']) / 100:.0f}"
        f" | {(p.get('description') or '')[:120]}"
        for p in products
    )


# ── REFERENCE search ─────────────────────────────────────────────────────────

async def _search_reference(profile: dict, modelo: str) -> tuple[dict | None, list[dict]]:
    """Search the catalog by the named model; return (matched_product, raw_candidates).

    Sprint 2.5 — when the Bling integration is wired (``BLING_CLIENT_ID``
    populated), we query ``bling_products`` directly. Otherwise we fall back
    to the legacy embedding-based ``search_products`` so existing tests + the
    pre-Bling pilot keep working unchanged.
    """
    settings = get_settings()
    if settings.bling_client_id:
        from app.sync.bling_repo import fetch_product_by_name, list_active_products
        try:
            candidates = await fetch_product_by_name(modelo)
            if not candidates:
                candidates = await list_active_products(limit=200)
        except Exception as exc:
            logger.warning("recommend_bling_lookup_failed: %s", exc)
            candidates = []
        matched = _find_name_match(candidates, modelo)
        return matched, candidates

    from app.rag.retriever import search_products
    from app.storage.db import get_session

    filters = _build_filters(profile)
    sport = _sport_value(profile) or "beach_tennis"
    query = f"raquete {modelo} {sport.replace('_', ' ')}"

    try:
        async with get_session() as session:
            candidates = await search_products(session, query, filters, k=5)
    except Exception as exc:
        logger.warning("recommend_reference_retrieval_failed: %s", exc)
        candidates = []

    matched = _find_name_match(candidates, modelo)
    return matched, candidates


def _build_reference_sim_context(
    profile: dict, customer_name: str | None, matched: dict
) -> str:
    name_line = f"Nome do cliente: {customer_name}\n" if customer_name else ""
    return (
        f"{name_line}"
        f"Modo: REFERENCE-SIM\n"
        f"Modelo solicitado pelo cliente: {matched.get('name')}\n"
        f"Status: TEMOS NO ESTOQUE\n\n"
        f"Perfil do cliente: {json.dumps(profile, ensure_ascii=False)}\n\n"
        f"Produto confirmado:\n{_format_products_for_context([matched])}"
    )


def _build_reference_nao_context(
    profile: dict, customer_name: str | None, modelo_solicitado: str
) -> str:
    settings = get_settings()
    preco = getattr(settings, "consultoria_preco", 350)
    name_line = f"Nome do cliente: {customer_name}\n" if customer_name else ""
    return (
        f"{name_line}"
        f"Modo: REFERENCE-NÃO\n"
        f"Modelo solicitado pelo cliente: {modelo_solicitado}\n"
        f"Status: NÃO TEMOS NO CATÁLOGO\n\n"
        f"Perfil do cliente: {json.dumps(profile, ensure_ascii=False)}\n\n"
        f"Investimento da Consultoria: R$ {preco} (100% abatido na compra "
        f"de raquete no mesmo dia)."
    )


# ── Fallback messages (used when the LLM returns empty) ──────────────────────

def _fallback_reference_sim(matched: dict) -> list[str]:
    name = matched.get("name", "essa raquete")
    return [
        f"Sim, temos a *{name}* aqui!",
        "Quer saber mais detalhes (preço, peso, indicação) ou já quer fechar?",
    ]


def _fallback_reference_nao(modelo: str) -> list[str]:
    settings = get_settings()
    preco = getattr(settings, "consultoria_preco", 350)
    return [
        f"A *{modelo}* específica a gente não tem no catálogo agora.",
        (
            f"Pra encontrar a raquete que realmente combina com seu jogo, "
            f"a gente prefere fazer com a *Consultoria Base Sports* — análise "
            f"do seu perfil + teste em quadra. Investimento de *R$ {preco}*, "
            f"100% abatido se comprar no mesmo dia."
        ),
        "Quer saber como funciona ou já agendar?",
    ]


# Sprint 2.1 — deterministic single-block replies for cliente determinado.
# No LLM, no bullets, no "qual delas". Frase natural com negrito no nome.

def _determined_reference_sim_text(matched: dict) -> str:
    name = matched.get("name", "essa raquete")
    return (
        f"Sim, temos a *{name}*! Quer saber preço, peso, ou tem alguma "
        f"dúvida específica?"
    )


def _determined_reference_nao_text(modelo: str) -> str:
    """Sprint 2.4 — REFERENCE-NÃO determined now offers to look for
    alternatives instead of jumping straight into the Consultoria pitch.
    The agent asks; if the customer agrees, triage flips them to exploring
    and diagnose runs next turn."""
    return (
        f"Infelizmente não temos a *{modelo}* em estoque no momento. "
        f"Posso te ajudar a ver outras opções que combinem com seu jogo?"
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────

async def recommend_node(state: AgentState) -> dict:
    profile = dict(state.get("player_profile") or {})
    customer_name = state.get("customer_name")
    messages = state.get("messages") or []

    # Anti-rerun guard (unchanged from Sprint 1.14).
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    last_msg = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )
    if should_block_rerun(state, "recommend", last_msg):
        fallback = fallback_message_for("recommend")
        logger.info(
            "recommend_blocked_by_anti_rerun phone_hash=%.8s",
            (state.get("phone_hash") or "")[:8],
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "response_blocks": [fallback],
            **stamp_node_execution("recommend"),
        }

    # ── PROFILE mode → delegate to consultoria_offer_node ────────────────────
    if not _has_model_reference(profile):
        logger.info("recommend mode=PROFILE delegating_to=consultoria_offer")
        offer_result = await consultoria_offer_node(state)
        # Stamp execution so anti-rerun applies to the next turn too.
        offer_result.update(stamp_node_execution("recommend"))
        return offer_result

    # ── REFERENCE mode ───────────────────────────────────────────────────────
    modelo = str(profile.get("modelo_desejado") or "").strip()
    matched, _ = await _search_reference(profile, modelo)
    is_sim = matched is not None
    mode = "REFERENCE-SIM" if is_sim else "REFERENCE-NÃO"
    is_determined = state.get("customer_intent_path") == "determined"

    # Sprint 2.1 — cliente determinado: deterministic single-block reply,
    # no diagnose needed, no LLM call. Subtle Consultoria pitch is appended
    # ONCE per conversation for REFERENCE-SIM (NÃO already mentions it).
    if is_determined:
        extra: dict = {}
        if is_sim:
            assert matched is not None  # is_sim guarantees a matched product
            # Sprint 2.4 — stock confirmation is pitch-free by design. The
            # subtle pitch only fires on PRICE / FITNESS / COMFORT or on the
            # 2nd+ technical question (DELAYED types).
            blocks = [_determined_reference_sim_text(matched)]
        else:
            # Sprint 2.4 — ask whether the customer wants alternatives;
            # flag is read by triage next turn to handle sim/não.
            blocks = [_determined_reference_nao_text(modelo)]
            extra["awaiting_alternatives_decision"] = True

        joined = "\n\n".join(blocks)
        logger.info(
            "recommend done mode=%s matched=%s path=determined blocks=%d",
            mode, (matched or {}).get("name"), len(blocks),
        )

        result: dict = {
            "messages": [AIMessage(content=joined)],
            "response_blocks": blocks,
            "produto_pesquisado": modelo,
            **stamp_node_execution("recommend"),
            **extra,
        }
        if is_sim:
            assert matched is not None
            result["recommended_products"] = [matched]
            result["last_recommendation_at"] = datetime.now(timezone.utc).isoformat()
        else:
            result["recommended_products"] = []
        return result

    # ── Exploring / unknown path — go through the LLM phrasing pipeline ─────
    if is_sim:
        assert matched is not None
        context = _build_reference_sim_context(profile, customer_name, matched)
    else:
        context = _build_reference_nao_context(profile, customer_name, modelo)

    settings = get_settings()
    system = build_recommend_prompt(settings)

    client = OpenAIClient()
    response = await client.chat(
        messages=[{"role": "user", "content": context}],
        system=system,
        max_tokens=600,
        temperature=0.5,
        json_mode=True,
    )

    blocks = parse_messages(response)
    needs_handoff = any(_HANDOFF_MARKER in b for b in blocks)
    blocks = [b.replace(_HANDOFF_MARKER, "").strip() for b in blocks]
    blocks = [b for b in blocks if b]

    # Defensive fallbacks — keep UX intact if LLM returns empty.
    if not blocks:
        if is_sim:
            assert matched is not None
            blocks = _fallback_reference_sim(matched)
        else:
            blocks = _fallback_reference_nao(modelo)
        logger.warning("recommend fallback_used mode=%s", mode)

    joined = "\n\n".join(blocks)

    logger.info(
        "recommend done mode=%s matched=%s blocks=%d handoff=%s",
        mode, (matched or {}).get("name"), len(blocks), needs_handoff,
    )

    result: dict = {
        "messages": [AIMessage(content=joined)],
        "response_blocks": blocks,
        "produto_pesquisado": modelo,
        **stamp_node_execution("recommend"),
    }

    if is_sim:
        # Only REFERENCE-SIM keeps a shortlist — the named product is the
        # active racket on the table for follow-up intents.
        assert matched is not None
        result["recommended_products"] = [matched]
        result["last_recommendation_at"] = datetime.now(timezone.utc).isoformat()
    else:
        # REFERENCE-NÃO is a Consultoria pivot — flip the interest flag for
        # later dossier rendering and clear any stale shortlist.
        result["consultoria_interest"] = True
        result["recommended_products"] = []

    if needs_handoff:
        result["needs_handoff"] = True
        result["handoff_reason"] = "recommend_no_match"

    return result
