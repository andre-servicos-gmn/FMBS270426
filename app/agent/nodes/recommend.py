"""Sprint 2.6 — product_inquiry handler.

The legacy "REFERENCE-SIM / REFERENCE-NÃO / PROFILE" three-mode machine
existed to navigate around the diagnose node. Sprint 2.6 removed diagnose,
so this node has exactly one job:

    Customer asked about a product → identify it in the Bling catalog
    (or fall back to the legacy local catalog), respond with a short
    confirmation + invite the next step.

Two outcomes:

- Product matched in catalog → "Sim, temos a *X*! Quer saber preço,
  detalhes ou já fechar?" (one block, deterministic).
- Product NOT matched → "Hmm, não encontrei essa raquete no nosso
  catálogo. Quer que eu te ajude a achar outra?" (one block).

No LLM call in either path. The node also records ``recommended_products``
so the subsequent ``price_inquiry`` / ``product_selection`` turns know
which product is on the table.
"""
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.anti_rerun import (
    fallback_message_for,
    should_block_rerun,
    stamp_node_execution,
)
from app.agent.nodes._product_match import match_product_tolerant
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


async def _list_catalog_candidates(last_text: str) -> list[dict[str, Any]]:
    """Resolve the candidate product list. Sprint 2.6.3 — Bling path now
    goes through the in-memory ``CatalogCache`` (full catalog, no LIMIT 200
    cap) instead of hitting Supabase per inbound message. Legacy local
    catalog path is preserved for dev / Bling-off mode.
    """
    settings = get_settings()
    if settings.bling_client_id:
        try:
            from app.sync.bling_catalog_cache import get_catalog_snapshot
            snapshot = await get_catalog_snapshot()
            return list(snapshot)
        except Exception as exc:
            logger.warning("recommend_bling_lookup_failed: %s", exc)
            return []

    # Legacy path — local catalog via embedding search.
    try:
        from app.rag.retriever import search_products
        from app.storage.db import get_session
        async with get_session() as session:
            return await search_products(session, last_text, {"min_stock": 1}, k=5)
    except Exception as exc:
        logger.warning("recommend_local_lookup_failed: %s", exc)
        return []


def _confirmation_text(matched: dict[str, Any]) -> str:
    """Sprint 2.6.1 / 2.6.4 — consultive tone, neutral noun.

    The 2.6.1 closing kept "ou já quer fechar?" out (purchase vocabulary
    reserved for explicit ``purchase_intent``). 2.6.4 replaces the
    "raquete" hard-coded fallback name with a neutral one so a customer
    asking about a non-racket product (sock, manguito, kit, etc.) doesn't
    see "Sim, temos a essa raquete!".
    """
    name = matched.get("name", "esse produto")
    return (
        f"Sim, temos a *{name}*! Posso te passar mais detalhes, ou "
        f"prefere ver pessoalmente na loja?"
    )


def _not_found_text(query: str) -> str:
    """Sprint 2.6.2 / 2.6.4 — neutral "not found" wording.

    Replaces "essa raquete" → "esse produto" so the line works for socks,
    manguitos, kits, etc. — the agent doesn't blow its cover when the
    customer asked about non-racket items.
    """
    return (
        "Hmm, não encontrei esse produto no nosso catálogo. "
        "Quer que eu te ajude a achar outro?"
    )


def _ambiguous_text(candidates: list[dict[str, Any]]) -> str:
    """Sprint 2.6.4 — render the ambiguous-list confirmation.

    Up to 3 candidates listed by bold name. The closing question is kept
    open ("Qual você procura?") so the customer can reply with name,
    position, or — for the multi-product price case (Sprint 2.6.4) —
    "as duas" / "ambas" / "todos".
    """
    names = [c.get("name", "") for c in candidates[:3] if c.get("name")]
    if len(names) >= 2:
        bulleted = "\n".join(f"• *{n}*" for n in names)
        return (
            "Temos algumas opções parecidas:\n"
            f"{bulleted}\n"
            "\n"
            "Qual você procura?"
        )
    if names:
        return f"Você quis dizer a *{names[0]}*?"
    return _not_found_text("")


def _fallback_top3_text(candidates: list[dict[str, Any]]) -> str:
    """Sprint 2.6.4 — when match returns ``none`` but the top candidates
    share at least one distinctive token with the query, offer them as a
    soft fallback instead of just "não encontrei"."""
    names = [c.get("name", "") for c in candidates[:3] if c.get("name")]
    bulleted = "\n".join(f"• *{n}*" for n in names)
    return (
        "Não encontrei exatamente isso, mas tenho modelos parecidos:\n"
        f"{bulleted}\n"
        "\n"
        "Algum desses é o que você procura?"
    )


def _candidates_share_tokens_with_query(
    query: str, candidates: list[dict[str, Any]]
) -> bool:
    """Did any top candidate share ≥1 distinctive token with the query?"""
    from app.agent.nodes._product_match import _distinctive_tokens, _tokenize

    q_tokens = _distinctive_tokens(query)
    if not q_tokens:
        return False
    for c in candidates:
        name_tokens = set(_tokenize(c.get("name") or ""))
        if q_tokens & name_tokens:
            return True
    return False


async def recommend_node(state: AgentState) -> dict:
    messages = state.get("messages") or []
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    last_text = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    # Anti-rerun guard — guards against the customer repeating the same
    # short follow-up within 60s.
    if should_block_rerun(state, "recommend", last_text):
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

    # Sprint 2.6.2 — confirmation flow: the previous turn asked "Você quis
    # dizer X?". Triage routes us back here when the customer answered yes;
    # we promote the stashed candidate to an active match and emit the
    # standard confirmation message.
    pending = state.get("awaiting_match_confirmation")
    if pending:
        text = _confirmation_text(pending)
        logger.info(
            "recommend confirmation_resolved product=%s "
            "awaiting_detail_choice=True",
            pending.get("name"),
        )
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            "recommended_products": [pending],
            "last_recommendation_at": datetime.now(timezone.utc).isoformat(),
            "produto_pesquisado": pending.get("name"),
            "awaiting_match_confirmation": None,
            # Sprint 2.6.10 — confirmation_text ends with "Posso te passar
            # mais detalhes, ou prefere ver pessoalmente na loja?". The next
            # turn must read this flag in triage and short-circuit "detalhes"
            # / "sim" / "manda" → attribute_inquiry (NOT help_request).
            "awaiting_detail_choice": True,
            **stamp_node_execution("recommend"),
        }

    candidates = await _list_catalog_candidates(last_text)
    match = match_product_tolerant(last_text, candidates) if candidates else None

    if match is None or match.status == "none":
        # Sprint 2.6.4 — when the matcher gives up but the top-3 candidates
        # share at least one distinctive token with the query, offer them
        # as a soft fallback (recovery) instead of "não encontrei". This
        # rescues queries where the customer paraphrases the product
        # ("kit bolinhas" → "Bola Beach Tennis ... Com 72 Bolinhas").
        top3 = (match.candidates if match else None) or []
        if top3 and _candidates_share_tokens_with_query(last_text, top3):
            text = _fallback_top3_text(top3)
            logger.info(
                "recommend not_matched_with_fallback query=%r n_candidates=%d",
                last_text[:80], len(top3),
            )
            return {
                "messages": [AIMessage(content=text)],
                "response_blocks": [text],
                "recommended_products": [],
                "last_product_candidates": top3,
                **stamp_node_execution("recommend"),
            }

        text = _not_found_text(last_text)
        logger.info("recommend not_matched query=%r", last_text[:80])
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            "recommended_products": [],
            "last_product_candidates": None,
            **stamp_node_execution("recommend"),
        }

    if match.status in ("exact", "fuzzy_high"):
        matched = match.product
        assert matched is not None  # status guarantees product is set
        text = _confirmation_text(matched)
        logger.info(
            "recommend matched product=%s status=%s "
            "awaiting_detail_choice=True",
            matched.get("name"), match.status,
        )
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            "recommended_products": [matched],
            "last_recommendation_at": datetime.now(timezone.utc).isoformat(),
            "produto_pesquisado": matched.get("name"),
            "last_product_candidates": None,
            # Sprint 2.6.10 — see comment in the confirmation_resolved branch.
            "awaiting_detail_choice": True,
            **stamp_node_execution("recommend"),
        }

    if match.status == "fuzzy_low":
        candidate = match.product
        assert candidate is not None
        name = candidate.get("name", "esse produto")
        text = f"Você quis dizer a *{name}*?"
        logger.info(
            "recommend confirmation_pending candidate=%s distance=%s",
            name, match.distance,
        )
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            "awaiting_match_confirmation": candidate,
            **stamp_node_execution("recommend"),
        }

    if match.status == "ambiguous":
        candidates = list(match.candidates or [])
        text = _ambiguous_text(candidates)
        logger.info(
            "recommend ambiguous n_candidates=%d", len(candidates),
        )
        return {
            "messages": [AIMessage(content=text)],
            "response_blocks": [text],
            # Sprint 2.6.4 — stash candidates so price_inquiry / triage can
            # resolve "as duas" / "ambas" / "o segundo" on the next turn.
            "last_product_candidates": candidates,
            **stamp_node_execution("recommend"),
        }

    # Defensive — unknown status falls through to "not found".
    text = _not_found_text(last_text)
    return {
        "messages": [AIMessage(content=text)],
        "response_blocks": [text],
        "recommended_products": [],
        "last_product_candidates": None,
        **stamp_node_execution("recommend"),
    }
