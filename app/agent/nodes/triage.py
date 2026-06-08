"""Sprint 2.6 — simplified triage.

The strategic refactor removed the ``diagnose`` flow entirely. Triage now
classifies the customer message into one of 9 declarative intents and the
router maps each to a single node. No more ``customer_intent_path``, no
more determined/exploring forks, no more purchase-pattern extraction, no
more opinion-seek override — all of that existed to navigate around the
diagnose node, which is gone.

Anything the LLM emits that ISN'T in ``_VALID_INTENTS`` falls back to
``smalltalk`` (safe default — the smalltalk node is always benign).
"""
import json
import logging
import re
import unicodedata

from langchain_core.messages import HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.prompts import SYSTEM_TRIAGE
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


_VALID_INTENTS = {
    "smalltalk",
    "product_inquiry",
    "price_inquiry",
    "purchase_intent",
    "scheduling_inquiry",
    "out_of_scope",
    "faq",
    "help_request",
    "close",
    # Sprint 2.6.6 — "qual o peso/balance/material dela?" → tecnical attribute
    # question about the ACTIVE product, NOT a search for a product whose
    # name contains "peso". Routed to attribute_inquiry_node.
    "attribute_inquiry",
}


# ── Sprint 2.6.2 — affirmative / negative detectors ─────────────────────────

def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


_AFFIRMATIVE_TOKENS = frozenset({
    "sim", "isso", "exato", "exatamente", "e isso", "eh isso",
    "ok", "claro", "pode", "uhum", "isso mesmo", "essa mesma",
    "yes", "afirmativo", "positivo", "perfeito", "show",
})

_NEGATIVE_TOKENS = frozenset({
    "nao", "n", "negativo", "nope", "nada disso", "outra",
    "nao e", "nao eh", "outra raquete", "outro produto",
})


# Sprint 2.6.10 — when recommend just emitted "Posso te passar mais
# detalhes, ou prefere ver pessoalmente na loja?", these short replies
# mean YES, send details. Normalized (no accent, lowercase).
_DETAIL_ACCEPT_TOKENS = frozenset({
    "detalhes", "detalhe", "quero detalhes", "detalhes por favor",
    "mais detalhes", "quero saber mais", "manda", "manda ai",
    "manda aí", "pode ser", "sim", "isso", "quero sim",
    "pode mandar", "me fala", "fala", "me conta", "quero",
    "pode passar", "passa", "passa ai", "vai", "uhum", "claro",
    "show", "perfeito", "ok",
})

# Same context, but customer pivots to price. Route to price_inquiry.
_DETAIL_PRICE_TOKENS = frozenset({
    "quanto custa", "quanto e", "quanto eh", "preco", "preço",
    "valor", "qual o preco", "qual o preço", "qual o valor",
    "quanto", "qual valor",
})


def _matches_short_token_set(text: str, tokens: frozenset[str]) -> bool:
    """True iff the normalized, punctuation-stripped text equals one of
    the multi-word tokens OR starts with one followed by punctuation/space.

    The matcher is intentionally STRICT — we only want to short-circuit
    on unambiguous short replies. Long sentences fall through to the LLM
    even if they contain a token, because they may carry other intent
    (e.g. "detalhes da minha conta bancária" should NOT route to
    attribute_inquiry).
    """
    if not text:
        return False
    norm = _strip_accents(text.lower()).strip(" .,!?\n\t").rstrip("!")
    if norm in tokens:
        return True
    # Allow a single trailing clause like "sim por favor" / "detalhes pf".
    head = norm.split(" por favor")[0].split(" pf")[0].strip()
    if head in tokens:
        return True
    # Multi-word tokens with a leading short verb form ("quero detalhes")
    # also match if the entire norm equals one of them.
    return False


def _is_affirmative_reply(text: str) -> bool:
    if not text:
        return False
    norm = _strip_accents(text.lower()).strip(" .,!?")
    if norm in _AFFIRMATIVE_TOKENS:
        return True
    first = re.split(r"[\s,;]+", norm, maxsplit=1)[0] if norm else ""
    return first in {"sim", "isso", "ok", "pode", "claro", "exato", "perfeito"}


def _is_negative_reply(text: str) -> bool:
    if not text:
        return False
    norm = _strip_accents(text.lower()).strip(" .,!?")
    if norm in _NEGATIVE_TOKENS:
        return True
    if norm.startswith("nao ") or norm.startswith("não "):
        return True
    return False


async def triage_node(state: AgentState) -> dict:
    messages = state.get("messages") or []
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {"intent": "smalltalk"}

    last_text = (
        last_human.content if isinstance(last_human.content, str)
        else str(last_human.content)
    )

    # ── Sprint 2.6.4 — multi-product reference short-circuit ──────────────
    # When the previous turn left ``last_product_candidates`` set (recommend
    # was ambiguous) AND the customer references the whole group ("as duas",
    # "ambas", "todos"), route DIRECTLY to price_inquiry so the multi-quote
    # path fires without the LLM having to classify the pronoun phrase.
    if state.get("last_product_candidates"):
        from app.agent.nodes.price_inquiry import is_multi_product_reference
        if is_multi_product_reference(last_text):
            logger.info("triage multi_product_reference → price_inquiry")
            return {"intent": "price_inquiry"}

    # ── Sprint 2.6.2 — fuzzy-match confirmation handler ───────────────────
    # If the previous turn ended with "Você quis dizer X?", the next user
    # reply is almost certainly yes/no. We short-circuit the LLM here:
    #   yes → route to recommend (which consumes the stashed candidate);
    #   no  → route to smalltalk with a canned "ok, qual então?" reply.
    # Ambiguous reply (neither) clears the flag and proceeds with normal triage.
    pending = state.get("awaiting_match_confirmation")
    if pending:
        if _is_affirmative_reply(last_text):
            logger.info(
                "triage match_confirmation=yes product=%s",
                (pending or {}).get("name"),
            )
            return {"intent": "product_inquiry"}
        if _is_negative_reply(last_text):
            logger.info("triage match_confirmation=no")
            return {
                "intent": "smalltalk",
                "awaiting_match_confirmation": None,
                "match_decline_pending": True,
            }
        # Anything else: clear the flag, let normal triage classify.
        logger.info("triage match_confirmation=ambiguous_reply — clearing flag")

    # ── Sprint 2.6.10 — detail-offer accept/price short-circuit ──────────
    # When recommend emits "Posso te passar mais detalhes, ou prefere ver
    # pessoalmente na loja?", it sets ``awaiting_detail_choice=True``.
    # Felipe production logs (June 2026) showed "detalhes" being routed to
    # out_of_scope / help_request by the LLM. This short-circuit catches
    # the common short replies before the LLM gets a chance to misroute.
    #
    # The diagnostic log line below is REQUIRED in every triage turn — it
    # makes the missing-flag failure mode visible without re-running the
    # request. If a future regression breaks the recommend→triage chain
    # again, the log will show ``awaiting_detail_choice=None`` and we'll
    # spot it in one grep.
    detail_flag = state.get("awaiting_detail_choice")
    logger.info(
        "detail_choice_check awaiting_detail_choice=%r msg=%.80r",
        detail_flag, last_text,
    )
    if detail_flag:
        if _matches_short_token_set(last_text, _DETAIL_ACCEPT_TOKENS):
            logger.info("triage detail_choice=accept → attribute_inquiry")
            return {
                "intent": "attribute_inquiry",
                "awaiting_detail_choice": False,
            }
        if _matches_short_token_set(last_text, _DETAIL_PRICE_TOKENS):
            logger.info("triage detail_choice=price → price_inquiry")
            return {
                "intent": "price_inquiry",
                "awaiting_detail_choice": False,
            }
        # Anything else: clear the flag, fall through to normal classify.
        logger.info("triage detail_choice=unrelated — clearing flag")

    client = OpenAIClient()
    response = await client.chat(
        messages=[{"role": "user", "content": last_text}],
        system=SYSTEM_TRIAGE,
        max_tokens=50,
        temperature=0.0,
        json_mode=True,
    )

    try:
        intent = json.loads(response).get("intent", "smalltalk")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("triage_parse_failed response=%.80r", response)
        intent = "smalltalk"

    if intent not in _VALID_INTENTS:
        logger.info("triage unknown_intent=%r → smalltalk", intent)
        intent = "smalltalk"

    logger.info("triage intent=%s", intent)
    update = {"intent": intent}
    # Sprint 2.6.2 — clear the awaiting flag when we fall through normal
    # triage (i.e. the customer's reply was neither yes nor no).
    if pending is not None:
        update["awaiting_match_confirmation"] = None
    # Sprint 2.6.10 — same clearing logic for the detail-choice flag: if
    # we reach the LLM classification, the short-circuit didn't fire,
    # which means the customer changed subject. The flag is stale.
    if detail_flag:
        update["awaiting_detail_choice"] = False
    return update
