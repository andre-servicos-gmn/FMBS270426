import json
import logging
import re
import unicodedata

from langchain_core.messages import HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.nodes._product_match import match_product_tolerant
from app.agent.prompts import SYSTEM_TRIAGE
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

# Sprint 1.10: classical + follow-up intents. Follow-up intents are only
# meaningful when state["recommended_products"] is non-empty AND
# last_recommendation_at is set. We accept any name here and let
# _triage_router demote follow-up intents to "smalltalk" if the state
# isn't post-recommendation.
_VALID_INTENTS = {
    # classical
    "faq",
    "diagnose",
    "recommend",
    "close",
    "consultoria",
    "handoff",
    "smalltalk",
    # Sprint 2.0 — pivot to qualifier: customer asks for a recommendation
    # without naming a specific racket. Routed to the diagnose flow which
    # ends in the Consultoria offer instead of an active recommendation.
    "bare_recommendation_request",
    # follow-ups (post-recommendation only)
    "price_inquiry",
    "product_selection",
    "re_recommendation",
    "product_detail",
    "out_of_scope",
    # Sprint 1.14 — customer wants to book the Consultoria (handoff)
    "scheduling_inquiry",
}


# Sprint 2.1 — opinion-seeking patterns. When a customer who was tagged as
# "determined" suddenly asks for our opinion ("essa serve mesmo pra mim?",
# "você acha que é boa?"), we flip them to "exploring" so the agent stops
# the close-flow and runs diagnose + Consultoria offer instead.
_OPINION_SEEKING_PATTERNS = (
    "voce acha",
    "vc acha",
    "tu acha",
    "voces acham",
    "voce indica",
    "vc indica",
    "voce recomenda",
    "vc recomenda",
    "serve pra mim",
    "serve mesmo",
    "boa pra mim",
    "boa pro meu",
    "essa e boa",
    "essa eh boa",
    "vale a pena",
    "compensa",
    "boa escolha",
    "e a melhor",
    "eh a melhor",
)


# Sprint 2.4 — purchase-intent phrases. When the matcher fails but the
# customer clearly wants to buy a named racket, we still mark them as
# "determined" so REFERENCE-NÃO determined can offer alternatives.
_PURCHASE_PATTERNS = (
    "quero comprar",
    "queria comprar",
    "vou comprar",
    "vou levar",
    "queria fechar",
    "quero fechar",
    "fechar com a",
    "fechar com o",
    "comprar a ",
    "comprar o ",
    "levar a ",
    "levar o ",
)

_PURCHASE_TARGET_RE = re.compile(
    r"\b(?:quero|queria|vou)\s+(?:comprar|levar|fechar(?:\s+com)?)\s+"
    r"(?:a|o|uma|um)?\s*([^\?\.,!\n]+)",
    re.IGNORECASE,
)


def _has_purchase_intent(text: str) -> bool:
    norm = _strip_accents(text.lower())
    return any(p in norm for p in _PURCHASE_PATTERNS)


def _extract_purchase_target(text: str) -> str | None:
    """Best-effort extraction of the product name from a purchase phrase.

    Returns None when the captured noun is too short, generic filler
    ("raquete", "alguma coisa"), or just punctuation. The conservative
    rejections protect against false positives like "quero comprar uma
    raquete" being treated as a determined customer.
    """
    m = _PURCHASE_TARGET_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip(" .,!?\n")
    if len(raw) < 3:
        return None
    norm = _strip_accents(raw.lower())
    if norm in ("raquete", "raquetes", "uma raquete", "alguma", "qualquer", "alguma coisa"):
        return None
    return raw


# Sprint 2.4 — affirmative / negative tokens for the yes/no follow-up that
# closes the REFERENCE-NÃO determined "posso ajudar a ver outras opções?"
# question.
_AFFIRMATIVE_EXACT = frozenset({
    "sim", "pode", "claro", "ok", "aceito", "manda", "manda bala",
    "bora", "vamos", "isso", "isso ai", "isso aí", "isso aí!",
    "por favor", "por favor!", "uhum", "yes", "blz", "beleza",
    "tranquilo", "vamo", "topo", "topa",
})

_NEGATIVE_EXACT = frozenset({
    "nao", "não", "n", "nope", "negativo", "nao obrigado",
    "não obrigado", "nao precisa", "não precisa", "deixa pra la",
    "deixa pra lá", "sem", "agradeco", "agradeço",
})


def _is_affirmative_reply(text: str) -> bool:
    norm = _strip_accents(text.lower()).strip(" .,!?")
    if norm in {_strip_accents(s) for s in _AFFIRMATIVE_EXACT}:
        return True
    # First token of the reply: "sim, pode" / "sim por favor" / "sim!" all
    # start with "sim". Split on commas + whitespace to isolate it.
    first = re.split(r"[\s,;]+", norm, maxsplit=1)[0] if norm else ""
    return first in {"sim", "pode", "claro", "manda", "bora", "topo", "ok", "uhum"}


def _is_negative_reply(text: str) -> bool:
    norm = _strip_accents(text.lower()).strip(" .,!?")
    if norm in {_strip_accents(s) for s in _NEGATIVE_EXACT}:
        return True
    if norm.startswith("nao ") or norm.startswith("não "):
        return True
    return False


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _post_recommendation_state(state: AgentState) -> bool:
    """True when the customer has just received a recommendation."""
    products = state.get("recommended_products") or []
    timestamp = state.get("last_recommendation_at")
    return bool(products) and bool(timestamp)


def _looks_like_opinion_seek(message: str) -> bool:
    """True when the message reads as 'you-tell-me' rather than 'I-want-this'."""
    if not message:
        return False
    norm = _strip_accents(message.lower())
    return any(p in norm for p in _OPINION_SEEKING_PATTERNS)


# Sprint 2.1.1 — generic words that should not count toward loose name-token
# overlap (otherwise "quero uma raquete" would match every racket name).
_GENERIC_WORDS = {
    "raquete", "raquetes", "pala", "palas", "de", "da", "do", "a", "o",
    "uma", "um", "para", "pra", "com", "sem", "tem", "tenho", "voces",
    "voce", "vc", "vcs", "oi", "ola", "ai", "tipo", "modelo",
}

# Distinctive tokens must be at least this long to count.
_LOOSE_MIN_TOKEN_LEN = 2
# Minimum number of overlapping distinctive tokens to call it a loose match.
_LOOSE_MIN_OVERLAP = 2


def _loose_name_match(message: str, candidates: list[dict]) -> dict | None:
    """Fallback for ``match_product_tolerant`` — counts distinctive token
    overlap between the message and each candidate name.

    Why this exists (Sprint 2.1.1): the tolerant matcher requires the
    product name to be (almost) entirely present in the query. Real users
    say things like "tem a Carbon X5 aí?" — only the "core" identifier
    ("Carbon X5") appears. The tolerant matcher misses; this fallback
    catches it by token overlap.

    Returns the candidate with the highest overlap among those that pass
    the ``_LOOSE_MIN_OVERLAP`` threshold; ties broken by the candidate's
    position in the input list (i.e. retriever's similarity ranking).
    """
    if not message or not candidates:
        return None

    msg_norm = _strip_accents(message.lower())
    msg_tokens = set(re.findall(r"[a-z0-9]+", msg_norm))
    if not msg_tokens:
        return None

    best: dict | None = None
    best_count = 0
    for product in candidates:
        name_norm = _strip_accents(str(product.get("name", "")).lower())
        name_tokens = [
            t for t in re.findall(r"[a-z0-9]+", name_norm)
            if t not in _GENERIC_WORDS and len(t) >= _LOOSE_MIN_TOKEN_LEN
        ]
        if len(name_tokens) < _LOOSE_MIN_OVERLAP:
            continue
        overlap = sum(1 for t in name_tokens if t in msg_tokens)
        if overlap >= _LOOSE_MIN_OVERLAP and overlap > best_count:
            best_count = overlap
            best = product
    return best


async def _try_match_named_product(message: str) -> dict | None:
    """Best-effort: search the catalog by the customer message; return product or None.

    Used in triage to decide whether the customer is "determined" (named a
    specific product that exists in the catalog). Strategy:

    1. Ask the retriever for the top-5 by embedding similarity. Empty list →
       no determined signal possible, return None.
    2. Run ``match_product_tolerant`` (Sprint 1.15 4-layer matcher) — handles
       full-name queries with typos/spaces.
    3. If the tolerant matcher misses (typical for short queries like
       "Carbon X5" embedded in noise like "tem a Carbon X5 aí?"), fall back
       to ``_loose_name_match`` — distinctive-token overlap.

    Any retriever/db failure falls through to None so triage never crashes.
    """
    if not message or len(message.strip()) < 3:
        return None
    try:
        from app.rag.retriever import search_products
        from app.storage.db import get_session

        async with get_session() as session:
            candidates = await search_products(
                session, message, {"min_stock": 1}, k=5
            )
        if not candidates:
            return None
        matched = match_product_tolerant(message, candidates).product
        if matched is not None:
            return matched
        # Tolerant matcher missed — try the loose fallback.
        loose = _loose_name_match(message, candidates)
        if loose is not None:
            logger.info(
                "triage loose_name_match_used product=%s",
                loose.get("name"),
            )
        return loose
    except Exception as exc:
        logger.warning("triage_catalog_match_failed: %s", exc)
        return None


async def triage_node(state: AgentState) -> dict:
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_human:
        return {"intent": "smalltalk"}

    last_text = (
        last_human.content if isinstance(last_human.content, str)
        else str(last_human.content)
    )

    # ── Sprint 2.4 — awaiting-alternatives decision (yes / no) ──────────
    # When REFERENCE-NÃO determined asked "posso ajudar a ver outras
    # opções?" the previous turn, we read the short reply here and either
    # transition to exploring (yes) or emit a graceful goodbye via
    # smalltalk (no). Either branch clears the flag so we don't loop.
    if state.get("awaiting_alternatives_decision"):
        if _is_affirmative_reply(last_text):
            logger.info("triage awaiting_alternatives → yes → exploring")
            profile = dict(state.get("player_profile") or {})
            profile["modelo_desejado"] = "nenhum"
            return {
                "intent": "bare_recommendation_request",
                "customer_intent_path": "exploring",
                "awaiting_alternatives_decision": False,
                "player_profile": profile,
                "recommended_products": [],
                "last_recommendation_at": None,
            }
        if _is_negative_reply(last_text):
            logger.info("triage awaiting_alternatives → no → graceful goodbye")
            return {
                "intent": "smalltalk",
                "awaiting_alternatives_decision": False,
                "goodbye_pending": True,
            }
        # Anything else: clear the flag and let normal triage proceed.
        logger.info("triage awaiting_alternatives → ambiguous → clearing flag")

    # Sprint 1.10: tell the LLM whether we're in post-recommendation state so
    # it picks the right intent space. We also surface the names of the
    # recommended products so price/selection/detail can match by name.
    post_rec = _post_recommendation_state(state)
    if post_rec:
        product_names = [
            str(p.get("name", "")) for p in (state.get("recommended_products") or [])
        ]
        state_block = (
            "Estado: post_recommendation\n"
            f"Raquetes já apresentadas ao cliente: {', '.join(filter(None, product_names))}"
        )
    else:
        state_block = "Estado: pré-recomendação"

    user_content = (
        f"{state_block}\n\n"
        f"Mensagem do cliente: {last_text}"
    )

    client = OpenAIClient()
    response = await client.chat(
        messages=[{"role": "user", "content": user_content}],
        system=SYSTEM_TRIAGE,
        max_tokens=50,
        temperature=0.0,
        json_mode=True,
    )

    try:
        intent = json.loads(response).get("intent", "smalltalk")
        if intent not in _VALID_INTENTS:
            intent = "smalltalk"
    except (json.JSONDecodeError, AttributeError):
        logger.warning("triage_parse_failed response=%.80r", response)
        intent = "smalltalk"

    update: dict = {"intent": intent}

    # Sprint 2.4 — if the awaiting-alternatives flag survived an ambiguous
    # reply above, clear it now so subsequent turns are clean.
    if state.get("awaiting_alternatives_decision"):
        update["awaiting_alternatives_decision"] = False

    # ── Sprint 2.1 — customer intent path detection ─────────────────────────
    # Opinion-seek override: determined customer suddenly asking for opinion
    # → flip to exploring + drop the determined shortcuts so the next router
    # pass lands on diagnose → consultoria_offer (the explorer path).
    if (
        state.get("customer_intent_path") == "determined"
        and _looks_like_opinion_seek(last_text)
    ):
        logger.info(
            "triage opinion_seek_detected — flipping determined → exploring"
        )
        update["customer_intent_path"] = "exploring"
        update["intent"] = "bare_recommendation_request"
        # Drop the named model so recommend reaches PROFILE mode (which
        # delegates to consultoria_offer) instead of confirming stock again.
        profile = dict(state.get("player_profile") or {})
        profile["modelo_desejado"] = "nenhum"
        update["player_profile"] = profile
        # Leave post-recommendation state so the router doesn't route us
        # through the rerun shortcut.
        update["recommended_products"] = []
        update["last_recommendation_at"] = None
        logger.info("triage intent=%s post_rec=%s path=exploring", update["intent"], post_rec)
        return update

    if intent == "bare_recommendation_request":
        update["customer_intent_path"] = "exploring"
    elif intent in ("recommend", "diagnose"):
        matched = await _try_match_named_product(last_text)
        # Sprint 2.1.1 — explicit debug log so production traces show
        # whether catalog match fired.
        logger.info(
            "triage determined_check intent=%s matched=%s",
            intent, bool(matched),
        )

        # Sprint 2.4 — when the matcher fails BUT the customer has clear
        # purchase intent + names something, we still mark determined and
        # let recommend's REFERENCE-NÃO determined branch ask about
        # alternatives. modelo_desejado gets the raw target so the message
        # echoes back exactly what the customer asked for.
        determined_name: str | None = None
        if matched is not None:
            determined_name = matched.get("name") or last_text.strip()
        elif _has_purchase_intent(last_text):
            extracted = _extract_purchase_target(last_text)
            if extracted:
                determined_name = extracted
                logger.info(
                    "triage purchase_no_match_detected target=%s", extracted
                )

        if determined_name:
            update["customer_intent_path"] = "determined"
            # Normalize intent so _start_router doesn't shortcut to diagnose
            # next turn (Sprint 2.1).
            if intent != "recommend":
                logger.info("triage intent rewrite: %s → recommend", intent)
            update["intent"] = "recommend"
            profile = dict(state.get("player_profile") or {})
            if not profile.get("modelo_desejado") or str(
                profile.get("modelo_desejado")
            ).strip().lower() in ("nenhum", "nenhuma", ""):
                profile["modelo_desejado"] = determined_name
                update["player_profile"] = profile
            logger.info(
                "triage determined_detected matched=%s",
                determined_name,
            )

    logger.info(
        "triage intent=%s post_rec=%s path=%s",
        update["intent"], post_rec,
        update.get("customer_intent_path") or state.get("customer_intent_path"),
    )
    return update
