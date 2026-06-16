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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.prompts import SYSTEM_TRIAGE
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# ── Sprint 2.7.1 — history helper (also used by smalltalk via import) ──────

# Last N messages we send to the LLM. 6 = 3 customer/agent exchanges,
# enough to remember "Qual você procura?" + the candidate list without
# bloating tokens. The window slides — we always include the current
# customer message at the end.
_TRIAGE_HISTORY_WINDOW = 6


def recent_chat_history(
    messages: list[BaseMessage] | None,
    *,
    window: int = _TRIAGE_HISTORY_WINDOW,
) -> list[dict[str, str]]:
    """Convert the last ``window`` Human/AI messages to OpenAI chat format.

    Sprint 2.7.1 — used by triage and smalltalk to give the LLM the
    immediate conversational context. SystemMessage / ToolMessage / etc.
    are skipped silently. Returns an empty list when there's no usable
    history (first turn, no messages).
    """
    if not messages:
        return []
    converted: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            converted.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            converted.append({"role": "assistant", "content": str(m.content)})
    return converted[-window:] if len(converted) > window else converted


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


# ── Sprint 2.7.1 — candidate-selection helper ─────────────────────────────
#
# When recommend emits "Qual você procura? • A • B • C", it persists the
# list to ``last_product_candidates`` AND flips ``awaiting_candidate_choice``.
# This deterministic matcher tries to resolve the customer's reply against
# the candidates BEFORE the LLM gets a chance to misclassify a short
# answer ("Primeira", "2026") as smalltalk. Conservative by design — only
# fires when the answer is unambiguously one of the listed candidates.

# Character class covers the ordinal indicators ``ª`` (U+00AA) and ``º``
# (U+00BA) — these are NOT touched by NFD-based accent stripping, so we
# match them explicitly. ``\b`` in Python's `re` treats `ª`/`º` as non-word
# characters, so a trailing ordinal indicator naturally ends the word.
_POSITIONAL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"(?:\bprimeira\b|\bprimeiro\b|\b1[aoªº]?\b|\ba\s*1\b|\bo\s*1\b)"), 0),
    (re.compile(r"(?:\bsegunda\b|\bsegundo\b|\b2[aoªº]?\b|\ba\s*2\b|\bo\s*2\b)"), 1),
    (re.compile(r"(?:\bterceira\b|\bterceiro\b|\b3[aoªº]?\b|\ba\s*3\b|\bo\s*3\b)"), 2),
    (re.compile(r"(?:\bquarta\b|\bquarto\b|\b4[aoªº]?\b|\ba\s*4\b|\bo\s*4\b)"), 3),
    (re.compile(r"(?:\bquinta\b|\bquinto\b|\b5[aoªº]?\b|\ba\s*5\b|\bo\s*5\b)"), 4),
    (re.compile(r"\b(?:ultima|ultimo)\b"), -1),
)


def _detect_positional_index(norm_text: str) -> int | None:
    """Return the index (0-based) the customer pointed at, or None.

    ``norm_text`` must already be lowercase + accent-stripped.
    """
    for pattern, idx in _POSITIONAL_PATTERNS:
        if pattern.search(norm_text):
            return idx
    return None


def _select_by_distinctive_token(
    norm_text: str, candidates: list[dict]
) -> dict | None:
    """If the customer's text contains a token that appears in EXACTLY ONE
    candidate's normalized name, return that candidate. Otherwise None.

    This catches year disambiguation ("2026" → only the Kronos 2026),
    partial-name selection ("Hugo Russo" → only the Kronos 2025 Hugo
    Russo Capa), and brand refinement when one candidate stands out.
    """
    tokens = [t for t in re.findall(r"[a-z0-9]+", norm_text) if len(t) >= 2]
    if not tokens:
        return None
    for token in tokens:
        owners = []
        for c in candidates:
            name_norm = _strip_accents(str(c.get("name") or "").lower())
            if re.search(rf"\b{re.escape(token)}\b", name_norm):
                owners.append(c)
        # Token must uniquely identify a candidate.
        if len(owners) == 1:
            return owners[0]
    return None


def try_select_candidate(text: str, candidates: list[dict]) -> dict | None:
    """Best-effort: return a single candidate from ``candidates`` if the
    customer's ``text`` clearly points at one, else None.

    Strategy (in order, first hit wins):
        1. Positional ("primeira" / "a 2" / "última") — index into list.
        2. Distinctive token — one token matches exactly one candidate name.

    None means "ambiguous or unrelated"; the caller should fall through
    to the LLM (which now has history per Part 1 to make the call).
    """
    if not text or not candidates:
        return None
    norm = _strip_accents(text.lower()).strip(" .,!?\n\t")

    idx = _detect_positional_index(norm)
    if idx is not None:
        try:
            return candidates[idx]
        except IndexError:
            # User said "terceira" but only 2 candidates — don't guess.
            return None

    chosen = _select_by_distinctive_token(norm, candidates)
    if chosen is not None:
        return chosen

    return None


# ── Sprint 2.7.3 — budget-mention detector ─────────────────────────────────
#
# Felipe's bug: "Quero uma raquete até 2k" — agent quoted a R$ 2.999
# racket. There's no price filter logic anywhere in the agent. The
# business rule (Sprint 2.6.9) forbids price-range vitrine: agent must
# NOT list rackets by budget. The right path is conducting the customer
# to the Consultoria, which IS the service designed for "perfil +
# orçamento + jogo → recomendação".
#
# Detector returns the parsed budget in REAIS, or None when no pattern
# fired. Caller (triage) further restricts: only fires when no product
# is active in the conversation (avoids hijacking "eu tenho R$3.500"
# said mid-conversation about something else).

_PRICE_WORD_NUMBERS: dict[str, int] = {
    "um": 1, "uma": 1,
    "dois": 2, "duas": 2,
    "tres": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
    "quinze": 15,
    "vinte": 20,
}

# Trigger words that ALMOST ALWAYS introduce a budget statement.
_BUDGET_TRIGGERS_RE = (
    r"\b(?:ate|maximo|max|no\s+maximo|"
    r"em\s+volta\s+de|por\s+volta\s+de|"
    r"uns?\b\s+(?:r\$\s*)?(?=\d))"
)

# digit + optional scale
_BUDGET_DIGIT_RE = re.compile(
    _BUDGET_TRIGGERS_RE
    + r"\s*(?:r\$\s*)?(\d{1,5})\s*(k|mil|reais|rs)?\b",
    flags=re.IGNORECASE,
)
# word number + "mil"
_BUDGET_WORD_RE = re.compile(
    r"\b(?:ate|maximo|max|no\s+maximo)\s+"
    r"(um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito|nove|dez|quinze|vinte)"
    r"\s+mil\b",
    flags=re.IGNORECASE,
)

# Minimum reais — below this, ignore (avoids "uns 50" in casual chat).
_BUDGET_MIN_REAIS = 100


def _extract_price_range(text: str) -> int | None:
    """Return the customer's stated max budget in REAIS, or None.

    Matches conservative patterns only — trigger words ("até", "máximo",
    "no máximo", "uns") followed by a number. Returns None for:
      - sentences without a budget trigger ("quanto custa a Proteo?")
      - parsed values < ``_BUDGET_MIN_REAIS`` ("uns 50" → None)
      - sentences where the trigger word is present but no number follows

    Detected scale: "k"/"mil" multiplies digits by 1000. Bare digits
    (≥100) treated as reais directly.
    """
    if not text:
        return None
    norm = _strip_accents(text.lower())
    # Squash multiple spaces so "  no   maximo  2k" matches.
    norm = re.sub(r"\s+", " ", norm).strip()

    # 1. Digit form ("até 2k", "no máximo 1500 reais", "uns 2000")
    m = _BUDGET_DIGIT_RE.search(norm)
    if m:
        n = int(m.group(1))
        scale = (m.group(2) or "").lower()
        if scale in ("k", "mil"):
            n *= 1000
        if n >= _BUDGET_MIN_REAIS:
            return n

    # 2. Word number + "mil" ("até dois mil", "no máximo cinco mil")
    m = _BUDGET_WORD_RE.search(norm)
    if m:
        n = _PRICE_WORD_NUMBERS.get(m.group(1), 0) * 1000
        if n >= _BUDGET_MIN_REAIS:
            return n

    return None


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

    # ── Sprint 2.7.1 — candidate-choice short-circuit ────────────────────
    # When the previous turn emitted "Qual você procura? • A • B • C",
    # recommend persists the list + flips ``awaiting_candidate_choice``.
    # Deterministic matcher tries to resolve positional ("primeira"),
    # year/token ("2026"), or partial-name ("Hugo Russo") selections
    # BEFORE the LLM. If unambiguous → route as if customer had nominated
    # the product (reuse ``awaiting_match_confirmation`` so recommend's
    # existing pending-resolution path takes over). If 0 or >1 → fall
    # through to the LLM, which now has history (Part 1) to make the call.
    awaiting_choice = state.get("awaiting_candidate_choice")
    candidates = state.get("last_product_candidates") or []
    logger.info(
        "candidate_choice_check awaiting_candidate_choice=%r n_candidates=%d msg=%.80r",
        awaiting_choice, len(candidates), last_text,
    )
    if awaiting_choice and len(candidates) >= 2:
        # Multi-product price reference ("as duas") wins over single-pick
        # — already handled above (line ~148). Here we only do single-pick.
        selected = try_select_candidate(last_text, candidates)
        if selected is not None:
            logger.info(
                "triage candidate_choice=selected name=%s",
                selected.get("name"),
            )
            return {
                # Route to recommend → the awaiting_match_confirmation
                # branch promotes the stashed candidate to active.
                "intent": "product_inquiry",
                "awaiting_match_confirmation": selected,
                "awaiting_candidate_choice": False,
                "last_product_candidates": None,
            }
        logger.info(
            "triage candidate_choice=ambiguous_or_unrelated — falling through to LLM"
        )

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

    # ── Sprint 2.7.3 — budget-mention short-circuit ──────────────────────
    # Customer mentions a price ceiling ("até 2k", "no máximo 1500") AND
    # there's no active product → route to help_request with a flag so
    # the Consultoria pitch acknowledges the budget. Restriction 1: no
    # active product (else "eu tenho R$3.500" mid-conversation would
    # hijack price_inquiry). Restriction 2: detector ignores < R$100.
    # First-message scenario ("oi, quero uma raquete até 2k") fires
    # because ``recommended_products`` is empty on conversation open.
    has_active_product = bool(state.get("recommended_products"))
    if not has_active_product:
        max_reais = _extract_price_range(last_text)
        if max_reais is not None:
            logger.info(
                "triage budget_mention max_reais=%d → help_request "
                "(price_range_mentioned=True)",
                max_reais,
            )
            return {
                "intent": "help_request",
                "price_range_mentioned": True,
            }

    # Sprint 2.7.1 — pass the last ~6 messages so the LLM can use the
    # immediate context. The current customer message is already the
    # tail of this list (add_messages reducer appended it on entry).
    # When there's no history (first turn), this collapses to a single
    # user message — same input the pre-2.7.1 code sent. Backward-safe.
    history = recent_chat_history(messages)
    if not history:
        history = [{"role": "user", "content": last_text}]

    client = OpenAIClient()
    response = await client.chat(
        messages=history,
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
    # Sprint 2.7.1 — same for the candidate-choice flag. If we got here,
    # the LLM is making the call (with history context). The stash is
    # consumed.
    if awaiting_choice:
        update["awaiting_candidate_choice"] = False
    return update
