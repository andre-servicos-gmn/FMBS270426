"""Sprint 1.14 — deterministic anti-rerun guard for expensive / repetitive nodes.

Sprint 1.15 — smarter "is there new info?" heuristic. The previous version
just looked at message length (≥20 chars → assume new info). That mis-fired
in production: customers often type long messages that are just repeats /
small talk. The new heuristic checks whether the message contains any of:

    a) a product reference (tolerant matcher: name match incl. typos / spaces)
    b) a positional reference ("a primeira", "2ª")
    c) a pronominal reference ("essa", "gostei dessa")
    d) re-recommendation keywords ("outras opções", "mais barato", …)

Any of those → genuine new input, allow rerun.
None of those → block as rerun cego.

Nodes that ARE NOT subject to blocking by design (whitelist enforced by
NOT calling should_block_rerun, not by a list in this file):
    diagnose            — must keep advancing the slot flow
    smalltalk / faq     — short, idempotent responses
    close               — terminal confirmation
    handoff / out_of_scope_handoff / scheduling_inquiry
                        — short canned answers, safe to rerun
    product_selection / price_inquiry / product_detail / re_recommendation
                        — follow-ups, already deterministic enough

Only ``recommend`` and ``pitch_consultoria`` opt-in to the block check.
"""
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage

from app.agent.nodes._positional_reference import detect_positional_reference
from app.agent.nodes._product_match import match_product_tolerant
from app.agent.nodes._pronominal_reference import detect_pronominal_reference
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD_S = 60

# Re-recommendation cues: when present, the customer is clearly asking for
# a NEW round — even within the rerun window, that's new info.
_REREC_KEYWORDS = (
    "outra opcao", "outra opção", "outras opcoes", "outras opções",
    "mais barat", "mais em conta", "menos cara", "menos caro", "acessivel",
    "acessível", "mais avanc", "mais avançada", "top de linha", "premium",
    "mais leve", "mais pesad", "diferente",
)


def stamp_node_execution(node_name: str) -> dict:
    """Return a state-update dict that bookmarks the node's execution.

    Nodes call this at the END of their body and merge the result into their
    return dict, so the NEXT invocation of the same node can see the
    timestamp and consider blocking.
    """
    return {
        "last_node_executed": node_name,
        "last_node_executed_at": datetime.now(timezone.utc).isoformat(),
    }


def is_recent_rerun(
    state: AgentState, node_name: str, threshold_seconds: int = _DEFAULT_THRESHOLD_S
) -> bool:
    """Return True when ``node_name`` ran within the last ``threshold_seconds``."""
    last_node = state.get("last_node_executed")
    last_at = state.get("last_node_executed_at")
    if last_node != node_name or not last_at:
        return False
    try:
        last_dt = datetime.fromisoformat(last_at)
    except (TypeError, ValueError):
        return False
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - last_dt).total_seconds() < threshold_seconds


# Sprint 2.6.3 — when the previous human turn and the current one are
# essentially the same prompt, the customer is repeating; that's the case
# anti_rerun was built for. When they're substantially different, the
# customer is asking something new and we must let the node run.
_QUERY_SIMILARITY_BLOCK_THRESHOLD = 0.75


def _previous_user_message(state: AgentState) -> str | None:
    """Return the SECOND-to-last HumanMessage content, or None."""
    messages = state.get("messages") or []
    humans: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            humans.append(text)
    if len(humans) < 2:
        return None
    return humans[-2]


def _query_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _carries_new_info(user_msg: str, state: AgentState) -> tuple[bool, str]:
    """Return (has_new_info, reason). Sprint 1.15 + 2.6.3 — relax bias:
    different query than the previous human turn = new info by default.

    'New info' is any of:
        - a product reference (tolerant matcher hit, any confidence)
        - a positional reference ("a primeira")
        - a pronominal reference ("essa", "gostei dessa")
        - a re-recommendation keyword ("outras opções", "mais barato", ...)
        - (Sprint 2.6.3) the message is meaningfully different from the
          previous human turn (SequenceMatcher ratio < threshold)
    """
    if not user_msg:
        return False, "empty"

    products = state.get("recommended_products") or []

    if products:
        match = match_product_tolerant(user_msg, products)
        if match.product is not None:
            return True, f"name_match_{match.method}"

        idx = detect_positional_reference(user_msg, len(products))
        if idx is not None:
            return True, f"positional_idx_{idx}"

        if detect_pronominal_reference(user_msg):
            return True, "pronominal"

    # Re-recommendation cues work even without products on the table.
    msg_lower = user_msg.lower()
    for kw in _REREC_KEYWORDS:
        if kw in msg_lower:
            return True, f"rerec_keyword={kw}"

    # Sprint 2.6.3 — fall-through unblocker: when the current and prior
    # human messages are substantially different, the customer is asking
    # something NEW (e.g. a different product name) and we must let the
    # node run. Without this, recommend's anti_rerun blocked a 2nd product
    # inquiry as if it were a vague follow-up.
    prev = _previous_user_message(state)
    if prev is not None:
        ratio = _query_similarity(prev, user_msg)
        if ratio < _QUERY_SIMILARITY_BLOCK_THRESHOLD:
            logger.info(
                "anti_rerun query_diverged prev=%.40r curr=%.40r ratio=%.2f",
                prev, user_msg, ratio,
            )
            return True, f"query_diverged_ratio_{ratio:.2f}"

    return False, "no_signal"


def should_block_rerun(
    state: AgentState,
    node_name: str,
    user_msg: str = "",
    threshold_seconds: int = _DEFAULT_THRESHOLD_S,
) -> bool:
    """Decide whether to short-circuit the node with a fallback response.

    Block when the same node ran within the threshold window AND the
    customer's message doesn't carry recognisable new information (no
    product reference, no positional/pronominal cue, no re-rec keyword).
    """
    if not is_recent_rerun(state, node_name, threshold_seconds):
        return False

    has_new, reason = _carries_new_info(user_msg, state)
    if has_new:
        logger.info(
            "anti_rerun allow_rerun node=%s reason=%s", node_name, reason
        )
        return False

    logger.info(
        "anti_rerun BLOCKED node=%s msg_preview=%.40r", node_name, user_msg
    )
    return True


# Canned fallback messages the orchestrator returns when blocking. Single
# block, short, contextual — gives the customer a path forward without
# spending another LLM call repeating the same content.
_FALLBACK_MESSAGES: dict[str, str] = {
    # Sprint 2.6.2 — the legacy recommend fallback claimed "te mostrei
    # algumas opções acima" which was a lie post-2.6 (recommend now answers
    # one product at a time, never lists multiples). Replaced with a neutral
    # ask for more detail.
    "recommend": (
        "Pode me dar mais detalhes do que você procura?"
    ),
    "pitch_consultoria": (
        "Te expliquei a Consultoria acima. Quer agendar, tirar dúvida sobre "
        "algum ponto específico, ou prefere outra coisa?"
    ),
}


def fallback_message_for(node_name: str) -> str:
    """Return the canned fallback used when blocking a rerun of ``node_name``."""
    return _FALLBACK_MESSAGES.get(
        node_name,
        "Pode me dar mais detalhes do que você procura?",
    )
