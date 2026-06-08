"""Sprint 2.0 — Consultoria offer node (qualifier pivot).

Replaces the old "active recommendation" in PROFILE mode. When the customer
finishes the diagnose without naming a specific racket (or asks "what do
you recommend?"), this node composes a personalized invitation to the
*Consultoria Base Sports* using their profile.

Deliberately AVOIDS recommending any specific racket — that's the whole
point of the Sprint 2.0 strategic pivot.

Sprint 2.1 also exports ``maybe_add_subtle_consultoria_offer`` — a tiny,
LLM-free helper that appends a 1-line "Consultoria as safety net" pitch
to a determined customer's reply, capped at one mention per conversation.
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.message_splitter import parse_messages
from app.agent.nodes._pitch_classification import (
    IMMEDIATE_TYPES,
    QuestionType,
    is_immediate,
)
from app.agent.prompts import build_consultoria_offer_prompt
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Sprint 2.3 — contextual pitch presets ───────────────────────────────────

def _preset_price(preco: int) -> str:
    return (
        f"É um bom investimento numa raquete que serve mesmo pro seu "
        f"perfil. Caso queira ter certeza absoluta antes de fechar, "
        f"a gente oferece a *Consultoria Base Sports* — você testa a "
        f"raquete em quadra antes da decisão. Investimento de *R$ {preco}*, "
        f"100% abatido na compra. Quer saber como funciona?"
    )


def _preset_fitness(preco: int) -> str:
    return (
        f"Pra essa pergunta especificamente, a *Consultoria Base Sports* "
        f"é o caminho certo. A gente faz uma análise técnica do seu perfil "
        f"+ teste em quadra com a raquete pra confirmar se realmente "
        f"combina. Investimento de *R$ {preco}*, abatido na compra. "
        f"Quer saber como funciona?"
    )


def _preset_comfort(preco: int) -> str:
    return (
        f"Conforto e prevenção de lesão são exatamente o tipo de coisa "
        f"que a *Consultoria Base Sports* analisa. A gente avalia seu "
        f"histórico, biomecânica e testa a raquete em quadra antes da "
        f"compra. Investimento de *R$ {preco}*, abatido se fechar. "
        f"Quer agendar?"
    )


def _preset_default(preco: int) -> str:
    return (
        f"Caso queira ter certeza absoluta que ela é a ideal pro seu jogo, "
        f"a gente oferece a *Consultoria Base Sports* com teste em quadra "
        f"(*R$ {preco}*, abatido na compra da raquete). Quer saber como funciona?"
    )


# Map: QuestionType → (preset_builder). All presets must mention the price
# and the abatimento, and end with a question — invariants asserted in tests.
# Sprint 2.4: STOCK moved to default preset (no longer a "specific" preset
# because REFERENCE-SIM determined doesn't call the helper anymore — stock
# confirmation is now pitch-free by design).
_PRESET_BUILDERS = {
    QuestionType.PRICE: _preset_price,
    QuestionType.FITNESS: _preset_fitness,
    QuestionType.COMFORT: _preset_comfort,
    # All DELAYED types share the default preset — less specific, less
    # aggressive (and the timing already filters when they fire).
    QuestionType.STOCK: _preset_default,
    QuestionType.WEIGHT: _preset_default,
    QuestionType.MATERIAL: _preset_default,
    QuestionType.BALANCE: _preset_default,
    QuestionType.OTHER: _preset_default,
}


def _build_preset(question_type: QuestionType, preco: int) -> str:
    builder = _PRESET_BUILDERS.get(question_type, _preset_default)
    return builder(preco)


# Back-compat alias for legacy code paths that still want the default text.
def _subtle_offer_text(preco: int) -> str:
    return _preset_default(preco)


# ── Sprint 2.1 / 2.3 — subtle Consultoria pitch helper ───────────────────────

def maybe_add_subtle_consultoria_offer(
    state: AgentState,
    response_blocks: list[str],
    question_type: QuestionType = QuestionType.OTHER,
    *,
    is_raquete: bool = True,
) -> tuple[list[str], dict]:
    """Append a one-line Consultoria pitch for determined customers.

    Sprint 2.3 adds CONTEXT and TIMING:

    - ``question_type`` decides WHICH preset text to use.
    - IMMEDIATE types (PRICE, STOCK, FITNESS, COMFORT) may emit on the
      first determined question; DELAYED types wait for the second.

    Hard guardrails (preserved from Sprint 2.1):
    - Only for ``customer_intent_path == "determined"``.
    - Cap of 1 mention per conversation (``consultoria_mentioned_count``).
    - Suppressed in handoff.
    - Suppressed when ``consultoria_enabled`` is False.

    Returns ``(new_blocks, state_update)``. ``state_update`` always carries
    the ``determined_question_count`` increment (so callers don't have to
    bookkeep it); ``consultoria_mentioned_count`` is set to 1 only when
    the pitch was actually appended.
    """
    update: dict = {}

    # Sprint 2.6 — ``customer_intent_path`` was removed. Every customer who
    # reaches a product follow-up (price_inquiry / product_detail / etc.)
    # is effectively a "determined" customer by definition, so the gate
    # collapses to the cap + handoff + config + raquete checks.
    new_count = int(state.get("determined_question_count") or 0) + 1
    update["determined_question_count"] = new_count

    if int(state.get("consultoria_mentioned_count") or 0) > 0:
        return response_blocks, update
    if state.get("needs_handoff"):
        return response_blocks, update

    settings = get_settings()
    if not getattr(settings, "consultoria_enabled", True):
        return response_blocks, update

    # Sprint 2.5 — Consultoria só faz sentido pra raquete de praia. Outros
    # produtos (bolas, vestuário, acessórios) não recebem o pitch sutil.
    if not is_raquete:
        logger.info(
            "subtle_consultoria_offer skipped product_not_raquete type=%s",
            question_type.value,
        )
        return response_blocks, update

    # Timing: DELAYED types only fire from the 2nd determined question on.
    if not is_immediate(question_type) and new_count < 2:
        logger.info(
            "subtle_consultoria_offer delayed_skip type=%s count=%d",
            question_type.value, new_count,
        )
        return response_blocks, update

    preco = getattr(settings, "consultoria_preco", 350)
    pitch_text = _build_preset(question_type, preco)
    new_blocks = list(response_blocks) + [pitch_text]
    update["consultoria_mentioned_count"] = 1
    logger.info(
        "subtle_consultoria_offer appended type=%s count=%d preco=%d",
        question_type.value, new_count, preco,
    )
    return new_blocks, update


def _profile_summary(profile: dict) -> str:
    """Render the customer's profile as a short PT-BR phrase for the LLM."""
    parts: list[str] = []

    level = (profile.get("nivel_jogo") or "").strip()
    if level:
        parts.append(f"nível {level}")

    lesoes = (profile.get("lesoes") or "").strip().lower()
    if lesoes and lesoes != "nenhuma":
        region = (profile.get("regiao_lesao") or "").strip()
        if region and region != "nenhuma":
            parts.append(f"com dor no {region.replace('_', ' ')}")
        else:
            parts.append(f"com {lesoes}")

    prev = (profile.get("esporte_raquete_previo") or "").strip().lower()
    if prev and prev not in ("nenhum", "nao_aplicavel"):
        parts.append(f"vindo de {prev}")

    return ", ".join(parts) if parts else "buscando uma raquete"


async def consultoria_offer_node(state: AgentState) -> dict:
    settings = get_settings()
    profile = dict(state.get("player_profile") or {})
    customer_name = state.get("customer_name")
    messages = state.get("messages") or []
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    last_text = last_human.content if last_human and isinstance(last_human.content, str) else ""

    if not getattr(settings, "consultoria_enabled", True):
        canned = (
            "No momento esta unidade não está com a Consultoria disponível. "
            "Posso te encaminhar pra um especialista humano que te ajuda a "
            "escolher a raquete. Em breve alguém da equipe entra em contato!"
        )
        logger.info("consultoria_offer skipped — consultoria_enabled=False")
        return {
            "messages": [AIMessage(content=canned)],
            "response_blocks": [canned],
            "needs_handoff": True,
            "handoff_reason": "consultoria_disabled",
        }

    system = build_consultoria_offer_prompt(settings)
    name_line = f"Nome do cliente: {customer_name}\n" if customer_name else ""
    profile_phrase = _profile_summary(profile)

    user_content = (
        f"{name_line}"
        f"Perfil resumido: {profile_phrase}.\n"
        f"Investimento da Consultoria: R$ {settings.consultoria_preco} "
        f"(100% abatido na compra de raquete no mesmo dia).\n\n"
        f"Última mensagem do cliente: {last_text}"
    )

    client = OpenAIClient()
    response = await client.chat(
        messages=[{"role": "user", "content": user_content}],
        system=system,
        max_tokens=500,
        temperature=0.5,
        json_mode=True,
    )

    blocks = parse_messages(response)
    if not blocks:
        # Defensive — never let the customer see empty.
        blocks = [
            f"Pra encontrar a raquete que combina com seu jogo, a gente "
            f"prefere fazer com a *Consultoria Base Sports* — análise "
            f"específica + teste em quadra. Investimento de R$ "
            f"{settings.consultoria_preco}, 100% abatido se comprar no "
            f"mesmo dia. Quer saber como funciona?"
        ]

    joined = "\n\n".join(blocks)
    logger.info(
        "consultoria_offer done blocks=%d has_name=%s",
        len(blocks),
        bool(customer_name),
    )
    return {
        "messages": [AIMessage(content=joined)],
        "response_blocks": blocks,
        "consultoria_interest": True,
    }
