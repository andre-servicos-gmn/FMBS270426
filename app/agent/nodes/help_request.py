"""Sprint 2.6 — generic help / orientation request.

Triggered when the customer asks for guidance WITHOUT naming a specific
product ("me ajuda a escolher", "qual vocês indicam?", "sou iniciante,
qual eu compro?"). The strategic stance: the agent never substitutes the
in-person Consultoria.

Sprint 2.6.9 — was 6 hardcoded strings (3 offers + 3 refusals). Now the
node calls the LLM with SYSTEM_HELP_REQUEST and validates the response
against business invariants (the "cerca"). The flow:

    1. Build the system prompt (principles + facts + red lines).
    2. Pass last customer message + already-offered flag in user block.
    3. LLM generates a free-form reply.
    4. ``_validate_help_response`` checks 4 invariants:
         (a) no "loja" mention (forbidden in this node since 2.6.9 —
             store is purchase-only, not choice/test).
         (b) no specific model recommendation pattern.
         (c) no budget question.
         (d) no follow-up promise.
    5. On violation: regenerate ONCE with a correction note appended,
       then re-validate.
    6. On persistent violation: use the SAFE FALLBACK (one hardcoded
       message that passes its own validation). Safety net, not the
       primary path.

The ``help_request_already_offered`` flag (Sprint 2.6.8) is still set on
every emission. It is now passed to the LLM as context so the model can
adopt a "refusal acknowledgment" tone on the second pass — no more
deterministic slot picking by phone_hash.
"""
import logging
import re
import unicodedata

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.prompts import build_help_request_prompt
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


# Sprint 2.6.9 — safety net. Used ONLY when the LLM violates invariants
# twice in a row. Carries every required signal (Consultoria, value,
# R$ 350, abatimento, opening to name a model) and respects every red
# line (no "loja", no model name, no budget, no follow-up promise). The
# tests assert this message passes _validate_help_response.
_SAFE_FALLBACK = (
    "Pra te ajudar a escolher a raquete certa, o ideal é a nossa "
    "*Consultoria* — a gente analisa seu jogo e você testa as raquetes "
    "em quadra antes de decidir (R$ 350, abatido se fechar). E se você "
    "já tem algum modelo em mente, é só me dizer o nome que eu te passo "
    "os detalhes!"
)


# ── Invariant validation (the "cerca") ──────────────────────────────────────

def _norm(text: str) -> str:
    """Lowercase + strip accents for invariant matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", (text or "").lower())
        if unicodedata.category(c) != "Mn"
    )


# (c) budget-question patterns — checked against normalized text.
_BUDGET_PATTERNS: tuple[str, ...] = (
    "quanto voce quer gastar",
    "quanto voce pretende gastar",
    "qual seu orcamento",
    "qual o seu orcamento",
    "qual o orcamento",
    "faixa de preco",
    "quanto pretende investir",
    "qual valor voce",
    "quanto quer investir",
)

# (d) follow-up promise — checked against normalized text.
_PROMISE_PATTERNS: tuple[str, ...] = (
    "entro em contato",
    "te retorno",
    "alguem da equipe entra",
    "alguem entra em contato",
    "te aviso depois",
    "te chamo depois",
    "te respondo depois",
)

# (b) model-recommendation verbs. Matched (case-insensitive) followed by
# an article + a Capitalized word in the ORIGINAL text (uppercase preserved).
# This catches "recomendo a Mormaii Sunset" / "Sugiro a BeachPro" but NOT
# generic verb usages where no product name follows. Known limitations
# documented in the report.
# Sprint 2.6.10 — IGNORECASE is applied INLINE only to the verb and
# article groups via (?i:...). Applying it globally would make
# [A-ZÀ-Ý] match lowercase letters too (Python `re` quirk), which
# caused "recomendo a nossa Consultoria" to capture target='nossa'
# and bypass the Consultoria allowlist check. Keep the target portion
# case-sensitive so only proper-noun-shaped tokens are flagged.
_RECOMMEND_VERB_RE = re.compile(
    r"\b(?i:recomendo|sugiro|indico|aconselho|recomendaria|sugeriria|"
    r"indicaria|aconselharia)\b\s+(?i:a|o|uma|um|nossa|nosso)?\s*"
    r"\*?(?P<target>[A-ZÀ-Ý][\wÀ-ÿ]+)",
    flags=re.UNICODE,
)

# Sprint 2.6.10 — allowlist of Capitalized words that ARE allowed after
# "recomendo a/o" because they are not racket models. "Consultoria" is the
# common case — Felipe's production log showed "recomendo a nossa
# *Consultoria*" being flagged as a model recommendation. Brand names of
# THE STORE ITSELF are also OK ("recomendo a Base Sports").
_ALLOWED_RECOMMEND_TARGETS: frozenset[str] = frozenset({
    "consultoria",
    "base",  # leading word of "Base Sports"
})

# (b) vitrine-list pattern: 2+ bullet/numbered items each starting with
# a *Bold* tag or capitalized word. Captures the obvious "shop window".
_LIST_ITEM_RE = re.compile(
    r"(?:^|\n)\s*(?:[•·\-\*]|\d+[.)])\s+\*?[A-ZÀ-Ý]",
    flags=re.MULTILINE | re.UNICODE,
)


def _validate_help_response(text: str) -> tuple[bool, list[str]]:
    """Check the response against the 4 business invariants.

    Returns ``(is_valid, violations)``. ``violations`` is a list of short
    machine-readable codes — used both for logging and for crafting the
    regeneration correction note.

    Invariants:
        (a) ``mentions_loja``        — any "loja" / "lojas" token appears.
        (b) ``recommends_specific_model`` — verb-of-recommendation + a
            capitalized noun (product name) in the original text.
        (b) ``looks_like_product_list``  — 2+ bullet/numbered items each
            starting with a bold/capitalized token (vitrine pattern).
        (c) ``asks_budget``          — known budget-question fragments.
        (d) ``promises_followup``    — known follow-up promise fragments.
    """
    if not text or not text.strip():
        return False, ["empty_response"]

    normalized = _norm(text)
    violations: list[str] = []

    # (a) Loja mention — any occurrence is a violation in this node.
    if re.search(r"\blojas?\b", normalized):
        violations.append("mentions_loja")

    # (b1) Specific model recommendation verb + Capitalized name.
    # Sprint 2.6.10 — iterate all matches and only count those whose
    # target is NOT in the allowlist. "recomendo a *Consultoria*" passes;
    # "recomendo a Mormaii" still violates.
    for m in _RECOMMEND_VERB_RE.finditer(text):
        target = (m.group("target") or "").lower()
        if target not in _ALLOWED_RECOMMEND_TARGETS:
            violations.append("recommends_specific_model")
            break

    # (b2) Vitrine list — 2+ items starting with capitalized noun.
    if len(_LIST_ITEM_RE.findall(text)) >= 2:
        violations.append("looks_like_product_list")

    # (c) Budget question.
    if any(p in normalized for p in _BUDGET_PATTERNS):
        violations.append("asks_budget")

    # (d) Follow-up promise.
    if any(p in normalized for p in _PROMISE_PATTERNS):
        violations.append("promises_followup")

    return (not violations, violations)


# ── User-message construction (the live conversational context) ────────────

_VIOLATION_HINTS = {
    "mentions_loja": (
        "Sua resposta anterior mencionou a loja. Neste contexto, escolha "
        "e teste são exclusivos da Consultoria — não cite a loja."
    ),
    "recommends_specific_model": (
        "Sua resposta anterior recomendou uma raquete específica. Você "
        "NUNCA recomenda modelo por conta própria; o cliente é quem "
        "nomeia, ou então o caminho é a Consultoria."
    ),
    "looks_like_product_list": (
        "Sua resposta anterior listou modelos como vitrine. Não liste "
        "raquetes; convide o cliente a nomear um modelo se ele já tiver "
        "um em mente."
    ),
    "asks_budget": (
        "Sua resposta anterior perguntou orçamento. Você NUNCA pergunta "
        "faixa de preço."
    ),
    "promises_followup": (
        "Sua resposta anterior prometeu retorno/contato posterior. Você "
        "NUNCA promete retorno nesta conversa."
    ),
    "empty_response": (
        "Sua resposta anterior estava vazia. Gere uma mensagem completa."
    ),
}


def _build_user_block(
    *,
    customer_name: str | None,
    last_text: str,
    already_offered: bool,
    price_range_mentioned: bool = False,
) -> str:
    """Compose the user-side block fed to the LLM (under the system prompt).

    The system prompt carries principles + facts + red lines. The user
    block carries the LIVE context: who the customer is, what they just
    said, whether the Consultoria was already pitched in this conversation,
    and (Sprint 2.7.3) whether the customer mentioned a budget.
    """
    lines: list[str] = []
    if customer_name:
        lines.append(f"Nome do cliente: {customer_name}")

    state_line = (
        "Estado da conversa: o cliente JÁ recebeu a oferta de Consultoria "
        "anteriormente nesta conversa e está insistindo / pedindo "
        "recomendação direta. Reconheça isso com naturalidade, reformule "
        "(NÃO repita a mensagem anterior), e abra pra ele nomear um modelo."
        if already_offered else
        "Estado da conversa: PRIMEIRA vez que o cliente pede ajuda pra "
        "escolher nesta conversa. Apresente a Consultoria com calor, "
        "explicando o valor (análise + teste em quadra), e abra a opção "
        "de ele nomear um modelo se já tiver um em mente."
    )
    lines.append(state_line)

    # Sprint 2.7.3 — budget hint. Customer voluntarily mentioned a price
    # ceiling. The agent MUST NOT list products by price (Sprint 2.6.9
    # red line) but SHOULD acknowledge the budget naturally and frame
    # the Consultoria as the place where perfil + jogo + faixa de valor
    # come together.
    if price_range_mentioned:
        lines.append(
            "ATENÇÃO: o cliente MENCIONOU faixa de preço / orçamento na "
            "mensagem. Reconheça isso de forma natural na resposta — "
            "explique que a Consultoria considera não só perfil e estilo "
            "de jogo, mas também faixa de investimento, pra recomendar "
            "uma raquete que faça sentido nas três frentes. NÃO ofereça "
            "um produto específico, NÃO liste opções, NÃO mencione "
            "valores de produto. Mantenha o tom da resposta dentro das "
            "linhas vermelhas habituais."
        )

    if last_text:
        lines.append(f"Última mensagem do cliente: {last_text}")

    return "\n\n".join(lines)


def _build_correction_note(violations: list[str]) -> str:
    """Craft a short correction note appended to the system prompt on
    regeneration. Lists ONLY the rules that were broken so the model
    isn't told to re-read every rule."""
    hints = [_VIOLATION_HINTS.get(v, "") for v in violations]
    hints = [h for h in hints if h]
    if not hints:
        return ""
    return (
        "\n\n[CORREÇÃO OBRIGATÓRIA — sua resposta anterior violou uma "
        "regra]\n" + "\n".join(f"- {h}" for h in hints) +
        "\nGere uma resposta NOVA que respeite todas as linhas vermelhas."
    )


# ── Node ───────────────────────────────────────────────────────────────────

async def help_request_node(state: AgentState) -> dict:
    settings = get_settings()
    phone_hash = state.get("phone_hash") or ""
    customer_name = state.get("customer_name")
    already_offered = bool(state.get("help_request_already_offered"))
    price_range_mentioned = bool(state.get("price_range_mentioned"))

    messages = state.get("messages") or []
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    last_text = (
        last_human.content
        if last_human and isinstance(last_human.content, str)
        else ""
    )

    system = build_help_request_prompt(settings)
    user_block = _build_user_block(
        customer_name=customer_name,
        last_text=last_text,
        already_offered=already_offered,
        price_range_mentioned=price_range_mentioned,
    )

    client = OpenAIClient()
    text = await client.chat(
        messages=[{"role": "user", "content": user_block}],
        system=system,
        max_tokens=300,
        temperature=0.7,
    )
    text = (text or "").strip()

    is_valid, violations = _validate_help_response(text)
    source = "llm"

    if not is_valid:
        logger.warning(
            "help_response_invariant_violation reasons=%s phone_hash=%.8s "
            "text=%.200r",
            violations, phone_hash, text,
        )
        correction = _build_correction_note(violations)
        regen_text = await client.chat(
            messages=[{"role": "user", "content": user_block}],
            system=system + correction,
            max_tokens=300,
            temperature=0.7,
        )
        regen_text = (regen_text or "").strip()
        is_valid_2, violations_2 = _validate_help_response(regen_text)

        if is_valid_2:
            text = regen_text
            source = "llm_regenerated"
        else:
            logger.error(
                "help_response_regen_also_violated reasons=%s phone_hash=%.8s "
                "text=%.200r",
                violations_2, phone_hash, regen_text,
            )
            text = _SAFE_FALLBACK
            source = "fallback"

    logger.info(
        "help_response source=%s already_offered=%s phone_hash=%.8s "
        "chars=%d",
        source, already_offered, phone_hash, len(text),
    )

    return {
        "messages": [AIMessage(content=text)],
        "response_blocks": [text],
        "consultoria_interest": True,
        "help_request_already_offered": True,
        # Sprint 2.7.3 — flag consumed; clear so a follow-up turn doesn't
        # keep re-injecting the budget instruction.
        "price_range_mentioned": False,
    }
