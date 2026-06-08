"""DEPRECATED em Sprint 2.6 — não está mais conectado ao grafo do WhatsApp.

Mantido pra possível futura Consultoria virtual (refatoração estratégica:
diagnose presencial pago é o produto principal, agente nunca substitui).

NÃO importar deste módulo nos nodes ativos. NÃO adicionar de volta ao grafo
sem revisão estratégica.

────────────────────────────────────────────────────────────────────────

Sprint 1.8 — diagnose_node em 4 fases.

Inversão parcial de controle: o código (Python) decide qual é a próxima pergunta
e aplica auto-fills determinísticos; o LLM continua sendo usado para (1) extrair
slots da mensagem livre do cliente e (2) refrasear a pergunta canônica com tom
natural. Isso elimina a fonte de inconsistência observada em produção (ordem
trocada de perguntas, guardrails ignorados).

Fluxo por turno:

    1. Meta-pergunta? ──sim──→ explain+re-ask (1 LLM call) ─────────→ retorna
                       └─não─→ 2. Extração (1 LLM call) — pega slots novos
                                3. Guardrails (Python puro) — auto-fills
                                4. Decisão (Python puro) — qual slot é o próximo?
                                   ├─ nenhum vazio → complete=True, vai pra recommend
                                   └─ um vazio    → 5. Fraseamento (1 LLM call) ─→ retorna
"""
import json
import logging
import unicodedata

from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.openai_client import OpenAIClient
from app.agent.prompts import (
    QUESTION_TEMPLATES,
    SLOT_ORDER,
    SYSTEM_DIAGNOSE_EXTRACT,
    SYSTEM_DIAGNOSE_META,
    SYSTEM_DIAGNOSE_PHRASE,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_accents(text: str) -> str:
    """Return ``text`` with Unicode combining accents removed (ç→c, ó→o)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _is_skip_level(level: str | None) -> bool:
    """True for nivel_jogo values that should skip esporte_raquete_previo."""
    if not level:
        return False
    norm = _strip_accents(level.strip().lower())
    return "intermedi" in norm or "avanc" in norm


_META_QUESTION_PATTERNS = (
    "isso importa",
    "por que pergunta",
    "por que voce pergunta",
    "porque pergunta",
    "porque voce pergunta",
    "preciso responder",
    "preciso falar",
    "tenho que responder",
    "isso vai mudar",
    "isso muda algo",
    "isso muda a recomendacao",
    "qual a importancia",
    "qual o motivo",
)


def is_meta_question(msg: str | None) -> bool:
    """Detect questions ABOUT the diagnose process rather than answers.

    Matches are case- and accent-insensitive substring checks against a curated
    set of phrases. Conservative on purpose: a customer's actual answer should
    almost never collide with these patterns.
    """
    if not msg:
        return False
    norm = _strip_accents(msg.strip().lower())
    return any(p in norm for p in _META_QUESTION_PATTERNS)


# ── Phase 2 — Deterministic guardrails ──────────────────────────────────────

def _apply_guardrails(profile: dict) -> dict:
    """Apply non-LLM auto-fills. Returns the same dict (mutated for convenience).

    Rules:
        - lesoes == "nenhuma" and regiao_lesao empty → regiao_lesao = "nenhuma"
        - nivel_jogo is intermediário/avançado and esporte_raquete_previo empty
          → esporte_raquete_previo = "nao_aplicavel"
    """
    lesoes = profile.get("lesoes")
    if (
        lesoes
        and str(lesoes).strip().lower() == "nenhuma"
        and not profile.get("regiao_lesao")
    ):
        profile["regiao_lesao"] = "nenhuma"
        logger.info("auto_fill slot=regiao_lesao value=nenhuma reason=lesoes_nenhuma")

    if _is_skip_level(profile.get("nivel_jogo")) and not profile.get(
        "esporte_raquete_previo"
    ):
        profile["esporte_raquete_previo"] = "nao_aplicavel"
        logger.info(
            "auto_fill slot=esporte_raquete_previo value=nao_aplicavel "
            "reason=nivel_%s",
            profile.get("nivel_jogo"),
        )

    return profile


# ── Phase 3 — Decide which slot is next ─────────────────────────────────────

def _next_pending_slot(profile: dict) -> str | None:
    """Iterate ``SLOT_ORDER`` and return the first slot still empty.

    Conditional skips (defense in depth — Phase 2 already pre-fills these
    in normal flow):
        - regiao_lesao: skipped if lesoes is "nenhuma" or empty
        - esporte_raquete_previo: skipped if nivel_jogo is intermed/avanc

    Returns None when every applicable slot has a value — caller flips to
    intent=recommend.
    """
    for slot in SLOT_ORDER:
        if profile.get(slot):
            continue
        if slot == "regiao_lesao":
            lesoes = profile.get("lesoes", "")
            if not lesoes or str(lesoes).strip().lower() == "nenhuma":
                continue
        if slot == "esporte_raquete_previo":
            if _is_skip_level(profile.get("nivel_jogo")):
                continue
        return slot
    return None


# ── Phase 1 — LLM extraction ────────────────────────────────────────────────

async def _extract_slots(messages: list, profile: dict) -> dict:
    """Call the LLM with the EXTRACT prompt and return slots dict (may be {})."""
    history = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content if isinstance(m.content, str) else str(m.content),
        }
        for m in messages[-10:]
    ]

    profile_ctx = f"[Perfil atual: {json.dumps(profile, ensure_ascii=False)}]"
    if history and history[-1]["role"] == "user":
        history[-1] = {
            "role": "user",
            "content": f"{profile_ctx}\n{history[-1]['content']}",
        }
    else:
        history.append({"role": "user", "content": profile_ctx})

    client = OpenAIClient()
    response = await client.chat(
        messages=history,
        system=SYSTEM_DIAGNOSE_EXTRACT,
        max_tokens=300,
        temperature=0.0,
        json_mode=True,
    )

    try:
        data = json.loads(response)
        extracted = data.get("extracted_slots") or {}
        if not isinstance(extracted, dict):
            extracted = {}
    except (json.JSONDecodeError, AttributeError):
        logger.warning("diagnose_extract_parse_failed response=%.150r", response)
        extracted = {}

    return extracted


# ── Phase 4 — LLM rephrasing ────────────────────────────────────────────────

async def _rephrase_question(canonical: str, last_user_msg: str) -> str:
    """Refrase the canned question with a natural tone. Falls back to canonical."""
    try:
        client = OpenAIClient()
        user_content = (
            f"Pergunta canônica a refrasear:\n{canonical}\n\n"
            f"Última mensagem do cliente para contexto de tom:\n{last_user_msg or '(início da conversa)'}"
        )
        response = await client.chat(
            messages=[{"role": "user", "content": user_content}],
            system=SYSTEM_DIAGNOSE_PHRASE,
            max_tokens=120,
            temperature=0.5,
        )
        cleaned = response.strip()
        return cleaned or canonical
    except Exception as exc:
        logger.warning("diagnose_phrase_failed (fallback to canned): %s", exc)
        return canonical


# ── Meta-question handler ───────────────────────────────────────────────────

async def _handle_meta_question(
    meta_msg: str, pending_canonical: str
) -> str:
    """Explain the original question's purpose AND re-ask it. LLM-driven."""
    try:
        client = OpenAIClient()
        user_content = (
            f"Pergunta original que estava em aberto:\n{pending_canonical}\n\n"
            f"Meta-pergunta do cliente:\n{meta_msg}"
        )
        response = await client.chat(
            messages=[{"role": "user", "content": user_content}],
            system=SYSTEM_DIAGNOSE_META,
            max_tokens=160,
            temperature=0.4,
        )
        cleaned = response.strip()
        return cleaned or pending_canonical
    except Exception as exc:
        logger.warning("diagnose_meta_failed (fallback to canned): %s", exc)
        return pending_canonical


# ── Orchestrator ────────────────────────────────────────────────────────────

async def diagnose_node(state: AgentState) -> dict:
    profile = dict(state.get("player_profile") or {})
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    last_msg = (
        last_human.content if last_human and isinstance(last_human.content, str) else ""
    )

    # ─── Meta-question short-circuit ─────────────────────────────────────────
    if is_meta_question(last_msg):
        # Defensive: apply guardrails before deciding which slot is pending,
        # so we never re-ask a slot that should have been auto-filled.
        snapshot = _apply_guardrails(dict(profile))
        pending = _next_pending_slot(snapshot)
        if pending is None:
            # Diagnose is already complete — meta-pergunta becomes a no-op slot-wise;
            # route to recommend.
            logger.info("diagnose meta_question on_complete_state → recommend")
            return {
                "player_profile": snapshot,
                "intent": "recommend",
            }
        canonical = QUESTION_TEMPLATES[pending]
        reply = await _handle_meta_question(last_msg, canonical)
        logger.info("diagnose meta_question pending_slot=%s", pending)
        return {
            "player_profile": snapshot,  # guardrails may have pre-filled — keep them
            "messages": [AIMessage(content=reply)],
            "intent": "diagnose",
        }

    # ─── Phase 1 — Extraction ─────────────────────────────────────────────────
    extracted = await _extract_slots(messages, profile)
    merged = {**profile, **extracted}

    # ─── Phase 2 — Guardrails ────────────────────────────────────────────────
    merged = _apply_guardrails(merged)

    # ─── Phase 3 — Decide next slot ──────────────────────────────────────────
    next_slot = _next_pending_slot(merged)
    if next_slot is None:
        logger.info("diagnose_complete slots=%s", sorted(merged.keys()))
        return {
            "player_profile": merged,
            "intent": "recommend",
        }

    # ─── Phase 4 — Rephrase canonical question ───────────────────────────────
    canonical = QUESTION_TEMPLATES[next_slot]
    phrased = await _rephrase_question(canonical, last_msg)

    logger.info(
        "diagnose_in_progress next_slot=%s slots_so_far=%s",
        next_slot,
        sorted(merged.keys()),
    )

    return {
        "player_profile": merged,
        "messages": [AIMessage(content=phrased)],
        "intent": "diagnose",
    }
