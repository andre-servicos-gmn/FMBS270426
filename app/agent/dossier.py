"""Sprint 2.0 / 2.2 — customer dossier builder + delivery.

When the agent hands a conversation off to a human (purchase closing,
scheduling, out-of-scope question, explicit request), we want the human
attendant to receive a structured summary of the lead instead of having to
scroll the raw WhatsApp transcript.

This module exposes:

    summarize_conversation(messages, phone_hash) -> str
        LLM-driven 2–3 sentence summary of the conversation so far.
        Cached per ``(phone_hash, len(messages))`` to avoid re-spending
        tokens when the same handoff fires twice in quick succession.

    build_dossier(state, summary=None) -> dict
        Pull the relevant slots from AgentState and assemble a structured
        dict. Optionally takes a pre-computed conversation summary.

    format_dossier_for_whatsapp(dossier) -> str
        Render the dict as a single WhatsApp-friendly message with light
        formatting (*bold* for headings, emojis as visual anchors). Empty
        fields are omitted; ``handoff_reason`` is translated to PT-BR.

    persist_dossier(phone_hash, dossier) -> None
        Save the dossier under ``Lead.profile['dossier']`` + flip
        ``needs_human=True`` and ``handoff_reason`` on the lead profile.

    send_dossier_to_recipient(state, settings, dossier) -> bool
        Sprint 2.2 — push the formatted dossier to the configured WhatsApp
        recipient (``DOSSIER_RECIPIENT_PHONE``). Failures are logged but
        never raised — handoff UX is unaffected.

    handoff_dossier_pipeline(state) -> dict
        Convenience orchestrator: summary → dossier → persist → send.
        Handoff nodes call this single function instead of stitching the
        four steps manually.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)


# ── PT-BR translation of handoff reasons (Sprint 2.2) ────────────────────────

_HANDOFF_REASON_LABELS: dict[str, str] = {
    "user_requested": "Pediu atendimento humano",
    "out_of_scope": "Pergunta fora do escopo",
    "scheduling": "Quer agendar Consultoria",
    "purchase_closing": "Quer comprar raquete",
    "recommend_no_match": "Modelo não disponível no catálogo",
    "consultoria_disabled": "Consultoria indisponível nesta unidade",
    "faq_escalation": "Dúvida operacional escalada",
}


def _humanize_handoff_reason(reason: str | None) -> str:
    """Return a PT-BR label for ``reason`` (falls back to the raw value)."""
    if not reason:
        return "—"
    return _HANDOFF_REASON_LABELS.get(reason, reason)


# ── Conversation summary (Sprint 2.2 — LLM-driven) ───────────────────────────

# Module-level cache of summaries — keyed by (phone_hash, message_count) so a
# repeated handoff at the same conversation state reuses the prior summary,
# but a longer conversation triggers a fresh call.
_SUMMARY_CACHE: dict[tuple[str, int], str] = {}

_SUMMARY_SYSTEM = (
    "Você é um assistente de pós-atendimento. Resuma uma conversa de "
    "WhatsApp de venda de raquetes em 2 ou 3 frases curtas, em PT-BR. "
    "Foque em: (a) o que o cliente quer, (b) onde a conversa parou, "
    "(c) próximo passo claro pro atendente. NÃO mencione dados sensíveis "
    "(telefone, CPF). Resposta direta — sem cumprimento, sem 'segue o resumo:'."
)


def _format_messages_for_summary(messages: list[Any]) -> str:
    """Render the message list as a compact dialogue for the LLM."""
    lines: list[str] = []
    for m in messages[-30:]:  # last 30 turns is plenty for a 2-3 sentence summary
        role = "Cliente" if isinstance(m, HumanMessage) else (
            "Agente" if isinstance(m, AIMessage) else "Sistema"
        )
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _clear_summary_cache() -> None:
    """Test helper — clears the module-level summary cache."""
    _SUMMARY_CACHE.clear()


async def summarize_conversation(
    messages: list[Any], phone_hash: str | None = None
) -> str:
    """Return a 2-3 sentence PT-BR summary of the conversation.

    Cached per ``(phone_hash, len(messages))`` — repeated calls during the
    same handoff don't re-spend tokens. Falls back to a deterministic
    message-count placeholder when the LLM call fails or messages is empty
    (so the dossier always has something readable in this field).
    """
    if not messages:
        return "Conversa sem mensagens registradas."

    cache_key = (phone_hash or "", len(messages))
    cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        logger.info(
            "summarize_conversation cache_hit phone_hash=%.8s count=%d",
            phone_hash or "", len(messages),
        )
        return cached

    user_block = _format_messages_for_summary(messages)
    try:
        from app.adapters.openai_client import OpenAIClient
        client = OpenAIClient()
        response = await client.chat(
            messages=[{"role": "user", "content": user_block}],
            system=_SUMMARY_SYSTEM,
            max_tokens=100,
            temperature=0.3,
        )
        summary = (response or "").strip()
        if not summary:
            raise ValueError("empty_summary")
    except Exception as exc:
        logger.warning("summarize_conversation_failed: %s", exc)
        msg_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        summary = (
            f"Conversa com {msg_count} mensagens do cliente. "
            f"Resumo automático indisponível."
        )

    _SUMMARY_CACHE[cache_key] = summary
    logger.info(
        "summarize_conversation done phone_hash=%.8s count=%d chars=%d",
        phone_hash or "", len(messages), len(summary),
    )
    return summary


# ── Dossier builder (Sprint 2.0) ─────────────────────────────────────────────

def _none_to_dash(value: Any) -> str:
    """Render a None/empty value as '—' (em-dash) for WhatsApp output."""
    if value is None or value == "":
        return "—"
    return str(value)


def build_dossier(
    state: dict[str, Any], summary: str | None = None
) -> dict[str, Any]:
    """Compose the dossier dict from the agent state.

    ``summary`` is the LLM-generated 2-3 sentence summary (Sprint 2.2). When
    None, a deterministic message-count placeholder is used — used by tests
    and any caller that doesn't go through ``handoff_dossier_pipeline``.
    """
    profile = dict(state.get("player_profile") or {})
    now = datetime.now(timezone.utc).isoformat()

    msgs = state.get("messages") or []
    if summary is None:
        msg_count = sum(1 for m in msgs if isinstance(m, HumanMessage))
        summary = (
            f"Conversa com {msg_count} mensagens. "
            f"Última intenção: {state.get('intent') or 'desconhecida'}."
        )

    dossier = {
        "nome": state.get("customer_name") or "—",
        "telefone_hash": state.get("phone_hash") or "",
        "nivel": _none_to_dash(profile.get("nivel_jogo")),
        "lesoes": _none_to_dash(profile.get("lesoes")),
        "regiao_lesao": _none_to_dash(profile.get("regiao_lesao")),
        "esporte_raquete_previo": _none_to_dash(
            profile.get("esporte_raquete_previo")
        ),
        "modelo_desejado": _none_to_dash(profile.get("modelo_desejado")),
        "produto_pesquisado": state.get("produto_pesquisado") or None,
        "consultoria_interesse": bool(state.get("consultoria_interest", False)),
        "needs_handoff_reason": state.get("handoff_reason"),
        "transcricao_resumo": summary,
        "timestamp": now,
    }
    return dossier


# ── WhatsApp formatter (Sprint 2.2 — emoji-rich + smart omissions) ───────────

_EMPTY_DASH_VALUES = {"—", "-", "", None}


def _line_or_skip(prefix: str, value: Any) -> str | None:
    """Return ``f"{prefix} {value}"`` only when ``value`` looks meaningful."""
    if value in _EMPTY_DASH_VALUES:
        return None
    return f"{prefix} {value}"


def _bullet_or_skip(label: str, value: Any) -> str | None:
    """Return ``f"• {label}: {value}"`` or None for empty/dash values."""
    if value in _EMPTY_DASH_VALUES:
        return None
    return f"• {label}: {value}"


def format_dossier_for_whatsapp(dossier: dict[str, Any]) -> str:
    """Render the dossier as a WhatsApp message ready for the human attendant.

    Empty values (None / "" / "—") are omitted line by line so the human
    never sees noise like "Lesões: —". ``handoff_reason`` is translated to
    a friendly PT-BR label.
    """
    nome = dossier.get("nome") or ""
    fone_hash = dossier.get("telefone_hash") or ""

    # Header
    parts: list[str] = ["📋 *NOVO LEAD — Base Sports*", ""]
    if nome and nome not in _EMPTY_DASH_VALUES:
        parts.append(f"👤 *Cliente:* {nome}")
    if fone_hash:
        parts.append(f"📱 *Hash:* {fone_hash[:12]}…")
    motivo = _humanize_handoff_reason(dossier.get("needs_handoff_reason"))
    if motivo and motivo != "—":
        parts.append(f"📌 *Motivo:* {motivo}")

    # ── Perfil — only render if at least one slot has content ────────────
    nivel = dossier.get("nivel")
    lesoes = dossier.get("lesoes")
    regiao = dossier.get("regiao_lesao")
    prev = dossier.get("esporte_raquete_previo")
    modelo = dossier.get("modelo_desejado")

    perfil_lines: list[str] = []
    nivel_line = _bullet_or_skip("Nível", nivel)
    if nivel_line:
        perfil_lines.append(nivel_line)
    # Combine lesoes + regiao in a single bullet when both are meaningful.
    if lesoes not in _EMPTY_DASH_VALUES and str(lesoes).lower() not in ("nenhuma", "nenhum"):
        if regiao not in _EMPTY_DASH_VALUES and str(regiao).lower() not in ("nenhuma", "nenhum"):
            perfil_lines.append(f"• Lesões: {lesoes} ({regiao})")
        else:
            perfil_lines.append(f"• Lesões: {lesoes}")
    prev_line = _bullet_or_skip("Esporte prévio", prev)
    if prev_line and str(prev).lower() not in (
        "nao_aplicavel", "nao", "não", "nenhum", "nenhuma"
    ):
        perfil_lines.append(prev_line)
    modelo_line = _bullet_or_skip("Modelo desejado", modelo)
    if modelo_line and str(modelo).lower() not in ("nenhum", "nenhuma"):
        perfil_lines.append(modelo_line)

    if perfil_lines:
        parts.append("")
        parts.append("🎾 *Perfil:*")
        parts.extend(perfil_lines)

    # ── Pesquisou + Consultoria interest ─────────────────────────────────
    extras: list[str] = []
    pesquisado_line = _line_or_skip("🔍 *Pesquisou:*", dossier.get("produto_pesquisado"))
    if pesquisado_line:
        extras.append(pesquisado_line)
    interesse = "Sim" if dossier.get("consultoria_interesse") else "Não"
    extras.append(f"💡 *Interesse em Consultoria:* {interesse}")
    if extras:
        parts.append("")
        parts.extend(extras)

    # ── Resumo da conversa ────────────────────────────────────────────────
    resumo = (dossier.get("transcricao_resumo") or "").strip()
    if resumo:
        parts.append("")
        parts.append("📝 *Resumo da conversa:*")
        parts.append(resumo)

    # ── Timestamp ─────────────────────────────────────────────────────────
    ts = dossier.get("timestamp") or ""
    pretty_ts = ts
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        pretty_ts = dt.strftime("%d/%m/%Y %H:%M")
    except (ValueError, AttributeError, TypeError):
        pass
    if pretty_ts:
        parts.append("")
        parts.append(f"🕐 {pretty_ts}")

    return "\n".join(parts)


# ── DB persistence (Sprint 2.0) ──────────────────────────────────────────────

async def persist_dossier(phone_hash: str, dossier: dict[str, Any]) -> None:
    """Save the dossier under ``Lead.profile['dossier']`` and flag the lead.

    Also stamps ``needs_human=True`` and ``handoff_reason=<reason>`` on the
    lead's profile so admin dashboards have a single source of truth — every
    dossier persistence in Sprint 2.0 happens because of a handoff.

    Best-effort: any DB error is logged but doesn't break the handoff flow.
    """
    if not phone_hash:
        return
    try:
        from sqlalchemy import select, update

        from app.storage.db import get_session
        from app.storage.models import Lead

        async with get_session() as session:
            result = await session.execute(
                select(Lead).where(Lead.phone_hash == phone_hash)
            )
            lead = result.scalar_one_or_none()
            if lead is None:
                logger.warning(
                    "dossier_persist_skipped no_lead phone_hash=%.8s", phone_hash
                )
                return
            merged = dict(lead.profile or {})
            merged["dossier"] = dossier
            merged["needs_human"] = True
            reason = dossier.get("needs_handoff_reason")
            if reason:
                merged["handoff_reason"] = reason
            await session.execute(
                update(Lead)
                .where(Lead.phone_hash == phone_hash)
                .values(profile=merged)
            )
            await session.commit()
            logger.info(
                "dossier_persisted phone_hash=%.8s reason=%s",
                phone_hash, reason,
            )
    except Exception as exc:
        logger.warning("dossier_persist_failed phone_hash=%.8s: %s", phone_hash, exc)


# ── WhatsApp delivery (Sprint 2.2) ───────────────────────────────────────────

async def send_dossier_to_recipient(
    state: dict[str, Any],
    settings: Any,
    dossier: dict[str, Any] | None = None,
) -> bool:
    """Send the formatted dossier to the configured WhatsApp recipient.

    Args:
        state:    Current AgentState (used as a fallback when ``dossier``
                  is None — we then call ``build_dossier(state)``).
        settings: A Settings-like object with ``dossier_recipient_phone``.
        dossier:  Pre-built dossier dict; when None we build it from state.

    Returns True on successful send, False otherwise. Never raises — the
    handoff flow continues regardless of delivery success.
    """
    recipient = (getattr(settings, "dossier_recipient_phone", "") or "").strip()
    if not recipient:
        logger.info("dossier_send_skipped reason=no_recipient_configured")
        return False

    if dossier is None:
        dossier = build_dossier(state)

    text = format_dossier_for_whatsapp(dossier)

    logger.info(
        "dossier_send begin recipient_hash=%.8s reason=%s",
        recipient[-4:].rjust(8, "*"),
        dossier.get("needs_handoff_reason"),
    )
    try:
        from app.adapters.evolution import EvolutionClient
        client = EvolutionClient()
        await client.send_text(recipient, text)
    except Exception as exc:
        logger.warning(
            "dossier_send_failed recipient_hash=%.8s error=%s",
            recipient[-4:].rjust(8, "*"), exc,
        )
        return False

    logger.info(
        "dossier_send done recipient_hash=%.8s",
        recipient[-4:].rjust(8, "*"),
    )
    return True


# ── Orchestrator (Sprint 2.2) ────────────────────────────────────────────────

async def handoff_dossier_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """End-to-end: summarize → build → persist → send. Returns the dossier.

    Handoff nodes call this single function instead of stitching the four
    steps manually. ``state`` MUST already carry the desired
    ``handoff_reason`` (so the persisted + sent dossier reflects it).
    """
    from app.config import get_settings

    summary = await summarize_conversation(
        state.get("messages") or [],
        state.get("phone_hash") or "",
    )
    dossier = build_dossier(state, summary=summary)
    await persist_dossier(state.get("phone_hash") or "", dossier)

    settings = get_settings()
    await send_dossier_to_recipient(state, settings, dossier=dossier)
    return dossier
