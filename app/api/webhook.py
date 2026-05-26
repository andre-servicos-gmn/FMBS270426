"""POST /webhook/whatsapp — receives messages from Evolution API (WhatsApp)."""
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from langchain_core.messages import AIMessage, HumanMessage

from app.adapters.evolution import EvolutionClient
from app.adapters.media_processor import transcribe_audio
from app.agent.graph import build_graph
from app.agent.message_splitter import parse_messages
from app.agent.reset import is_reset_command, reset_conversation  # DEV_RESET_HOOK
from app.config import get_settings
from app.security.audit_log import log_access
from app.security.pii_masker import hash_phone, mask_pii
from app.storage.redis_session import is_message_processed, mark_message_processed

# Canned responses for unsupported / partially supported media kinds.
_IMAGE_RESPONSE = (
    "Vi que você mandou uma imagem. Por aqui ainda não consigo analisar fotos, "
    "mas se você me contar em texto qual modelo te interessa, eu te ajudo a buscar!"
)
_DOCUMENT_RESPONSE = (
    "Ainda não consigo abrir documentos por aqui. Se quiser, manda como foto "
    "da tela ou em texto que eu te ajudo!"
)
_AUDIO_EMPTY_RESPONSE = (
    "Não consegui entender o áudio. Pode mandar por texto ou tentar de novo?"
)
_AUDIO_FAILURE_RESPONSE = (
    "Tive um problema pra processar seu áudio. Pode mandar por texto pra eu te ajudar?"
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Graph singleton (MemorySaver keeps state in-process) ──────────────────────

_graph = None


def _get_graph():  # type: ignore[return]
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Auth ──────────────────────────────────────────────────────────────────────

_APIKEY_HEADER = APIKeyHeader(name="apikey", auto_error=False)


async def _require_token(apikey: str | None = Depends(_APIKEY_HEADER)) -> str | None:
    """Validate the inbound webhook token.

    When ``EVOLUTION_WEBHOOK_TOKEN`` is unset (empty string), auth is BYPASSED
    and the request is accepted. A warning is logged on every call so the
    operator notices and re-enables auth before production. This exists
    because some self-hosted Evolution builds do not allow configuring a
    custom header on outbound webhooks — the pilot URL is obscure (ngrok
    subdomain, unpublished), so the risk is bounded for dev.
    """
    token = get_settings().evolution_webhook_token
    if not token:
        logger.warning(
            "webhook auth DISABLED (no EVOLUTION_WEBHOOK_TOKEN configured) "
            "— INSECURE, dev/pilot only"
        )
        return None
    if not apikey or apikey != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return apikey


# ── Payload helpers ───────────────────────────────────────────────────────────

def _classify_message(message: dict[str, Any]) -> dict[str, Any]:
    """Identify what kind of WhatsApp message this is.

    Returns a dict ``{"kind": <kind>, "text": <str | None>}`` where ``kind``
    is one of: ``text``, ``audio``, ``image``, ``document``, ``sticker``,
    ``video``, ``unknown``. Only ``text`` carries a non-None ``text`` value.

    Mapping of Evolution / Baileys payload shapes:
        message.conversation                 → text (plain)
        message.extendedTextMessage.text     → text (reply/quote/format)
        message.audioMessage                 → audio
        message.imageMessage                 → image
        message.documentMessage              → document
        message.stickerMessage               → sticker
        message.videoMessage                 → video
    """
    conv = message.get("conversation")
    if conv:
        return {"kind": "text", "text": str(conv)}
    ext = message.get("extendedTextMessage")
    if isinstance(ext, dict) and ext.get("text"):
        return {"kind": "text", "text": str(ext["text"])}
    if "audioMessage" in message:
        return {"kind": "audio", "text": None}
    if "imageMessage" in message:
        return {"kind": "image", "text": None}
    if "documentMessage" in message:
        return {"kind": "document", "text": None}
    if "stickerMessage" in message:
        return {"kind": "sticker", "text": None}
    if "videoMessage" in message:
        return {"kind": "video", "text": None}
    return {"kind": "unknown", "text": None}


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/webhook/whatsapp", status_code=200)
async def webhook_whatsapp(
    request: Request,
    background_tasks: BackgroundTasks,
    _token: str | None = Depends(_require_token),
) -> dict[str, Any]:
    # Debug-level dump of incoming headers — helps reverse-engineer what
    # Evolution actually sends so we can re-enable auth on the correct header.
    # At LOG_LEVEL=INFO (default) this is silent; flip to DEBUG to see it.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "webhook_inbound_headers method=%s path=%s headers=%s",
            request.method,
            request.url.path,
            dict(request.headers),
        )

    body: dict[str, Any] = await request.json()

    # Only handle incoming text messages
    if body.get("event") != "messages.upsert":
        return {"status": "ignored", "reason": "event_type"}

    data: dict[str, Any] = body.get("data") or {}
    key: dict[str, Any] = data.get("key") or {}

    if key.get("fromMe"):
        return {"status": "ignored", "reason": "from_me"}

    raw_jid: str = key.get("remoteJid", "")
    raw_phone: str = raw_jid.split("@")[0]
    message_id: str = key.get("id", "")

    if not raw_phone or not message_id:
        return {"status": "ignored", "reason": "missing_required_fields"}

    # Idempotency — applies to ALL message kinds (audio, image, document too)
    try:
        if await is_message_processed(message_id):
            logger.info("webhook_duplicate message_id=%s", message_id)
            return {"status": "ok", "duplicate": True}
        await mark_message_processed(message_id)
    except Exception as exc:
        logger.warning("redis_idempotency_unavailable: %s — proceeding without dedup", exc)

    phone_hash = hash_phone(raw_phone)

    # ── Sprint 1.12 — classify message kind ─────────────────────────────────
    message_dict = data.get("message") or {}
    classified = _classify_message(message_dict)
    kind = classified["kind"]
    message_text = classified["text"]

    # Silently ignore stickers and videos (no canned reply — spam-prone).
    if kind in ("sticker", "video"):
        logger.info("webhook_ignored kind=%s phone_hash=%.8s", kind, phone_hash)
        return {"status": "ignored", "reason": kind}

    if kind == "unknown":
        logger.info("webhook_ignored_unknown phone_hash=%.8s", phone_hash)
        return {"status": "ignored", "reason": "unknown_message_type"}

    # TODO Sprint 1.13: vision support — analyse image with gpt-4o.
    if kind == "image":
        logger.info("webhook_image_received phone_hash=%.8s", phone_hash)
        background_tasks.add_task(_send_canned_text, raw_phone=raw_phone, text=_IMAGE_RESPONSE)
        return {"status": "ok", "kind": "image"}

    if kind == "document":
        logger.info("webhook_document_received phone_hash=%.8s", phone_hash)
        background_tasks.add_task(_send_canned_text, raw_phone=raw_phone, text=_DOCUMENT_RESPONSE)
        return {"status": "ok", "kind": "document"}

    if kind == "audio":
        logger.info("audio_received message_id=%s phone_hash=%.8s", message_id, phone_hash)
        background_tasks.add_task(
            _handle_audio_message,
            raw_phone=raw_phone,
            phone_hash=phone_hash,
            message_key=key,
        )
        return {"status": "ok", "kind": "audio"}

    # Text branch — guard against empty text payloads after classify.
    if not message_text:
        return {"status": "ignored", "reason": "empty_text"}

    # ── DEV_RESET_HOOK — magic /reset command (REMOVE BEFORE PRODUCTION) ──────
    if is_reset_command(message_text):
        background_tasks.add_task(
            _handle_reset_command,
            raw_phone=raw_phone,
            phone_hash=phone_hash,
        )
        return {"status": "ok", "reset": True}

    background_tasks.add_task(
        _process_message,
        raw_phone=raw_phone,
        phone_hash=phone_hash,
        message_text=message_text,
    )

    return {"status": "ok"}


# ── Canned-response helpers (Sprint 1.12) ─────────────────────────────────────

async def _send_canned_text(raw_phone: str, text: str) -> None:
    """Send a fixed text reply via Evolution, ignoring transport errors."""
    try:
        await EvolutionClient().send_text(raw_phone, text)
    except Exception as exc:
        logger.error("canned_response_failed phone=%.8s: %s", raw_phone, exc)


async def _handle_audio_message(
    raw_phone: str, phone_hash: str, message_key: dict[str, Any]
) -> None:
    """Download the audio from Evolution, transcribe via Whisper, dispatch as text.

    On any infrastructure failure (download or transcription), send a polite
    canned response inviting the customer to switch to text. On an empty
    transcription (silent / inaudible audio), send a different canned
    message so the customer knows their attempt arrived but didn't carry
    intelligible speech.
    """
    try:
        audio_bytes, mime_type = await EvolutionClient().get_media_base64(message_key)
    except Exception as exc:
        logger.error("audio_download_failed phone_hash=%.8s: %s", phone_hash, exc)
        await _send_canned_text(raw_phone, _AUDIO_FAILURE_RESPONSE)
        return

    try:
        text = await transcribe_audio(audio_bytes, mime_type)
    except Exception as exc:
        logger.error("audio_transcription_failed phone_hash=%.8s: %s", phone_hash, exc)
        await _send_canned_text(raw_phone, _AUDIO_FAILURE_RESPONSE)
        return

    if not text.strip():
        logger.info("audio_empty_transcription phone_hash=%.8s", phone_hash)
        await _send_canned_text(raw_phone, _AUDIO_EMPTY_RESPONSE)
        return

    # DECISION: do NOT prefix the text with "[ÁUDIO TRANSCRITO]" — the agent
    # should treat the transcription as a normal customer message. The fact
    # that it came from audio is recorded in the structured log below.
    logger.info("audio_to_text phone_hash=%.8s text=%.100s", phone_hash, text)
    await _process_message(
        raw_phone=raw_phone,
        phone_hash=phone_hash,
        message_text=text,
    )


# ── DEV_RESET_HOOK background task (REMOVE BEFORE PRODUCTION) ─────────────────

async def _handle_reset_command(raw_phone: str, phone_hash: str) -> None:
    """Wipe Redis state for the caller and send the canned confirmation.

    DEV/PILOT ONLY. See app/agent/reset.py docstring before shipping.
    """
    try:
        deleted = await reset_conversation(phone_hash)
        logger.info(
            "webhook_reset_command phone_hash=%.8s keys_deleted=%d",
            phone_hash,
            deleted,
        )
    except Exception as exc:
        logger.error("webhook_reset_failed phone_hash=%.8s: %s", phone_hash, exc)

    try:
        await EvolutionClient().send_text(raw_phone, "Conversa resetada ✅")
    except Exception as exc:
        logger.error("webhook_reset_reply_failed phone_hash=%.8s: %s", phone_hash, exc)


# ── Background processing ─────────────────────────────────────────────────────

async def _process_message(
    raw_phone: str,
    phone_hash: str,
    message_text: str,
) -> None:
    """Invoke the agent graph and send the reply via WhatsApp.

    This runs in a FastAPI BackgroundTask so the webhook returns 200 immediately
    even though graph inference + network I/O can take several seconds.
    """
    from sqlalchemy import func
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.storage.db import get_session
    from app.storage.models import ConversationLog, Lead

    # Upsert lead on every contact — creates on first touch, updates timestamp on return
    try:
        async with get_session() as session:
            stmt = (
                pg_insert(Lead)
                .values(id=uuid.uuid4(), phone_hash=phone_hash, profile={})
                .on_conflict_do_update(
                    index_elements=["phone_hash"],
                    set_={"last_interaction_at": func.now()},
                )
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as exc:
        logger.warning("lead_upsert_failed phone_hash=%.8s: %s", phone_hash, exc)

    graph = _get_graph()
    config = {"configurable": {"thread_id": phone_hash}}

    # Reset turn-scoped flags; other state (player_profile, messages) is kept
    # by the MemorySaver checkpointer across turns for the same phone_hash.
    # Sprint 1.16 — response_blocks MUST be cleared each turn. The active
    # node is expected to fill it (or leave it empty for the splitter
    # fallback). Without this reset, a node that only returns "messages"
    # (e.g. close) would inherit response_blocks from a previous turn's
    # recommend and the webhook would send those stale blocks.
    state_update: dict[str, Any] = {
        "messages": [HumanMessage(content=message_text)],
        "phone_hash": phone_hash,
        "needs_handoff": False,
        "handoff_reason": None,
        "response_blocks": [],
    }

    try:
        result: dict[str, Any] = await graph.ainvoke(state_update, config=config)
    except Exception as exc:
        logger.error("graph_invoke_failed phone_hash=%.8s: %s", phone_hash, exc)
        return

    # Extract last AI message
    ai_response = ""
    for m in reversed(result.get("messages") or []):
        if isinstance(m, AIMessage):
            ai_response = m.content if isinstance(m.content, str) else str(m.content)
            break

    if not ai_response:
        logger.warning("graph_no_ai_response phone_hash=%.8s", phone_hash)
        return

    # Sprint 1.6: prefer explicit response_blocks from the node; if absent
    # (older nodes still on single-string outputs), run the splitter on the
    # joined AIMessage content to recover a sensible block list.
    blocks: list[str] = result.get("response_blocks") or []
    if not blocks:
        blocks = parse_messages(ai_response) or [ai_response]

    try:
        await EvolutionClient().send_text_blocks(raw_phone, blocks)
    except Exception as exc:
        logger.error("evolution_send_failed phone_hash=%.8s: %s", phone_hash, exc)

    # Persist conversation logs — always mask before writing
    try:
        async with get_session() as session:
            session.add(
                ConversationLog(
                    id=uuid.uuid4(),
                    phone_hash=phone_hash,
                    message_role="user",
                    content_masked=mask_pii(message_text),
                )
            )
            session.add(
                ConversationLog(
                    id=uuid.uuid4(),
                    phone_hash=phone_hash,
                    message_role="assistant",
                    content_masked=mask_pii(ai_response),
                )
            )
            await session.commit()
    except Exception as exc:
        logger.error("conversation_log_failed phone_hash=%.8s: %s", phone_hash, exc)

    await log_access(
        actor="webhook",
        action="process_message",
        target_hash=phone_hash,
    )
