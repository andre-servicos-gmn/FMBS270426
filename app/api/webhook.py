"""POST /webhook/whatsapp — receives messages from Evolution API (WhatsApp)."""
import hashlib
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.adapters.evolution import EvolutionClient
from app.adapters.media_processor import identify_racket_image, transcribe_audio
from app.agent.graph import build_graph
from app.agent.message_splitter import parse_messages
from app.agent.reset import (
    is_reset_authorized,
    is_reset_command,
    reset_conversation,
)  # DEV_RESET_HOOK
from app.api.debounce_buffer import DebounceBuffer
from app.config import get_settings
from app.security.audit_log import log_access
from app.security.pii_masker import hash_phone, mask_pii
from app.storage.redis_session import (
    cache_image_id,
    cache_transcript,
    count_audio_message,
    count_image_message,
    get_cached_image_id,
    get_cached_transcript,
    is_message_processed,
    mark_message_processed,
)

# Canned responses for unsupported / partially supported media kinds.
# Sprint 3.11 — photos of rackets ARE supported now (vision identification);
# the canned replies below cover the failure/edge branches of that flow.
_IMAGE_NOT_RACKET_RESPONSE = (
    "Recebi sua foto! Mas não consegui identificar uma raquete nela. "
    "Se você me disser em texto qual modelo te interessa, eu te ajudo a buscar!"
)
_IMAGE_UNIDENTIFIED_RESPONSE = (
    "Vi que é uma raquete, mas não consegui identificar o modelo pela foto. "
    "Sabe me dizer a marca ou o nome dela? Aí eu vejo aqui pra você!"
)
_IMAGE_FAILURE_RESPONSE = (
    "Tive um problema pra processar sua foto. Pode me dizer em texto qual "
    "modelo te interessa que eu te ajudo!"
)
_IMAGE_TOO_BIG_RESPONSE = (
    "Essa imagem ficou um pouco pesada pra eu processar por aqui. Consegue "
    "mandar como foto normal (sem ser em alta resolução) ou me dizer o "
    "modelo em texto?"
)
_IMAGE_RATE_LIMIT_RESPONSE = (
    "Recebi várias fotos suas em pouco tempo. Pra eu conseguir te ajudar "
    "melhor agora, me diz por texto qual modelo você procura?"
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
_AUDIO_TOO_LONG_RESPONSE = (
    "Esse áudio ficou um pouco longo pra eu processar por aqui. Consegue mandar "
    "um áudio mais curto ou resumir em texto?"
)
_AUDIO_RATE_LIMIT_RESPONSE = (
    "Recebi vários áudios seus em pouco tempo. Pra eu conseguir te ajudar melhor "
    "agora, me manda por texto?"
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Graph singleton (MemorySaver keeps state in-process) ──────────────────────

_graph = None
# Fase 0 — supervisor V2 singleton, compilado sob demanda APENAS quando
# settings.use_v2 está ligado. Default OFF → este caminho nunca é tocado e o
# comportamento do webhook é idêntico ao de hoje.
_graph_v2 = None


def _get_graph():  # type: ignore[return]
    # Fase 0 — branch gated pela feature flag. Com use_v2=False (default), cai
    # direto no grafo legado abaixo, sem nenhuma mudança de comportamento. Com
    # use_v2=True, compila/retorna o grafo do supervisor reaproveitando o MESMO
    # checkpointer (AsyncRedisSaver) que o grafo legado usa.
    if get_settings().use_v2:
        global _graph_v2
        if _graph_v2 is None:
            from app.agent.checkpointer import get_checkpointer
            from app.agent.supervisor import build_supervisor_graph
            _graph_v2 = build_supervisor_graph(get_checkpointer())
            logger.info("supervisor_v2 graph compiled (USE_V2 enabled)")
        return _graph_v2

    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Sprint 2.7.2 — debounce buffer singleton ──────────────────────────────────
#
# Groups rapid text messages from the same phone_hash into ONE graph
# invocation. Bypassed for non-text (audio/image/document) and /reset.
# The buffer's on_flush callback is _process_message — the merged text
# flows through the existing pipeline unchanged.

_debounce_buffer: DebounceBuffer | None = None


async def _on_first_buffered_message(raw_phone: str) -> None:
    """Sprint 2.7.2 Ajuste 1 — fire the WhatsApp 'typing...' presence
    when a new buffer opens, so the debounce window doesn't look like
    the agent froze. Cosmetic; failures are swallowed inside
    ``send_presence``."""
    await EvolutionClient().send_presence(
        raw_phone,
        presence="composing",
        # Hint a bit beyond the debounce window so the indicator stays
        # visible across the buffer wait + initial graph latency.
        delay_ms=get_settings().message_debounce_ms + 2000,
    )


def _get_debounce_buffer() -> DebounceBuffer:
    """Lazy singleton — uses current settings for cap/window/ttl."""
    global _debounce_buffer
    if _debounce_buffer is None:
        settings = get_settings()
        _debounce_buffer = DebounceBuffer(
            window_ms=settings.message_debounce_ms,
            cap=settings.message_debounce_cap,
            hard_ttl_ms=settings.message_debounce_hard_ttl_ms,
            on_flush=_dispatch_flushed_message,
            on_first_message=_on_first_buffered_message,
        )
    return _debounce_buffer


def _reset_debounce_buffer() -> None:
    """Used by tests to inject custom windows / fakes."""
    global _debounce_buffer
    _debounce_buffer = None


async def _dispatch_flushed_message(
    raw_phone: str, phone_hash: str, merged_text: str
) -> None:
    """Sprint 2.7.2 — buffer's on_flush callback. Hands the merged
    message off to the existing single-turn pipeline."""
    await _process_message(
        raw_phone=raw_phone,
        phone_hash=phone_hash,
        message_text=merged_text,
    )


# ── Sprint 2.6.5 — resilient retry layer + honest fallback ──────────────────

# Fallback wording — discreet, human-sounding. The spec FORBIDS the words
# "erro", "falha", "problema técnico", "sistema", "bug", "indisponível".
# Enforced by test_redis_resilience::test_fallback_message_has_no_forbidden_words.
_FALLBACK_CLIENT_MESSAGE = "Opa, me dá só um segundinho que já te respondo 🙌"

# Anti-repeat marker: same phone_hash → fallback at most once per window.
# Module-level dict survives within the process; if the process dies the
# state resets, which is acceptable behaviour (the next failure cycle is
# treated fresh after a restart).
_FALLBACK_REPEAT_WINDOW_S = 60
_fallback_last_sent: dict[str, float] = {}


def _is_redis_connection_error(exc: BaseException) -> bool:
    """Heuristic: looks like a dead/idle/closed Redis connection?

    Matches both real ``redis-py`` exceptions AND the
    ``'NoneType' object is not callable`` symptom we observed in
    production — that string surfaces when the internal connection pool
    of the saver was torn down but the wrapper object still exists, so
    method lookups (.exists, .setex, .aput) resolve to None.
    """
    if isinstance(exc, (RedisConnectionError, RedisTimeoutError, RedisError)):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    s = str(exc).lower()
    if "nonetype" in s and "callable" in s:
        return True
    if any(token in s for token in ("connection", "timeout", "redis", "closed")):
        return True
    return False


async def _reconnect_redis_singletons() -> None:
    """Discard the Redis-backed singletons so the next call rebuilds them.

    Replaces:
    - ``app.storage.redis_session._redis_client`` (idempotency + session store)
    - ``app.agent.checkpointer._saver`` (LangGraph checkpointer)
    - ``app.api.webhook._graph`` (compiled graph holds the old saver)

    The new graph is recompiled lazily on the next ``_get_graph()`` call.
    """
    global _graph, _graph_v2
    from app.agent.checkpointer import init_checkpointer, reset_checkpointer
    from app.storage.redis_session import reset_redis_client

    await reset_redis_client()
    await reset_checkpointer()
    # Rebuild checkpointer immediately so the next graph compile uses it.
    await init_checkpointer()
    _graph = None
    # Fase 0 — também invalida o grafo v2 (segura o mesmo saver) se a flag
    # estiver ligada; com use_v2=False ele já é None e isto é no-op.
    _graph_v2 = None
    logger.info("redis_singletons_reconnected (next request rebuilds graph)")


async def _ainvoke_with_retry(state_update: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Sprint 2.6.5 Camada 2 — retry the graph invocation once when the
    first attempt fails with a Redis connection error. Between attempts
    we tear down + rebuild every Redis-backed singleton so the second
    attempt uses a fresh client + fresh checkpointer.
    """
    max_attempts = 2
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        graph = _get_graph()
        try:
            result = await graph.ainvoke(state_update, config=config)
            if attempt > 1:
                logger.info("graph_retry_success attempt=%d", attempt)
            return result
        except Exception as exc:  # noqa: BLE001 — we re-raise non-redis
            last_exc = exc
            if attempt == max_attempts:
                logger.error("graph_retry_exhausted attempts=%d: %s", attempt, exc)
                raise
            if not _is_redis_connection_error(exc):
                # Non-Redis error → don't retry, propagate immediately.
                raise
            logger.info(
                "graph_retry_attempt reason=redis_connection attempt=%d err=%.120s",
                attempt + 1, str(exc),
            )
            try:
                await _reconnect_redis_singletons()
            except Exception as reconn_exc:
                logger.error("graph_retry_reconnect_failed: %s", reconn_exc)
                raise last_exc from reconn_exc
    # Unreachable — the loop either returns or raises.
    raise last_exc  # type: ignore[misc]


def _fallback_should_emit(phone_hash: str) -> bool:
    """Anti-repeat gate. Returns True when we may send a fallback now."""
    now = time.time()
    last = _fallback_last_sent.get(phone_hash, 0.0)
    if now - last < _FALLBACK_REPEAT_WINDOW_S:
        return False
    _fallback_last_sent[phone_hash] = now
    return True


async def _send_fallback_and_alert(
    raw_phone: str, phone_hash: str, exc: BaseException
) -> None:
    """Sprint 2.6.5 Camada 3 — last resort when retry is exhausted.

    Sends:
    1. A discreet, human-sounding hold-on to the CLIENT (no error vocab,
       no technical emojis).
    2. A technical alert to the configured ``DOSSIER_RECIPIENT_PHONE``
       (Andre's WhatsApp) so the gap doesn't go unnoticed.

    Both sends are best-effort: if Evolution itself is down too, we just
    log and move on — the alternative (re-raising) drops a 500 in the
    background task which serves nobody.
    """
    if not _fallback_should_emit(phone_hash):
        logger.info(
            "fallback_suppressed phone_hash=%.8s window_s=%d",
            phone_hash, _FALLBACK_REPEAT_WINDOW_S,
        )
        return

    # 1) Client message
    try:
        await EvolutionClient().send_text(raw_phone, _FALLBACK_CLIENT_MESSAGE)
        logger.info("fallback_client_msg_sent phone_hash=%.8s", phone_hash)
    except Exception as send_exc:  # noqa: BLE001 — best effort
        logger.error(
            "fallback_client_msg_send_failed phone_hash=%.8s: %s",
            phone_hash, send_exc,
        )

    # 2) Andre alert
    recipient = (get_settings().dossier_recipient_phone or "").strip()
    if not recipient:
        logger.info("fallback_alert_skipped reason=no_recipient_configured")
        return

    from datetime import datetime
    alert = (
        f"⚠️ ALERTA TÉCNICO — Base Sports\n"
        f"Cliente phone_hash={phone_hash[:12]}… teve mensagem não respondida.\n"
        f"Motivo: falha de conexão (Redis) após retry.\n"
        f"Horário: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"Erro: {str(exc)[:200]}\n"
        f"Ação: verificar conexão / cliente pode precisar reenviar."
    )
    try:
        await EvolutionClient().send_text(recipient, alert)
        logger.info("fallback_alert_sent")
    except Exception as alert_exc:  # noqa: BLE001 — best effort
        logger.error("fallback_alert_send_failed: %s", alert_exc)


# ── Auth ──────────────────────────────────────────────────────────────────────

_APIKEY_HEADER = APIKeyHeader(name="apikey", auto_error=False)


async def _require_token(apikey: str | None = Depends(_APIKEY_HEADER)) -> str | None:
    """Validate the inbound webhook token.

    Sprint 2.7 — explicit log lines for every path so operators can debug
    Evolution misconfiguration from the logs alone:
      * token unset           → ``webhook_auth_disabled``  (warning per call)
      * header missing        → ``webhook_auth_failed reason=missing_header``
      * header wrong          → ``webhook_auth_failed reason=token_mismatch``
      * header matches        → ``webhook_auth_ok``

    When ``EVOLUTION_WEBHOOK_TOKEN`` is unset, auth is BYPASSED and the
    request is accepted (a warning is logged on every call so the operator
    notices). This exists because some self-hosted Evolution builds don't
    allow configuring custom outbound headers; the pilot URL is obscure
    (ngrok subdomain, unpublished), so the risk is bounded for dev.

    Evolution sends the token in the ``apikey`` HTTP header — same header
    name Evolution uses for inbound auth on its own API. Configure it in
    the Evolution panel: Instance Settings → Webhook → Custom Headers →
    ``apikey: <same value as EVOLUTION_WEBHOOK_TOKEN>``.
    """
    token = get_settings().evolution_webhook_token
    if not token:
        logger.warning(
            "webhook_auth_disabled reason=no_token_configured "
            "— INSECURE, dev/pilot only"
        )
        return None
    if not apikey:
        logger.warning(
            "webhook_auth_failed reason=missing_header "
            "(expected `apikey` header from Evolution)"
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if apikey != token:
        logger.warning(
            "webhook_auth_failed reason=token_mismatch "
            "(Evolution sent a value but it doesn't match EVOLUTION_WEBHOOK_TOKEN)"
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    logger.info("webhook_auth_ok")
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

    # Sprint 3.11 — vision support: identify the racket in the photo and
    # answer like a product inquiry. Bypasses the debounce buffer (same
    # decision as audio — Sprint 2.7.2).
    if kind == "image":
        image_meta = message_dict.get("imageMessage") or {}
        caption = str(image_meta.get("caption") or "")
        logger.info(
            "webhook_image_received phone_hash=%.8s has_caption=%s",
            phone_hash,
            bool(caption),
        )
        background_tasks.add_task(
            _handle_image_message,
            raw_phone=raw_phone,
            phone_hash=phone_hash,
            message_key=key,
            caption=caption,
        )
        return {"status": "ok", "kind": "image"}

    if kind == "document":
        logger.info("webhook_document_received phone_hash=%.8s", phone_hash)
        background_tasks.add_task(_send_canned_text, raw_phone=raw_phone, text=_DOCUMENT_RESPONSE)
        return {"status": "ok", "kind": "document"}

    if kind == "audio":
        # Sprint 3.10 — duration guard BEFORE download: WhatsApp voice notes
        # carry their length in audioMessage.seconds, so an over-limit audio
        # is rejected without paying the Evolution download or Whisper.
        audio_meta = message_dict.get("audioMessage") or {}
        seconds = audio_meta.get("seconds")
        max_seconds = get_settings().audio_max_seconds
        if isinstance(seconds, (int, float)) and seconds > max_seconds:
            logger.info(
                "audio_rejected_too_long phone_hash=%.8s seconds=%s max=%d",
                phone_hash,
                seconds,
                max_seconds,
            )
            background_tasks.add_task(
                _send_canned_text, raw_phone=raw_phone, text=_AUDIO_TOO_LONG_RESPONSE
            )
            return {"status": "ok", "kind": "audio", "rejected": "too_long"}

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

    # ── DEV_RESET_HOOK — magic /reset command, gated by allowlist ─────────
    if is_reset_command(message_text):
        if not is_reset_authorized(raw_phone):
            # Silent ignore: ack the webhook (Evolution won't retry) but
            # send NOTHING to the client and dispatch NOTHING to the
            # graph. This makes "/reset" indistinguishable from any other
            # silently-dropped message for an unauthorized sender — they
            # can't probe the command's existence.
            logger.warning(
                "reset_denied phone_hash=%.8s reason=unauthorized",
                phone_hash,
            )
            return {"status": "ok", "reset": "denied"}

        logger.info(
            "reset_authorized phone_hash=%.8s — wiping state", phone_hash,
        )
        background_tasks.add_task(
            _handle_reset_command,
            raw_phone=raw_phone,
            phone_hash=phone_hash,
        )
        return {"status": "ok", "reset": True}

    # Sprint 2.7.2 — TEXT messages go through the debounce buffer so a
    # burst from the same customer is grouped into ONE graph invocation.
    # Media (audio/image/document) and the /reset magic command are
    # dispatched directly above and never reach this branch.
    background_tasks.add_task(
        _buffer_text_message,
        raw_phone=raw_phone,
        phone_hash=phone_hash,
        message_text=message_text,
    )

    return {"status": "ok"}


async def _buffer_text_message(
    raw_phone: str, phone_hash: str, message_text: str
) -> None:
    """Tiny shim — yields control to the BackgroundTask runner before
    awaiting the buffer's lock so the webhook can return 200 with
    minimal latency."""
    await _get_debounce_buffer().add(
        raw_phone=raw_phone,
        phone_hash=phone_hash,
        message_text=message_text,
    )


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

    Sprint 3.10 guards, in order: per-customer rate limit (before download),
    post-download size cap, transcript cache by content hash. Redis failures
    in the guards are fail-open — a Redis blip must not take the audio path
    down.

    On any infrastructure failure (download or transcription), send a polite
    canned response inviting the customer to switch to text. On an empty
    transcription (silent / inaudible audio), send a different canned
    message so the customer knows their attempt arrived but didn't carry
    intelligible speech.
    """
    settings = get_settings()

    # Guard 1 — rate limit per phone_hash, fixed 1h window. 0 disables.
    limit = settings.audio_rate_limit_per_hour
    if limit > 0:
        try:
            count = await count_audio_message(phone_hash)
        except Exception as exc:
            logger.warning("audio_rate_limit_unavailable: %s — proceeding", exc)
            count = 0
        if count > limit:
            logger.warning(
                "audio_rate_limited phone_hash=%.8s count=%d limit=%d",
                phone_hash,
                count,
                limit,
            )
            await _send_canned_text(raw_phone, _AUDIO_RATE_LIMIT_RESPONSE)
            return

    try:
        audio_bytes, mime_type = await EvolutionClient().get_media_base64(message_key)
    except Exception as exc:
        logger.error("audio_download_failed phone_hash=%.8s: %s", phone_hash, exc)
        await _send_canned_text(raw_phone, _AUDIO_FAILURE_RESPONSE)
        return

    # Guard 2 — size cap after download. Covers payloads without .seconds
    # (forwarded audio files) that slipped past the webhook duration guard.
    if len(audio_bytes) > settings.audio_max_bytes:
        logger.warning(
            "audio_rejected_too_big phone_hash=%.8s bytes=%d max=%d",
            phone_hash,
            len(audio_bytes),
            settings.audio_max_bytes,
        )
        await _send_canned_text(raw_phone, _AUDIO_TOO_LONG_RESPONSE)
        return

    # Guard 3 — transcript cache: identical audio (forwarded voice note,
    # retry with a new message_id) never pays Whisper twice.
    audio_sha = hashlib.sha256(audio_bytes).hexdigest()
    text: str | None = None
    try:
        text = await get_cached_transcript(audio_sha)
    except Exception as exc:
        logger.warning("audio_cache_read_failed: %s — transcribing", exc)

    if text is None:
        try:
            text = await transcribe_audio(audio_bytes, mime_type)
        except Exception as exc:
            logger.error("audio_transcription_failed phone_hash=%.8s: %s", phone_hash, exc)
            await _send_canned_text(raw_phone, _AUDIO_FAILURE_RESPONSE)
            return
        try:
            await cache_transcript(audio_sha, text, ttl=settings.audio_transcript_cache_ttl)
        except Exception as exc:
            logger.warning("audio_cache_write_failed: %s (ignored)", exc)
    else:
        logger.info(
            "audio_transcript_cache_hit phone_hash=%.8s sha=%.12s", phone_hash, audio_sha
        )

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


async def _handle_image_message(
    raw_phone: str, phone_hash: str, message_key: dict[str, Any], caption: str
) -> None:
    """Sprint 3.11 — download the photo, identify the racket via vision,
    dispatch as a product inquiry.

    Guards mirror the audio hardening (Sprint 3.10), in order: per-customer
    rate limit (before download), post-download size cap, identification
    cache by content hash. Redis failures in the guards are fail-open.

    Outcomes:
    - vision says it's not a racket        → canned "não identifiquei raquete"
    - racket but brand AND model unreadable → canned "me diz a marca/modelo"
    - identified                            → synthetic "brand model" query
      flows through ``_process_message`` with ``image_product_query=True``;
      triage short-circuits to recommend, which confirms with photo-aware
      wording and registers the product for price/detail follow-ups.
    """
    settings = get_settings()

    # Guard 1 — rate limit per phone_hash, fixed 1h window. 0 disables.
    limit = settings.image_rate_limit_per_hour
    if limit > 0:
        try:
            count = await count_image_message(phone_hash)
        except Exception as exc:
            logger.warning("image_rate_limit_unavailable: %s — proceeding", exc)
            count = 0
        if count > limit:
            logger.warning(
                "image_rate_limited phone_hash=%.8s count=%d limit=%d",
                phone_hash,
                count,
                limit,
            )
            await _send_canned_text(raw_phone, _IMAGE_RATE_LIMIT_RESPONSE)
            return

    try:
        image_bytes, mime_type = await EvolutionClient().get_media_base64(message_key)
    except Exception as exc:
        logger.error("image_download_failed phone_hash=%.8s: %s", phone_hash, exc)
        await _send_canned_text(raw_phone, _IMAGE_FAILURE_RESPONSE)
        return

    # Guard 2 — size cap after download.
    if len(image_bytes) > settings.image_max_bytes:
        logger.warning(
            "image_rejected_too_big phone_hash=%.8s bytes=%d max=%d",
            phone_hash,
            len(image_bytes),
            settings.image_max_bytes,
        )
        await _send_canned_text(raw_phone, _IMAGE_TOO_BIG_RESPONSE)
        return

    # Guard 3 — identification cache: identical photo (forwarded image,
    # retry with a new message_id) never pays the vision call twice.
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    identification: dict[str, Any] | None = None
    try:
        identification = await get_cached_image_id(image_sha)
    except Exception as exc:
        logger.warning("image_cache_read_failed: %s — identifying", exc)

    if identification is None:
        try:
            identification = await identify_racket_image(image_bytes, mime_type, caption)
        except Exception as exc:
            logger.error("image_identification_failed phone_hash=%.8s: %s", phone_hash, exc)
            await _send_canned_text(raw_phone, _IMAGE_FAILURE_RESPONSE)
            return
        try:
            await cache_image_id(image_sha, identification, ttl=settings.image_id_cache_ttl)
        except Exception as exc:
            logger.warning("image_cache_write_failed: %s (ignored)", exc)
    else:
        logger.info(
            "image_id_cache_hit phone_hash=%.8s sha=%.12s", phone_hash, image_sha
        )

    if not identification.get("is_racket"):
        logger.info("image_not_racket phone_hash=%.8s", phone_hash)
        await _send_canned_text(raw_phone, _IMAGE_NOT_RACKET_RESPONSE)
        return

    brand = (identification.get("brand") or "").strip()
    model = (identification.get("model") or "").strip()
    query = f"{brand} {model}".strip()
    if not query:
        logger.info("image_racket_unidentified phone_hash=%.8s", phone_hash)
        await _send_canned_text(raw_phone, _IMAGE_UNIDENTIFIED_RESPONSE)
        return

    logger.info(
        "image_to_product_query phone_hash=%.8s query=%.80s confidence=%s",
        phone_hash,
        query,
        identification.get("confidence"),
    )
    await _process_message(
        raw_phone=raw_phone,
        phone_hash=phone_hash,
        message_text=query,
        extra_state={"image_product_query": True},
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
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Invoke the agent graph and send the reply via WhatsApp.

    This runs in a FastAPI BackgroundTask so the webhook returns 200 immediately
    even though graph inference + network I/O can take several seconds.

    ``extra_state`` (Sprint 3.11) lets media handlers inject turn-scoped
    flags into the graph input (e.g. ``image_product_query`` for photo-born
    product queries). Keys MUST exist in the AgentState TypedDict.
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

    config = {"configurable": {"thread_id": phone_hash}}

    # Reset turn-scoped flags; other state (player_profile, messages) is kept
    # by the checkpointer across turns for the same phone_hash.
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
    if extra_state:
        state_update.update(extra_state)

    try:
        # Sprint 2.6.5 — retry once on Redis-connection errors with a full
        # singleton reconnect between attempts. If both attempts fail,
        # Camada 3 emits a discreet fallback to the client + alert to Andre.
        result: dict[str, Any] = await _ainvoke_with_retry(state_update, config)
    except Exception as exc:
        logger.error("graph_invoke_failed phone_hash=%.8s: %s", phone_hash, exc)
        await _send_fallback_and_alert(raw_phone, phone_hash, exc)
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
