"""Media processors for inbound WhatsApp attachments.

Sprint 1.12 — ``transcribe_audio(bytes, mime)`` returns the transcribed text
in PT-BR. Uses ``whisper-1`` with ``language="pt"`` so the recognition is
optimised for Brazilian Portuguese.

Sprint 3.11 — ``identify_racket_image(bytes, mime, caption)`` sends the photo
to the vision model (``OPENAI_VISION_MODEL``, default gpt-4o) and returns a
structured identification dict: is the photo a racket, and if so which
brand/model is visible.

The OpenAI SDK's ``audio.transcriptions.create()`` accepts a (filename,
bytes, mime_type) tuple as the ``file`` parameter — convenient because the
audio comes in as a bytes blob from the Evolution adapter, not a file on
disk.

Cost note: whisper-1 is billed at ~$0.006/minute. A typical WhatsApp voice
note (≤30 s) costs about $0.003. A gpt-4o vision call at high detail on a
WhatsApp-compressed photo runs ~$0.005. We log every call so consumption can
be tracked from the structured logs.
"""
import base64
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from app.agent.prompts import SYSTEM_RACKET_VISION
from app.config import get_settings
from app.security.pii_masker import mask_pii

logger = logging.getLogger(__name__)

# Default to .ogg because every WhatsApp voice note we've seen in pilot
# arrives as "audio/ogg; codecs=opus". Other formats below cover the rare
# cases (uploaded mp3, m4a, etc.). Whisper accepts all of these.
_MIME_TO_EXT: dict[str, str] = {
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/flac": "flac",
}

_TRANSCRIPTION_TIMEOUT_S = 30.0


def _mime_to_extension(mime_type: str) -> str:
    """Return a file extension Whisper accepts, given the source mimetype.

    WhatsApp sends ``audio/ogg; codecs=opus`` — we trim parameters before
    looking up the table.
    """
    base = (mime_type or "").split(";", 1)[0].strip().lower()
    return _MIME_TO_EXT.get(base, "ogg")


async def transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    """Transcribe an audio blob to text using Whisper.

    Args:
        audio_bytes: Raw audio file bytes (any Whisper-supported codec).
        mime_type:   Source mimetype from the Evolution payload — drives the
                     filename extension we hand to the OpenAI SDK.

    Returns:
        Transcribed text trimmed of leading/trailing whitespace. Empty string
        when the audio is silent / inaudible — the caller decides the
        user-facing fallback in that case.

    Raises:
        Propagates OpenAI SDK errors (RateLimitError, APIError, timeout)
        without modification so the caller can pick the right user message.
    """
    extension = _mime_to_extension(mime_type)
    filename = f"audio.{extension}"

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=_TRANSCRIPTION_TIMEOUT_S)

    t0 = time.perf_counter()
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes, mime_type or "audio/ogg"),
        language="pt",
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # The default response_format is "json"; .text holds the transcription.
    text = (getattr(response, "text", None) or "").strip()

    logger.info(
        "whisper_transcribed bytes=%d mime=%s latency_ms=%.0f text_length=%d",
        len(audio_bytes),
        mime_type,
        latency_ms,
        len(text),
    )
    return text


# ── Sprint 3.11 — racket identification from photos ──────────────────────────

_VISION_TIMEOUT_S = 30.0

# WhatsApp photos arrive as jpeg in every pilot capture; the other entries
# cover gallery uploads. Anything unknown falls back to jpeg — the vision
# API sniffs the real content anyway.
_IMAGE_MIME_ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _normalize_image_mime(mime_type: str) -> str:
    base = (mime_type or "").split(";", 1)[0].strip().lower()
    return base if base in _IMAGE_MIME_ALLOWED else "image/jpeg"


async def identify_racket_image(
    image_bytes: bytes, mime_type: str, caption: str = ""
) -> dict[str, Any]:
    """Identify the racket in a customer photo using the vision model.

    Args:
        image_bytes: Raw image bytes from the Evolution adapter.
        mime_type:   Source mimetype from the payload (drives the data URL).
        caption:     Optional caption the customer sent with the photo. It is
                     PII-masked before leaving the process (the image itself
                     carries no regex-maskable PII).

    Returns:
        Dict with keys ``is_racket`` (bool), ``brand`` (str | None),
        ``model`` (str | None), ``confidence`` ("high" | "low" | None).
        Missing keys in the model output default to the negative case.

    Raises:
        Propagates OpenAI SDK errors (RateLimitError, APIError, timeout).
        ValueError when the model reply is not parseable JSON — the caller
        treats it like any other infrastructure failure.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=_VISION_TIMEOUT_S)

    data_url = (
        f"data:{_normalize_image_mime(mime_type)};base64,"
        f"{base64.b64encode(image_bytes).decode()}"
    )
    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
    ]
    masked_caption = mask_pii(caption or "").strip()
    if masked_caption:
        user_content.append(
            {"type": "text", "text": f"Legenda enviada pelo cliente: {masked_caption}"}
        )

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=settings.openai_vision_model,
        messages=[
            {"role": "system", "content": SYSTEM_RACKET_VISION},
            {"role": "user", "content": user_content},
        ],
        max_tokens=150,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("vision_parse_failed raw=%.120r", raw)
        raise ValueError("vision model returned non-JSON output") from exc
    if not isinstance(parsed, dict):
        raise ValueError("vision model returned non-object JSON")

    result: dict[str, Any] = {
        "is_racket": bool(parsed.get("is_racket", False)),
        "brand": (str(parsed["brand"]).strip() or None) if parsed.get("brand") else None,
        "model": (str(parsed["model"]).strip() or None) if parsed.get("model") else None,
        "confidence": parsed.get("confidence") if parsed.get("confidence") in ("high", "low") else None,
    }

    # Structured log — identification outcome only, never raw content.
    logger.info(
        "vision_racket_id bytes=%d mime=%s latency_ms=%.0f is_racket=%s "
        "has_brand=%s has_model=%s confidence=%s",
        len(image_bytes),
        mime_type,
        latency_ms,
        result["is_racket"],
        result["brand"] is not None,
        result["model"] is not None,
        result["confidence"],
    )
    return result
