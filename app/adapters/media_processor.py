"""Sprint 1.12 — OpenAI Whisper transcription for inbound WhatsApp audio.

Single entry point ``transcribe_audio(bytes, mime)`` that returns the
transcribed text in PT-BR. Uses ``whisper-1`` with ``language="pt"`` so the
recognition is optimised for Brazilian Portuguese.

The OpenAI SDK's ``audio.transcriptions.create()`` accepts a (filename,
bytes, mime_type) tuple as the ``file`` parameter — convenient because the
audio comes in as a bytes blob from the Evolution adapter, not a file on
disk.

Cost note: whisper-1 is billed at ~$0.006/minute. A typical WhatsApp voice
note (≤30 s) costs about $0.003. We log every call so consumption can be
tracked from the structured logs.
"""
import logging
import time

from openai import AsyncOpenAI

from app.config import get_settings

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
