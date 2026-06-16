"""Evolution API client — sends WhatsApp messages with exponential-backoff retry."""
import asyncio
import base64
import logging
from typing import Any

import httpx

from app.agent.message_splitter import compute_typing_delay
from app.config import get_settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0


class EvolutionClient:
    """Async wrapper for the Evolution API /message/sendText endpoint.

    Retries on 5xx errors and network failures with exponential backoff
    (1 s → 2 s). 4xx errors are not retried and re-raise immediately.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.evolution_api_url.rstrip("/")
        self._instance = settings.evolution_instance
        self._headers = {
            "apikey": settings.evolution_api_key,
            "Content-Type": "application/json",
        }

    async def send_text(self, phone: str, text: str) -> None:
        """Send a plain-text WhatsApp message.

        Args:
            phone: Destination number, digits only (e.g. "5511999999999").
            text:  Message body — must NOT contain raw PII.
        """
        url = f"{self._base_url}/message/sendText/{self._instance}"
        payload = {"number": phone, "text": text}

        # Pre-send structured log. We log destination + payload size only — the
        # API key lives in self._headers and is never serialized to a log line.
        logger.info(
            "evolution_send_text begin phone=%.8s instance=%s text_len=%d",
            phone,
            self._instance,
            len(text),
        )

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload, headers=self._headers)
                    if resp.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"server error {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()
                    logger.info(
                        "evolution_send_text phone=%.8s status=%d attempt=%d",
                        phone,
                        resp.status_code,
                        attempt + 1,
                    )
                    return

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise  # 4xx — not a server fault, don't retry
                last_exc = exc
            except httpx.RequestError as exc:
                last_exc = exc

            if attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY_S * (2**attempt)  # 1 s, 2 s
                logger.warning(
                    "evolution_send_text retry attempt=%d delay=%.1fs phone=%.8s err=%s",
                    attempt + 1,
                    delay,
                    phone,
                    last_exc,
                )
                await asyncio.sleep(delay)

        logger.error(
            "evolution_send_text failed after %d attempts phone=%.8s",
            _MAX_RETRIES,
            phone,
        )
        raise last_exc  # type: ignore[misc]

    async def get_media_base64(self, message_key: dict[str, Any]) -> tuple[bytes, str]:
        """Download an incoming media attachment as raw bytes + mime type.

        Calls the Evolution self-hosted endpoint
        ``POST /chat/getBase64FromMediaMessage/<instance>`` with the full
        ``message.key`` dict from the inbound webhook payload (most Evolution
        builds also accept just ``{"key": {"id": ...}}`` but echoing the full
        key keeps the request compatible across builds).

        The endpoint returns JSON with a ``base64`` field plus ``mimetype``.
        Different builds use slightly different key casing — we tolerate both
        ``mimetype`` and ``mimeType``.

        Raises:
            httpx.HTTPError on transport / 4xx / 5xx (caller decides fallback).
            ValueError when the response lacks the expected base64 payload.
        """
        url = f"{self._base_url}/chat/getBase64FromMediaMessage/{self._instance}"
        payload = {"message": {"key": message_key}}

        logger.info(
            "evolution_get_media begin message_id=%s instance=%s",
            (message_key.get("id") or "")[:24],
            self._instance,
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=self._headers)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        b64 = data.get("base64") or data.get("media") or ""
        if not b64:
            raise ValueError(
                f"Evolution getBase64FromMediaMessage returned no base64; keys={sorted(data.keys())}"
            )
        mimetype = (
            data.get("mimetype")
            or data.get("mimeType")
            or "application/octet-stream"
        )
        raw = base64.b64decode(b64)

        logger.info(
            "evolution_get_media done bytes=%d mimetype=%s message_id=%s",
            len(raw),
            mimetype,
            (message_key.get("id") or "")[:24],
        )
        return raw, mimetype

    async def send_presence(
        self,
        phone: str,
        *,
        presence: str = "composing",
        delay_ms: int = 0,
    ) -> bool:
        """Sprint 2.7.2 — send a presence indicator ("typing...", "recording",
        "online") to mask debounce-window latency.

        Evolution V2 exposes this at ``POST /chat/sendPresence/{instance}``.
        Different self-hosted builds vary in payload shape; we send the
        common-denominator JSON: ``{"number", "presence", "delay"}``.

        Best-effort: returns True on 2xx, False on any failure. Never
        raises — typing indicator is cosmetic, must NOT break the message
        flow. Caller may ignore the return value.
        """
        url = f"{self._base_url}/chat/sendPresence/{self._instance}"
        payload: dict[str, Any] = {"number": phone, "presence": presence}
        if delay_ms:
            payload["delay"] = delay_ms

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers)
            ok = 200 <= resp.status_code < 300
            logger.info(
                "evolution_send_presence phone=%.8s presence=%s status=%d",
                phone, presence, resp.status_code,
            )
            return ok
        except Exception as exc:  # noqa: BLE001 — best effort, cosmetic
            logger.info(
                "evolution_send_presence_failed phone=%.8s presence=%s err=%.80s "
                "(non-fatal — typing indicator is cosmetic)",
                phone, presence, str(exc),
            )
            return False

    async def send_text_blocks(self, phone: str, blocks: list[str]) -> None:
        """Send a sequence of WhatsApp messages with humanizing delays.

        Each block is sent via ``send_text``. Before every block EXCEPT the
        first one, we sleep for ``compute_typing_delay(block)`` (≈1–3 s) so
        the messages arrive as a person typing rather than a single dump.

        Args:
            phone:  Destination number, digits only.
            blocks: Ordered list of message strings. Empty list is a no-op.
        """
        if not blocks:
            return

        for index, block in enumerate(blocks):
            if index > 0:
                delay = compute_typing_delay(block)
                logger.info(
                    "evolution_send_blocks pause phone=%.8s index=%d delay_s=%.2f",
                    phone,
                    index,
                    delay,
                )
                await asyncio.sleep(delay)
            await self.send_text(phone, block)

        logger.info(
            "evolution_send_blocks done phone=%.8s count=%d", phone, len(blocks)
        )
