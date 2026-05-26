import logging
import time
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.security.pii_masker import is_clean, mask_pii

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async wrapper around AsyncOpenAI with mandatory PII masking before every call.

    PII masking is applied to all message content and the system prompt.
    In development mode a defense-in-depth assertion verifies the masker
    left no detectable PII after cleaning; failures raise immediately rather
    than silently leaking data to the API.
    """

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(api_key=get_settings().openai_api_key)

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request with PII-safe content.

        Args:
            messages: Conversation history as list of {role, content} dicts.
            system:   System prompt (versioned string from prompts.py).
            max_tokens: Upper bound on completion length.
            temperature: Sampling temperature.
            json_mode: When True, sets response_format=json_object.
                       The system prompt must contain the word "json" for the
                       API to accept this mode.

        Returns:
            The assistant message content string.
        """
        settings = get_settings()

        # ── 1. Mask PII in every content field ─────────────────────────────
        masked_system = mask_pii(system)
        masked_messages: list[dict[str, str]] = [
            {"role": msg["role"], "content": mask_pii(msg.get("content", ""))}
            for msg in messages
        ]

        # ── 2. Defense-in-depth assertion (dev only) ────────────────────────
        # If mask_pii has a gap, we refuse to send rather than leak PII.
        if settings.app_env == "development" and settings.pii_mask_enabled:
            if not is_clean(masked_system):
                raise ValueError("PII leak detected after masking in system prompt")
            for msg in masked_messages:
                if not is_clean(msg["content"]):
                    raise ValueError(
                        f"PII leak detected after masking in message role={msg['role']}"
                    )

        # ── 3. Build final API payload ──────────────────────────────────────
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": masked_system},
            *masked_messages,
        ]

        kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # ── 4. Call the API ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # ── 5. Structured log — content is NEVER included ──────────────────
        logger.info(
            "openai_chat model=%s tokens_in=%d tokens_out=%d latency_ms=%.0f",
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            latency_ms,
        )

        return response.choices[0].message.content or ""
