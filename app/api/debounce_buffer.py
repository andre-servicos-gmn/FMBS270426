"""Sprint 2.7.2 — in-memory debounce buffer for fast-burst text messages.

When the customer sends multiple WhatsApp messages within
``window_ms`` of each other ("Quero a Proteo" + "Vc tem?" 200 ms later),
the agent used to process them as 2 concurrent background tasks. Both
read the same starting state, both wrote a new checkpoint — last writer
won, and the second response was a generic smalltalk reply ("E aí
Felipe!") because its state snapshot didn't include the first turn's
flag updates.

This buffer fixes the symptom AT THE ROOT: it groups rapid messages
into a single processing turn. The graph runs ONCE with the concatenated
input ("Quero a Proteo. Vc tem?") and emits a single coherent reply.

Design choices:
  - In-memory dict keyed by phone_hash. Single-replica deploy on
    easypanel makes Redis-backed coordination unnecessary; refactor only
    if scaling to >1 replica.
  - A single ``asyncio.Lock`` guards mutations so the dict + per-entry
    state stay consistent under concurrent ``add()`` calls.
  - Cap and hard-TTL are defensive circuit-breakers, not the primary
    flush trigger. Primary trigger is the debounce timer (``window_ms``).
  - Tests get a ``flush_now()`` API that bypasses the timer — keeps
    tests deterministic without real time.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


# Callback signatures.
#   - ``OnFlush``: invoked with (raw_phone, phone_hash, merged_text) when
#     the buffer flushes. Must await the downstream processing.
#   - ``OnFirstMessage``: optional, invoked with (raw_phone) when a new
#     buffer opens. Used for the "typing..." presence indicator. Errors
#     here are swallowed — purely cosmetic.
OnFlush = Callable[[str, str, str], Awaitable[None]]
OnFirstMessage = Callable[[str], Awaitable[None]]


@dataclass
class _BufferEntry:
    raw_phone: str = ""
    messages: list[str] = field(default_factory=list)
    started_at: float = 0.0  # ``time.monotonic()`` seconds at open
    task: asyncio.Task[None] | None = None


def _merge_messages(messages: list[str]) -> str:
    """Join with ``". "`` after stripping noise.

    Strips trailing dots/whitespace from each message (except the last —
    that one keeps "?" or "!" so the LLM sees the customer's tone). Empty
    fragments are dropped. The result reads like a single coherent
    sentence-pair the customer wrote out.
    """
    if not messages:
        return ""
    cleaned: list[str] = []
    for m in messages:
        s = (m or "").strip().rstrip(". ")
        if s:
            cleaned.append(s)
    return ". ".join(cleaned)


class DebounceBuffer:
    """Async, in-memory message buffer with per-phone debounce.

    Invariants:
        - At most one ``_BufferEntry`` per phone_hash in ``_entries``.
        - The entry's ``task`` (if not None and not done) is the only
          scheduled flush; ``add()`` cancels it on each new message and
          schedules a fresh one (true debounce: timer resets).
        - All flushes invoke ``on_flush`` exactly once with the merged
          text. The merge is deterministic — same input order produces
          the same output.
    """

    def __init__(
        self,
        *,
        window_ms: int,
        cap: int,
        hard_ttl_ms: int,
        on_flush: OnFlush,
        on_first_message: OnFirstMessage | None = None,
    ) -> None:
        if window_ms <= 0:
            raise ValueError("window_ms must be > 0")
        if cap < 1:
            raise ValueError("cap must be >= 1")
        if hard_ttl_ms <= 0:
            raise ValueError("hard_ttl_ms must be > 0")
        # NOTE: production config typically has hard_ttl_ms >> window_ms
        # (e.g. 8000 vs 1500), but we don't enforce that here — tests want
        # to set hard_ttl tiny to exercise the circuit-breaker path.
        self._window_ms = window_ms
        self._cap = cap
        self._hard_ttl_ms = hard_ttl_ms
        self._on_flush = on_flush
        self._on_first_message = on_first_message
        self._lock = asyncio.Lock()
        self._entries: dict[str, _BufferEntry] = {}

    # ── Public API ──────────────────────────────────────────────────────

    async def add(
        self, *, raw_phone: str, phone_hash: str, message_text: str
    ) -> bool:
        """Add a message to the buffer for ``phone_hash``.

        Returns ``True`` if this opened a NEW buffer (caller may use the
        return to fire a typing indicator manually — though
        ``on_first_message`` already does that automatically if
        configured).

        Cap or hard-TTL triggers force an IMMEDIATE flush; the merged
        text is dispatched via ``on_flush`` in a fire-and-forget task so
        ``add()`` returns quickly to the webhook.
        """
        flush_payload: tuple[str, str, str, str, int] | None = None
        was_new = False

        async with self._lock:
            entry = self._entries.get(phone_hash)
            if entry is None:
                entry = _BufferEntry(
                    raw_phone=raw_phone,
                    started_at=time.monotonic(),
                )
                self._entries[phone_hash] = entry
                was_new = True
            else:
                # True debounce: cancel pending flush so the new timer wins.
                if entry.task is not None and not entry.task.done():
                    entry.task.cancel()

            entry.messages.append(message_text)
            elapsed_ms = (time.monotonic() - entry.started_at) * 1000.0

            if (
                len(entry.messages) >= self._cap
                or elapsed_ms >= self._hard_ttl_ms
            ):
                # Force flush — pop from the dict so a late timer can't
                # also flush the same entry.
                self._entries.pop(phone_hash, None)
                reason = (
                    "cap" if len(entry.messages) >= self._cap else "hard_ttl"
                )
                merged = _merge_messages(entry.messages)
                flush_payload = (
                    entry.raw_phone,
                    phone_hash,
                    merged,
                    reason,
                    len(entry.messages),
                )
            else:
                # Schedule a delayed flush. ``entry`` is captured by ref;
                # the task compares it to the current entry on wake to
                # bail out if it was replaced/cancelled meanwhile.
                entry.task = asyncio.create_task(
                    self._delayed_flush(phone_hash, entry)
                )

        # Outside the lock — typing indicator is fire-and-forget (cosmetic).
        if was_new and flush_payload is None and self._on_first_message is not None:
            asyncio.create_task(self._safe_call_first(raw_phone))

        # Cap/TTL-triggered flush is awaited INLINE here. The webhook has
        # already returned 200 to Evolution by the time this code runs
        # (we live inside a FastAPI BackgroundTask), so awaiting the graph
        # doesn't add customer-facing latency — and it keeps the call
        # boundary deterministic for tests + simpler in production.
        if flush_payload is not None:
            raw_phone_fl, ph_fl, merged_fl, reason_fl, n_fl = flush_payload
            logger.info(
                "debounce_flush phone_hash=%.8s reason=%s n_messages=%d chars=%d",
                ph_fl, reason_fl, n_fl, len(merged_fl),
            )
            await self._safe_call_flush(raw_phone_fl, ph_fl, merged_fl)

        logger.info(
            "debounce_add phone_hash=%.8s was_new=%s buffer_size=%d msg_len=%d",
            phone_hash, was_new,
            0 if flush_payload is not None
            else len(self._entries.get(phone_hash, _BufferEntry()).messages),
            len(message_text),
        )
        return was_new

    async def flush_now(self, phone_hash: str) -> bool:
        """Force-flush the buffer for ``phone_hash`` immediately.

        Used by tests to avoid relying on real timers. Returns True if
        there was something to flush, False if no buffer existed.
        """
        async with self._lock:
            entry = self._entries.pop(phone_hash, None)
            if entry is None:
                return False
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
            merged = _merge_messages(entry.messages)
            raw_phone = entry.raw_phone
            n = len(entry.messages)

        logger.info(
            "debounce_flush phone_hash=%.8s reason=force n_messages=%d chars=%d",
            phone_hash, n, len(merged),
        )
        await self._safe_call_flush(raw_phone, phone_hash, merged)
        return True

    def has_buffered(self, phone_hash: str) -> bool:
        """True if there's a pending buffer for ``phone_hash``."""
        return phone_hash in self._entries

    def buffered_count(self, phone_hash: str) -> int:
        """How many messages are currently buffered for ``phone_hash``."""
        entry = self._entries.get(phone_hash)
        return len(entry.messages) if entry is not None else 0

    async def shutdown(self) -> None:
        """Cancel every pending timer (best-effort) and clear state.
        Called at app shutdown."""
        async with self._lock:
            for entry in self._entries.values():
                if entry.task is not None and not entry.task.done():
                    entry.task.cancel()
            self._entries.clear()

    # ── Internals ───────────────────────────────────────────────────────

    async def _delayed_flush(
        self, phone_hash: str, entry_ref: _BufferEntry
    ) -> None:
        try:
            await asyncio.sleep(self._window_ms / 1000.0)
        except asyncio.CancelledError:
            return  # superseded by a newer message

        flush_payload: tuple[str, str, str, int] | None = None
        async with self._lock:
            current = self._entries.get(phone_hash)
            # Guard: if the entry was already popped (cap/ttl flush) or a
            # newer buffer replaced it, do nothing.
            if current is not entry_ref:
                return
            self._entries.pop(phone_hash, None)
            merged = _merge_messages(current.messages)
            flush_payload = (
                current.raw_phone,
                phone_hash,
                merged,
                len(current.messages),
            )

        raw_phone, ph, merged_text, n = flush_payload
        logger.info(
            "debounce_flush phone_hash=%.8s reason=window n_messages=%d chars=%d",
            ph, n, len(merged_text),
        )
        await self._safe_call_flush(raw_phone, ph, merged_text)

    async def _safe_call_flush(
        self, raw_phone: str, phone_hash: str, merged: str
    ) -> None:
        try:
            await self._on_flush(raw_phone, phone_hash, merged)
        except Exception as exc:  # noqa: BLE001 — must not crash buffer
            logger.error(
                "debounce_flush_callback_failed phone_hash=%.8s: %s",
                phone_hash, exc,
            )

    async def _safe_call_first(self, raw_phone: str) -> None:
        try:
            assert self._on_first_message is not None
            await self._on_first_message(raw_phone)
        except Exception as exc:  # noqa: BLE001 — typing indicator is cosmetic
            logger.info(
                "debounce_on_first_message_failed phone=%.8s err=%.80s "
                "(non-fatal — typing indicator is cosmetic)",
                raw_phone, str(exc),
            )
