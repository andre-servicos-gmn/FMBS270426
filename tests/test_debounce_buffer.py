"""Sprint 2.7.2 — debounce buffer tests.

Felipe's production bug: two messages in <1.5s ("Quero a Proteo" + "Vc
tem?") arrived as 2 concurrent background tasks, races the checkpointer,
2nd response goes generic. The buffer groups rapid bursts so the graph
runs ONCE with the concatenated input.

Tests are DETERMINISTIC:
  - Most use ``flush_now()`` to force the flush — no real-time waiting.
  - A few use a tiny window (10 ms) + a small sleep when the goal is to
    cover the timer path itself. These tolerate ~50 ms of slack so
    they don't flake on slow CI.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.debounce_buffer import DebounceBuffer, _merge_messages


_PHONE_A = "5511987654321"
_PHONE_B = "5511999999999"
_HASH_A = "hashAhashA" * 6 + "hashA"
_HASH_B = "hashBhashB" * 6 + "hashB"
_TOKEN = "test-webhook-token-abc"


def _make_buffer(
    *,
    window_ms: int = 50,
    cap: int = 10,
    hard_ttl_ms: int | None = None,
    on_first_message=None,
) -> tuple[DebounceBuffer, AsyncMock]:
    """Helper — returns (buffer, on_flush_mock).

    Window defaults to 50 ms so timer-path tests are quick when used.
    cap/ttl are loose so they don't fire unintentionally. ``hard_ttl_ms``
    auto-scales to ``window_ms * 4`` unless specified, so large windows
    in flush_now tests don't violate the ``hard_ttl >= window`` invariant.
    """
    if hard_ttl_ms is None:
        hard_ttl_ms = max(200, window_ms * 4)
    on_flush = AsyncMock()
    buffer = DebounceBuffer(
        window_ms=window_ms,
        cap=cap,
        hard_ttl_ms=hard_ttl_ms,
        on_flush=on_flush,
        on_first_message=on_first_message,
    )
    return buffer, on_flush


# ════════════════════════════════════════════════════════════════════════════
# UNIT — _merge_messages
# ════════════════════════════════════════════════════════════════════════════

def test_merge_basic_concat():
    assert _merge_messages(["A", "B"]) == "A. B"


def test_merge_three_messages_in_order():
    assert _merge_messages(["um", "dois", "três"]) == "um. dois. três"


def test_merge_strips_trailing_dot_and_space():
    """The customer's '.' at end of msg shouldn't double up after our
    ". " join."""
    assert _merge_messages(["Quero a Proteo.", "Vc tem?"]) == "Quero a Proteo. Vc tem?"


def test_merge_preserves_question_mark_and_exclamation():
    assert _merge_messages(["Tem kronos?", "Manda aí!"]) == "Tem kronos?. Manda aí!"


def test_merge_drops_empty_fragments():
    assert _merge_messages(["", "  ", "ok"]) == "ok"


def test_merge_felipe_scenario():
    """The exact production case."""
    merged = _merge_messages(["Quero a Proteo", "Vc tem?"])
    assert merged == "Quero a Proteo. Vc tem?"


# ════════════════════════════════════════════════════════════════════════════
# FLUSH_NOW — deterministic dispatching
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_flush_now_dispatches_merged_messages():
    buffer, on_flush = _make_buffer(window_ms=10_000)  # large — won't fire
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="Quero a Proteo")
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="Vc tem?")

    assert buffer.has_buffered(_HASH_A)
    assert buffer.buffered_count(_HASH_A) == 2

    flushed = await buffer.flush_now(_HASH_A)
    assert flushed is True
    assert not buffer.has_buffered(_HASH_A)

    on_flush.assert_awaited_once()
    args = on_flush.await_args.args
    assert args[0] == _PHONE_A           # raw_phone
    assert args[1] == _HASH_A            # phone_hash
    assert args[2] == "Quero a Proteo. Vc tem?"


@pytest.mark.asyncio
async def test_flush_now_returns_false_when_empty():
    buffer, on_flush = _make_buffer(window_ms=10_000)
    assert await buffer.flush_now(_HASH_A) is False
    on_flush.assert_not_awaited()


# ════════════════════════════════════════════════════════════════════════════
# BURST GROUPING
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_two_rapid_messages_grouped():
    """The Felipe case end-to-end through the buffer."""
    buffer, on_flush = _make_buffer(window_ms=10_000)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="Quero a Proteo")
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="Vc tem?")
    await buffer.flush_now(_HASH_A)

    on_flush.assert_awaited_once()
    assert on_flush.await_args.args[2] == "Quero a Proteo. Vc tem?"


@pytest.mark.asyncio
async def test_three_messages_burst_grouped_in_order():
    buffer, on_flush = _make_buffer(window_ms=10_000)
    for msg in ["um", "dois", "três"]:
        await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text=msg)
    await buffer.flush_now(_HASH_A)
    assert on_flush.await_args.args[2] == "um. dois. três"


# ════════════════════════════════════════════════════════════════════════════
# TIMER PATH — uses real time but with a tiny window
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_timer_flushes_after_window():
    buffer, on_flush = _make_buffer(window_ms=30, hard_ttl_ms=200)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="oi")
    # Wait > window for the timer to fire.
    await asyncio.sleep(0.1)
    on_flush.assert_awaited_once()
    assert on_flush.await_args.args[2] == "oi"
    assert not buffer.has_buffered(_HASH_A)


@pytest.mark.asyncio
async def test_timer_resets_on_new_message():
    """Each new message cancels the prior timer; the flush is keyed to
    the LAST arrival, not the first."""
    buffer, on_flush = _make_buffer(window_ms=40, hard_ttl_ms=500)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="a")
    await asyncio.sleep(0.02)  # Within window — won't flush
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="b")
    await asyncio.sleep(0.02)  # Still within window — won't flush
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="c")
    # After 0.04s we still have all 3 buffered (timer was reset twice).
    assert buffer.buffered_count(_HASH_A) == 3
    on_flush.assert_not_awaited()
    # Now wait > window for the final timer to fire.
    await asyncio.sleep(0.08)
    on_flush.assert_awaited_once()
    assert on_flush.await_args.args[2] == "a. b. c"


@pytest.mark.asyncio
async def test_message_outside_window_processed_separately():
    """After the first burst flushes, a NEW message opens a fresh
    buffer."""
    buffer, on_flush = _make_buffer(window_ms=30, hard_ttl_ms=200)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="a")
    await asyncio.sleep(0.1)  # First flush fires
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="b")
    await asyncio.sleep(0.1)  # Second flush fires
    assert on_flush.await_count == 2
    assert on_flush.await_args_list[0].args[2] == "a"
    assert on_flush.await_args_list[1].args[2] == "b"


# ════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKERS — cap and hard_ttl
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_cap_forces_flush():
    """When buffer hits cap, flush immediately without waiting for timer."""
    buffer, on_flush = _make_buffer(window_ms=10_000, cap=3, hard_ttl_ms=20_000)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="a")
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="b")
    # 3rd message hits cap → immediate flush.
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="c")
    # Give the fire-and-forget flush task a tick to run.
    await asyncio.sleep(0.01)
    on_flush.assert_awaited_once()
    assert on_flush.await_args.args[2] == "a. b. c"
    assert not buffer.has_buffered(_HASH_A)


@pytest.mark.asyncio
async def test_hard_ttl_forces_flush():
    """If the customer keeps resetting the timer for hard_ttl, force flush."""
    buffer, on_flush = _make_buffer(window_ms=10_000, cap=99, hard_ttl_ms=50)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="a")
    await asyncio.sleep(0.06)  # exceed hard_ttl
    # Next add should trigger the hard-ttl flush.
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="b")
    await asyncio.sleep(0.01)
    on_flush.assert_awaited_once()
    assert on_flush.await_args.args[2] == "a. b"


# ════════════════════════════════════════════════════════════════════════════
# MULTI-CLIENT ISOLATION
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_two_clients_independent_buffers():
    buffer, on_flush = _make_buffer(window_ms=10_000)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="cliente A diz oi")
    await buffer.add(raw_phone=_PHONE_B, phone_hash=_HASH_B, message_text="cliente B diz oi")

    assert buffer.buffered_count(_HASH_A) == 1
    assert buffer.buffered_count(_HASH_B) == 1

    await buffer.flush_now(_HASH_A)
    # B's buffer untouched
    assert buffer.has_buffered(_HASH_B)
    assert buffer.buffered_count(_HASH_B) == 1

    # A flushed with its own content
    assert on_flush.await_args.args[1] == _HASH_A
    assert "cliente A" in on_flush.await_args.args[2]
    assert "cliente B" not in on_flush.await_args.args[2]


# ════════════════════════════════════════════════════════════════════════════
# TYPING INDICATOR — fires on first message of a new buffer
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_first_message_callback_fires_for_new_buffer():
    on_first = AsyncMock()
    buffer, _ = _make_buffer(window_ms=10_000, on_first_message=on_first)

    was_new = await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="oi")
    # Give the create_task a chance to run.
    await asyncio.sleep(0.01)

    assert was_new is True
    on_first.assert_awaited_once_with(_PHONE_A)


@pytest.mark.asyncio
async def test_first_message_callback_does_not_fire_on_subsequent_messages():
    on_first = AsyncMock()
    buffer, _ = _make_buffer(window_ms=10_000, on_first_message=on_first)

    was_new_1 = await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="oi")
    was_new_2 = await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="tudo bem?")
    await asyncio.sleep(0.01)

    assert was_new_1 is True
    assert was_new_2 is False
    on_first.assert_awaited_once_with(_PHONE_A)  # ONLY ONCE


@pytest.mark.asyncio
async def test_first_message_callback_swallows_errors():
    """Typing indicator failures must NOT crash add()."""
    async def boom(_phone: str) -> None:
        raise RuntimeError("Evolution presence endpoint down")

    buffer, _ = _make_buffer(window_ms=10_000, on_first_message=boom)
    # Should not raise.
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="oi")
    await asyncio.sleep(0.01)


# ════════════════════════════════════════════════════════════════════════════
# RESILIENCE — flush callback errors don't break the buffer
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_flush_callback_error_does_not_corrupt_buffer():
    on_flush = AsyncMock(side_effect=RuntimeError("graph crashed"))
    buffer = DebounceBuffer(
        window_ms=10_000,
        cap=10,
        hard_ttl_ms=20_000,
        on_flush=on_flush,
    )
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="oi")
    # flush_now does NOT raise even though the callback throws.
    await buffer.flush_now(_HASH_A)
    # Buffer is empty now — new message starts fresh.
    assert not buffer.has_buffered(_HASH_A)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="tudo bem?")
    assert buffer.has_buffered(_HASH_A)


# ════════════════════════════════════════════════════════════════════════════
# SHUTDOWN
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_shutdown_cancels_pending_timers():
    buffer, on_flush = _make_buffer(window_ms=10_000)
    await buffer.add(raw_phone=_PHONE_A, phone_hash=_HASH_A, message_text="oi")
    await buffer.shutdown()
    assert not buffer.has_buffered(_HASH_A)
    # No flush happened (timer cancelled, buffer cleared).
    on_flush.assert_not_awaited()


# ════════════════════════════════════════════════════════════════════════════
# WEBHOOK INTEGRATION — non-text bypasses, /reset bypasses, text goes via buffer
# ════════════════════════════════════════════════════════════════════════════


import fakeredis.aioredis as fakeredis_aioredis  # noqa: E402
from app.main import app  # noqa: E402


def _payload(text: str | None, *, message_id: str = "MSG-001", kind: str = "text") -> dict:
    """Build an Evolution-shaped webhook payload."""
    base_msg: dict
    if kind == "text":
        base_msg = {"conversation": text}
    elif kind == "image":
        base_msg = {"imageMessage": {"caption": "x"}}
    elif kind == "audio":
        base_msg = {"audioMessage": {}}
    elif kind == "document":
        base_msg = {"documentMessage": {}}
    else:
        base_msg = {}
    return {
        "event": "messages.upsert",
        "instance": "test-instance",
        "data": {
            "key": {
                "remoteJid": f"{_PHONE_A}@s.whatsapp.net",
                "fromMe": False,
                "id": message_id,
            },
            "message": base_msg,
            "messageType": "conversation",
            "messageTimestamp": 1700000000,
        },
    }


@pytest.fixture
def webhook_env(monkeypatch):
    monkeypatch.setenv("EVOLUTION_WEBHOOK_TOKEN", _TOKEN)
    monkeypatch.setenv("RESET_ALLOWED_PHONES", "")
    monkeypatch.setenv("MESSAGE_DEBOUNCE_MS", "10000")  # large — won't fire on its own
    monkeypatch.setenv("MESSAGE_DEBOUNCE_CAP", "10")
    monkeypatch.setenv("MESSAGE_DEBOUNCE_HARD_TTL_MS", "60000")
    from app.config import get_settings
    get_settings.cache_clear()
    # Reset the singleton so it picks up the new settings.
    from app.api.webhook import _reset_debounce_buffer
    _reset_debounce_buffer()
    yield
    get_settings.cache_clear()
    _reset_debounce_buffer()


@pytest.mark.asyncio
async def test_text_messages_go_via_buffer(webhook_env):
    """Two text messages from the same customer in <1.5s → 1 graph
    invocation with merged input."""
    from app.api import webhook as webhook_mod

    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)
    process_mock = AsyncMock()

    with (
        patch.object(webhook_mod, "_process_message", new=process_mock),
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
        patch.object(webhook_mod, "EvolutionClient") as MockEvo,
    ):
        MockEvo.return_value.send_presence = AsyncMock(return_value=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post(
                "/webhook/whatsapp",
                json=_payload("Quero a Proteo", message_id="MSG-001"),
                headers={"apikey": _TOKEN},
            )
            r2 = await client.post(
                "/webhook/whatsapp",
                json=_payload("Vc tem?", message_id="MSG-002"),
                headers={"apikey": _TOKEN},
            )

        assert r1.status_code == 200
        assert r2.status_code == 200
        # Let the background buffer.add() tasks run.
        await asyncio.sleep(0.05)

        # _process_message should NOT have fired yet (window is 10s).
        process_mock.assert_not_called()

        # Force flush — single invocation with concatenated text.
        buffer = webhook_mod._get_debounce_buffer()
        # Compute phone_hash the same way webhook does.
        from app.security.pii_masker import hash_phone
        phone_hash = hash_phone(_PHONE_A)
        flushed = await buffer.flush_now(phone_hash)
        assert flushed is True
        await asyncio.sleep(0.01)

    process_mock.assert_awaited_once()
    call_kwargs = process_mock.await_args.kwargs
    assert call_kwargs["message_text"] == "Quero a Proteo. Vc tem?"
    assert call_kwargs["raw_phone"] == _PHONE_A


@pytest.mark.asyncio
async def test_image_message_bypasses_buffer(webhook_env):
    """Image messages must NEVER hit the text debounce buffer."""
    from app.api import webhook as webhook_mod

    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)
    process_mock = AsyncMock()

    with (
        patch.object(webhook_mod, "_process_message", new=process_mock),
        patch.object(webhook_mod, "_send_canned_text", new_callable=AsyncMock) as send_canned,
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload(None, message_id="MSG-IMG", kind="image"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("kind") == "image"
    # Image path uses the canned-response helper, NOT the buffer.
    await asyncio.sleep(0.01)
    process_mock.assert_not_called()
    send_canned.assert_awaited_once()

    # Buffer is empty for this phone.
    from app.security.pii_masker import hash_phone
    buffer = webhook_mod._get_debounce_buffer()
    assert not buffer.has_buffered(hash_phone(_PHONE_A))


@pytest.mark.asyncio
async def test_audio_message_bypasses_buffer(webhook_env):
    """Audio messages must NEVER hit the text debounce buffer either."""
    from app.api import webhook as webhook_mod

    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)
    process_mock = AsyncMock()
    audio_handler = AsyncMock()

    with (
        patch.object(webhook_mod, "_process_message", new=process_mock),
        patch.object(webhook_mod, "_handle_audio_message", new=audio_handler),
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/webhook/whatsapp",
                json=_payload(None, message_id="MSG-AUDIO", kind="audio"),
                headers={"apikey": _TOKEN},
            )

    assert resp.status_code == 200
    assert resp.json().get("kind") == "audio"
    await asyncio.sleep(0.01)
    process_mock.assert_not_called()
    audio_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_idempotency_still_applies_before_buffer(webhook_env):
    """Same message_id 2x → only 1 entry in the buffer.

    The check happens at the webhook entrypoint, so the buffer never
    sees the duplicate. Mirrors the existing webhook contract.
    """
    from app.api import webhook as webhook_mod

    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)
    process_mock = AsyncMock()

    with (
        patch.object(webhook_mod, "_process_message", new=process_mock),
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
        patch.object(webhook_mod, "EvolutionClient") as MockEvo,
    ):
        MockEvo.return_value.send_presence = AsyncMock(return_value=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="MSG-DUP"),
                headers={"apikey": _TOKEN},
            )
            r2 = await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="MSG-DUP"),  # SAME id
                headers={"apikey": _TOKEN},
            )

        assert r1.status_code == 200
        assert r1.json() == {"status": "ok"}
        assert r2.status_code == 200
        assert r2.json().get("duplicate") is True

        await asyncio.sleep(0.05)

        from app.security.pii_masker import hash_phone
        buffer = webhook_mod._get_debounce_buffer()
        assert buffer.buffered_count(hash_phone(_PHONE_A)) == 1


@pytest.mark.asyncio
async def test_typing_indicator_fires_on_first_buffered_message(webhook_env):
    """First message of a new buffer triggers the Evolution 'composing'
    presence call."""
    from app.api import webhook as webhook_mod

    fake = fakeredis_aioredis.FakeRedis(decode_responses=True)

    with (
        patch("app.storage.redis_session._get_redis_client", return_value=fake),
        patch.object(webhook_mod, "EvolutionClient") as MockEvo,
    ):
        send_presence_mock = AsyncMock(return_value=True)
        MockEvo.return_value.send_presence = send_presence_mock
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/webhook/whatsapp",
                json=_payload("oi", message_id="MSG-PRES"),
                headers={"apikey": _TOKEN},
            )

        # The on_first_message callback fires as a background task.
        await asyncio.sleep(0.05)

    send_presence_mock.assert_awaited_once()
    call_kwargs = send_presence_mock.await_args.kwargs
    assert call_kwargs.get("presence") == "composing"
