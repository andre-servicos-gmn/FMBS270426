"""Sprint 2.6.5 — Redis resilience: keepalive + retry + honest fallback.

Three layers under test:
1. Every Redis client is created with the resilient kwargs (keepalive,
   health_check_interval, retry config).
2. ``_ainvoke_with_retry`` retries once on Redis-connection errors,
   reconnecting Redis singletons between attempts.
3. ``_send_fallback_and_alert`` emits the client hold-on message + the
   Andre alert, capped at 1 per minute per phone_hash, with no forbidden
   vocabulary.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Camada 1 — resilient client kwargs ──────────────────────────────────────

def test_resilient_kwargs_has_keepalive_and_health_check():
    from app.storage.redis_resilient import resilient_connection_kwargs

    kwargs = resilient_connection_kwargs()
    assert kwargs.get("socket_keepalive") is True
    # health_check_interval is the SINGLE most important fix for the
    # idle-kill symptom observed in production.
    assert kwargs.get("health_check_interval") == 30


def test_resilient_kwargs_has_retry_config():
    from redis.asyncio.retry import Retry
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import TimeoutError as RedisTimeoutError

    from app.storage.redis_resilient import resilient_connection_kwargs

    kwargs = resilient_connection_kwargs()
    assert kwargs.get("retry_on_timeout") is True
    assert RedisConnectionError in (kwargs.get("retry_on_error") or [])
    assert RedisTimeoutError in (kwargs.get("retry_on_error") or [])
    assert isinstance(kwargs.get("retry"), Retry)


def test_make_resilient_redis_passes_kwargs(monkeypatch):
    """The factory must forward the resilient kwargs to Redis.from_url."""
    captured: dict[str, object] = {}

    def fake_from_url(url, **kwargs):
        captured.update(kwargs)
        captured["__url"] = url
        return MagicMock()

    monkeypatch.setattr("app.storage.redis_resilient.Redis.from_url", fake_from_url)

    from app.storage.redis_resilient import make_resilient_redis
    make_resilient_redis("redis://localhost:6379/0")

    assert captured["__url"] == "redis://localhost:6379/0"
    assert captured.get("socket_keepalive") is True
    assert captured.get("health_check_interval") == 30
    assert captured.get("retry_on_timeout") is True


def test_checkpointer_initialized_with_resilient_connection_args(monkeypatch):
    """The AsyncRedisSaver must receive connection_args with the resilient kwargs."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    from app.config import get_settings
    get_settings.cache_clear()

    import asyncio

    from app.agent import checkpointer as cp

    # Force a clean state for this test.
    cp._saver = None

    captured: dict[str, object] = {}

    class _FakeSaver:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self._owns_its_client = False

        async def asetup(self):
            return None

        async def aset_client_info(self):
            return None

    monkeypatch.setattr(cp, "AsyncRedisSaver", _FakeSaver)
    asyncio.get_event_loop().run_until_complete(cp.init_checkpointer())

    conn_args = captured.get("connection_args") or {}
    assert conn_args.get("socket_keepalive") is True
    assert conn_args.get("health_check_interval") == 30
    # Reset so other tests get a clean singleton.
    cp._saver = None


# ── Camada 2 — retry helper ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_invoke_retries_on_connection_error(monkeypatch):
    """1st call raises ConnectionError, 2nd succeeds → result returned."""
    from app.api import webhook

    fake_graph_first = MagicMock()
    fake_graph_first.ainvoke = AsyncMock(side_effect=ConnectionError("conn closed"))
    fake_graph_second = MagicMock()
    fake_graph_second.ainvoke = AsyncMock(return_value={"messages": ["ok"]})

    # _get_graph returns the first graph (broken), then after reconnect we
    # rebuild and return the second graph.
    graph_iter = iter([fake_graph_first, fake_graph_second])

    def _fake_get_graph():
        return next(graph_iter)

    monkeypatch.setattr(webhook, "_get_graph", _fake_get_graph)
    with patch(
        "app.api.webhook._reconnect_redis_singletons", new_callable=AsyncMock
    ) as reconnect:
        result = await webhook._ainvoke_with_retry({"x": 1}, {"configurable": {}})

    assert result == {"messages": ["ok"]}
    reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_invoke_does_not_retry_on_non_redis_error(monkeypatch):
    """Non-Redis errors propagate immediately, no reconnect, no retry."""
    from app.api import webhook

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=ValueError("business logic"))

    monkeypatch.setattr(webhook, "_get_graph", lambda: fake_graph)
    with patch(
        "app.api.webhook._reconnect_redis_singletons", new_callable=AsyncMock
    ) as reconnect:
        with pytest.raises(ValueError):
            await webhook._ainvoke_with_retry({"x": 1}, {"configurable": {}})

    reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_invoke_falls_back_after_exhausted(monkeypatch):
    """Both attempts fail with Redis-like errors → raises so caller fallback fires."""
    from app.api import webhook

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=ConnectionError("dead"))

    monkeypatch.setattr(webhook, "_get_graph", lambda: fake_graph)
    with patch(
        "app.api.webhook._reconnect_redis_singletons", new_callable=AsyncMock
    ) as reconnect:
        with pytest.raises(ConnectionError):
            await webhook._ainvoke_with_retry({"x": 1}, {"configurable": {}})

    # Exactly one reconnect (between attempts 1 and 2).
    assert reconnect.await_count == 1
    # Two invocation attempts.
    assert fake_graph.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_graph_invoke_recognizes_nonetype_callable_symptom(monkeypatch):
    """The exact production symptom — 'NoneType' object is not callable —
    must be classified as a Redis connection error."""
    from app.api import webhook

    err = TypeError("'NoneType' object is not callable")
    fake_graph_first = MagicMock()
    fake_graph_first.ainvoke = AsyncMock(side_effect=err)
    fake_graph_second = MagicMock()
    fake_graph_second.ainvoke = AsyncMock(return_value={"messages": ["ok"]})

    graph_iter = iter([fake_graph_first, fake_graph_second])
    monkeypatch.setattr(webhook, "_get_graph", lambda: next(graph_iter))

    with patch(
        "app.api.webhook._reconnect_redis_singletons", new_callable=AsyncMock
    ):
        result = await webhook._ainvoke_with_retry({}, {"configurable": {}})

    assert result == {"messages": ["ok"]}


def test_is_redis_connection_error_detects_known_signals():
    from app.api.webhook import _is_redis_connection_error
    from redis.exceptions import ConnectionError as RedisConnectionError

    assert _is_redis_connection_error(RedisConnectionError("..."))
    assert _is_redis_connection_error(ConnectionError("..."))
    assert _is_redis_connection_error(TimeoutError("..."))
    assert _is_redis_connection_error(OSError("connection reset"))
    assert _is_redis_connection_error(TypeError("'NoneType' object is not callable"))
    assert _is_redis_connection_error(RuntimeError("redis client closed"))
    # Negative
    assert not _is_redis_connection_error(ValueError("bad input"))
    assert not _is_redis_connection_error(KeyError("missing"))


# ── Camada 3 — fallback wording + anti-repeat ───────────────────────────────

def test_fallback_message_has_no_forbidden_words():
    from app.api.webhook import _FALLBACK_CLIENT_MESSAGE

    msg = _FALLBACK_CLIENT_MESSAGE.lower()
    for forbidden in ("erro", "falha", "problema técnico", "sistema", "bug", "indisponível"):
        assert forbidden not in msg, f"forbidden word leaked: {forbidden!r}"
    # No alert / warning emojis.
    for emoji in ("⚠️", "🔧", "❌", "🚫"):
        assert emoji not in _FALLBACK_CLIENT_MESSAGE, (
            f"forbidden emoji leaked: {emoji!r}"
        )


@pytest.mark.asyncio
async def test_fallback_sends_client_message_and_andre_alert(monkeypatch):
    monkeypatch.setenv("DOSSIER_RECIPIENT_PHONE", "5511999999999")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.api import webhook

    # Clear anti-repeat marker for this test.
    webhook._fallback_last_sent.clear()

    sent: list[tuple[str, str]] = []

    async def _fake_send(self, phone, text):
        sent.append((phone, text))

    monkeypatch.setattr(
        "app.adapters.evolution.EvolutionClient.send_text", _fake_send
    )
    await webhook._send_fallback_and_alert(
        "5511888888888",
        "phone_hash_abc12345",
        RuntimeError("redis down"),
    )

    assert len(sent) == 2
    # First send → client (uses raw phone), second → Andre.
    client_phone, client_msg = sent[0]
    assert client_phone == "5511888888888"
    assert client_msg == webhook._FALLBACK_CLIENT_MESSAGE

    alert_phone, alert_msg = sent[1]
    assert alert_phone == "5511999999999"
    assert "ALERTA TÉCNICO" in alert_msg
    # Hash is truncated to 12 chars in the alert (phone_hash[:12]).
    assert "phone_hash_a" in alert_msg
    assert "Redis" in alert_msg


@pytest.mark.asyncio
async def test_fallback_not_repeated_within_window(monkeypatch):
    from app.api import webhook

    webhook._fallback_last_sent.clear()

    sends: list[tuple[str, str]] = []

    async def _fake_send(self, phone, text):
        sends.append((phone, text))

    monkeypatch.setattr(
        "app.adapters.evolution.EvolutionClient.send_text", _fake_send
    )
    monkeypatch.setenv("DOSSIER_RECIPIENT_PHONE", "")
    from app.config import get_settings
    get_settings.cache_clear()

    # First call → emits.
    await webhook._send_fallback_and_alert("5511888", "ph1", RuntimeError("x"))
    # Second call within window → suppressed.
    await webhook._send_fallback_and_alert("5511888", "ph1", RuntimeError("x"))

    # Only one client message (recipient empty → no alert either way).
    assert len(sends) == 1


@pytest.mark.asyncio
async def test_fallback_skips_andre_alert_when_recipient_empty(monkeypatch):
    monkeypatch.setenv("DOSSIER_RECIPIENT_PHONE", "")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.api import webhook
    webhook._fallback_last_sent.clear()

    sends: list[tuple[str, str]] = []

    async def _fake_send(self, phone, text):
        sends.append((phone, text))

    monkeypatch.setattr(
        "app.adapters.evolution.EvolutionClient.send_text", _fake_send
    )
    await webhook._send_fallback_and_alert("5511888", "ph2", RuntimeError("x"))
    # Only the client message, no Andre alert.
    assert len(sends) == 1


@pytest.mark.asyncio
async def test_fallback_swallows_evolution_failure(monkeypatch):
    """If Evolution itself is down too, the fallback must not raise."""
    monkeypatch.setenv("DOSSIER_RECIPIENT_PHONE", "5511999999999")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.api import webhook
    webhook._fallback_last_sent.clear()

    async def _fail(self, *a, **kw):
        raise RuntimeError("evolution down")

    monkeypatch.setattr(
        "app.adapters.evolution.EvolutionClient.send_text", _fail
    )
    # Should not raise.
    await webhook._send_fallback_and_alert("5511888", "ph3", RuntimeError("y"))


# ── Smoke — reconnect_redis_singletons calls all three reset hooks ──────────

@pytest.mark.asyncio
async def test_reconnect_calls_all_singleton_resets(monkeypatch):
    from app.api import webhook

    called: dict[str, int] = {}

    async def _reset_redis():
        called["redis"] = called.get("redis", 0) + 1

    async def _reset_checkpointer():
        called["checkpointer"] = called.get("checkpointer", 0) + 1

    async def _init_checkpointer():
        called["init"] = called.get("init", 0) + 1

    monkeypatch.setattr(
        "app.storage.redis_session.reset_redis_client", _reset_redis
    )
    monkeypatch.setattr(
        "app.agent.checkpointer.reset_checkpointer", _reset_checkpointer
    )
    monkeypatch.setattr(
        "app.agent.checkpointer.init_checkpointer", _init_checkpointer
    )

    await webhook._reconnect_redis_singletons()
    assert called.get("redis") == 1
    assert called.get("checkpointer") == 1
    assert called.get("init") == 1
    assert webhook._graph is None  # graph cleared so next request rebuilds
