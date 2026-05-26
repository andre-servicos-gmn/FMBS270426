"""Tests for app.adapters.evolution.EvolutionClient.send_text_blocks (Sprint 1.6)."""
from unittest.mock import AsyncMock, patch

import pytest

from app.adapters.evolution import EvolutionClient


# Mock the settings so EvolutionClient can be instantiated without real env.
@pytest.fixture
def evo_client(monkeypatch):
    monkeypatch.setenv("EVOLUTION_API_URL", "http://localhost:9999")
    monkeypatch.setenv("EVOLUTION_API_KEY", "test-key")
    monkeypatch.setenv("EVOLUTION_INSTANCE", "test-instance")
    from app.config import get_settings
    get_settings.cache_clear()
    yield EvolutionClient()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_send_text_blocks_sends_each_message(evo_client):
    """Every block in the list must be sent in order via send_text."""
    blocks = ["Primeiro bloco.", "Segundo bloco.", "Terceiro bloco com CTA?"]

    with (
        patch.object(evo_client, "send_text", new_callable=AsyncMock) as mock_send,
        patch("app.adapters.evolution.asyncio.sleep", new_callable=AsyncMock),
    ):
        await evo_client.send_text_blocks("5511999999999", blocks)

    assert mock_send.call_count == 3
    sent_texts = [call.args[1] for call in mock_send.call_args_list]
    assert sent_texts == blocks


@pytest.mark.asyncio
async def test_send_text_blocks_applies_delay_between_blocks(evo_client):
    """asyncio.sleep is called between blocks — once less than block count."""
    blocks = ["bloco 1", "bloco 2", "bloco 3", "bloco 4"]

    with (
        patch.object(evo_client, "send_text", new_callable=AsyncMock),
        patch("app.adapters.evolution.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await evo_client.send_text_blocks("5511999999999", blocks)

    # 4 blocks → 3 pauses (between 1-2, 2-3, 3-4). No pause before block 1.
    assert mock_sleep.call_count == 3
    # Each delay must be inside the documented 1.0–3.0s range.
    for call in mock_sleep.call_args_list:
        delay = call.args[0]
        assert 1.0 <= delay <= 3.0


@pytest.mark.asyncio
async def test_send_text_blocks_first_block_has_no_delay(evo_client):
    """The very first block must be sent immediately, no pre-sleep."""
    with (
        patch.object(evo_client, "send_text", new_callable=AsyncMock) as mock_send,
        patch("app.adapters.evolution.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await evo_client.send_text_blocks("5511999999999", ["unico bloco"])

    mock_send.assert_called_once()
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_send_text_blocks_empty_list_is_noop(evo_client):
    with (
        patch.object(evo_client, "send_text", new_callable=AsyncMock) as mock_send,
        patch("app.adapters.evolution.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await evo_client.send_text_blocks("5511999999999", [])

    mock_send.assert_not_called()
    mock_sleep.assert_not_called()
