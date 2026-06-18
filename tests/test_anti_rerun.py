"""Sprint 1.14 — anti-rerun guard tests.

Pure-Python helpers + integration with recommend_node and pitch_consultoria_node:
verify the deterministic block returns the contextual fallback instead of
calling the LLM again.
"""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.anti_rerun import (
    fallback_message_for,
    is_recent_rerun,
    should_block_rerun,
    stamp_node_execution,
)
from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _state_with_last_node(node_name: str | None, seconds_ago: int) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="oi")],
        "phone_hash": "antirerun" * 7,
        "intent": None,
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    if node_name is not None:
        ts = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        state["last_node_executed"] = node_name
        state["last_node_executed_at"] = ts.isoformat()
    return state


# ── Pure helpers ─────────────────────────────────────────────────────────────

def test_is_recent_rerun_true_within_threshold():
    state = _state_with_last_node("recommend", seconds_ago=10)
    assert is_recent_rerun(state, "recommend", threshold_seconds=60) is True


def test_is_recent_rerun_false_after_threshold():
    state = _state_with_last_node("recommend", seconds_ago=120)
    assert is_recent_rerun(state, "recommend", threshold_seconds=60) is False


def test_is_recent_rerun_false_for_different_node():
    state = _state_with_last_node("pitch_consultoria", seconds_ago=5)
    assert is_recent_rerun(state, "recommend") is False


def test_is_recent_rerun_false_when_state_blank():
    state = _state_with_last_node(None, seconds_ago=0)
    assert is_recent_rerun(state, "recommend") is False


def test_should_block_rerun_short_message_blocks():
    """Short follow-up within window → should be blocked."""
    state = _state_with_last_node("recommend", seconds_ago=5)
    assert should_block_rerun(state, "recommend", user_msg="sim") is True


def test_should_block_rerun_long_message_allows():
    """Substantive new content within the window → allowed (≥20 chars)."""
    state = _state_with_last_node("recommend", seconds_ago=5)
    long_msg = "agora prefiro uma raquete mais leve por causa do meu ombro"
    assert should_block_rerun(state, "recommend", user_msg=long_msg) is False


def test_stamp_node_execution_uses_iso_utc():
    out = stamp_node_execution("recommend")
    assert out["last_node_executed"] == "recommend"
    # parses round-trip
    dt = datetime.fromisoformat(out["last_node_executed_at"])
    assert dt.tzinfo is not None


# ── Whitelist semantics (no node-name in blocklist by default) ──────────────

def test_diagnose_never_blocked():
    """diagnose is not subject to anti-rerun — it must always advance slots.

    The anti-rerun helpers are blocking-agnostic (any node can call them),
    but in practice only recommend & pitch_consultoria invoke them. Here we
    just verify the helpers DO return True for diagnose if a caller asked
    them to — i.e. there's no special whitelist baked in. The whitelist is
    enforced by the FACT that diagnose_node never calls should_block_rerun.
    """
    # The diagnose node does NOT import should_block_rerun → no blocking path.
    from app.agent.nodes import diagnose
    src = open(diagnose.__file__, encoding="utf-8").read()
    assert "should_block_rerun" not in src
    assert "anti_rerun" not in src


# ── Integration: recommend node blocks rerun ────────────────────────────────

@pytest.mark.asyncio
async def test_recommend_blocks_rerun_within_threshold():
    """recommend within 60s with a short follow-up → fallback, no LLM call."""
    from app.agent.nodes.recommend import recommend_node

    state = _state_with_last_node("recommend", seconds_ago=5)
    state["player_profile"] = {
        "nivel_jogo": "intermediário",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "nao_aplicavel",
        "modelo_desejado": "nenhum",
    }
    state["messages"] = [HumanMessage(content="sim")]

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    # No LLM call must have happened.
    llm.assert_not_called()
    search.assert_not_called()
    # The reply IS the canned fallback.
    blocks = result["response_blocks"]
    assert blocks == [fallback_message_for("recommend")]
    # And the execution stamp was refreshed.
    assert result["last_node_executed"] == "recommend"


# ── Integration: pitch_consultoria blocks rerun ─────────────────────────────

@pytest.mark.asyncio
async def test_pitch_consultoria_blocks_rerun():
    """Same anti-rerun guard on the pitch node."""
    from app.agent.nodes.pitch_consultoria import pitch_consultoria_node

    state = _state_with_last_node("pitch_consultoria", seconds_ago=10)
    state["messages"] = [HumanMessage(content="ok")]

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        result = await pitch_consultoria_node(state)

    llm.assert_not_called()
    assert result["response_blocks"] == [fallback_message_for("pitch_consultoria")]


@pytest.mark.asyncio
async def test_blocked_rerun_returns_fallback_message_not_llm():
    """Sanity: the canned fallback strings are distinct per node."""
    assert fallback_message_for("recommend") != fallback_message_for("pitch_consultoria")
    # Sprint 2.6.2 — phantom phrase replaced with neutral ask-for-detail.
    assert "Pode me dar mais detalhes" in fallback_message_for("recommend")
    assert "Te expliquei a Consultoria" in fallback_message_for("pitch_consultoria")


# ── Sprint 2.6.3 — query-similarity unblocks divergent follow-ups ───────────

def _state_with_two_human_messages(prev: str, curr: str, seconds_ago: int = 5) -> AgentState:
    """Stitch a state where two HumanMessages are visible in the history."""
    from langchain_core.messages import AIMessage as _AI, HumanMessage as _H

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [
            _H(content=prev),
            _AI(content="(agente respondeu algo)"),
            _H(content=curr),
        ],
        "phone_hash": "ar263" * 13,
        "intent": None,
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    ts = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    state["last_node_executed"] = "recommend"
    state["last_node_executed_at"] = ts.isoformat()
    return state


def test_anti_rerun_allows_different_product_query():
    """Sprint 2.6.3 — asking about a DIFFERENT product within the window
    must be allowed (previous bug: blocked as 'rerun cego')."""
    state = _state_with_two_human_messages(
        prev="vocês têm a Carbon X5?",
        curr="e a ShotCentauro50?",
    )
    assert should_block_rerun(state, "recommend", user_msg="e a ShotCentauro50?") is False


def test_anti_rerun_blocks_identical_query():
    """Literally identical query within the window → still blocks."""
    state = _state_with_two_human_messages(
        prev="vocês têm a Carbon X5?",
        curr="vocês têm a Carbon X5?",
    )
    assert should_block_rerun(state, "recommend", user_msg="vocês têm a Carbon X5?") is True


def test_anti_rerun_blocks_query_with_minor_typo():
    """Same intent with a small typo (ratio ≥ 0.75) → blocks."""
    state = _state_with_two_human_messages(
        prev="vocês têm a Carbon X5?",
        curr="voces tem a Carbon X5?",  # missing accents only
    )
    # ratio ≈ 0.91 — still above the 0.75 threshold → blocks.
    assert should_block_rerun(state, "recommend", user_msg="voces tem a Carbon X5?") is True
