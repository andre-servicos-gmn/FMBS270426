"""Sprint 2.7.1 — fixes for the 'conversational amnesia' bug.

Felipe's production failure: agent shows "Qual você procura? • Kronos
2026 • Kronos 2025"; customer replies "Primeira" / "2026" / "Raquete";
old triage classified these short replies as smalltalk (LLM saw only
the isolated word) → smalltalk produced the generic "E aí, Felipe!".

Two fixes shipped together:
  - Part 1: triage now sends the last ~6 messages to the LLM. The system
    prompt teaches it to use the context.
  - Part 2: recommend sets ``awaiting_candidate_choice=True`` whenever it
    shows a candidate list. Triage uses a deterministic positional /
    distinctive-token matcher BEFORE the LLM as a safety net.
  - Part 3: smalltalk Phase 3 also gets history + a contextual prompt.

Tests cover:
  - Schema guard (lesson 2.6.10 — without the field in the TypedDict the
    checkpointer drops it silently).
  - Recommend sets/clears the flag correctly across every return path.
  - Deterministic selector: positional, year/token, ambiguous-falls-
    through cases.
  - Triage receives history + clears flag after consuming.
  - Existing short-circuits still fire (no regression on match_confirmation
    / detail_choice).
  - Smalltalk now receives history.
"""
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.recommend import recommend_node
from app.agent.nodes.triage import (
    _detect_positional_index,
    _select_by_distinctive_token,
    recent_chat_history,
    triage_node,
    try_select_candidate,
)
from app.agent.state import AgentState


# ── Fixtures ────────────────────────────────────────────────────────────────

_PHONE = "amnesia" * 9


def _kronos_2026() -> dict:
    return {
        "id": 101,
        "name": "Raquete Beach Tennis Ama Sport Kronos 6th Generation 2026",
        "price_cents": 89900,
        "external_id": "kronos-2026",
        "is_raquete_praia": True,
    }


def _kronos_2025_hugo() -> dict:
    return {
        "id": 102,
        "name": "Raquete Beach Tennis Ama Sport Kronos 2025 Hugo Russo Capa",
        "price_cents": 84900,
        "external_id": "kronos-2025",
        "is_raquete_praia": True,
    }


def _drop_shot_short() -> dict:
    return {
        "id": 201,
        "name": "Short Drop Shot Padel Carbon",
        "price_cents": 19900,
        "external_id": "drop-short",
        "is_raquete_praia": False,
    }


def _drop_shot_top() -> dict:
    return {
        "id": 202,
        "name": "Top Drop Shot Feminino Branco",
        "price_cents": 14900,
        "external_id": "drop-top",
        "is_raquete_praia": False,
    }


def _drop_shot_raquete() -> dict:
    return {
        "id": 203,
        "name": "Raquete Drop Shot Legacy 12k",
        "price_cents": 119900,
        "external_id": "drop-raquete",
        "is_raquete_praia": True,
    }


def _base_state(
    *,
    message: str,
    candidates: list[dict] | None = None,
    awaiting_candidate_choice: bool | None = None,
    awaiting_match_confirmation: dict | None = None,
    awaiting_detail_choice: bool | None = None,
    history_msgs: list | None = None,
) -> AgentState:
    msgs = list(history_msgs or [])
    msgs.append(HumanMessage(content=message))
    state: dict = {
        "messages": msgs,
        "phone_hash": _PHONE,
        "intent": None,
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    if candidates is not None:
        state["last_product_candidates"] = candidates
    if awaiting_candidate_choice is not None:
        state["awaiting_candidate_choice"] = awaiting_candidate_choice
    if awaiting_match_confirmation is not None:
        state["awaiting_match_confirmation"] = awaiting_match_confirmation
    if awaiting_detail_choice is not None:
        state["awaiting_detail_choice"] = awaiting_detail_choice
    return state  # type: ignore[return-value]


# ════════════════════════════════════════════════════════════════════════════
# SCHEMA — lesson 2.6.10
# ════════════════════════════════════════════════════════════════════════════

def test_awaiting_candidate_choice_in_schema():
    """If this field is missing from AgentState's __annotations__, the
    LangGraph checkpointer drops it silently and the short-circuit
    never fires. Same failure mode as 2.6.10 — guard explicitly."""
    annotations = getattr(AgentState, "__annotations__", {})
    assert "awaiting_candidate_choice" in annotations, (
        "awaiting_candidate_choice MUST be declared in AgentState"
    )


# ════════════════════════════════════════════════════════════════════════════
# RECOMMEND — sets/clears the flag across return paths
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_recommend_ambiguous_sets_flag_and_candidates():
    """The 'Qual você procura?' route MUST set the flag and stash the list."""
    cands = [_kronos_2026(), _kronos_2025_hugo()]

    # Build a fake catalog so the matcher returns 'ambiguous' with both.
    state: AgentState = _base_state(message="kronos")
    with patch(
        "app.agent.nodes.recommend._list_catalog_candidates",
        new_callable=AsyncMock,
        return_value=cands,
    ):
        result = await recommend_node(state)

    assert result.get("awaiting_candidate_choice") is True
    assert result.get("last_product_candidates") == cands


@pytest.mark.asyncio
async def test_recommend_single_match_clears_flag():
    """A clean single match must NOT leave the flag set."""
    cands = [_kronos_2026()]
    state: AgentState = _base_state(
        message="Kronos 6th Generation 2026",
        awaiting_candidate_choice=True,
    )
    with patch(
        "app.agent.nodes.recommend._list_catalog_candidates",
        new_callable=AsyncMock,
        return_value=cands,
    ):
        result = await recommend_node(state)

    assert result.get("awaiting_candidate_choice") is False


@pytest.mark.asyncio
async def test_recommend_confirmation_resolved_clears_flag():
    """When the selected candidate is promoted via awaiting_match_confirmation,
    the candidate-choice flag must be explicitly cleared."""
    state: AgentState = _base_state(
        message="",
        awaiting_match_confirmation=_kronos_2026(),
        awaiting_candidate_choice=True,
    )
    result = await recommend_node(state)
    assert result.get("awaiting_candidate_choice") is False
    assert result.get("recommended_products") == [_kronos_2026()]


# ════════════════════════════════════════════════════════════════════════════
# SELECTOR — positional / token (unit tests)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text,expected_idx", [
    ("primeira", 0),
    ("Primeira", 0),
    ("a primeira", 0),
    ("1", 0),
    ("1ª", 0),
    ("a 1", 0),
    ("segunda", 1),
    ("Segunda por favor", 1),
    ("a 2", 1),
    ("2", 1),
    ("2º", 1),
    ("terceira", 2),
    ("a 3", 2),
    ("última", -1),
    ("a ultima", -1),
])
def test_detect_positional(text, expected_idx):
    from app.agent.nodes.triage import _strip_accents
    norm = _strip_accents(text.lower())
    assert _detect_positional_index(norm) == expected_idx


@pytest.mark.parametrize("text", [
    "qualquer uma", "não sei", "tô em dúvida", "tem outra?", "kronos",
])
def test_detect_positional_returns_none_for_non_positional(text):
    from app.agent.nodes.triage import _strip_accents
    norm = _strip_accents(text.lower())
    assert _detect_positional_index(norm) is None


def test_select_by_distinctive_token_year():
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    chosen = _select_by_distinctive_token("2026", cands)
    assert chosen is not None
    assert "2026" in chosen["name"]


def test_select_by_distinctive_token_partial_name():
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    chosen = _select_by_distinctive_token("hugo russo", cands)
    assert chosen is not None
    assert "Hugo Russo" in chosen["name"]


def test_select_by_distinctive_token_ambiguous_returns_none():
    """A token that appears in MULTIPLE candidates → None (don't guess)."""
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    chosen = _select_by_distinctive_token("kronos", cands)
    assert chosen is None  # both have "kronos"


def test_try_select_candidate_positional():
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    assert try_select_candidate("primeira", cands) == _kronos_2026()
    assert try_select_candidate("Segunda", cands) == _kronos_2025_hugo()


def test_try_select_candidate_year_token():
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    chosen = try_select_candidate("2026", cands)
    assert chosen == _kronos_2026()


def test_try_select_candidate_category_refinement_to_one():
    """When 'raquete' uniquely matches one candidate (because the others
    are short/top), the selector picks it."""
    cands = [_drop_shot_short(), _drop_shot_top(), _drop_shot_raquete()]
    chosen = try_select_candidate("raquete", cands)
    # The Raquete Drop Shot Legacy is the only one with "raquete" in the
    # name; the matcher picks it.
    assert chosen == _drop_shot_raquete()


def test_try_select_candidate_ambiguous_returns_none():
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    assert try_select_candidate("kronos", cands) is None
    assert try_select_candidate("não sei", cands) is None
    assert try_select_candidate("qualquer", cands) is None


def test_try_select_candidate_out_of_range_returns_none():
    """User says 'quarta' but only 2 candidates → None, don't guess."""
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    assert try_select_candidate("quarta", cands) is None
    assert try_select_candidate("5", cands) is None


# ════════════════════════════════════════════════════════════════════════════
# TRIAGE — short-circuit fires + clears flag
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_triage_primeira_routes_to_recommend_with_selected():
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    state = _base_state(
        message="Primeira",
        candidates=cands,
        awaiting_candidate_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "product_inquiry"
    assert result.get("awaiting_match_confirmation") == _kronos_2026()
    assert result.get("awaiting_candidate_choice") is False
    assert result.get("last_product_candidates") is None


@pytest.mark.asyncio
async def test_triage_year_token_routes_to_recommend():
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    state = _base_state(
        message="2026",
        candidates=cands,
        awaiting_candidate_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "product_inquiry"
    assert result.get("awaiting_match_confirmation") == _kronos_2026()


@pytest.mark.asyncio
async def test_triage_ambiguous_choice_falls_through_to_llm_with_history():
    """When the selector returns None, triage falls through to the LLM
    classification (which now sees history)."""
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    state = _base_state(
        message="não sei qual",
        candidates=cands,
        awaiting_candidate_choice=True,
    )
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "help_request"}',
    ) as llm:
        result = await triage_node(state)

    # Selector returned None → LLM ran with history → flag cleared.
    assert result["intent"] == "help_request"
    assert result.get("awaiting_candidate_choice") is False
    # LLM saw a message list, not a single string.
    call_args = llm.call_args
    sent_messages = call_args.kwargs.get("messages") or call_args.args[0]
    assert isinstance(sent_messages, list)
    assert len(sent_messages) >= 1


@pytest.mark.asyncio
async def test_triage_no_flag_no_short_circuit():
    """Without ``awaiting_candidate_choice``, the candidate selector NEVER
    fires even if ``last_product_candidates`` is populated."""
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    state = _base_state(
        message="primeira",
        candidates=cands,
        # Flag NOT set.
    )
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "smalltalk"}',
    ):
        result = await triage_node(state)
    # No short-circuit → goes through LLM → smalltalk per the mock.
    assert result["intent"] == "smalltalk"
    assert result.get("awaiting_match_confirmation") is None


# ════════════════════════════════════════════════════════════════════════════
# TRIAGE — history is sent to the LLM
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_triage_sends_recent_history_to_llm():
    """The LLM call must receive the last ~N messages, not just the
    current one. This is what fixes the amnesia bug at the root."""
    history = [
        HumanMessage(content="oi"),
        AIMessage(content="Olá! Qual seu nome?"),
        HumanMessage(content="Felipe"),
        AIMessage(content="E aí Felipe! Como posso ajudar?"),
        HumanMessage(content="tem kronos?"),
        AIMessage(
            content="Temos algumas opções parecidas:\n• Kronos 2026\n• Kronos 2025\n\nQual você procura?"
        ),
    ]
    state = _base_state(message="Primeira", history_msgs=history)

    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "smalltalk"}',
    ) as llm:
        await triage_node(state)

    call_args = llm.call_args
    sent_messages = call_args.kwargs.get("messages") or call_args.args[0]
    # We should see multiple messages, with the LAST one being "Primeira".
    assert len(sent_messages) > 1
    assert sent_messages[-1]["role"] == "user"
    assert sent_messages[-1]["content"] == "Primeira"
    # And the previous assistant turn should be the agent's question.
    assistant_msgs = [m for m in sent_messages if m["role"] == "assistant"]
    assert any("Qual você procura" in m["content"] for m in assistant_msgs)


@pytest.mark.asyncio
async def test_triage_history_no_prior_falls_back_to_single_message():
    """First-turn (only current customer message) still works — falls back
    to the legacy single-message format."""
    state = _base_state(message="oi")

    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "smalltalk"}',
    ) as llm:
        await triage_node(state)

    call_args = llm.call_args
    sent_messages = call_args.kwargs.get("messages") or call_args.args[0]
    assert sent_messages == [{"role": "user", "content": "oi"}]


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION — existing short-circuits intact, clear messages still classify
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_existing_match_confirmation_still_works():
    """Sprint 2.6.2 short-circuit: 'sim' after 'Você quis dizer X?' → product_inquiry."""
    pending = _kronos_2026()
    state = _base_state(
        message="sim",
        awaiting_match_confirmation=pending,
    )
    result = await triage_node(state)
    assert result["intent"] == "product_inquiry"


@pytest.mark.asyncio
async def test_existing_detail_choice_still_works():
    """Sprint 2.6.10 short-circuit: 'detalhes' after the offer → attribute_inquiry."""
    state = _base_state(
        message="detalhes por favor",
        awaiting_detail_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "attribute_inquiry"


@pytest.mark.asyncio
async def test_existing_multi_product_reference_still_works():
    """Sprint 2.6.4 short-circuit: 'as duas' → price_inquiry."""
    cands = [_kronos_2026(), _kronos_2025_hugo()]
    state = _base_state(
        message="as duas",
        candidates=cands,
        awaiting_candidate_choice=True,
    )
    result = await triage_node(state)
    # multi_product_reference checks BEFORE candidate-choice short-circuit.
    assert result["intent"] == "price_inquiry"


@pytest.mark.asyncio
async def test_clear_oi_still_smalltalk():
    """Regression: 'oi' without context still routes to smalltalk."""
    state = _base_state(message="oi")
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "smalltalk"}',
    ):
        result = await triage_node(state)
    assert result["intent"] == "smalltalk"


# ════════════════════════════════════════════════════════════════════════════
# HISTORY HELPER unit test
# ════════════════════════════════════════════════════════════════════════════

def test_recent_chat_history_converts_and_truncates():
    msgs = [
        HumanMessage(content=f"user-{i}") for i in range(10)
    ]
    # Interleave with AI messages
    interleaved = []
    for i in range(10):
        interleaved.append(HumanMessage(content=f"user-{i}"))
        interleaved.append(AIMessage(content=f"ai-{i}"))

    history = recent_chat_history(interleaved, window=6)
    assert len(history) == 6
    # Last message should be the most recent
    assert history[-1]["content"] == "ai-9"
    assert history[-1]["role"] == "assistant"
    # Roles alternate
    for i, m in enumerate(history):
        assert m["role"] in {"user", "assistant"}


def test_recent_chat_history_empty():
    assert recent_chat_history([]) == []
    assert recent_chat_history(None) == []


# ════════════════════════════════════════════════════════════════════════════
# SMALLTALK — receives history (Part 3)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_smalltalk_receives_history():
    """_smalltalk_node Phase 3 must pass recent history to the LLM, not
    just the last message."""
    from app.agent.graph import _smalltalk_node

    history = [
        HumanMessage(content="oi"),
        AIMessage(content="Olá Felipe! Como posso ajudar?"),
        HumanMessage(content="tem mormaii sunset?"),
        AIMessage(content="Sim, temos a *Mormaii Sunset Plus*!"),
        HumanMessage(content="ok valeu"),  # current message — smalltalk
    ]
    state: dict = {
        "messages": history,
        "phone_hash": _PHONE,
        "intent": "smalltalk",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "customer_name": "Felipe",
        "name_asked": False,
    }

    with patch(
        "app.agent.graph.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value="De nada! Se quiser saber mais sobre a Mormaii Sunset, é só me chamar.",
    ) as llm:
        await _smalltalk_node(state)  # type: ignore[arg-type]

    call_args = llm.call_args
    sent_messages = call_args.kwargs.get("messages") or call_args.args[0]
    assert isinstance(sent_messages, list)
    assert len(sent_messages) > 1, (
        "Smalltalk should see history, not just the last message"
    )
    # The last user turn (with name hint injected) carries "ok valeu".
    assert "ok valeu" in sent_messages[-1]["content"]
