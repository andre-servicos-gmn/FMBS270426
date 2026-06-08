"""Sprint 2.6.9 — help_request node tests (LLM-driven version).

Sprint 2.6.8 hardcoded 6 strings (3 offers + 3 refusals). Sprint 2.6.9
replaced that with an LLM call + invariant validator (the "cerca"). The
LLM is non-deterministic, so these tests mock its output and focus on:

  FLOW
    - node calls the LLM with the principles-based system prompt
    - the already_offered flag is reflected in the user block
    - the help_request_already_offered flag is set on first emission

  CERCA (invariant validation)
    - each red line (loja, model recommendation, vitrine list, budget,
      follow-up promise) is detected and triggers regeneration
    - regeneration that succeeds is used as the response
    - persistent violation falls back to the safe message
    - the safe fallback itself passes its own validation
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.help_request import (
    _SAFE_FALLBACK,
    _validate_help_response,
    help_request_node,
)
from app.agent.state import AgentState


def _state(
    message: str,
    *,
    phone_hash: str = "helptest" * 8,
    already_offered: bool = False,
    customer_name: str | None = None,
) -> AgentState:
    return AgentState(  # type: ignore[typeddict-item]
        messages=[HumanMessage(content=message)],
        phone_hash=phone_hash,
        intent="help_request",
        player_profile={},
        recommended_products=[],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
        help_request_already_offered=already_offered,
        customer_name=customer_name,
    )


# Example of a GOOD LLM response — passes every invariant.
_GOOD_LLM_REPLY = (
    "Pra te indicar a raquete que combina com seu jogo, o caminho é a "
    "*Consultoria* — a gente conhece seu perfil e você testa as opções "
    "em quadra antes de fechar (R$ 350, 100% abatido na compra). Mas se "
    "você já tem algum modelo em mente, me manda o nome que eu te passo "
    "tudo: preço, estoque e detalhes!"
)


# ════════════════════════════════════════════════════════════════════════════
# FLOW — prompt construction, LLM invocation, flag handling
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_help_request_calls_llm_with_principles_in_system():
    """The system prompt sent to the LLM must contain the principles-key
    fragments: Consultoria, teste em quadra, R$ 350, no-loja red line."""
    captured: dict = {}

    async def fake_chat(*, messages, system, **kwargs):
        captured["system"] = system
        captured["user"] = messages[0]["content"]
        return _GOOD_LLM_REPLY

    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        side_effect=fake_chat,
    ):
        await help_request_node(_state("me ajuda a escolher"))

    system = captured["system"]
    # Core principles are present.
    assert "Consultoria" in system
    assert "teste" in system.lower() and "quadra" in system.lower()
    assert "R$ 350" in system or "R$350" in system
    # Red lines are spelled out.
    assert "loja" in system.lower()  # at least mentioned in the rule itself
    assert "orcamento" in system.lower() or "orçamento" in system.lower()


@pytest.mark.asyncio
async def test_help_request_first_call_marks_state_as_not_already_offered():
    """First call: the user block tells the model this is the FIRST help-ask."""
    captured: dict = {}

    async def fake_chat(*, messages, system, **kwargs):
        captured["user"] = messages[0]["content"]
        return _GOOD_LLM_REPLY

    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        side_effect=fake_chat,
    ):
        await help_request_node(_state("qual vocês indicam?"))

    user_block = captured["user"]
    assert "PRIMEIRA" in user_block or "primeira" in user_block.lower()


@pytest.mark.asyncio
async def test_help_request_already_offered_marks_state_as_refusal():
    """Second call: the user block tells the model this is the REFUSAL pass."""
    captured: dict = {}

    async def fake_chat(*, messages, system, **kwargs):
        captured["user"] = messages[0]["content"]
        return _GOOD_LLM_REPLY

    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        side_effect=fake_chat,
    ):
        await help_request_node(
            _state("não quero, me indica algo", already_offered=True)
        )

    user_block = captured["user"]
    assert (
        "JÁ recebeu" in user_block
        or "ja recebeu" in user_block.lower()
        or "insistindo" in user_block.lower()
    )


@pytest.mark.asyncio
async def test_help_request_sets_offered_flag_on_first_call():
    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value=_GOOD_LLM_REPLY,
    ):
        result = await help_request_node(_state("preciso de ajuda"))
    assert result.get("help_request_already_offered") is True
    assert result.get("consultoria_interest") is True


@pytest.mark.asyncio
async def test_help_request_preserves_customer_name_in_user_block():
    captured: dict = {}

    async def fake_chat(*, messages, system, **kwargs):
        captured["user"] = messages[0]["content"]
        return _GOOD_LLM_REPLY

    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        side_effect=fake_chat,
    ):
        await help_request_node(_state("me ajuda", customer_name="Felipe"))

    assert "Felipe" in captured["user"]


@pytest.mark.asyncio
async def test_help_request_returns_llm_text_unchanged_when_valid():
    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value=_GOOD_LLM_REPLY,
    ):
        result = await help_request_node(_state("me ajuda"))
    assert result["response_blocks"] == [_GOOD_LLM_REPLY]


# ════════════════════════════════════════════════════════════════════════════
# CERCA — invariant validation
# ════════════════════════════════════════════════════════════════════════════

def test_invariant_blocks_loja_mention():
    """ANY 'loja' reference is a violation in this node."""
    text = "Passa na nossa loja que o time te ajuda a testar raquetes."
    is_valid, violations = _validate_help_response(text)
    assert not is_valid
    assert "mentions_loja" in violations


def test_invariant_blocks_recommend_specific_model():
    text = "Eu recomendo a Mormaii Sunset Plus, é ótima pra iniciante."
    is_valid, violations = _validate_help_response(text)
    assert not is_valid
    assert "recommends_specific_model" in violations


def test_invariant_blocks_sugiro_specific_model():
    text = "Sugiro a BeachPro Carbon X5 — é uma opção sólida."
    is_valid, violations = _validate_help_response(text)
    assert not is_valid
    assert "recommends_specific_model" in violations


def test_invariant_blocks_vitrine_list():
    """2+ bullets each starting with a capitalized name → vitrine list."""
    text = (
        "Algumas opções:\n"
        "• Mormaii Sunset Plus\n"
        "• BeachPro Carbon X5\n"
        "• Drop Shot Legacy"
    )
    is_valid, violations = _validate_help_response(text)
    assert not is_valid
    assert "looks_like_product_list" in violations


def test_invariant_blocks_budget_question():
    text = "Pra eu te indicar bem, qual seu orçamento?"
    is_valid, violations = _validate_help_response(text)
    assert not is_valid
    assert "asks_budget" in violations


def test_invariant_blocks_followup_promise():
    text = "Vou anotar aqui, alguém da equipe entra em contato com você logo."
    is_valid, violations = _validate_help_response(text)
    assert not is_valid
    assert "promises_followup" in violations


def test_invariant_blocks_empty_response():
    is_valid, violations = _validate_help_response("")
    assert not is_valid
    assert "empty_response" in violations

    is_valid, violations = _validate_help_response("   \n  ")
    assert not is_valid


def test_valid_response_passes_clean():
    """A response that follows every rule passes without violations."""
    is_valid, violations = _validate_help_response(_GOOD_LLM_REPLY)
    assert is_valid, f"Expected clean pass, got violations: {violations}"
    assert violations == []


def test_fallback_message_respects_all_invariants():
    """The hardcoded safe fallback MUST pass its own validation. If a
    future refactor breaks this, the node would loop on its safety net."""
    is_valid, violations = _validate_help_response(_SAFE_FALLBACK)
    assert is_valid, (
        f"_SAFE_FALLBACK violates its own invariants: {violations}"
    )


def test_fallback_mentions_required_signals():
    """Sanity: the fallback must carry Consultoria, value, and the
    'name a product' opening — the same payload the LLM is supposed to
    produce."""
    lower = _SAFE_FALLBACK.lower()
    assert "consultoria" in lower
    assert "r$ 350" in lower
    assert "quadra" in lower
    assert "modelo em mente" in lower or "me dizer o nome" in lower


# ════════════════════════════════════════════════════════════════════════════
# Regeneration flow
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_violation_triggers_regeneration():
    """Bad LLM reply (mentions loja) → node regenerates with a correction
    note → 2nd reply is clean → that one is returned."""
    bad = "Passa lá na loja que o pessoal te orienta."
    good = _GOOD_LLM_REPLY

    chat_mock = AsyncMock(side_effect=[bad, good])
    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        chat_mock,
    ):
        result = await help_request_node(_state("me ajuda"))

    assert chat_mock.call_count == 2
    assert result["response_blocks"] == [good]
    # 2nd call's system prompt should carry the correction note.
    regen_system = chat_mock.call_args_list[1].kwargs["system"]
    assert "CORREÇÃO OBRIGATÓRIA" in regen_system
    assert "loja" in regen_system.lower()


@pytest.mark.asyncio
async def test_regeneration_also_violates_falls_back_to_safe():
    """LLM violates twice → safe fallback is used."""
    bad1 = "Recomendo a Mormaii Sunset."
    bad2 = "Qual seu orçamento? Posso indicar algumas opções."

    chat_mock = AsyncMock(side_effect=[bad1, bad2])
    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        chat_mock,
    ):
        result = await help_request_node(_state("me ajuda"))

    assert chat_mock.call_count == 2
    assert result["response_blocks"] == [_SAFE_FALLBACK]


@pytest.mark.asyncio
async def test_clean_first_response_no_regeneration():
    """LLM produces a valid response on first try → no regeneration."""
    chat_mock = AsyncMock(return_value=_GOOD_LLM_REPLY)
    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        chat_mock,
    ):
        result = await help_request_node(_state("me ajuda"))

    assert chat_mock.call_count == 1
    assert result["response_blocks"] == [_GOOD_LLM_REPLY]


# ════════════════════════════════════════════════════════════════════════════
# TRIAGE ROUTING (regression — must still wire help_request)
# ════════════════════════════════════════════════════════════════════════════

def test_triage_intent_help_request_in_valid_set():
    from app.agent.nodes.triage import _VALID_INTENTS
    assert "help_request" in _VALID_INTENTS


def test_triage_prompt_lists_help_request():
    from app.agent.prompts import SYSTEM_TRIAGE
    assert "help_request" in SYSTEM_TRIAGE


def test_triage_prompt_no_longer_lists_diagnose():
    from app.agent.prompts import SYSTEM_TRIAGE
    assert "- diagnose " not in SYSTEM_TRIAGE


def test_router_maps_help_request_to_help_request_node():
    from app.agent.graph import _INTENT_TO_NODE
    assert _INTENT_TO_NODE["help_request"] == "help_request"


def test_router_maps_product_inquiry_to_recommend():
    from app.agent.graph import _INTENT_TO_NODE
    assert _INTENT_TO_NODE["product_inquiry"] == "recommend"


def test_router_maps_purchase_intent_to_product_selection():
    from app.agent.graph import _INTENT_TO_NODE
    assert _INTENT_TO_NODE["purchase_intent"] == "product_selection"


def test_router_close_falls_through_smalltalk():
    from app.agent.graph import _INTENT_TO_NODE
    assert _INTENT_TO_NODE["close"] == "smalltalk"


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION — specific product inquiry never lands in help_request
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_specific_product_inquiry_skips_help_request():
    from app.agent.graph import _triage_router

    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="vocês têm a Carbon X5?")],
        "phone_hash": "skipx" * 13,
        "intent": "product_inquiry",
        "player_profile": {}, "recommended_products": [],
        "needs_handoff": False, "handoff_reason": None,
        "consultoria_interest": False,
    }
    assert _triage_router(state) == "recommend"


@pytest.mark.asyncio
async def test_help_request_then_consultoria_acceptance_routes_to_scheduling():
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent.graph import build_graph

    graph = build_graph(checkpointer=MemorySaver())
    initial: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quero a consultoria, como agendo?")],
        "phone_hash": "schedaft" * 8,
        "intent": None, "player_profile": {},
        "recommended_products": [], "needs_handoff": False,
        "handoff_reason": None, "consultoria_interest": False,
        "customer_name": "Andre",
    }
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "scheduling_inquiry"}']
        with patch(
            "app.agent.nodes.scheduling_inquiry.handoff_dossier_pipeline",
            new_callable=AsyncMock,
        ):
            result = await graph.ainvoke(initial, {"configurable": {"thread_id": "tsched"}})

    assert result["intent"] == "scheduling_inquiry"
    assert result.get("needs_handoff") is True
    assert result["handoff_reason"] == "scheduling"
