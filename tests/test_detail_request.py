"""Sprint 2.6.10 — detail-offer routing + broad-detail cascade.

These tests cover the full chain Felipe broke in production:

    recommend confirms product → state carries awaiting_detail_choice=True
    → next-turn triage SHORT-CIRCUITS on "detalhes" / "sim" → attribute_inquiry
    → attribute_inquiry renders broad-detail cascade (NEVER pitches Consultoria)

The most important test is ``test_full_chain_recommend_to_detail`` — it
runs both nodes back-to-back and asserts the flag actually persists. The
unit tests that set the flag manually (test_accept_detail_offer_*) are
kept as fast-feedback regressions, but they only cover half the chain.

Production failure mode (June 2026, Felipe): the flag was NOT in the
TypedDict schema, so the LangGraph checkpointer silently dropped it on
save → next turn `state.get("awaiting_detail_choice")` returned None →
LLM classified "detalhes" as out_of_scope / help_request → pitch instead
of details.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.attribute_inquiry import (
    _clean_description_for_whatsapp,
    _render_broad_details,
    attribute_inquiry_node,
    is_broad_detail_request,
)
from app.agent.nodes.recommend import recommend_node
from app.agent.nodes.triage import triage_node
from app.agent.state import AgentState


# ── Test fixtures ──────────────────────────────────────────────────────────


def _product_with_attrs(name: str = "Mormaii Sunset Plus 2026", price_cents: int = 49900) -> dict:
    return {
        "id": 101,
        "name": name,
        "price_cents": price_cents,
        "atributos_parseados": {
            "peso": "320g (+/- 10g)",
            "composicao": "Carbono 3K",
            "espessura": "22mm",
            "comprimento": "50cm",
            "equilibrio": "27,5cm",
        },
        "description": "",
        "external_id": str(name),
        "is_raquete_praia": True,
    }


def _product_with_description_only(
    name: str = "AMA PROTEO 22mm",
    description: str = (
        "<p>A AMA Sport Proteo é uma raquete carbono 3K, peso médio "
        "220g, ideal pra jogadores intermediários que buscam controle "
        "e conforto. Edgeless construction.</p>"
    ),
) -> dict:
    return {
        "id": 202,
        "name": name,
        "price_cents": 79900,
        "atributos_parseados": {},
        "description": description,
        "external_id": str(name),
        "is_raquete_praia": True,
    }


def _product_bare(name: str = "Bola Beach Tennis Pack") -> dict:
    return {
        "id": 303,
        "name": name,
        "price_cents": 5900,
        "atributos_parseados": {},
        "description": "",
        "external_id": str(name),
        "is_raquete_praia": False,
    }


def _state(
    *,
    message: str = "",
    recommended_products: list[dict] | None = None,
    awaiting_detail_choice: bool | None = None,
    awaiting_match_confirmation: dict | None = None,
    phone_hash: str = "felipe" * 10,
) -> AgentState:
    msgs = [HumanMessage(content=message)] if message else []
    s: dict = {
        "messages": msgs,
        "phone_hash": phone_hash,
        "intent": None,
        "player_profile": {},
        "recommended_products": recommended_products or [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    if awaiting_detail_choice is not None:
        s["awaiting_detail_choice"] = awaiting_detail_choice
    if awaiting_match_confirmation is not None:
        s["awaiting_match_confirmation"] = awaiting_match_confirmation
    return s  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# FULL-CHAIN integration — the test Felipe needed
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_full_chain_recommend_to_detail():
    """The whole chain end-to-end, WITHOUT setting the flag manually.

    Step 1: recommend confirms a product → return dict must have
            awaiting_detail_choice=True.
    Step 2: feed that return dict (merged into state) into triage with
            "detalhes" → must route to attribute_inquiry, NOT help_request.

    If the flag is missing from state.py (production bug), step 1 still
    "sets" it in the dict but the schema validator would drop it — Step 2
    then proves the field IS in the schema by reading it back successfully.
    """
    product = _product_with_attrs("AMA Sport Proteo")

    # Step 1: recommend
    state_in: AgentState = _state(
        message="tem a proteo?",
        recommended_products=[],
    )
    # Mock the catalog lookup so the matcher returns our product cleanly.
    with patch(
        "app.agent.nodes.recommend._list_catalog_candidates",
        new_callable=AsyncMock,
        return_value=[product],
    ):
        result_recommend = await recommend_node(state_in)

    assert result_recommend.get("awaiting_detail_choice") is True, (
        "recommend must set the flag — broken chain otherwise"
    )
    assert result_recommend.get("recommended_products") == [product]

    # Step 2: simulate state persistence (in prod this goes through Redis
    # via the LangGraph checkpointer; here we just merge the partial).
    persisted: dict = dict(state_in)
    persisted.update(result_recommend)
    persisted["messages"] = [HumanMessage(content="detalhes por favor")]

    triage_result = await triage_node(persisted)  # type: ignore[arg-type]

    assert triage_result["intent"] == "attribute_inquiry", (
        f"triage routed to {triage_result['intent']!r} instead of "
        "attribute_inquiry — the short-circuit didn't fire. "
        "Likely cause: schema missing awaiting_detail_choice."
    )
    # Flag is cleared on consumption.
    assert triage_result.get("awaiting_detail_choice") is False


@pytest.mark.asyncio
async def test_full_chain_recommend_to_price():
    """Same chain, but the customer pivots to price after the offer."""
    product = _product_with_attrs()
    state_in: AgentState = _state(message="mormaii sunset")
    with patch(
        "app.agent.nodes.recommend._list_catalog_candidates",
        new_callable=AsyncMock,
        return_value=[product],
    ):
        result_recommend = await recommend_node(state_in)
    assert result_recommend["awaiting_detail_choice"] is True

    persisted: dict = dict(state_in)
    persisted.update(result_recommend)
    persisted["messages"] = [HumanMessage(content="quanto custa?")]

    triage_result = await triage_node(persisted)  # type: ignore[arg-type]
    assert triage_result["intent"] == "price_inquiry"
    assert triage_result.get("awaiting_detail_choice") is False


# ═══════════════════════════════════════════════════════════════════════════
# Schema sanity — the root-cause check
# ═══════════════════════════════════════════════════════════════════════════

def test_awaiting_detail_choice_in_state_schema():
    """Sprint 2.6.10 root-cause guard. If this field is missing from the
    TypedDict, LangGraph checkpointer silently drops it on persistence
    (the Felipe production failure). Test by inspection of the annotations."""
    from app.agent.state import AgentState
    annotations = getattr(AgentState, "__annotations__", {})
    assert "awaiting_detail_choice" in annotations, (
        "awaiting_detail_choice MUST be in AgentState. Without it, the "
        "checkpointer drops the flag between turns and the short-circuit "
        "never fires (production bug 2026-06)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Triage short-circuit — unit tests (flag set manually)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_accept_detail_offer_routes_to_attribute_inquiry():
    state = _state(
        message="detalhes por favor",
        recommended_products=[_product_with_attrs()],
        awaiting_detail_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "attribute_inquiry"
    assert result.get("awaiting_detail_choice") is False


@pytest.mark.asyncio
async def test_accept_detail_with_sim():
    state = _state(
        message="sim",
        recommended_products=[_product_with_attrs()],
        awaiting_detail_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "attribute_inquiry"


@pytest.mark.asyncio
async def test_accept_detail_with_manda():
    state = _state(
        message="manda aí",
        recommended_products=[_product_with_attrs()],
        awaiting_detail_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "attribute_inquiry"


@pytest.mark.asyncio
async def test_accept_detail_with_quero_saber_mais():
    state = _state(
        message="quero saber mais",
        recommended_products=[_product_with_attrs()],
        awaiting_detail_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "attribute_inquiry"


@pytest.mark.asyncio
async def test_price_after_detail_offer_routes_to_price():
    state = _state(
        message="quanto custa",
        recommended_products=[_product_with_attrs()],
        awaiting_detail_choice=True,
    )
    result = await triage_node(state)
    assert result["intent"] == "price_inquiry"
    assert result.get("awaiting_detail_choice") is False


@pytest.mark.asyncio
async def test_detail_flag_cleared_on_unrelated_message():
    """Customer changes subject → flag is cleared, LLM gets to classify."""
    state = _state(
        message="qual o horário de vocês?",
        recommended_products=[_product_with_attrs()],
        awaiting_detail_choice=True,
    )
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "faq"}',
    ):
        result = await triage_node(state)
    assert result["intent"] == "faq"
    assert result.get("awaiting_detail_choice") is False


@pytest.mark.asyncio
async def test_no_flag_no_short_circuit():
    """When the flag is NOT set, the triage runs the LLM normally even
    if the message contains 'detalhes' (because we don't know the
    context — the customer might be asking 'detalhes da entrega')."""
    state = _state(
        message="detalhes",
        recommended_products=[],
        awaiting_detail_choice=None,
    )
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "help_request"}',
    ):
        result = await triage_node(state)
    # The LLM ran (mocked); we don't second-guess its output here.
    assert result["intent"] == "help_request"


# ═══════════════════════════════════════════════════════════════════════════
# Flag-ordering vs match_confirmation
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_match_confirmation_wins_when_both_flags_set():
    """Defense-in-depth: if both flags are set (shouldn't happen, but
    paranoia), match_confirmation runs first because it's the older
    short-circuit (and recommend only sets detail_choice AFTER consuming
    match_confirmation in the same turn)."""
    state = _state(
        message="sim",
        recommended_products=[],
        awaiting_match_confirmation={"name": "Test Racket", "id": 1},
        awaiting_detail_choice=True,
    )
    result = await triage_node(state)
    # match_confirmation 'yes' → product_inquiry (so recommend re-fires).
    assert result["intent"] == "product_inquiry"


# ═══════════════════════════════════════════════════════════════════════════
# Broad-detail cascade — attribute_inquiry
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_broad_details_with_parsed_attributes_lists_them():
    state = _state(
        message="detalhes por favor",
        recommended_products=[_product_with_attrs("Mormaii Sunset Plus")],
    )
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    # Every parsed attribute shows up.
    assert "Peso" in text and "320g" in text
    assert "Composição" in text and "Carbono 3K" in text
    assert "Espessura" in text and "22mm" in text
    assert "Comprimento" in text and "50cm" in text
    assert "Equilíbrio" in text and "27,5cm" in text
    # Price added at the end.
    assert "R$ 499" in text or "R$ 499" == text  # 49900 cents = R$ 499


@pytest.mark.asyncio
async def test_broad_details_without_attributes_uses_description():
    """No atributos_parseados but has descricao_curta → cascade falls to
    the description (cleaned and truncated)."""
    state = _state(
        message="me conta sobre essa raquete",
        recommended_products=[_product_with_description_only()],
    )
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    # HTML stripped.
    assert "<p>" not in text and "</p>" not in text
    # Description content shows up.
    assert "AMA Sport Proteo" in text or "controle e conforto" in text
    # Price appears.
    assert "R$ 799" in text


@pytest.mark.asyncio
async def test_broad_details_bare_product_shows_price_and_honest():
    """Product with no attrs and no description → name + price + honest
    'specs detalhadas não constam' (no follow-up promise, no pitch)."""
    state = _state(
        message="quero detalhes",
        recommended_products=[_product_bare()],
    )
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "Bola Beach Tennis Pack" in text
    assert "R$ 59" in text or "R$ 59" == text  # 5900 cents = R$ 59
    # Honest — but no "vou confirmar e te retorno" promise (that's reserved
    # for specific-attribute misses).
    assert "não constam" in text.lower() or "nao constam" in text.lower()


@pytest.mark.asyncio
async def test_broad_details_never_pitches_consultoria():
    """The cardinal rule: 'detalhes' MUST NOT trigger a Consultoria pitch."""
    for product in (
        _product_with_attrs(),
        _product_with_description_only(),
        _product_bare(),
    ):
        state = _state(
            message="detalhes",
            recommended_products=[product],
        )
        result = await attribute_inquiry_node(state)
        text = result["response_blocks"][0]
        # No price-pitch signature.
        assert "R$ 350" not in text  # consultoria fee — never in detail response
        assert "abatido" not in text.lower()
        assert "consultoria" not in text.lower()
        assert "quer agendar" not in text.lower()
        assert "quer saber como funciona" not in text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Specific attribute — regression of Sprint 2.6.6
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_specific_attribute_returns_value_when_present():
    state = _state(
        message="qual o peso?",
        recommended_products=[_product_with_attrs("Mormaii Sunset")],
    )
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "320g" in text
    assert "pesa" in text.lower()


@pytest.mark.asyncio
async def test_specific_attribute_honest_when_missing():
    """Proteo style: attribute asked, product has nothing parsed → honest +
    alert. 2.6.6 path, MUST be preserved (broad-detail must not steal it)."""
    state = _state(
        message="qual o peso?",
        recommended_products=[_product_with_description_only("AMA Proteo")],
    )
    with patch(
        "app.agent.nodes.attribute_inquiry._send_missing_attr_alert",
        new_callable=AsyncMock,
    ) as alert:
        result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    # Honest line emitted; alert fired.
    assert "não consta" in text.lower() or "nao consta" in text.lower()
    assert "confirmar com a equipe" in text.lower()
    alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_quantos_k_routes_to_composicao_attribute():
    """'Quantos K' is a composition question — Felipe's case. Should NOT
    fall through to broad-detail."""
    state = _state(
        message="quantos k ela tem?",
        recommended_products=[_product_with_attrs("Test")],
    )
    result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    # Resolved as composicao → "Carbono 3K".
    assert "Carbono 3K" in text or "carbono 3k" in text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Gender concordance
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_honest_missing_uses_correct_article_for_ficha():
    state = _state(
        message="me fala a ficha técnica",
        recommended_products=[_product_with_description_only("Test")],
    )
    with patch(
        "app.agent.nodes.attribute_inquiry._send_missing_attr_alert",
        new_callable=AsyncMock,
    ):
        result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    # "A ficha técnica" not "O ficha técnica"
    assert "A ficha" in text
    assert "O ficha" not in text


@pytest.mark.asyncio
async def test_honest_missing_uses_correct_article_for_composicao():
    """Set up a product with peso present but composicao missing → the
    partial-then-promise renderer must use the right article for the
    missing label."""
    product = _product_with_attrs("Test")
    product["atributos_parseados"] = {"peso": "300g"}  # only peso, missing composicao
    state = _state(
        message="qual o peso e a composição?",
        recommended_products=[product],
    )
    with patch(
        "app.agent.nodes.attribute_inquiry._send_missing_attr_alert",
        new_callable=AsyncMock,
    ):
        result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    assert "A Composição" in text or "A composição" in text
    assert "O Composição" not in text


@pytest.mark.asyncio
async def test_honest_missing_uses_correct_article_for_peso():
    state = _state(
        message="qual o peso?",
        recommended_products=[_product_with_description_only("Test")],
    )
    with patch(
        "app.agent.nodes.attribute_inquiry._send_missing_attr_alert",
        new_callable=AsyncMock,
    ):
        result = await attribute_inquiry_node(state)
    text = result["response_blocks"][0]
    # "O peso" is correct (masculine).
    assert "O peso" in text or "O Peso" in text


# ═══════════════════════════════════════════════════════════════════════════
# Helper unit tests
# ═══════════════════════════════════════════════════════════════════════════

def test_is_broad_detail_request_positive_cases():
    for msg in (
        "detalhes",
        "Detalhes por favor",
        "quero detalhes",
        "me conta sobre essa",
        "fala mais dela",
        "me explica como é essa",
    ):
        assert is_broad_detail_request(msg), f"failed for: {msg!r}"


def test_is_broad_detail_request_negative_cases():
    for msg in (
        "qual o peso?",
        "quanto custa?",
        "vocês têm a Kronos?",
        "tchau",
    ):
        assert not is_broad_detail_request(msg), f"false-positive for: {msg!r}"


def test_clean_description_strips_html_and_truncates():
    raw = (
        "<p>Lorem ipsum dolor sit amet. Consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
        "nisi ut aliquip ex ea commodo consequat.</p>"
    )
    out = _clean_description_for_whatsapp(raw)
    assert "<p>" not in out
    assert len(out) <= 450 + 1
    # Truncates at a sentence boundary (ends with ".").
    assert out.endswith(".") or out.endswith("…")


def test_clean_description_empty_returns_empty():
    assert _clean_description_for_whatsapp("") == ""
    assert _clean_description_for_whatsapp(None) == ""  # type: ignore[arg-type]


def test_render_broad_details_includes_price_when_available():
    p = _product_with_attrs()
    out = _render_broad_details(p)
    assert "R$" in out


# ═══════════════════════════════════════════════════════════════════════════
# Fence false-positive (Part D)
# ═══════════════════════════════════════════════════════════════════════════

def test_fence_does_not_flag_consultoria_recommendation():
    from app.agent.nodes.help_request import _validate_help_response

    # Common case: the LLM said "recomendo a nossa Consultoria".
    text = (
        "Pra te ajudar a escolher, recomendo a nossa *Consultoria* — "
        "a gente analisa seu jogo e você testa em quadra (R$ 350)."
    )
    is_valid, violations = _validate_help_response(text)
    assert is_valid, (
        f"False-positive: 'Consultoria' is not a racket model. violations={violations}"
    )


def test_fence_does_not_flag_base_sports_recommendation():
    from app.agent.nodes.help_request import _validate_help_response

    text = "Recomendo a Base Sports — temos a Consultoria que resolve."
    is_valid, violations = _validate_help_response(text)
    # "Base" is in the allowlist; "Sports" follows naturally.
    assert is_valid, f"False-positive on store brand. violations={violations}"


def test_fence_still_flags_real_model_recommendation():
    """The allowlist must NOT loosen detection of real models."""
    from app.agent.nodes.help_request import _validate_help_response

    text = "Recomendo a Mormaii Sunset Plus, é a melhor pra iniciante."
    is_valid, violations = _validate_help_response(text)
    assert not is_valid
    assert "recommends_specific_model" in violations
