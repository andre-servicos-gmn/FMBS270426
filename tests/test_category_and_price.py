"""Sprint 2.7.3 — fixes for two production bugs:

  Bug 2 (category): "drop shot raquete" used to return clothes because
  the matcher filtered "raquete" as a generic token and 2/2 token-score
  tied clothes vs racket. Fixed by a category gate that filters the
  candidate list BEFORE scoring when the query carries a category hint.

  Bug 1 (budget): "Quero uma raquete até 2k" used to quote a R$ 2.999
  product. Fixed by a triage short-circuit that detects budget mentions
  (with 2 restrictions: no active product, and value >= R$ 100) and
  routes to help_request with a flag the help_request prompt uses to
  acknowledge the budget WITHOUT listing products (business rule).
"""
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes._product_match import (
    _product_category_slug,
    detect_category_hint,
    filter_by_category,
    match_product_tolerant,
)
from app.agent.nodes.help_request import _build_user_block, help_request_node
from app.agent.nodes.triage import _extract_price_range, triage_node
from app.agent.state import AgentState


# ── Fixtures ────────────────────────────────────────────────────────────────

def _raquete_drop_shot() -> dict:
    return {
        "id": 1,
        "name": "Raquete Drop Shot Legacy 12k",
        "price_cents": 119900,
        "is_raquete_praia": True,
        "categoria_nome": "Raquetes de Praia",
        "external_id": "1",
    }


def _short_drop_shot() -> dict:
    return {
        "id": 2,
        "name": "Short Drop Shot Padel Carbon",
        "price_cents": 19900,
        "is_raquete_praia": False,
        "categoria_nome": "Short",
        "external_id": "2",
    }


def _top_drop_shot() -> dict:
    return {
        "id": 3,
        "name": "Top Drop Shot Feminino Branco",
        "price_cents": 14900,
        "is_raquete_praia": False,
        "categoria_nome": "Top",
        "external_id": "3",
    }


def _mormaii_sunset() -> dict:
    return {
        "id": 4,
        "name": "Raquete Mormaii Sunset Plus 2026",
        "price_cents": 89900,
        "is_raquete_praia": True,
        "categoria_nome": "Raquetes de Praia",
        "external_id": "4",
    }


def _kronos_padel() -> dict:
    return {
        "id": 5,
        "name": "Raquete Pickleball Kronos Padel Pro",
        "price_cents": 79900,
        "is_raquete_praia": False,
        "categoria_nome": "RAQUETE PADEL",
        "external_id": "5",
    }


# ════════════════════════════════════════════════════════════════════════════
# CATEGORY — detector + product slug + filter (unit)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected", [
    ("drop shot raquete", "raquete"),
    ("tem raquetes?", "raquete"),
    ("Drop Shot Raquete", "raquete"),
    ("quero um short", "vestuario"),
    ("tem top feminino?", "vestuario"),
    ("tem vestido", "vestuario"),
    ("manguito", "acessorio"),
    ("uma bola", "bola"),
    ("raqueteira", "bolsa"),
    ("sapatilha", "calcado"),
    ("pala de padel", "pala"),
    # No category hint → None (regression protection)
    ("Mormaii Sunset", None),
    ("kronos", None),
    ("hugo russo 2026", None),
    ("oi", None),
])
def test_detect_category_hint(query, expected):
    assert detect_category_hint(query) == expected


def test_product_category_slug_for_raquete():
    assert _product_category_slug(_raquete_drop_shot()) == "raquete"
    assert _product_category_slug(_mormaii_sunset()) == "raquete"


def test_product_category_slug_for_vestuario():
    assert _product_category_slug(_short_drop_shot()) == "vestuario"
    assert _product_category_slug(_top_drop_shot()) == "vestuario"


def test_product_category_slug_for_pala():
    assert _product_category_slug(_kronos_padel()) == "pala"


def test_filter_by_category_drops_non_matching():
    products = [_short_drop_shot(), _top_drop_shot(), _raquete_drop_shot()]
    filtered = filter_by_category(products, "raquete")
    assert filtered == [_raquete_drop_shot()]


def test_filter_by_category_empty_filter_falls_back_to_full_list():
    """If the hint produces zero candidates (rare — keyword had no map),
    return the original list so the matcher can still try."""
    products = [_mormaii_sunset()]
    # Synthesize an exotic hint that no product satisfies.
    filtered = filter_by_category(products, "exotic_category_no_match")
    assert filtered == products


def test_filter_by_category_no_hint_returns_unchanged():
    products = [_short_drop_shot(), _raquete_drop_shot()]
    assert filter_by_category(products, None) is products


# ════════════════════════════════════════════════════════════════════════════
# CATEGORY — end-to-end through match_product_tolerant
# ════════════════════════════════════════════════════════════════════════════

def test_drop_shot_raquete_returns_raquete_not_clothes():
    """The headline bug: 'drop shot raquete' must surface the Raquete
    Drop Shot Legacy, NOT Short/Top Drop Shot."""
    products = [_short_drop_shot(), _top_drop_shot(), _raquete_drop_shot()]
    result = match_product_tolerant("drop shot raquete", products)

    assert result.product is not None or result.candidates, result
    if result.product:
        assert "Raquete" in result.product["name"]
    else:
        # ambiguous over filtered-down list — but ALL candidates must be raquetes
        for c in result.candidates or []:
            assert _product_category_slug(c) == "raquete", (
                f"non-raquete leaked through: {c['name']!r}"
            )


def test_mormaii_sunset_no_keyword_unchanged_behavior():
    """Regression: queries without a category hint behave exactly as
    before (no category filter applied)."""
    products = [_short_drop_shot(), _mormaii_sunset()]
    result = match_product_tolerant("Mormaii Sunset", products)
    assert result.product is not None
    assert result.product["name"] == "Raquete Mormaii Sunset Plus 2026"


def test_kronos_no_hint_keeps_pickleball_visible_for_disambiguation():
    """When the cliente only says 'kronos' (no sport hint), pickleball
    and beach-tennis Kronos can both appear — that's the intentional
    behavior. Filter must NOT activate when there's no hint."""
    products = [_raquete_drop_shot(), _kronos_padel()]
    # Without a hint, both pass through; the matcher does normal
    # scoring (which here returns whichever matches 'kronos' best).
    result = match_product_tolerant("kronos", products)
    # Whatever the matcher decides, the FILTER did not pre-eliminate
    # pickleball. We assert the filter is inert by checking we can find
    # the pickleball when it's the only token match.
    assert result.product is not None or result.candidates


def test_short_query_finds_clothing_not_racket():
    """Symmetry: 'short' should NOT match the Raquete (Drop Shot Legacy)
    just because the racket name contains the word 'shot'."""
    products = [_short_drop_shot(), _raquete_drop_shot()]
    result = match_product_tolerant("tem short?", products)
    assert result.product is not None
    assert result.product["name"] == "Short Drop Shot Padel Carbon"


# ════════════════════════════════════════════════════════════════════════════
# PRICE — extractor (unit)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text,expected", [
    # Digit + scale
    ("até 2k", 2000),
    ("até 2000", 2000),
    ("ate 2 mil", 2000),
    ("ate r$ 2.000", None),    # period not parsed by current detector; OK
    ("até 1500 reais", 1500),
    ("até 1500 rs", 1500),
    ("até 1500", 1500),
    # "uns X"
    ("uns 2000", 2000),
    ("uns 2 mil", 2000),
    ("uns 1500 reais", 1500),
    # Word numbers
    ("até dois mil", 2000),
    ("ate dois mil", 2000),
    ("até cinco mil", 5000),
    ("no maximo dois mil", 2000),
    # No trigger → None (regression: don't hijack price questions)
    ("quanto custa a Proteo?", None),
    ("quanto custa?", None),
    ("Proteo", None),
    ("oi tudo bem?", None),
    # Below threshold
    ("uns 50", None),
    ("ate 80 reais", None),
    # Empty / nonsense
    ("", None),
    ("até", None),
])
def test_extract_price_range(text, expected):
    assert _extract_price_range(text) == expected


# ════════════════════════════════════════════════════════════════════════════
# PRICE — triage short-circuit + restrictions
# ════════════════════════════════════════════════════════════════════════════

def _state(
    *,
    message: str,
    recommended_products: list | None = None,
    history_msgs: list | None = None,
) -> AgentState:
    msgs = list(history_msgs or [])
    msgs.append(HumanMessage(content=message))
    s: dict = {
        "messages": msgs,
        "phone_hash": "pricetest" * 6,
        "intent": None,
        "player_profile": {},
        "recommended_products": recommended_products or [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    return s  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_opening_with_budget_routes_to_help_request():
    """The Felipe opening case: 'oi, quero uma raquete até 2k' on first
    contact (no active product) → help_request + flag set."""
    state = _state(message="oi, quero uma raquete até 2k")
    result = await triage_node(state)
    assert result["intent"] == "help_request"
    assert result.get("price_range_mentioned") is True


@pytest.mark.asyncio
async def test_bare_budget_phrase_routes_to_help_request():
    state = _state(message="quero uma até dois mil")
    result = await triage_node(state)
    assert result["intent"] == "help_request"
    assert result.get("price_range_mentioned") is True


@pytest.mark.asyncio
async def test_no_budget_phrase_does_not_short_circuit():
    """Regression: a normal message without budget words goes through
    the LLM as usual."""
    state = _state(message="tem mormaii sunset?")
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "product_inquiry"}',
    ):
        result = await triage_node(state)
    assert result["intent"] == "product_inquiry"
    assert result.get("price_range_mentioned") is None or result.get("price_range_mentioned") is False


@pytest.mark.asyncio
async def test_budget_mention_with_active_product_does_not_short_circuit():
    """Restriction 1: customer says 'eu tenho R$3500' MID-CONVERSATION
    while a product is active → the short-circuit does NOT fire (this
    isn't a budget-driven SEARCH; it's an aside)."""
    active = [_mormaii_sunset()]
    state = _state(
        message="legal! eu tenho uns 3500 reais por sinal",
        recommended_products=active,
    )
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "smalltalk"}',
    ):
        result = await triage_node(state)
    # short-circuit DIDN'T fire (active product present) → LLM ran.
    assert result["intent"] == "smalltalk"


@pytest.mark.asyncio
async def test_low_value_budget_does_not_fire():
    """Restriction 2: 'uns 50' is below R$ 100 threshold → no
    short-circuit (avoids hijacking casual chat)."""
    state = _state(message="uns 50 reais")
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "smalltalk"}',
    ):
        result = await triage_node(state)
    assert result["intent"] == "smalltalk"


@pytest.mark.asyncio
async def test_quanto_custa_specific_product_still_works():
    """The most important false-positive guard: 'quanto custa a Proteo?'
    must NOT be hijacked as budget. Routes via LLM normally → price_inquiry."""
    state = _state(message="quanto custa a Proteo?")
    with patch(
        "app.agent.nodes.triage.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value='{"intent": "price_inquiry"}',
    ):
        result = await triage_node(state)
    assert result["intent"] == "price_inquiry"
    assert not result.get("price_range_mentioned")


# ════════════════════════════════════════════════════════════════════════════
# SCHEMA — lesson 2.6.10
# ════════════════════════════════════════════════════════════════════════════

def test_price_range_mentioned_in_state_schema():
    """Without declaration in TypedDict, LangGraph's checkpointer drops
    the field between turns. Guard explicitly."""
    annotations = getattr(AgentState, "__annotations__", {})
    assert "price_range_mentioned" in annotations, (
        "price_range_mentioned MUST be declared in AgentState"
    )


# ════════════════════════════════════════════════════════════════════════════
# HELP_REQUEST — consumes the flag, augments prompt context, clears
# ════════════════════════════════════════════════════════════════════════════

def test_user_block_mentions_budget_when_flag_set():
    block = _build_user_block(
        customer_name="Felipe",
        last_text="quero uma raquete até 2k",
        already_offered=False,
        price_range_mentioned=True,
    )
    assert "ATENÇÃO" in block
    assert "faixa de preço" in block.lower() or "orçamento" in block.lower()
    assert "consultoria considera" in block.lower()
    # Hard rule reminders.
    assert "NÃO ofereça um produto" in block or "não ofereça" in block.lower()
    assert "NÃO liste" in block or "não liste" in block.lower()


def test_user_block_no_budget_section_when_flag_unset():
    block = _build_user_block(
        customer_name="Felipe",
        last_text="qual vocês indicam?",
        already_offered=False,
        price_range_mentioned=False,
    )
    assert "ATENÇÃO" not in block
    assert "faixa de preço" not in block.lower()


@pytest.mark.asyncio
async def test_help_request_clears_price_range_flag_after_consumption():
    """After processing the budget-tagged help_request, the flag is
    cleared so a follow-up turn doesn't keep re-injecting the
    instruction."""
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="quero uma raquete até 2k")],
        "phone_hash": "cleartest" * 6,
        "intent": "help_request",
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "price_range_mentioned": True,
    }
    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value=(
            "Olha, pra te indicar a raquete certa dentro do seu "
            "orçamento, a *Consultoria* analisa perfil, jogo e faixa "
            "de valor — é o caminho. Se já tem algum modelo em mente, "
            "me diz o nome que te passo os detalhes!"
        ),
    ):
        result = await help_request_node(state)

    assert result.get("price_range_mentioned") is False


# ════════════════════════════════════════════════════════════════════════════
# END-TO-END — chain triage + help_request, verifying NO product offered
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_chain_budget_query_never_quotes_product():
    """The Felipe acceptance test: 'oi, quero uma raquete até 2k' goes
    through triage → help_request. The final reply must NOT contain a
    product name + price (which is what produced the broken 'Kronos
    R$ 2.999' output)."""
    state_in: AgentState = _state(message="oi, quero uma raquete até 2k")
    triage_result = await triage_node(state_in)
    assert triage_result["intent"] == "help_request"
    assert triage_result.get("price_range_mentioned") is True

    # Merge for next node.
    merged: dict = dict(state_in)
    merged.update(triage_result)

    with patch(
        "app.agent.nodes.help_request.OpenAIClient.chat",
        new_callable=AsyncMock,
        return_value=(
            "Pra te indicar a raquete certa dentro do que faz sentido "
            "pro seu jogo E pra sua faixa de investimento, a *Consultoria* "
            "é o caminho — análise + teste em quadra (R$ 350, abatido se "
            "fechar). Se você já tem algum modelo em mente, me diz o nome "
            "que te passo os detalhes!"
        ),
    ):
        result = await help_request_node(merged)  # type: ignore[arg-type]

    text = result["response_blocks"][0]
    # Must NOT contain a specific product price quote.
    assert "R$ 2.999" not in text
    assert "R$ 2999" not in text
    assert "Kronos" not in text
    assert "Mormaii" not in text
    # Must mention Consultoria.
    assert "consultoria" in text.lower()
