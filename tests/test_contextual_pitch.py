"""Sprint 2.3 — contextual pitch tests.

Three areas:
- ``classify_question`` matches each ``QuestionType`` correctly.
- The 5 presets (PRICE, STOCK, FITNESS, COMFORT, default) all mention
  R$<preco> + abatimento and end with a question.
- Timing: IMMEDIATE types may fire on the 1st determined question;
  DELAYED types only from the 2nd on.
- Cap + path guards (carried over from Sprint 2.1) still hold.
- Integration: each node passes the right ``question_type``.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes._pitch_classification import (
    DELAYED_TYPES,
    IMMEDIATE_TYPES,
    QuestionType,
    classify_question,
    is_immediate,
)
from app.agent.nodes.consultoria_offer import (
    _PRESET_BUILDERS,
    maybe_add_subtle_consultoria_offer,
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


def _state(**overrides) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="msg")],
        "phone_hash": "ctxpitch" * 8,
        "intent": None,
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
        "customer_intent_path": "determined",
        "consultoria_mentioned_count": 0,
        "determined_question_count": 0,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _product(name: str, **extra) -> dict:
    base = {
        "id": f"id-{name}", "name": name, "sport": "beach_tennis",
        "level": "intermediário", "price_cents": 89900, "stock": 5,
        "description": f"desc {name}", "similarity": 0.9,
        "external_id": name.replace(" ", "-"), "url": None, "image_url": None,
        "updated_at": None, "is_active": True, "weight_g": 350,
        "balance": "médio", "material": "carbono", "category": "raquete",
    }
    base.update(extra)
    return base


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFY
# ════════════════════════════════════════════════════════════════════════════

def test_classify_question_price():
    for text in ("quanto custa?", "qual o preço?", "qual o valor?", "quanto sai?"):
        assert classify_question(text) is QuestionType.PRICE, text


def test_classify_question_stock():
    for text in ("vocês têm a Carbon X5?", "tem em estoque?", "está disponível?"):
        assert classify_question(text) is QuestionType.STOCK, text


def test_classify_question_fitness():
    for text in (
        "serve pra mim?",
        "é boa pro meu nível?",
        "combina comigo?",
    ):
        assert classify_question(text) is QuestionType.FITNESS, text


def test_classify_question_comfort():
    for text in (
        "tem antivibração?",
        "isso machuca o cotovelo?",
        "evita lesão?",
        "absorve impacto?",
    ):
        assert classify_question(text) is QuestionType.COMFORT, text


def test_classify_question_weight():
    for text in ("qual o peso?", "ela é leve?", "quantos gramas?"):
        assert classify_question(text) is QuestionType.WEIGHT, text
    # WEIGHT is DELAYED.
    assert QuestionType.WEIGHT in DELAYED_TYPES
    assert not is_immediate(QuestionType.WEIGHT)


def test_classify_question_material():
    for text in ("qual o material?", "é de carbono?", "tem fibra de vidro?"):
        assert classify_question(text) is QuestionType.MATERIAL, text
    assert QuestionType.MATERIAL in DELAYED_TYPES


def test_classify_question_other_is_default():
    assert classify_question("e a cor?") is QuestionType.OTHER
    assert classify_question("") is QuestionType.OTHER
    assert classify_question("me conta mais") is QuestionType.OTHER


def test_immediate_and_delayed_sets_are_disjoint_and_cover_all_types():
    assert not (IMMEDIATE_TYPES & DELAYED_TYPES)
    assert (IMMEDIATE_TYPES | DELAYED_TYPES) == set(QuestionType)


# ════════════════════════════════════════════════════════════════════════════
# PRESETS — invariants (price + abatimento + ends-with-question)
# ════════════════════════════════════════════════════════════════════════════

def test_each_question_type_has_specific_preset():
    """Every QuestionType maps to a non-empty preset text."""
    for qt in QuestionType:
        assert qt in _PRESET_BUILDERS
        text = _PRESET_BUILDERS[qt](350)
        assert text and isinstance(text, str)


@pytest.mark.parametrize("qt", list(QuestionType))
def test_all_presets_mention_350_and_abatimento(qt):
    text = _PRESET_BUILDERS[qt](350)
    assert "350" in text, f"{qt} preset missing price"
    assert "abatido" in text.lower() or "abate" in text.lower(), (
        f"{qt} preset missing abatimento language"
    )


@pytest.mark.parametrize("qt", list(QuestionType))
def test_all_presets_end_with_question(qt):
    text = _PRESET_BUILDERS[qt](350).rstrip()
    assert text.endswith("?"), f"{qt} preset must end with a question"


def test_immediate_and_delayed_presets_differ():
    """The four IMMEDIATE presets use distinct copy; the DELAYED ones share
    the default text (intentional — they're catch-all)."""
    immediate_texts = {qt: _PRESET_BUILDERS[qt](350) for qt in IMMEDIATE_TYPES}
    # All immediate presets are distinct strings.
    assert len(set(immediate_texts.values())) == len(immediate_texts)
    # All delayed share the default preset.
    delayed_texts = {_PRESET_BUILDERS[qt](350) for qt in DELAYED_TYPES}
    assert len(delayed_texts) == 1


# ════════════════════════════════════════════════════════════════════════════
# TIMING — immediate vs delayed
# ════════════════════════════════════════════════════════════════════════════

def test_immediate_pitch_appears_on_first_question():
    state = _state(determined_question_count=0)
    blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    assert len(blocks) == 2
    assert update["consultoria_mentioned_count"] == 1
    assert update["determined_question_count"] == 1


def test_delayed_pitch_waits_for_second_question():
    state = _state(determined_question_count=0)
    blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta sobre peso"], QuestionType.WEIGHT
    )
    # No pitch on the 1st delayed question.
    assert blocks == ["resposta sobre peso"]
    assert "consultoria_mentioned_count" not in update
    # But the counter still moved.
    assert update["determined_question_count"] == 1


def test_delayed_pitch_appears_after_count_2():
    state = _state(determined_question_count=1)
    blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta sobre material"], QuestionType.MATERIAL
    )
    assert len(blocks) == 2
    assert update["consultoria_mentioned_count"] == 1
    assert update["determined_question_count"] == 2


def test_delayed_pitch_uses_default_preset():
    state = _state(determined_question_count=1)
    blocks, _ = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.OTHER
    )
    assert "Consultoria Base Sports" in blocks[1]
    # Default preset's distinctive opening.
    assert "Caso queira ter certeza absoluta" in blocks[1]


def test_immediate_price_preset_text_matches():
    state = _state(determined_question_count=0)
    blocks, _ = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    # PRICE preset's distinctive opening.
    assert "investimento numa raquete" in blocks[1].lower()


def test_immediate_fitness_preset_text_matches():
    state = _state(determined_question_count=0)
    blocks, _ = maybe_add_subtle_consultoria_offer(
        state, ["sim, serve"], QuestionType.FITNESS
    )
    assert "Pra essa pergunta especificamente" in blocks[1]


# ════════════════════════════════════════════════════════════════════════════
# CAP / PATH / HANDOFF
# ════════════════════════════════════════════════════════════════════════════

def test_pitch_appears_max_once_per_conversation():
    """Even with IMMEDIATE question types, only one pitch per convo."""
    state = _state(determined_question_count=0)
    blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["primeira"], QuestionType.PRICE
    )
    assert len(blocks) == 2

    # Simulate state after the helper's update propagated.
    state_after = {**state, **update}
    blocks2, update2 = maybe_add_subtle_consultoria_offer(
        state_after, ["segunda"], QuestionType.STOCK
    )
    assert blocks2 == ["segunda"]  # no pitch the 2nd time
    assert "consultoria_mentioned_count" not in update2


def test_pitch_not_emitted_for_exploring_path():
    state = _state(customer_intent_path="exploring")
    blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    assert blocks == ["resposta"]
    # Exploring path doesn't even tick the determined counter.
    assert "determined_question_count" not in update


def test_pitch_not_emitted_during_handoff():
    state = _state(needs_handoff=True)
    blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    assert blocks == ["resposta"]
    assert "consultoria_mentioned_count" not in update


def test_pitch_not_emitted_when_consultoria_disabled(monkeypatch):
    monkeypatch.setenv("CONSULTORIA_ENABLED", "false")
    from app.config import get_settings
    get_settings.cache_clear()
    state = _state(determined_question_count=0)
    blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    assert blocks == ["resposta"]
    assert "consultoria_mentioned_count" not in update


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION — each node passes the correct QuestionType
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_price_inquiry_emits_immediate_pitch_PRICE():
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _state(
        messages=[HumanMessage(content="quanto custa?")],
        recommended_products=[_product("Raquete X")],
        determined_question_count=0,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    # Price preset's distinctive opening (PRICE is IMMEDIATE → fires on 1st turn).
    assert "investimento numa raquete" in full.lower()
    assert result.get("consultoria_mentioned_count") == 1
    assert result.get("determined_question_count") == 1


@pytest.mark.asyncio
async def test_product_detail_weight_emits_delayed_pitch():
    from app.agent.nodes.product_detail import product_detail_node

    # 1st turn — WEIGHT is DELAYED, no pitch yet.
    state_first = _state(
        messages=[HumanMessage(content="qual o peso da Raquete X?")],
        recommended_products=[_product("Raquete X")],
        determined_question_count=0,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    r1 = await product_detail_node(state_first)
    blob1 = " ".join(r1["response_blocks"])
    assert "Consultoria Base Sports" not in blob1
    assert r1.get("determined_question_count") == 1
    assert r1.get("consultoria_mentioned_count") is None

    # 2nd turn — MATERIAL is also DELAYED, but now count >= 2 → pitch fires.
    state_second = _state(
        messages=[HumanMessage(content="qual o material?")],
        recommended_products=[_product("Raquete X")],
        determined_question_count=1,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    r2 = await product_detail_node(state_second)
    blob2 = " ".join(r2["response_blocks"])
    assert "Consultoria Base Sports" in blob2
    assert r2.get("consultoria_mentioned_count") == 1


@pytest.mark.asyncio
async def test_recommend_reference_sim_no_longer_emits_pitch():
    """Sprint 2.4 — stock confirmation is pitch-free now (STOCK is DELAYED
    and REFERENCE-SIM determined no longer calls the pitch helper)."""
    from app.agent.nodes.recommend import recommend_node

    state = _state(
        messages=[HumanMessage(content="vocês têm a Raquete X?")],
        player_profile={"modelo_desejado": "Raquete X"},
        determined_question_count=0,
    )
    with patch(
        "app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock
    ):
        with patch(
            "app.rag.retriever.search_products", new_callable=AsyncMock
        ) as search:
            search.return_value = [_product("Raquete X")]
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    blocks = result["response_blocks"]
    # Exactly one block — the stock confirmation — and no pitch.
    assert len(blocks) == 1
    assert "Sim, temos" in blocks[0]
    full = " ".join(blocks)
    assert "Consultoria" not in full
    assert result.get("consultoria_mentioned_count") is None


@pytest.mark.asyncio
async def test_product_detail_comfort_emits_immediate_pitch():
    """COMFORT is IMMEDIATE — pitch fires on the very first determined question."""
    from app.agent.nodes.product_detail import product_detail_node

    state = _state(
        messages=[HumanMessage(content="tem antivibração?")],
        recommended_products=[_product("Raquete X", description="leve com bom controle")],
        determined_question_count=0,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await product_detail_node(state)
    full = " ".join(result["response_blocks"])
    assert "Conforto e prevenção de lesão" in full  # COMFORT preset opener
    assert result.get("consultoria_mentioned_count") == 1


@pytest.mark.asyncio
async def test_price_inquiry_no_pitch_for_exploring_customer():
    """Regression: exploring customers never see the subtle pitch."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _state(
        customer_intent_path="exploring",
        messages=[HumanMessage(content="quanto custa?")],
        recommended_products=[_product("Raquete X")],
        determined_question_count=0,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"])
    assert "Consultoria Base Sports" not in full
