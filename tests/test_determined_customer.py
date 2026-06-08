"""Sprint 2.1 — cliente determinado test suite (Sprint 2.6: SKIPPED).

Sprint 2.6 removed the determined / exploring fork because diagnose was
removed entirely. Every customer who reaches a product node is "determined"
by definition; there's no flag to flip and no opinion-seek override. The
tests below remain as historical reference but are skipped at module level.

Covers:
- Triage path detection (determined / exploring / opinion-flip).
- Recommend skipping diagnose + natural single-block confirmation.
- price_inquiry natural format for a single product (no bullet, no "qual delas").
- The subtle Consultoria pitch helper — appears once, then never again.
- REFERENCE-NÃO determined → canned offer with no diagnose.
- Transition from determined → exploring when customer asks our opinion.
- Regression: exploring path still runs the full diagnose; bare_recommendation
  still works; pitch_consultoria full pitch is unaffected.

All external I/O (OpenAI, DB, retriever) is mocked.
"""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState

pytestmark = pytest.mark.skip(
    reason="diagnose deprecated in Sprint 2.6 — customer_intent_path removed"
)


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


def _product(name: str, *, price_cents: int = 89900, **extra) -> dict:
    base = {
        "id": f"id-{name}", "name": name, "sport": "beach_tennis",
        "level": "intermediário", "price_cents": price_cents, "stock": 5,
        "description": f"desc {name}", "similarity": 0.9,
        "external_id": name.replace(" ", "-"), "url": None, "image_url": None,
        "updated_at": None, "is_active": True, "weight_g": 350,
        "balance": "médio", "material": "carbono", "category": "raquete",
    }
    base.update(extra)
    return base


def _base_state(**overrides) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="msg")],
        "phone_hash": "determined" * 6,
        "intent": None,
        "player_profile": {},
        "recommended_products": [],
        "needs_handoff": False,
        "handoff_reason": None,
        "consultoria_interest": False,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ════════════════════════════════════════════════════════════════════════════
# TRIAGE — detection of intent path
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_triage_marks_determined_when_named_product_matches():
    """LLM says recommend/diagnose + catalog match → customer_intent_path=determined."""
    from app.agent.nodes.triage import triage_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(
        messages=[HumanMessage(content="vocês têm a beach pro carbon x5?")],
    )

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "diagnose"}'
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await triage_node(state)

    assert result["customer_intent_path"] == "determined"
    # modelo_desejado is pre-populated so recommend can enter REFERENCE-SIM directly.
    assert result["player_profile"]["modelo_desejado"] == "Raquete BeachPro Carbon X5"


@pytest.mark.asyncio
async def test_triage_marks_exploring_when_bare_recommendation():
    from app.agent.nodes.triage import triage_node

    state = _base_state(messages=[HumanMessage(content="qual vocês indicam?")])
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "bare_recommendation_request"}'
        with patch("app.storage.db.get_session", _mock_db_session):
            result = await triage_node(state)

    assert result["customer_intent_path"] == "exploring"


@pytest.mark.asyncio
async def test_triage_no_path_when_no_catalog_match():
    """Generic recommend message with no catalog match → path NOT set."""
    from app.agent.nodes.triage import triage_node

    state = _base_state(messages=[HumanMessage(content="quero uma raquete")])
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "diagnose"}'
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []  # catalog has nothing relevant
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await triage_node(state)

    assert "customer_intent_path" not in result


# ════════════════════════════════════════════════════════════════════════════
# Sprint 2.1.1 — robust determined detection (even with noisy / partial names)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_determined_detected_when_intent_is_diagnose():
    """The LLM often labels 'vocês têm a X?' as diagnose; we still detect determined."""
    from app.agent.nodes.triage import triage_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(messages=[HumanMessage(content="vocês tem a beach pro carbon x5?")])
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "diagnose"}'  # LLM picks diagnose
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await triage_node(state)

    assert result["customer_intent_path"] == "determined"
    assert result["intent"] == "recommend"  # rewritten away from diagnose


@pytest.mark.asyncio
async def test_determined_detected_with_combined_message_oi_plus_request():
    """'oi, vocês têm a Carbon X5?' — greeting + short product name."""
    from app.agent.nodes.triage import triage_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(messages=[HumanMessage(content="oi, vocês têm a Carbon X5?")])
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "diagnose"}'
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await triage_node(state)

    assert result["customer_intent_path"] == "determined"


@pytest.mark.asyncio
async def test_determined_NOT_detected_for_generic_raquete_request():
    """'quero uma raquete' should NOT trigger determined even if the retriever
    returns something — the loose matcher requires distinctive token overlap."""
    from app.agent.nodes.triage import triage_node

    # The retriever can still return candidates by similarity, but the
    # message has no distinctive tokens → loose match returns None.
    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(messages=[HumanMessage(content="quero uma raquete")])
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "diagnose"}'
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await triage_node(state)

    assert result.get("customer_intent_path") != "determined"


@pytest.mark.asyncio
async def test_determined_detected_for_eu_queria_saber_se():
    """Polite long form: 'eu queria saber se vocês têm a Carbon X5'."""
    from app.agent.nodes.triage import triage_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(
        messages=[HumanMessage(content="eu queria saber se vocês têm a Carbon X5")],
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "diagnose"}'
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await triage_node(state)

    assert result["customer_intent_path"] == "determined"


@pytest.mark.asyncio
async def test_determined_detection_logs_explicit(caplog):
    """The new ``determined_check`` log line must appear for every detection run."""
    import logging

    from app.agent.nodes.triage import triage_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(messages=[HumanMessage(content="tem a Carbon X5?")])
    with caplog.at_level(logging.INFO, logger="app.agent.nodes.triage"):
        with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
            llm.return_value = '{"intent": "diagnose"}'
            with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
                search.return_value = candidates
                with patch("app.storage.db.get_session", _mock_db_session):
                    await triage_node(state)

    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "determined_check" in log_text
    assert "intent=diagnose" in log_text
    assert "matched=True" in log_text
    # And the rewrite log fires.
    assert "intent rewrite" in log_text


@pytest.mark.asyncio
async def test_triage_opinion_seek_flips_determined_to_exploring():
    """A determined customer asking 'você acha?' is re-routed to exploring."""
    from app.agent.nodes.triage import triage_node

    state = _base_state(
        messages=[HumanMessage(content="você acha que ela serve mesmo pra mim?")],
        customer_intent_path="determined",
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "product_detail"}'
        with patch("app.storage.db.get_session", _mock_db_session):
            result = await triage_node(state)

    assert result["customer_intent_path"] == "exploring"
    assert result["intent"] == "bare_recommendation_request"


# ════════════════════════════════════════════════════════════════════════════
# RECOMMEND — determined skips diagnose
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_recommend_skips_diagnose_when_determined():
    """customer_intent_path=determined → no LLM call (deterministic answer).

    Triage normally populates ``modelo_desejado`` with the full catalog name
    (because the matcher returns the matched product's ``name``). We mirror
    that here so the in-node ``_find_name_match`` sees the full token.
    """
    from app.agent.nodes.recommend import recommend_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(
        messages=[HumanMessage(content="vocês têm a carbon x5?")],
        customer_intent_path="determined",
        player_profile={"modelo_desejado": "Raquete BeachPro Carbon X5"},
    )

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    llm.assert_not_called()
    assert result["recommended_products"][0]["name"] == "Raquete BeachPro Carbon X5"


@pytest.mark.asyncio
async def test_recommend_confirms_stock_in_single_block():
    """REFERENCE-SIM + determined → exactly 1 confirmation block (subtle pitch is a 2nd)."""
    from app.agent.nodes.recommend import recommend_node

    candidates = [_product("Raquete BeachPro Carbon X5")]
    state = _base_state(
        customer_intent_path="determined",
        player_profile={"modelo_desejado": "Raquete BeachPro Carbon X5"},
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock):
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    # Block 1 = confirmation. Block 2 (optional) = subtle pitch (first turn).
    assert 1 <= len(result["response_blocks"]) <= 2
    confirmation = result["response_blocks"][0]
    assert "Raquete BeachPro Carbon X5" in confirmation
    assert "•" not in confirmation  # NO bullet
    assert "qual delas" not in confirmation.lower()


@pytest.mark.asyncio
async def test_recommend_NOT_asks_level_when_determined():
    """No 'nível' question slips into a determined-path reply."""
    from app.agent.nodes.recommend import recommend_node

    candidates = [_product("Raquete Carbon X5")]
    state = _base_state(
        customer_intent_path="determined",
        player_profile={"modelo_desejado": "Raquete Carbon X5"},
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock):
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    full = " ".join(result["response_blocks"]).lower()
    assert "nível" not in full and "nivel" not in full


# ════════════════════════════════════════════════════════════════════════════
# REFERENCE-NÃO determined
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_reference_not_determined_skips_diagnose():
    """Sprint 2.4 — REFERENCE-NÃO determined offers alternatives (no LLM call)."""
    from app.agent.nodes.recommend import recommend_node

    state = _base_state(
        customer_intent_path="determined",
        player_profile={"modelo_desejado": "Wilson Pro Staff"},
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []  # nothing matches
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    llm.assert_not_called()
    text = result["response_blocks"][0]
    assert "Wilson Pro Staff" in text
    # Sprint 2.4 — no Consultoria pitch on the no-match reply; just an
    # offer to look at alternatives.
    assert "outras opções" in text or "outras opcoes" in text


@pytest.mark.asyncio
async def test_reference_not_sets_awaiting_alternatives_flag():
    """REFERENCE-NÃO determined: shortlist cleared + ``awaiting_alternatives_decision`` set."""
    from app.agent.nodes.recommend import recommend_node

    state = _base_state(
        customer_intent_path="determined",
        player_profile={"modelo_desejado": "Inexistente"},
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock):
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []
            with patch("app.storage.db.get_session", _mock_db_session):
                result = await recommend_node(state)

    assert result["recommended_products"] == []
    assert result["awaiting_alternatives_decision"] is True


# ════════════════════════════════════════════════════════════════════════════
# PRICE_INQUIRY — natural format for 1 product
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_price_inquiry_natural_format_for_single_product():
    """1 product → 'A *X* sai por R$Y.' (no bullet, no listing)."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    products = [_product("Raquete BeachPro Carbon X5", price_cents=89900)]
    state = _base_state(
        messages=[HumanMessage(content="quanto custa?")],
        recommended_products=products,
        customer_intent_path="determined",
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await price_inquiry_node(state)
    first = result["response_blocks"][0]
    assert first.startswith("A *Raquete BeachPro Carbon X5*")
    assert "•" not in first  # no bullet
    assert "899" in first  # the price


@pytest.mark.asyncio
async def test_price_inquiry_no_qual_delas_for_single():
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state = _base_state(
        messages=[HumanMessage(content="quanto sai?")],
        recommended_products=[_product("Raquete A", price_cents=70000)],
        customer_intent_path="determined",
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await price_inquiry_node(state)
    full = " ".join(result["response_blocks"]).lower()
    assert "qual delas" not in full


@pytest.mark.asyncio
async def test_price_inquiry_bullet_kept_for_multiple():
    """Multi-product fallback still uses the bullet format."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    products = [
        _product("Raquete A", price_cents=70000),
        _product("Raquete B", price_cents=90000),
    ]
    state = _base_state(
        messages=[HumanMessage(content="quanto custa?")],
        recommended_products=products,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    result = await price_inquiry_node(state)
    text = "\n".join(result["response_blocks"])
    assert "• *Raquete A*" in text
    assert "• *Raquete B*" in text
    # Reworded trailing prompt — no more "qual delas".
    assert "qual delas" not in text.lower()


# ════════════════════════════════════════════════════════════════════════════
# SUBTLE CONSULTORIA PITCH HELPER
# ════════════════════════════════════════════════════════════════════════════

def test_subtle_consultoria_appears_once_for_determined():
    from app.agent.nodes._pitch_classification import QuestionType
    from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer

    state = _base_state(
        customer_intent_path="determined",
        consultoria_mentioned_count=0,
    )
    new_blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta principal"], QuestionType.PRICE
    )
    assert len(new_blocks) == 2
    assert "Consultoria Base Sports" in new_blocks[1]
    # Sprint 2.3 — update carries BOTH counters now.
    assert update.get("consultoria_mentioned_count") == 1
    assert update.get("determined_question_count") == 1


def test_subtle_consultoria_NOT_appears_for_exploring():
    from app.agent.nodes._pitch_classification import QuestionType
    from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer

    state = _base_state(
        customer_intent_path="exploring",
        consultoria_mentioned_count=0,
    )
    new_blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    assert new_blocks == ["resposta"]
    assert update == {}  # exploring path doesn't even tick the counter


def test_subtle_consultoria_NOT_repeated_in_same_conversation():
    from app.agent.nodes._pitch_classification import QuestionType
    from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer

    state = _base_state(
        customer_intent_path="determined",
        consultoria_mentioned_count=1,  # already mentioned once
    )
    new_blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    assert new_blocks == ["resposta"]
    assert "consultoria_mentioned_count" not in update
    # The determined question counter still ticks (informational).
    assert update.get("determined_question_count") == 1


def test_subtle_consultoria_NOT_appears_during_handoff():
    from app.agent.nodes._pitch_classification import QuestionType
    from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer

    state = _base_state(
        customer_intent_path="determined",
        consultoria_mentioned_count=0,
        needs_handoff=True,
    )
    new_blocks, update = maybe_add_subtle_consultoria_offer(
        state, ["resposta"], QuestionType.PRICE
    )
    assert new_blocks == ["resposta"]
    assert "consultoria_mentioned_count" not in update


def test_consultoria_mentioned_count_increments():
    """price_inquiry returns the increment so state carries 1 mention after this turn."""
    from app.agent.nodes._pitch_classification import QuestionType
    from app.agent.nodes.consultoria_offer import maybe_add_subtle_consultoria_offer

    state = _base_state(
        customer_intent_path="determined", consultoria_mentioned_count=0,
    )
    _, update = maybe_add_subtle_consultoria_offer(state, ["x"], QuestionType.PRICE)
    assert update.get("consultoria_mentioned_count") == 1


@pytest.mark.asyncio
async def test_price_inquiry_appends_subtle_pitch_first_time_only():
    """1st price inquiry for determined customer → pitch appended. 2nd → silent."""
    from app.agent.nodes.price_inquiry import price_inquiry_node

    state_first = _base_state(
        messages=[HumanMessage(content="quanto custa?")],
        recommended_products=[_product("Raquete X", price_cents=80000)],
        customer_intent_path="determined",
        consultoria_mentioned_count=0,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )
    r1 = await price_inquiry_node(state_first)
    full1 = " ".join(r1["response_blocks"])
    assert "Consultoria Base Sports" in full1
    assert r1.get("consultoria_mentioned_count") == 1

    state_second = dict(state_first)
    state_second["consultoria_mentioned_count"] = 1
    state_second["messages"] = [HumanMessage(content="e quanto custa em 10x?")]
    r2 = await price_inquiry_node(state_second)
    full2 = " ".join(r2["response_blocks"])
    assert "Consultoria Base Sports" not in full2


# ════════════════════════════════════════════════════════════════════════════
# TRANSITION — determined → exploring
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_determined_to_exploring_triggers_diagnose():
    """Through the graph: determined → opinion-seek → exploring → diagnose."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent.graph import build_graph

    graph = build_graph(checkpointer=MemorySaver())
    candidates = [_product("Raquete Carbon X5")]
    initial = _base_state(
        messages=[HumanMessage(content="vocês têm a carbon x5?")],
        customer_intent_path=None,
        player_profile={},
    )

    # Turn 1: determined → REFERENCE-SIM canned confirmation.
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = ['{"intent": "diagnose"}']
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = candidates
            with patch("app.storage.db.get_session", _mock_db_session):
                r1 = await graph.ainvoke(initial, {"configurable": {"thread_id": "t-transition"}})

    assert r1["customer_intent_path"] == "determined"

    # Turn 2: opinion-seeking question — triage flips → exploring → diagnose runs.
    state2 = dict(r1)
    state2["messages"] = [HumanMessage(content="você acha que ela serve mesmo pra mim?")]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            '{"intent": "product_detail"}',  # triage; opinion-seek overrides → bare_recommendation
            json.dumps({"extracted_slots": {}}),  # diagnose extract
            "Pra te ajudar a escolher, qual seu nível?",  # diagnose phrase (1 LLM call)
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock):
            with patch("app.storage.db.get_session", _mock_db_session):
                r2 = await graph.ainvoke(state2, {"configurable": {"thread_id": "t-transition"}})

    assert r2["customer_intent_path"] == "exploring"
    # Diagnose actually fired — the LLM was called more than once on this turn.
    last_text = r2["messages"][-1].content
    assert "nível" in last_text.lower() or "nivel" in last_text.lower()


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_exploring_path_still_does_full_diagnose():
    """customer_intent_path=exploring + empty profile → diagnose asks a question."""
    from app.agent.nodes.diagnose import diagnose_node

    state = _base_state(
        messages=[HumanMessage(content="quero indicação")],
        customer_intent_path="exploring",
        player_profile={},
    )
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.side_effect = [
            json.dumps({"extracted_slots": {}}),  # nothing extracted
            "E qual é o seu nível de jogo?",       # phrase
        ]
        result = await diagnose_node(state)

    text = result["messages"][-1].content.lower()
    assert "nível" in text or "nivel" in text


@pytest.mark.asyncio
async def test_bare_recommendation_still_works():
    """bare_recommendation_request still flips intent path to exploring."""
    from app.agent.nodes.triage import triage_node

    state = _base_state(messages=[HumanMessage(content="me indica uma")])
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = '{"intent": "bare_recommendation_request"}'
        with patch("app.storage.db.get_session", _mock_db_session):
            r = await triage_node(state)

    assert r["intent"] == "bare_recommendation_request"
    assert r["customer_intent_path"] == "exploring"


@pytest.mark.asyncio
async def test_pitch_consultoria_full_still_works():
    """The full pitch node is unaffected (still 3 blocks via LLM)."""
    from app.agent.nodes.pitch_consultoria import pitch_consultoria_node

    state = _base_state(messages=[HumanMessage(content="me explica a consultoria")])
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as llm:
        llm.return_value = json.dumps({
            "messages": ["bloco 1", "bloco 2 R$350", "agendar?"],
        })
        result = await pitch_consultoria_node(state)

    blocks = result["response_blocks"]
    assert len(blocks) >= 2
    assert result.get("consultoria_interest") is True
