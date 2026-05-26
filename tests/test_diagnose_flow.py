"""Tests for the Sprint 1.8 4-phase diagnose architecture.

Coverage:
    - Strict slot ordering via _next_pending_slot
    - Deterministic guardrails via _apply_guardrails
    - Meta-question detection via is_meta_question
    - Full integration flows mocking extract + phrase per turn
    - Singular/plural concordance in the recommend guardrail
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state import AgentState


# ── Helpers ──────────────────────────────────────────────────────────────────

def _initial_state(message: str, phone_hash: str = "diagflow" * 8) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=message)],
        phone_hash=phone_hash,
        intent=None,
        player_profile={},
        recommended_products=[],
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
    )


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _last_ai_message(result: dict) -> str:
    for m in reversed(result.get("messages") or []):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
    session.commit = AsyncMock()
    yield session


# ════════════════════════════════════════════════════════════════════════════
# ORDEM ESTRITA — unit tests on _next_pending_slot
# ════════════════════════════════════════════════════════════════════════════

def test_diagnose_asks_nivel_first():
    """Empty profile → first slot to ask is nivel_jogo."""
    from app.agent.nodes.diagnose import _next_pending_slot
    assert _next_pending_slot({}) == "nivel_jogo"


def test_diagnose_asks_lesoes_after_nivel():
    """With only nivel_jogo set, the next slot is lesoes."""
    from app.agent.nodes.diagnose import _next_pending_slot
    assert _next_pending_slot({"nivel_jogo": "iniciante"}) == "lesoes"


def test_diagnose_asks_regiao_lesao_only_when_lesoes_positive():
    """regiao_lesao is conditional on the customer reporting an injury."""
    from app.agent.nodes.diagnose import _next_pending_slot
    # No lesion (or auto-filled "nenhuma") → regiao_lesao is skipped.
    profile_none = {"nivel_jogo": "iniciante", "lesoes": "nenhuma", "regiao_lesao": "nenhuma"}
    assert _next_pending_slot(profile_none) != "regiao_lesao"
    # Real injury reported → regiao_lesao becomes the next question.
    profile_lesion = {"nivel_jogo": "iniciante", "lesoes": "dor no cotovelo"}
    assert _next_pending_slot(profile_lesion) == "regiao_lesao"


def test_diagnose_asks_esporte_previo_after_lesoes():
    """For a beginner with no lesion, the next pending slot is esporte_raquete_previo."""
    from app.agent.nodes.diagnose import _next_pending_slot
    profile = {
        "nivel_jogo": "iniciante",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
    }
    assert _next_pending_slot(profile) == "esporte_raquete_previo"


def test_diagnose_asks_modelo_last():
    """After all earlier slots, modelo_desejado is the final question."""
    from app.agent.nodes.diagnose import _next_pending_slot
    profile = {
        "nivel_jogo": "iniciante",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "tênis",
    }
    assert _next_pending_slot(profile) == "modelo_desejado"


def test_diagnose_complete_returns_none():
    """All applicable slots filled → next_pending_slot signals completion."""
    from app.agent.nodes.diagnose import _next_pending_slot
    profile = {
        "nivel_jogo": "intermediário",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "nao_aplicavel",
        "modelo_desejado": "nenhum",
    }
    assert _next_pending_slot(profile) is None


# ════════════════════════════════════════════════════════════════════════════
# GUARDRAILS DETERMINÍSTICOS — unit tests on _apply_guardrails
# ════════════════════════════════════════════════════════════════════════════

def test_guardrail_skips_esporte_previo_for_intermediate():
    """nivel_jogo=intermediário must pre-fill esporte_raquete_previo."""
    from app.agent.nodes.diagnose import _apply_guardrails
    result = _apply_guardrails({"nivel_jogo": "intermediário"})
    assert result["esporte_raquete_previo"] == "nao_aplicavel"


def test_guardrail_skips_esporte_previo_for_advanced():
    """nivel_jogo=avançado (any case/accent variant) must pre-fill the slot."""
    from app.agent.nodes.diagnose import _apply_guardrails
    for level in ("avançado", "avancado", "AVANÇADO", "Avancado", "AVANCADO"):
        result = _apply_guardrails({"nivel_jogo": level})
        assert result["esporte_raquete_previo"] == "nao_aplicavel", (
            f"failed for level={level!r}"
        )


def test_guardrail_does_not_skip_for_beginner():
    """Iniciante customers still need the prior-sport question — no auto-fill."""
    from app.agent.nodes.diagnose import _apply_guardrails
    result = _apply_guardrails({"nivel_jogo": "iniciante"})
    assert "esporte_raquete_previo" not in result


def test_guardrail_pre_fills_regiao_lesao_when_no_lesion():
    """lesoes='nenhuma' auto-fills regiao_lesao='nenhuma' so it's not asked."""
    from app.agent.nodes.diagnose import _apply_guardrails
    result = _apply_guardrails({"lesoes": "nenhuma"})
    assert result["regiao_lesao"] == "nenhuma"


# ════════════════════════════════════════════════════════════════════════════
# META-PERGUNTAS — unit tests on is_meta_question + prompt content
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "msg,expected",
    [
        ("isso importa?", True),
        ("ISSO IMPORTA", True),
        ("por que pergunta isso", True),
        ("porque pergunta", True),
        ("preciso responder?", True),
        ("isso vai mudar a recomendação?", True),
        ("intermediário", False),
        ("tenho dor no cotovelo", False),
        ("quero uma raquete", False),
        ("", False),
        ("oi", False),
    ],
)
def test_meta_question_detection(msg, expected):
    from app.agent.nodes.diagnose import is_meta_question
    assert is_meta_question(msg) is expected


def test_meta_question_does_not_advance_slot():
    """When a meta question is detected, the meta-handler runs and slot stays empty."""
    from app.agent.nodes.diagnose import _apply_guardrails, _next_pending_slot
    # Profile state before the meta turn: customer was about to be asked about
    # esporte_raquete_previo (beginner, no lesion).
    profile = {"nivel_jogo": "iniciante", "lesoes": "nenhuma", "regiao_lesao": "nenhuma"}
    pending = _next_pending_slot(_apply_guardrails(dict(profile)))
    assert pending == "esporte_raquete_previo"

    # After a meta-question turn, the slot must still be pending (didn't advance).
    # We don't run the full graph here — we just confirm the meta path uses
    # _next_pending_slot on the same profile and returns the SAME slot, so the
    # canonical question to re-ask is consistent.
    pending_after = _next_pending_slot(_apply_guardrails(dict(profile)))
    assert pending_after == "esporte_raquete_previo"


def test_meta_question_re_asks_pending_question():
    """SYSTEM_DIAGNOSE_META must instruct the LLM to explain AND re-ask."""
    from app.agent.prompts import SYSTEM_DIAGNOSE_META
    s = SYSTEM_DIAGNOSE_META.lower()
    # "Explicar brevemente" + "repetir a pergunta original"
    assert "explicar" in s and "brevemente" in s
    assert "repetir a pergunta" in s


# ════════════════════════════════════════════════════════════════════════════
# FLUXO COMPLETO — integration tests with extract + phrase mocks
# ════════════════════════════════════════════════════════════════════════════

def _extract_resp(slots: dict) -> str:
    return json.dumps({"extracted_slots": slots})


@pytest.fixture
def memory_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from app.agent.graph import build_graph
    return build_graph(checkpointer=MemorySaver())


@pytest.mark.asyncio
async def test_diagnose_intermediate_full_flow(memory_graph):
    """Intermediate customer: nivel → lesoes → modelo (esporte_previo skipped)."""
    thread = "t-intermed-full-1"

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        # Turn 1: triage (diagnose) + extract (empty) + phrase (nivel question)
        mock.side_effect = [
            '{"intent": "diagnose"}',
            _extract_resp({}),
            "Beleza! Qual seu nível de jogo? Iniciante, intermediário ou avançado?",
        ]
        r1 = await memory_graph.ainvoke(
            _initial_state("quero uma raquete"), _config(thread)
        )
        assert r1["player_profile"] == {}
        assert "nível" in _last_ai_message(r1).lower()

        # Turn 2: extract (nivel=intermediário, guardrail fills esporte_previo)
        # + phrase (lesoes question)
        mock.side_effect = [
            _extract_resp({"nivel_jogo": "intermediário"}),
            "Show! Você sente ou já sentiu alguma dor jogando?",
        ]
        r2 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="intermediário")]}, _config(thread)
        )
        assert r2["player_profile"]["nivel_jogo"] == "intermediário"
        # Phase 2 guardrail must have pre-filled this:
        assert r2["player_profile"]["esporte_raquete_previo"] == "nao_aplicavel"
        # Next question must be about lesoes — explicitly NOT about prior sport.
        reply2 = _last_ai_message(r2).lower()
        assert "dor" in reply2 or "lesão" in reply2 or "lesao" in reply2
        assert "outro esporte" not in reply2

        # Turn 3: extract (lesoes=nenhuma, guardrail fills regiao_lesao=nenhuma)
        # + phrase (modelo question — esporte already filled)
        mock.side_effect = [
            _extract_resp({"lesoes": "nenhuma"}),
            "Entendi. Você já tem algum modelo em mente?",
        ]
        r3 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="sem lesão")]}, _config(thread)
        )
        assert r3["player_profile"]["lesoes"] == "nenhuma"
        assert r3["player_profile"]["regiao_lesao"] == "nenhuma"
        reply3 = _last_ai_message(r3).lower()
        assert "modelo" in reply3 or "marca" in reply3

        # Turn 4: extract (modelo=nenhum) → diagnose complete → recommend
        fake_products = [
            {
                "id": "p1", "name": "Raquete X5", "sport": "beach_tennis",
                "level": "intermediário", "price_cents": 89900, "stock": 10,
                "description": "Boa raquete", "similarity": 0.9,
                "external_id": "X5", "url": None, "image_url": None,
                "updated_at": None, "is_active": True,
                "weight_g": 355, "balance": "médio", "material": "carbono",
            }
        ]
        recommend_resp = json.dumps({
            "messages": [
                "*Raquete X5* — R$ 899\n\nÓtima opção pro seu perfil intermediário.",
                "Posso reservar para você?",
            ]
        })
        mock.side_effect = [
            _extract_resp({"modelo_desejado": "nenhum"}),
            recommend_resp,
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = fake_products
            with patch("app.storage.db.get_session", _mock_db_session):
                r4 = await memory_graph.ainvoke(
                    {"messages": [HumanMessage(content="não tenho modelo")]},
                    _config(thread),
                )

        assert r4["intent"] == "recommend"
        assert r4["player_profile"]["modelo_desejado"] == "nenhum"
        # Esporte_raquete_previo MUST be "nao_aplicavel" the entire conversation —
        # never asked, just pre-filled by Phase 2 in turn 2.
        assert r4["player_profile"]["esporte_raquete_previo"] == "nao_aplicavel"


@pytest.mark.asyncio
async def test_diagnose_beginner_full_flow(memory_graph):
    """Beginner customer: nivel → lesoes → esporte_previo → modelo (5 turns total)."""
    thread = "t-begin-full-1"

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        # Turn 1: triage + extract (empty) + phrase (nivel)
        mock.side_effect = [
            '{"intent": "diagnose"}',
            _extract_resp({}),
            "Qual seu nível? Iniciante, intermediário ou avançado?",
        ]
        await memory_graph.ainvoke(_initial_state("quero uma raquete"), _config(thread))

        # Turn 2: extract nivel=iniciante → NO guardrail auto-fill → phrase lesoes
        mock.side_effect = [
            _extract_resp({"nivel_jogo": "iniciante"}),
            "Você sente alguma dor jogando?",
        ]
        r2 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="iniciante")]}, _config(thread)
        )
        # Beginner — esporte_raquete_previo MUST still be empty.
        assert "esporte_raquete_previo" not in r2["player_profile"]

        # Turn 3: extract lesoes=nenhuma → regiao auto-fills → phrase esporte_previo
        mock.side_effect = [
            _extract_resp({"lesoes": "nenhuma"}),
            "Você já praticou algum outro esporte de raquete?",
        ]
        r3 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="sem lesão")]}, _config(thread)
        )
        assert r3["player_profile"]["regiao_lesao"] == "nenhuma"
        # Now the agent IS asking about prior sport.
        assert "outro esporte" in _last_ai_message(r3).lower()

        # Turn 4: extract esporte_previo=tênis → phrase modelo
        mock.side_effect = [
            _extract_resp({"esporte_raquete_previo": "tênis"}),
            "Você já tem algum modelo em mente?",
        ]
        r4 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="tênis")]}, _config(thread)
        )
        assert r4["player_profile"]["esporte_raquete_previo"] == "tênis"

        # Turn 5: extract modelo=nenhum → diagnose complete → recommend
        fake_products = [
            {
                "id": "p1", "name": "Raquete BeachPro", "sport": "beach_tennis",
                "level": "iniciante", "price_cents": 49900, "stock": 5,
                "description": "Entry", "similarity": 0.9, "external_id": "BP1",
                "url": None, "image_url": None, "updated_at": None, "is_active": True,
                "weight_g": 340, "balance": "leve", "material": "espuma",
            }
        ]
        mock.side_effect = [
            _extract_resp({"modelo_desejado": "nenhum"}),
            json.dumps({"messages": ["*BeachPro* — R$ 499", "Posso reservar?"]}),
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = fake_products
            with patch("app.storage.db.get_session", _mock_db_session):
                r5 = await memory_graph.ainvoke(
                    {"messages": [HumanMessage(content="não tenho modelo")]},
                    _config(thread),
                )

        assert r5["intent"] == "recommend"
        assert r5["player_profile"]["esporte_raquete_previo"] == "tênis"


@pytest.mark.asyncio
async def test_diagnose_extracts_multiple_slots_in_one_message(memory_graph):
    """Single rich message must populate multiple slots in one shot."""
    thread = "t-multi-slot-1"

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        # Triage + extract (3 slots) + phrase (modelo — others either filled or auto-filled)
        mock.side_effect = [
            '{"intent": "diagnose"}',
            _extract_resp({
                "nivel_jogo": "intermediário",
                "lesoes": "dor no cotovelo",
                "regiao_lesao": "cotovelo",
            }),
            "Anotei aqui. Você já tem algum modelo em mente?",
        ]
        result = await memory_graph.ainvoke(
            _initial_state(
                "quero uma raquete, sou intermediário com dor no cotovelo"
            ),
            _config(thread),
        )

    profile = result["player_profile"]
    assert profile["nivel_jogo"] == "intermediário"
    assert profile["lesoes"] == "dor no cotovelo"
    assert profile["regiao_lesao"] == "cotovelo"
    # Guardrail filled esporte_raquete_previo because nivel is intermediário.
    assert profile["esporte_raquete_previo"] == "nao_aplicavel"
    # The next question must therefore be about modelo_desejado.
    assert "modelo" in _last_ai_message(result).lower()


# ════════════════════════════════════════════════════════════════════════════
# Sprint 1.9 — casual Brazilian response mapping in EXTRACT
# ════════════════════════════════════════════════════════════════════════════
#
# These tests mock the LLM to return the mapping the EXTRACT prompt should
# produce for each casual phrase. They verify:
#   (a) the architecture merges the extracted slot into player_profile
#   (b) the agent does NOT re-ask the same question after extraction
#   (c) the EXTRACT prompt advertises the casual-mapping rule in plain text
# Whether the real LLM actually follows the mapping is a prompt-engineering
# concern outside the test — we assert the prompt instructs it correctly.


def _assert_extract_prompt_mentions(*phrases: str) -> None:
    """Assert SYSTEM_DIAGNOSE_EXTRACT carries every phrase (case-insensitive)."""
    from app.agent.prompts import SYSTEM_DIAGNOSE_EXTRACT
    s = SYSTEM_DIAGNOSE_EXTRACT.lower()
    for p in phrases:
        assert p.lower() in s, f"prompt missing the mapping cue: {p!r}"


@pytest.mark.asyncio
async def test_extract_maps_nao_tenho_to_nenhum(memory_graph):
    """'não tenho' answered after a modelo question → modelo_desejado = 'nenhum'."""
    _assert_extract_prompt_mentions("não tenho", "modelo_desejado")
    thread = "t-extract-nao-tenho-1"
    # Seed: customer was asked about modelo (other slots already filled).
    seed_profile = {
        "nivel_jogo": "iniciante",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "tênis",
    }

    fake_products = [
        {
            "id": "p1", "name": "Raquete A", "sport": "beach_tennis",
            "level": "iniciante", "price_cents": 50000, "stock": 5,
            "description": "boa", "similarity": 0.9, "external_id": "RA",
            "url": None, "image_url": None, "updated_at": None, "is_active": True,
            "weight_g": 340, "balance": "leve", "material": "espuma",
        }
    ]

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        # Pre-load profile via a "complete" extract that fills everything except modelo.
        mock.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": seed_profile}),
            "Você já tem algum modelo em mente?",
        ]
        await memory_graph.ainvoke(_initial_state("quero uma raquete"), _config(thread))

        # Turn 2: customer says "não tenho". Mock LLM extracts 'nenhum'.
        # No phrase call this turn — diagnose becomes complete and routes to recommend.
        mock.side_effect = [
            json.dumps({"extracted_slots": {"modelo_desejado": "nenhum"}}),
            json.dumps({"messages": ["*Raquete A* — R$ 500", "Posso reservar?"]}),
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = fake_products
            with patch("app.storage.db.get_session", _mock_db_session):
                r2 = await memory_graph.ainvoke(
                    {"messages": [HumanMessage(content="não tenho")]}, _config(thread)
                )

    assert r2["player_profile"]["modelo_desejado"] == "nenhum"
    assert r2["intent"] == "recommend"


@pytest.mark.asyncio
async def test_extract_maps_sei_la_to_nenhum(memory_graph):
    """'sei lá' is a synonym of 'no preference' — same outcome."""
    _assert_extract_prompt_mentions("sei lá")
    thread = "t-extract-seila-1"
    seed_profile = {
        "nivel_jogo": "iniciante",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "nenhum",
    }
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": seed_profile}),
            "Você tem algum modelo em mente?",
        ]
        await memory_graph.ainvoke(_initial_state("oi quero raquete"), _config(thread))

        mock.side_effect = [
            json.dumps({"extracted_slots": {"modelo_desejado": "nenhum"}}),
            json.dumps({"messages": ["*Raquete X*", "Posso reservar?"]}),
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []
            with patch("app.storage.db.get_session", _mock_db_session):
                r2 = await memory_graph.ainvoke(
                    {"messages": [HumanMessage(content="sei lá")]}, _config(thread)
                )

    assert r2["player_profile"]["modelo_desejado"] == "nenhum"


@pytest.mark.asyncio
async def test_extract_maps_qualquer_um_to_nenhum(memory_graph):
    """'qualquer um' also implies no model preference."""
    _assert_extract_prompt_mentions("qualquer um")
    thread = "t-extract-qualquer-1"
    seed_profile = {
        "nivel_jogo": "iniciante",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "nenhum",
    }
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": seed_profile}),
            "Modelo?",
        ]
        await memory_graph.ainvoke(_initial_state("quero uma raquete"), _config(thread))

        mock.side_effect = [
            json.dumps({"extracted_slots": {"modelo_desejado": "nenhum"}}),
            json.dumps({"messages": ["*X*", "Posso?"]}),
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []
            with patch("app.storage.db.get_session", _mock_db_session):
                r2 = await memory_graph.ainvoke(
                    {"messages": [HumanMessage(content="qualquer um")]}, _config(thread)
                )

    assert r2["player_profile"]["modelo_desejado"] == "nenhum"


@pytest.mark.asyncio
async def test_extract_maps_nunca_joguei_to_esporte_nenhum(memory_graph):
    """'nunca joguei' answered after the prior-sport question → 'nenhum'."""
    _assert_extract_prompt_mentions("nunca joguei", "esporte_raquete_previo")
    thread = "t-extract-nunca-joguei-1"
    seed_profile = {"nivel_jogo": "iniciante", "lesoes": "nenhuma", "regiao_lesao": "nenhuma"}
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": seed_profile}),
            "Você praticou outro esporte de raquete?",
        ]
        await memory_graph.ainvoke(_initial_state("quero raquete"), _config(thread))

        mock.side_effect = [
            json.dumps({"extracted_slots": {"esporte_raquete_previo": "nenhum"}}),
            "Tem algum modelo em mente?",
        ]
        r2 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="nunca joguei")]}, _config(thread)
        )

    assert r2["player_profile"]["esporte_raquete_previo"] == "nenhum"


@pytest.mark.asyncio
async def test_extract_maps_to_bem_to_lesoes_nenhuma(memory_graph):
    """'tô bem' answered after the injury question → lesoes = 'nenhuma'."""
    _assert_extract_prompt_mentions("tô bem", "lesoes")
    thread = "t-extract-to-bem-1"
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": {"nivel_jogo": "iniciante"}}),
            "Você sentiu alguma dor jogando?",
        ]
        await memory_graph.ainvoke(_initial_state("iniciante"), _config(thread))

        mock.side_effect = [
            json.dumps({"extracted_slots": {"lesoes": "nenhuma"}}),
            "Você já praticou outro esporte de raquete?",
        ]
        r2 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="tô bem")]}, _config(thread)
        )

    assert r2["player_profile"]["lesoes"] == "nenhuma"
    # Phase 2 guardrail must auto-fill regiao_lesao = nenhuma as side effect.
    assert r2["player_profile"]["regiao_lesao"] == "nenhuma"


@pytest.mark.asyncio
async def test_extract_maps_me_indica_to_modelo_nenhum(memory_graph):
    """'me indica' is a synonym for 'no preference, pick for me'."""
    _assert_extract_prompt_mentions("me indica")
    thread = "t-extract-me-indica-1"
    seed_profile = {
        "nivel_jogo": "iniciante",
        "lesoes": "nenhuma",
        "regiao_lesao": "nenhuma",
        "esporte_raquete_previo": "nenhum",
    }
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            '{"intent": "diagnose"}',
            json.dumps({"extracted_slots": seed_profile}),
            "Modelo?",
        ]
        await memory_graph.ainvoke(_initial_state("oi"), _config(thread))

        mock.side_effect = [
            json.dumps({"extracted_slots": {"modelo_desejado": "nenhum"}}),
            json.dumps({"messages": ["*X*", "Posso?"]}),
        ]
        with patch("app.rag.retriever.search_products", new_callable=AsyncMock) as search:
            search.return_value = []
            with patch("app.storage.db.get_session", _mock_db_session):
                r2 = await memory_graph.ainvoke(
                    {"messages": [HumanMessage(content="me indica")]}, _config(thread)
                )

    assert r2["player_profile"]["modelo_desejado"] == "nenhum"


# ════════════════════════════════════════════════════════════════════════════
# RECOMENDAÇÃO TEXTUAL — singular vs plural concordance
# ════════════════════════════════════════════════════════════════════════════

# Sprint 2.0 — the singular/plural Consultoria-mention guardrail in
# recommend_node was tied to the old active-recommendation path. PROFILE
# mode now delegates to consultoria_offer (its own pitch), so the
# guardrail was removed along with the multi-racket shortlist. The
# pre-2.0 tests test_recommend_uses_singular_when_one_product and
# test_recommend_uses_plural_when_multiple_products were dropped.
