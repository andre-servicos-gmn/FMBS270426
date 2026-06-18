"""Integration tests for the LangGraph agent.

All external I/O (OpenAI, DB, retriever) is mocked so tests run in-process
with no API keys or live services required.

Strategy
--------
- patch OpenAIClient.chat at the class level so all node instances share the mock.
- use side_effect lists to feed different responses to different LLM calls
  (triage always first, then the node that handles the routed intent).
- patch app.storage.db.get_session for the handoff node.
- patch app.rag.retriever.search_products for the recommend node.
- each test builds a fresh graph (MemorySaver is in-process, cheap to create).
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state import AgentState

# ── helpers ───────────────────────────────────────────────────────────────────

def _initial_state(message: str, phone_hash: str = "deadbeef" * 8) -> AgentState:
    """Full initial state for the first invocation of a thread."""
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
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m.content
    return ""


@asynccontextmanager
async def _mock_db_session():
    """Async context manager that yields a no-op mock session."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
    session.commit = AsyncMock()
    yield session


# ── smalltalk ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_smalltalk_returns_friendly_response(memory_graph):
    # Sprint 2.4 — first interaction now uses a canned brand greeting
    # ("Bem-vindo à Base Sports"). Only triage hits the LLM on this turn.
    side_effects = ['{"intent": "smalltalk"}']
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = side_effects
        result = await memory_graph.ainvoke(_initial_state("oi tudo bem"), _config("t-smalltalk-1"))

    assert result["intent"] == "smalltalk"
    text = _last_ai_message(result)
    assert "Base Sports" in text  # brand greeting
    assert "nome" in text.lower()
    assert mock.call_count == 1


# ── FAQ ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_faq_returns_answer(memory_graph):
    side_effects = [
        '{"intent": "faq"}',
        "O prazo de entrega é de 5 a 7 dias úteis.",
    ]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = side_effects
        result = await memory_graph.ainvoke(
            _initial_state("qual o prazo de entrega?"), _config("t-faq-1")
        )

    assert result["intent"] == "faq"
    assert "prazo" in _last_ai_message(result).lower()
    assert result["needs_handoff"] is False


@pytest.mark.asyncio
async def test_faq_handoff_marker_sets_flag_and_is_stripped(memory_graph):
    """[HANDOFF] marker must set needs_handoff=True and not appear in the reply."""
    side_effects = [
        '{"intent": "faq"}',
        "Não tenho essa informação, mas um atendente pode ajudar! [HANDOFF]",
    ]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = side_effects
        result = await memory_graph.ainvoke(
            _initial_state("quero devolver meu pedido"), _config("t-faq-handoff-1")
        )

    assert result["needs_handoff"] is True
    assert result["handoff_reason"] == "faq_escalation"
    assert "[HANDOFF]" not in _last_ai_message(result)
    assert "atendente" in _last_ai_message(result)


# ── diagnose ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
# NOTE: Sprint 1.8 deletion. The Sprint 1.2 / 1.5 integration tests that ran
# the diagnose end-to-end with a single LLM mock (`updated_profile` + `next_message`
# + `complete`) no longer match production — each turn now calls the LLM twice
# (extract + phrase). The replacements live in tests/test_diagnose_flow.py with
# updated mock contracts.


# ── Sprint 1.4 — close prompt injects store info ─────────────────────────────

def _last_system_prompt(mock) -> str:
    """Return the system prompt passed to the most recent OpenAIClient.chat call."""
    last_call = mock.call_args_list[-1]
    return last_call.kwargs.get("system") or last_call.args[1]


# NOTE: Sprint 1.8 deletion. test_diagnose_complete_routes_to_recommend and
# test_recommend_intent_from_triage_goes_through_diagnose both relied on the
# old "1 LLM call = full diagnose answer" contract. They are replaced by
# integration tests in tests/test_diagnose_flow.py that mock the extract +
# phrase pair per turn.


# ── handoff ───────────────────────────────────────────────────────────────────


# ── triage unknown intent falls back to smalltalk ─────────────────────────────

@pytest.mark.asyncio
async def test_unknown_intent_falls_back_to_smalltalk(memory_graph):
    """If triage returns garbage JSON the router must fall back to smalltalk."""
    side_effects = [
        "isso não é um json válido",                     # triage → fallback to smalltalk
        "Olá! Pode me perguntar sobre beach tennis!",    # smalltalk node
    ]
    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        mock.side_effect = side_effects
        result = await memory_graph.ainvoke(
            _initial_state("???"), _config("t-fallback-1")
        )

    assert result["intent"] == "smalltalk"
    assert _last_ai_message(result) != ""


# ── multi-turn FAQ conversation ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_faq_turns_on_same_thread(memory_graph):
    """Messages from two separate FAQ questions accumulate in the same thread."""
    thread = "t-faq-multi-1"

    with patch("app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock) as mock:
        # First question
        mock.side_effect = ['{"intent": "faq"}', "Aceitamos cartão, PIX e boleto."]
        r1 = await memory_graph.ainvoke(_initial_state("quais formas de pagamento?"), _config(thread))

        # Second question
        mock.side_effect = ['{"intent": "faq"}', "A garantia é de 90 dias."]
        r2 = await memory_graph.ainvoke(
            {"messages": [HumanMessage(content="qual a garantia?")]}, _config(thread)
        )

    assert "pagamento" in _last_ai_message(r1).lower() or "cartão" in _last_ai_message(r1).lower()
    assert "garantia" in _last_ai_message(r2).lower() or "90" in _last_ai_message(r2).lower()
    # Thread accumulated all messages
    assert len(r2["messages"]) > len(r1["messages"])


# ── Sprint 1.5/1.8 — protected slots + canonical questions ──────────────────
#
# After Sprint 1.8 the diagnose became a Python-driven slot machine. The
# protection of forbidden slots (orcamento, frequência, estilo, etc.) is now
# enforced by the absence of those slots from QUESTION_TEMPLATES — there's
# physically no canned question to ask. The EXTRACT prompt still captures
# them silently if the customer volunteers the value.

def test_diagnose_does_not_ask_about_budget():
    """orcamento is not a question the agent asks — but is captured if mentioned."""
    from app.agent.prompts import QUESTION_TEMPLATES, SYSTEM_DIAGNOSE_EXTRACT
    assert "orcamento" not in QUESTION_TEMPLATES
    # Extract prompt still mentions it as a spontaneous-capture slot.
    assert "orcamento" in SYSTEM_DIAGNOSE_EXTRACT.lower()
    # And the PROIBIDOS label must be present so future readers know the rule.
    assert "proibido" in SYSTEM_DIAGNOSE_EXTRACT.lower()


def test_diagnose_does_not_ask_about_frequency():
    from app.agent.prompts import QUESTION_TEMPLATES, SYSTEM_DIAGNOSE_EXTRACT
    assert "frequencia_pratica" not in QUESTION_TEMPLATES
    assert "frequencia_pratica" in SYSTEM_DIAGNOSE_EXTRACT.lower()


def test_diagnose_does_not_ask_about_playstyle():
    from app.agent.prompts import QUESTION_TEMPLATES, SYSTEM_DIAGNOSE_EXTRACT
    assert "estilo_jogo" not in QUESTION_TEMPLATES
    assert "estilo_jogo" in SYSTEM_DIAGNOSE_EXTRACT.lower()


def test_diagnose_assumes_beach_tennis_by_default():
    """The EXTRACT prompt must NOT extract esporte_praticado=beach without explicit signal."""
    from app.agent.prompts import SYSTEM_DIAGNOSE_EXTRACT
    s = SYSTEM_DIAGNOSE_EXTRACT.lower()
    assert "default" in s and "beach tennis" in s
    # The rule about only extracting padel on explicit signal must be present.
    assert "padel" in s and ("explicit" in s or "sinal" in s)


def test_diagnose_confirms_padel_when_client_hints():
    """EXTRACT prompt must mention padel cues so the LLM can pick them up."""
    from app.agent.prompts import SYSTEM_DIAGNOSE_EXTRACT
    s = SYSTEM_DIAGNOSE_EXTRACT.lower()
    assert "pala" in s
    assert "joguei padel" in s


def test_diagnose_asks_prior_sport_when_beginner():
    """The slot must be in QUESTION_TEMPLATES so beginners get asked about it."""
    from app.agent.prompts import QUESTION_TEMPLATES
    q = QUESTION_TEMPLATES["esporte_raquete_previo"].lower()
    assert "outro esporte de raquete" in q


def test_diagnose_skips_prior_sport_when_intermediate():
    """Deterministic guardrail auto-fills the slot for intermediário."""
    from app.agent.nodes.diagnose import _apply_guardrails
    result = _apply_guardrails({"nivel_jogo": "intermediário"})
    assert result["esporte_raquete_previo"] == "nao_aplicavel"


def test_diagnose_skips_prior_sport_when_advanced():
    """Deterministic guardrail also covers all accent/case variants of avançado."""
    from app.agent.nodes.diagnose import _apply_guardrails
    for level in ("avançado", "avancado", "AVANÇADO", "Avancado"):
        result = _apply_guardrails({"nivel_jogo": level})
        assert result["esporte_raquete_previo"] == "nao_aplicavel", level


# Tests 8-9: recommend behaviour after Sprint 1.5 changes.

def test_recommend_mentions_consultoria_in_final_message():
    """Built recommend prompt MUST instruct LLM to mention the Consultoria."""
    from app.agent.prompts import build_recommend_prompt

    class _FakeSettings:
        consultoria_enabled = True

    prompt = build_recommend_prompt(_FakeSettings())
    assert "Consultoria Base Sports" in prompt
    assert "teste em quadra" in prompt or "testa em quadra" in prompt
    # When disabled, mention must be suppressed.

    class _FakeSettingsOff:
        consultoria_enabled = False

    prompt_off = build_recommend_prompt(_FakeSettingsOff())
    assert "NÃO é oferecida" in prompt_off or "Não mencione consultoria" in prompt_off


# NOTE: Sprint 1.8 — the pre-fill guardrail tests moved to test_diagnose_flow.py
# as direct unit tests on _apply_guardrails. The old integration form (running
# the full graph with the legacy single-call diagnose mock) no longer matches
# production, which now calls extract + phrase per turn.


def test_diagnose_handles_meta_question_and_re_asks():
    """SYSTEM_DIAGNOSE_META instructs explain + re-ask of the pending question."""
    from app.agent.prompts import SYSTEM_DIAGNOSE_META
    s = SYSTEM_DIAGNOSE_META.lower()
    # The three obligatory behaviours.
    assert "explicar brevemente" in s or "explicar" in s and "1 frase" in s
    assert "repetir a pergunta" in s or "repetir" in s
    # Tom must be friendly and Brazilian (no markdown).
    assert "sem markdown" in s


def test_recommend_uses_formatted_blocks():
    """SYSTEM_RECOMMEND must instruct visual formatting (title in *bold*,
    'Ideal pra:' tail, italic placeholder). Sprint 1.13 made the template
    leaner — the placeholder names are now '*Nome da Raquete*' and
    '_perfil curto_' but the structural rules still apply.
    """
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND
    # Bold-name placeholder for the racket title.
    assert "*Nome da Raquete*" in s
    # Tail line and italic placeholder for the profile pitch.
    assert "Ideal pra:" in s
    assert "_perfil curto_" in s
    # Forbid WhatsApp-incompatible markdown.
    assert "tabelas" in s.lower() or "##" in s


def test_recommend_consultoria_uses_strategic_positioning():
    """The default consultoria block must reflect the new strategic positioning:
    'especificamente'/'perfil geral' contrast with 'Consultoria Base Sports'
    in bold (Sprint 2.6.9 brand cleanup). The legacy phrase 'se quiser ter
    ainda mais certeza' may still appear in the prompt — but ONLY inside a
    'NÃO use' negative example block, not as part of the model phrase to emit.
    """
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND
    s_low = s.lower()

    # New positioning vocabulary present.
    assert "especificamente" in s_low or "personalizada" in s_low
    # Contrast between "perfil geral" and "specific"-style framing.
    assert "perfil geral" in s_low
    # Brand name highlighted (*Consultoria Base Sports*)
    assert "*Consultoria Base Sports*" in s

    # Legacy phrase is only acceptable inside a "NÃO use" forbid clause.
    if "se quiser ter ainda mais certeza" in s_low:
        # Find the surrounding ~150-char window and verify it sits inside a
        # negative-example construct.
        idx = s_low.index("se quiser ter ainda mais certeza")
        window = s_low[max(0, idx - 150):idx]
        assert "não use" in window or "nao use" in window or "desvaloriz" in window, (
            "legacy phrase must only appear in a forbid clause, never as the "
            "recommended phrasing"
        )
