"""Sprint 2.2 — dossier WhatsApp delivery test suite.

Covers:
- ``send_dossier_to_recipient``: success, skip when no recipient, graceful
  failure, doesn't bubble exceptions to the handoff flow.
- ``format_dossier_for_whatsapp``: omits empty fields, emoji structure,
  PT-BR handoff_reason translation, includes the LLM summary.
- Integration: each of the 4 handoff nodes (handoff, out_of_scope,
  scheduling, product_selection/purchase_closing) calls the pipeline.
- ``summarize_conversation``: LLM-driven, cached per ``(phone_hash, count)``.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.dossier import (
    _clear_summary_cache,
    build_dossier,
    format_dossier_for_whatsapp,
    send_dossier_to_recipient,
    summarize_conversation,
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


def _settings(recipient: str = "5511987654321") -> object:
    """Tiny settings stub with the two fields the dossier module reads."""
    class _S:
        dossier_recipient_phone = recipient

    return _S()


def _base_state(**overrides) -> AgentState:
    state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content="oi")],
        "phone_hash": "doss" + "x" * 60,
        "intent": "handoff",
        "player_profile": {
            "nivel_jogo": "intermediário",
            "lesoes": "tendinite",
            "regiao_lesao": "cotovelo",
            "esporte_raquete_previo": "tênis",
            "modelo_desejado": "BeachPro Carbon X5",
        },
        "recommended_products": [],
        "needs_handoff": True,
        "handoff_reason": "purchase_closing",
        "consultoria_interest": True,
        "customer_name": "Andre",
        "produto_pesquisado": "BeachPro Carbon X5",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ════════════════════════════════════════════════════════════════════════════
# SEND DOSSIER
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_send_dossier_when_config_set():
    """Recipient configured + EvolutionClient.send_text succeeds → returns True."""
    state = _base_state()
    settings = _settings("5511987654321")

    with patch(
        "app.adapters.evolution.EvolutionClient.send_text", new_callable=AsyncMock
    ) as send:
        result = await send_dossier_to_recipient(state, settings)

    assert result is True
    send.assert_called_once()
    phone_arg, text_arg = send.call_args.args
    assert phone_arg == "5511987654321"
    assert "NOVO LEAD" in text_arg
    assert "Andre" in text_arg


@pytest.mark.asyncio
async def test_send_dossier_skips_when_no_config():
    """Empty recipient → no Evolution call, returns False (gracioso)."""
    state = _base_state()
    settings = _settings("")

    with patch(
        "app.adapters.evolution.EvolutionClient.send_text", new_callable=AsyncMock
    ) as send:
        result = await send_dossier_to_recipient(state, settings)

    assert result is False
    send.assert_not_called()


@pytest.mark.asyncio
async def test_send_dossier_logs_on_failure(caplog):
    """Evolution raises → function returns False and logs a warning."""
    import logging

    state = _base_state()
    settings = _settings("5511987654321")

    with caplog.at_level(logging.WARNING, logger="app.agent.dossier"):
        with patch(
            "app.adapters.evolution.EvolutionClient.send_text",
            new_callable=AsyncMock,
        ) as send:
            send.side_effect = RuntimeError("connection refused")
            result = await send_dossier_to_recipient(state, settings)

    assert result is False
    assert "dossier_send_failed" in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_send_dossier_does_not_block_handoff_on_error():
    """Even if Evolution raises, the handoff node still returns its response."""
    from app.agent.nodes.handoff import handoff_node

    state = _base_state(intent="handoff", handoff_reason=None)
    with patch(
        "app.agent.dossier.persist_dossier", new_callable=AsyncMock
    ), patch(
        "app.agent.dossier.summarize_conversation", new_callable=AsyncMock
    ) as summ, patch(
        "app.adapters.evolution.EvolutionClient.send_text",
        new_callable=AsyncMock,
    ) as send:
        summ.return_value = "Resumo qualquer."
        send.side_effect = RuntimeError("evolution down")
        result = await handoff_node(state)

    assert result["needs_handoff"] is True
    assert result["handoff_reason"] == "user_requested"
    assert result["response_blocks"]


# ════════════════════════════════════════════════════════════════════════════
# FORMAT WHATSAPP
# ════════════════════════════════════════════════════════════════════════════

def test_format_omits_empty_fields():
    """Empty / dash fields are omitted line by line (no 'Lesões: —')."""
    dossier = build_dossier(
        _base_state(
            customer_name=None,
            produto_pesquisado=None,
            player_profile={
                "nivel_jogo": "iniciante",
                "lesoes": "nenhuma",
                "regiao_lesao": "nenhuma",
                "esporte_raquete_previo": "nenhum",
                "modelo_desejado": "nenhum",
            },
            handoff_reason="user_requested",
            consultoria_interest=False,
        ),
        summary="Resumo qualquer.",
    )
    text = format_dossier_for_whatsapp(dossier)
    # No empty-data placeholder lines (header em-dash is allowed).
    assert "Cliente: —" not in text
    assert "Lesões: —" not in text
    assert "Modelo desejado: —" not in text
    # No "Lesões:" because the profile says "nenhuma".
    assert "Lesões:" not in text
    # No "Esporte prévio:" because "nenhum"/"nao_aplicavel" suppresses.
    assert "Esporte prévio:" not in text
    # No "Modelo desejado:" because "nenhum" suppresses.
    assert "Modelo desejado:" not in text
    # No "Pesquisou:" because field is None.
    assert "Pesquisou:" not in text
    # No "Cliente:" line at all because the customer name is empty.
    assert "*Cliente:*" not in text


def test_format_includes_emoji_structure():
    """The visual scaffolding (📋, 👤, 🎾, 📝, 🕐) must be present."""
    dossier = build_dossier(_base_state(), summary="Resumo.")
    text = format_dossier_for_whatsapp(dossier)
    for emoji in ("📋", "👤", "🎾", "📝", "🕐"):
        assert emoji in text, f"missing visual anchor {emoji!r}"


@pytest.mark.parametrize(
    "reason,label",
    [
        ("scheduling", "Quer agendar Consultoria"),
        ("purchase_closing", "Quer comprar raquete"),
        ("out_of_scope", "Pergunta fora do escopo"),
        ("user_requested", "Pediu atendimento humano"),
    ],
)
def test_format_translates_handoff_reason(reason, label):
    dossier = build_dossier(
        _base_state(handoff_reason=reason), summary="Resumo qualquer."
    )
    text = format_dossier_for_whatsapp(dossier)
    assert label in text
    # Raw reason must NOT leak into the output.
    assert reason not in text


def test_format_includes_summary():
    summary = "Cliente quer fechar a Carbon X5. Aguardando atendente."
    dossier = build_dossier(_base_state(), summary=summary)
    text = format_dossier_for_whatsapp(dossier)
    assert "Resumo da conversa" in text
    assert summary in text


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION — handoff nodes call the pipeline
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_handoff_node_sends_dossier():
    from app.agent.nodes.handoff import handoff_node

    state = _base_state()
    with patch(
        "app.agent.nodes.handoff.handoff_dossier_pipeline", new_callable=AsyncMock
    ) as pipeline:
        await handoff_node(state)

    pipeline.assert_called_once()
    assert pipeline.call_args.args[0]["handoff_reason"] == "user_requested"


@pytest.mark.asyncio
async def test_purchase_closing_no_longer_sends_dossier():
    """Sprint 2.4 — product_selection emits a short pickup invite (no dossier
    sent for routine purchases). The dossier remains only for actual human
    handoffs (user_requested, out_of_scope, scheduling)."""
    from app.agent.nodes.product_selection import product_selection_node

    state = _base_state(
        messages=[HumanMessage(content="vou de Carbon X5")],
        intent="product_selection",
        recommended_products=[{
            "id": "p1", "name": "Raquete BeachPro Carbon X5",
            "price_cents": 89900, "sport": "beach_tennis",
            "level": "intermediário", "stock": 5, "description": "",
            "similarity": 0.9, "external_id": "Raquete-BeachPro-Carbon-X5",
            "is_active": True, "category": "raquete",
            "weight_g": 350, "balance": "médio", "material": "carbono",
            "url": None, "image_url": None, "updated_at": None,
        }],
        needs_handoff=False, handoff_reason=None,
    )

    # The module no longer imports persist_dossier / send_dossier_to_recipient /
    # summarize_conversation. Patching Evolution + dossier helpers at their
    # source confirms NO calls happen.
    with patch(
        "app.agent.dossier.persist_dossier", new_callable=AsyncMock,
    ) as persist, patch(
        "app.agent.dossier.send_dossier_to_recipient", new_callable=AsyncMock,
    ) as send, patch(
        "app.agent.dossier.summarize_conversation", new_callable=AsyncMock,
    ) as summ:
        result = await product_selection_node(state)

    persist.assert_not_called()
    send.assert_not_called()
    summ.assert_not_called()
    assert result.get("needs_handoff") is not True
    invite = result["response_blocks"][0]
    assert "Raquete BeachPro Carbon X5" in invite


@pytest.mark.asyncio
async def test_scheduling_inquiry_sends_dossier():
    from app.agent.nodes.scheduling_inquiry import scheduling_inquiry_node

    state = _base_state(intent="scheduling_inquiry")
    with patch(
        "app.agent.nodes.scheduling_inquiry.handoff_dossier_pipeline",
        new_callable=AsyncMock,
    ) as pipeline:
        await scheduling_inquiry_node(state)

    pipeline.assert_called_once()
    assert pipeline.call_args.args[0]["handoff_reason"] == "scheduling"


@pytest.mark.asyncio
async def test_out_of_scope_sends_dossier():
    from app.agent.nodes.out_of_scope_handoff import out_of_scope_handoff_node

    state = _base_state(intent="out_of_scope")
    with patch(
        "app.agent.nodes.out_of_scope_handoff.handoff_dossier_pipeline",
        new_callable=AsyncMock,
    ) as pipeline:
        await out_of_scope_handoff_node(state)

    pipeline.assert_called_once()
    assert pipeline.call_args.args[0]["handoff_reason"] == "out_of_scope"


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY (LLM + cache)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_summary_generated_via_llm():
    _clear_summary_cache()
    messages = [
        HumanMessage(content="oi"),
        AIMessage(content="oi! qual seu nome?"),
        HumanMessage(content="Andre"),
        AIMessage(content="show, Andre, em que posso ajudar?"),
        HumanMessage(content="quero a Carbon X5"),
    ]

    with patch(
        "app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock
    ) as llm:
        llm.return_value = "Cliente quer a Carbon X5. Esperando confirmação de estoque."
        out = await summarize_conversation(messages, phone_hash="abc12345")

    assert "Carbon X5" in out
    llm.assert_called_once()


@pytest.mark.asyncio
async def test_summary_caches_per_conversation():
    _clear_summary_cache()
    messages = [HumanMessage(content="oi"), HumanMessage(content="quero")]
    with patch(
        "app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock
    ) as llm:
        llm.return_value = "Resumo."
        s1 = await summarize_conversation(messages, phone_hash="cache-test")
        s2 = await summarize_conversation(messages, phone_hash="cache-test")

    assert s1 == s2
    assert llm.call_count == 1  # 2nd call served from cache


@pytest.mark.asyncio
async def test_summary_cache_invalidated_when_message_count_changes():
    """A longer conversation triggers a fresh LLM call (different cache key)."""
    _clear_summary_cache()
    msgs_short = [HumanMessage(content="oi")]
    msgs_long = [HumanMessage(content="oi"), HumanMessage(content="quero")]
    with patch(
        "app.adapters.openai_client.OpenAIClient.chat", new_callable=AsyncMock
    ) as llm:
        llm.side_effect = ["resumo curto", "resumo maior"]
        s1 = await summarize_conversation(msgs_short, phone_hash="invtest")
        s2 = await summarize_conversation(msgs_long, phone_hash="invtest")

    assert s1 != s2
    assert llm.call_count == 2
