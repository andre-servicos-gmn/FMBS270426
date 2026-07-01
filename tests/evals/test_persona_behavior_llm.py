"""T4 / T5 / T6 — behavior evals (hit the real gpt-4o-mini).

Marked ``llm`` so the deterministic gate excludes them:
    pytest -m "not llm"   # fast gate
    pytest -m llm         # these (needs OPENAI_API_KEY)

Assertions are on BEHAVIOR/KEYWORDS, never exact text. The data layer (catalog +
KB) is patched so only the LLM is live. A MemorySaver + fixed thread_id
accumulates history across turns.
"""
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import get_settings
from tests.evals._helpers import requires_key

pytestmark = pytest.mark.llm


# ── Patched data layer (catalog + KB) ────────────────────────────────────────

_RACKET = {
    "id": 16700000001,
    "name": "Raquete Beach Tennis Drop Shot Conqueror 12K",
    "marca": "Drop Shot",
    "modelo": "Conqueror 12K",
    "price_cents": 129900,
    "categoria_nome": "Raquetes de Praia",
    "is_raquete_praia": True,
    "stock": 7,
    "weight_g": 350,
    "description": "Raquete de beach tennis com carbono 12K e núcleo de EVA.",
    "campos_customizados": {"Material": "Carbono 12K", "Núcleo": "EVA Soft"},
}
_CATALOG = [_RACKET]
_BY_ID = {p["id"]: p for p in _CATALOG}

_RACKET_DOC = {
    "title": "Materiais da raquete: carbono, EVA e furação (o que muda no jogo)",
    "content": (
        "Carbono: vai de 1k a 24k. Quanto mais carbono (24k), mais dura e rígida a "
        "raquete — devolve mais potência mas perde conforto. Menos carbono (1k) é mais "
        "flexível e perdoa erro. "
        "EVA (núcleo): Soft é macio, dá conforto e impulsão; Tech é duro, dá controle e "
        "batida seca; Pro é o meio termo, equilíbrio entre os dois. "
        "Furação: mais furos deixam a raquete mais macia e elástica; menos furos deixam "
        "mais firme e dura."
    ),
    "category": "faq",
}
_CONSULTORIA_DOC = {
    "title": "Consultoria Base Sports",
    "content": (
        "Investimento de R$ 350, gratuita ao comprar a raquete. Etapas: Diagnóstico "
        "(entendemos seu jogo) e Teste em quadra."
    ),
    "category": "store",
}


async def _fake_snapshot():
    return list(_CATALOG)


async def _fake_fetch_by_id(pid):
    return _BY_ID.get(int(pid))


async def _fake_stock(pid):
    return 7


async def _fake_kb(session, query, k=4):
    q = (query or "").lower()
    if any(w in q for w in ("consultoria", "agendar", "diagn", "350")):
        return [_CONSULTORIA_DOC]
    return [_RACKET_DOC]


@pytest.fixture
def patched_layer(monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("STORE_NAME", "Base Sports")
    monkeypatch.setenv("STORE_ADDRESS", "Av. Beira Mar, 1234, Florianópolis")
    monkeypatch.setenv("STORE_HOURS", "seg a sáb, 9h às 19h")
    monkeypatch.setenv("ECOMMERCE_URL", "https://loja.basesports.com.br")
    get_settings.cache_clear()
    with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", _fake_snapshot), patch(
        "app.sync.bling_repo.fetch_product_by_id", _fake_fetch_by_id
    ), patch("app.sync.bling_stock.get_stock", _fake_stock), patch(
        "app.rag.retriever.search_knowledge_base", _fake_kb
    ):
        yield
    get_settings.cache_clear()


def _final_text(result) -> str:
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _graph():
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent.supervisor import build_supervisor_graph

    return build_supervisor_graph(MemorySaver())


# ── T4: welcome ──────────────────────────────────────────────────────────────

@requires_key
@pytest.mark.asyncio
async def test_welcome_first_turn_has_signature(patched_layer):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-welcome"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="oi")], "phone_hash": "h", "thread_id": "eval-welcome"},
        config=cfg,
    )
    final = _final_text(result).lower()
    assert "assistente base" in final, f"first turn missing welcome signature: {final[:200]}"


@requires_key
@pytest.mark.asyncio
async def test_welcome_not_repeated_second_turn(patched_layer):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-welcome2"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="oi")], "phone_hash": "h", "thread_id": "eval-welcome2"},
        config=cfg,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="quero uma raquete")], "phone_hash": "h", "thread_id": "eval-welcome2"},
        config=cfg,
    )
    final = _final_text(result).lower()
    assert "sou o assistente base" not in final, f"welcome repeated on 2nd turn: {final[:200]}"
    assert "bem vindo a base sports" not in final, f"welcome repeated on 2nd turn: {final[:200]}"


# ── T5: humanization (output) ────────────────────────────────────────────────

_TURNS = ["oi", "qual o preço da Conqueror?", "me explica os materiais dela", "tem em estoque?"]


@requires_key
@pytest.mark.asyncio
async def test_replies_have_no_travessao(patched_layer):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-dash"}}
    for msg in _TURNS:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=msg)], "phone_hash": "h", "thread_id": "eval-dash"},
            config=cfg,
        )
        final = _final_text(result)
        assert "—" not in final, f"em-dash in reply to {msg!r}: {final[:200]}"
        assert "–" not in final, f"en-dash in reply to {msg!r}: {final[:200]}"


@requires_key
@pytest.mark.asyncio
async def test_replies_are_short(patched_layer):
    """Heuristic tripwire — store-attendant replies stay short on WhatsApp."""
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-short"}}
    for msg in _TURNS:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=msg)], "phone_hash": "h", "thread_id": "eval-short"},
            config=cfg,
        )
        final = _final_text(result)
        n_lines = len([ln for ln in final.splitlines() if ln.strip()])
        assert len(final) <= 600 or n_lines <= 8, \
            f"reply too long ({len(final)} chars / {n_lines} lines) to {msg!r}: {final[:120]}"


@requires_key
@pytest.mark.asyncio
async def test_no_ai_self_declaration_behaviour(patched_layer):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-ai"}}
    for msg in ["você é um robô?", "qual o preço da Conqueror?"]:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=msg)], "phone_hash": "h", "thread_id": "eval-ai"},
            config=cfg,
        )
        final = _final_text(result).lower()
        for bad in ("assistente virtual", "inteligência artificial", "sou uma ia", "modelo de linguagem"):
            assert bad not in final, f"AI self-declaration in reply to {msg!r}: {final[:200]}"


# ── T6: spec translated to practical impact ──────────────────────────────────

_IMPACT_WORDS = (
    "controle", "potência", "potencia", "conforto", "rígid", "rigid", "dura",
    "macia", "elástic", "elastic", "firme", "absor", "impuls", "perdoa", "flexív", "flexiv",
)


@requires_key
@pytest.mark.asyncio
async def test_racket_description_covers_three_dimensions(patched_layer):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-mat"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="me explica o que muda o carbono, o EVA e a furação numa raquete")],
         "phone_hash": "h", "thread_id": "eval-mat"},
        config=cfg,
    )
    final = _final_text(result).lower()
    assert "carbono" in final, f"description dropped Carbono: {final[:300]}"
    assert "eva" in final, f"description dropped EVA: {final[:300]}"
    assert "fura" in final, f"description dropped Furação: {final[:300]}"


@requires_key
@pytest.mark.asyncio
async def test_terms_translated_to_impact(patched_layer):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-impact"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="o que muda na prática entre mais carbono e menos carbono?")],
         "phone_hash": "h", "thread_id": "eval-impact"},
        config=cfg,
    )
    final = _final_text(result).lower()
    matched = [w for w in _IMPACT_WORDS if w in final]
    assert len(matched) >= 2, f"spec not translated into practical impact (matched={matched}): {final[:300]}"


@requires_key
@pytest.mark.asyncio
async def test_spec_question_pulls_from_kb(patched_layer):
    """A materials question must be grounded — the agent calls a knowledge/detail
    tool before asserting spec, not answered from memory."""
    graph = _graph()
    cfg = {"configurable": {"thread_id": "eval-grounded"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="o que significa carbono 12k e EVA na raquete?")],
         "phone_hash": "h", "thread_id": "eval-grounded"},
        config=cfg,
    )
    tool_names = [
        tc["name"]
        for m in result["messages"] if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    assert any(t in tool_names for t in ("buscar_conhecimento", "detalhes_produto", "buscar_catalogo")), \
        f"materials question answered from memory (tools={tool_names})"


# ── T6: material description must NOT trip the personalized-rec fence ─────────

@requires_key
@pytest.mark.asyncio
async def test_material_description_not_flagged_by_fence():
    """Describing Carbono/EVA/Furação in impact terms is FACTUAL product info,
    not a personalized recommendation — the fence must not flag it (viola=False),
    so the answer is never replaced by the Consultoria pivot."""
    from app.agent.supervisor import _classify_personalized_rec

    contexto = "- o que muda o carbono e o eva numa raquete?"
    resposta = (
        "Mais carbono deixa a raquete mais dura e devolve mais potência; menos carbono é "
        "mais flexível e dá conforto. O EVA Soft é macio e dá conforto; o Tech é mais duro "
        "e dá controle."
    )
    verdict = await _classify_personalized_rec(contexto, resposta)
    assert verdict["viola"] is False, f"factual material description wrongly fenced: {verdict}"
