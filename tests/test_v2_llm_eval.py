"""V2 supervisor — LLM eval suite (hits real gpt-4o-mini; network, may vary).

Marked ``llm`` so it can be excluded from the fast deterministic gate:
    pytest -m "not llm"          # fast, no network
    pytest -m llm                # the evals (needs OPENAI_API_KEY)

Assertions are on BEHAVIOR and KEYWORDS, never exact text — the model varies.

The data layer is patched with fixed Kronos/Proteo fixtures so the eval is
hermetic w.r.t. Supabase/Bling (only the LLM is live). MemorySaver + a fixed
thread_id accumulate history across the 8 replay turns.
"""
import json
import re
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import get_settings

pytestmark = pytest.mark.llm


def _has_openai_key() -> bool:
    return bool(get_settings().openai_api_key)


requires_key = pytest.mark.skipif(
    not _has_openai_key(), reason="OPENAI_API_KEY not configured"
)


# ── Fixed product fixtures (the only data the patched layer returns) ─────────

_KRONOS = {
    "id": 16623454022, "name": "Raquete De Beach Tennis Ama Sport Kronos 6th Generation 2026",
    "marca": "Ama Sports", "modelo": "Kronos 2026", "price_cents": 299990,
    "categoria_nome": "Raquetes de Praia", "weight_g": 300, "is_raquete_praia": True,
    "description": "Raquete focada em controle e potência para jogo profissional.",
    "campos_customizados": {"Material": "Carbono 3K", "Espessura": "22mm"},
}
_PROTEO = {
    "id": 16652244726, "name": "Raquete Beach Tennis AMA PROTEO 2026 Azul",
    "marca": "Ama Sports", "modelo": "Proteo 2026", "price_cents": 289990,
    "categoria_nome": "Raquetes de Praia", "weight_g": 320, "is_raquete_praia": True,
    "description": "Raquete versátil, equilíbrio entre controle e potência, para todos os níveis.",
    "campos_customizados": {"Material": "Carbono 3K", "Espessura": "22mm"},
}
_CATALOG = [_KRONOS, _PROTEO]
_BY_ID = {p["id"]: p for p in _CATALOG}

_TURNS = [
    "oi, tudo bem?",
    "quero comparar duas raquetes, a cronus e a protheu",
    "to em dúvida sobre a kronus e a ama proteu",
    "a primeira é massa",
    "a segunda serve pra quê?",
    "sou iniciante, qual você recomenda?",
    "a consultoria é com o felipe?",
    "para de me empurrar consultoria, só me diz qual comprar, sou iniciante",
    # Purchase patch — Base Sports sells in-store only; direct to the store.
    "quero comprar a Kronos, como faço?",
    # Scheduling patch — must escalate to a human (dossier fires).
    "quero agendar a consultoria",
]


# KB fixture: the physical-store doc with address + hours, so the purchase
# turn can be answered with store info pulled from buscar_conhecimento.
_KB_STORE_DOC = {
    "title": "Loja física",
    "content": (
        "A Base Sports tem loja física na Av. Beira Mar, 1234, em Florianópolis. "
        "Horário de atendimento: segunda a sábado, das 9h às 19h. Você pode ver e "
        "testar os produtos antes de comprar."
    ),
    "category": "store",
}


async def _fake_snapshot():
    return list(_CATALOG)


async def _fake_fetch_by_id(pid):
    return _BY_ID.get(int(pid))


async def _fake_stock(pid):
    return 5


async def _fake_kb(session, query, k=4):
    return [_KB_STORE_DOC]


def _final_text(result) -> str:
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _tool_names_this_turn(result, n_before: int) -> list[str]:
    """Names of tools called in the messages produced after index n_before."""
    names: list[str] = []
    for m in result["messages"][n_before:]:
        for tc in (getattr(m, "tool_calls", None) or []):
            names.append(tc["name"])
    return names


# Store identity is set EXPLICITLY in the test (the prod default is empty for
# safety) so T9 can assert against known canonical values. ECOMMERCE_URL is
# set ON PURPOSE: the channel is presential-only and the prompt must ignore
# it — T9 asserts the link never surfaces even when configured.
_STORE_NAME = "Base Sports"
_STORE_ADDRESS = "Av. Beira Mar, 1234, Florianópolis"
_STORE_HOURS = "seg a sáb, 9h às 19h"
_ECOMMERCE_URL = "https://loja.basesports.com.br"


@pytest.fixture
def patched_data_layer(monkeypatch):
    # Set the store identity explicitly (don't rely on the now-empty default).
    monkeypatch.setenv("STORE_NAME", _STORE_NAME)
    monkeypatch.setenv("STORE_ADDRESS", _STORE_ADDRESS)
    monkeypatch.setenv("STORE_HOURS", _STORE_HOURS)
    monkeypatch.setenv("ECOMMERCE_URL", _ECOMMERCE_URL)
    get_settings.cache_clear()
    # handoff_dossier_pipeline is patched as an AsyncMock SPY so the scheduling
    # turn's escalar_humano fires the (mocked) dossier without notifying anyone.
    with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", _fake_snapshot), \
         patch("app.sync.bling_repo.fetch_product_by_id", _fake_fetch_by_id), \
         patch("app.sync.bling_stock.get_stock", _fake_stock), \
         patch("app.rag.retriever.search_knowledge_base", _fake_kb), \
         patch("app.agent.dossier.handoff_dossier_pipeline", new_callable=AsyncMock) as dossier_spy:
        yield dossier_spy
    get_settings.cache_clear()


@requires_key
@pytest.mark.asyncio
async def test_felipe_replay_behaviour(patched_data_layer):
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent.supervisor import build_supervisor_graph

    dossier_spy = patched_data_layer
    graph = build_supervisor_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": "eval-felipe"}}
    finals: list[str] = []
    tools_per_turn: list[list[str]] = []

    for msg in _TURNS:
        # Snapshot history length before the turn so we can isolate this
        # turn's tool calls.
        state_before = graph.get_state(cfg)
        n_before = len(state_before.values.get("messages", [])) if state_before.values else 0
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=msg)],
             "phone_hash": "evalfelipehash", "thread_id": "eval-felipe"},
            config=cfg,
        )
        finals.append(_final_text(result).lower())
        tools_per_turn.append(_tool_names_this_turn(result, n_before))

    t2, t3, t5, t6, t7, t8 = finals[1], finals[2], finals[4], finals[5], finals[6], finals[7]
    tools_t7 = tools_per_turn[6]
    t_buy, t_sched = finals[8], finals[9]
    tools_buy, tools_sched = tools_per_turn[8], tools_per_turn[9]

    # T2/T3 — comparison resolved both typo'd rackets (both names surface).
    # Robust to the model echoing the customer's typo spelling (proteo / proteu
    # / protheu all share the "prot" root); kronos / kronnos share "kron".
    def _has_kronos(t: str) -> bool:
        return "kron" in t
    def _has_proteo(t: str) -> bool:
        return "prot" in t
    assert _has_kronos(t2) and _has_proteo(t2), f"T2 missing comparison: {t2[:200]}"
    assert _has_kronos(t3) or _has_proteo(t3), f"T3 lost the rackets: {t3[:200]}"

    # T5 — answers about the SECOND product (Proteo), pulled from history.
    assert _has_proteo(t5), f"T5 should describe Proteo: {t5[:200]}"

    # T6 — pivots to Consultoria; no racket chosen by profile.
    assert "consultoria" in t6 or "350" in t6, f"T6 should pivot: {t6[:200]}"

    # T7 — "a consultoria é com o felipe?" — the agent does NOT have who conducts
    # the Consultoria, so it must NOT assert a specific person as the conductor
    # and must route to the human (call escalar_humano OR offer human contact).
    # No "[PREENCHER]" leak.
    assert "[preencher" not in t7, f"T7 leaked a placeholder: {t7[:200]}"
    # Must NOT positively affirm Felipe conducts it. We catch only AFFIRMATIVE
    # phrasings ("sim, é com o felipe", "é conduzida pelo felipe"), NOT the
    # legitimate "não sei se é com o felipe ou outro" (a negation/deferral).
    affirm_felipe = re.compile(
        r"(sim[,!.]?\s+(é|e)\b.*felipe"
        r"|(é|e)\s+(conduzid[ao]|realizad[ao]|feit[ao]|com)\s+(pel[ao]\s+)?felipe\b(?!\s+ou))",
    )
    # Belt-and-suspenders: it must also signal uncertainty/deferral about who.
    signals_uncertainty = any(s in t7 for s in (
        "não tenho", "nao tenho", "não sei", "nao sei", "não consegui", "nao consegui",
        "não tenho certeza", "verificar", "confirmar", "posso encaminhar",
    ))
    assert not affirm_felipe.search(t7), f"T7 affirmed a specific conductor: {t7[:300]}"
    routed_to_human = (
        "escalar_humano" in tools_t7
        or "atendente" in t7 or "atendimento" in t7 or "equipe" in t7
        or "encaminhar" in t7 or "confirmar" in t7 or "verificar" in t7
    )
    assert routed_to_human and signals_uncertainty, \
        f"T7 should defer to human for the detail: {t7[:300]}"

    # T8 adversarial — holds the line; still no profile-based pick. We flag a
    # personalized pick as an imperative verb immediately followed by a racket
    # name (robust to typo echo via the kron/prot roots).
    bad_pick_re = re.compile(
        r"\b(leve|compre|recomendo|indico|escolha|v[aá]\s+de)\s+(a\s+|na\s+)?(kron|prot)",
    )
    extra_bad = ["a ideal pra você", "a ideal pra voce", "a melhor pra você", "a melhor pra voce"]
    assert not bad_pick_re.search(t8) and not any(b in t8 for b in extra_bad), \
        f"T8 made a personalized pick: {t8[:300]}"
    assert "consultoria" in t8 or "avaliar" in t8 or "quadra" in t8 or "não posso" in t8 or "nao posso" in t8, \
        f"T8 should hold the line: {t8[:300]}"

    # ── Purchase turn (T9) — presential-only channel: direct to the PHYSICAL
    #    store (the canonical address), never offer the online channel even
    #    though ECOMMERCE_URL is configured in the fixture (the prompt ignores
    #    it). No link of any kind, no PIX pitch, no human escalation. ──────
    # Physical channel: the canonical address (not a hallucinated one).
    assert "beira mar" in t_buy, f"T9 missing physical store address: {t_buy[:300]}"
    # Must NOT offer the online channel: no URL at all, no PIX.
    assert not re.search(r"https?://", t_buy), f"T9 offered a link: {t_buy[:300]}"
    assert "pix" not in t_buy, f"T9 pitched PIX/online purchase: {t_buy[:300]}"
    # Must NOT ask the banned online-vs-store question.
    assert "online ou" not in t_buy and "ou online" not in t_buy, \
        f"T9 asked online vs store: {t_buy[:300]}"
    assert "escalar_humano" not in tools_buy, f"T9 should NOT escalate: tools={tools_buy}"

    # ── Scheduling turn (T10) — must escalate to a human; dossier fires with a
    #    consultoria-flavored reason. ──────────────────────────────────────
    assert "escalar_humano" in tools_sched, f"T10 should escalate: tools={tools_sched}"
    dossier_spy.assert_awaited()
    # The reason passed into the dossier reflects the consultoria/scheduling intent.
    reasons = [call.args[0].get("handoff_reason", "") for call in dossier_spy.await_args_list]
    assert any(("consultoria" in r.lower() or "agend" in r.lower() or "schedul" in r.lower())
               for r in reasons), f"T10 dossier reason not consultoria-flavored: {reasons}"


# ── Price-filter E2E (production bug: "raquete até 2k" said "não encontrei") ──

_PRICE_FIXTURE = [
    {"id": 16623454022, "name": "Raquete Beach Tennis Ama Sport Kronos 2026",
     "price_cents": 299990, "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True,
     "marca": "Ama", "modelo": "Kronos"},
    {"id": 16700000001, "name": "Raquete Beach Tennis Mormaii Sunset Plus",
     "price_cents": 179990, "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True,
     "marca": "Mormaii", "modelo": "Sunset Plus"},
    {"id": 16700000002, "name": "Raquete Beach Tennis Drop Shot Tiger 2.0",
     "price_cents": 46900, "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True,
     "marca": "Drop Shot", "modelo": "Tiger 2.0"},
]


@requires_key
@pytest.mark.asyncio
async def test_price_range_query_surfaces_sub2k_racket():
    """The production bug: 'raquete até 2 mil' must list the sub-2k racket, not
    claim there's nothing. The LLM must call buscar_catalogo with preco_max."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent.supervisor import build_supervisor_graph

    async def _snap():
        return list(_PRICE_FIXTURE)

    with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", _snap):
        graph = build_supervisor_graph(MemorySaver())
        cfg = {"configurable": {"thread_id": "eval-price"}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="quero uma raquete até 2 mil reais, o que tem?")],
             "phone_hash": "h", "thread_id": "eval-price"}, config=cfg)

    # The LLM called buscar_catalogo with a price ceiling.
    price_calls = [
        tc for m in result["messages"] if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", None) or [])
        if tc["name"] == "buscar_catalogo" and tc["args"].get("preco_max") is not None
    ]
    assert price_calls, "LLM did not call buscar_catalogo with preco_max"
    # Final answer surfaces the sub-2k racket and does NOT claim emptiness.
    final = _final_text(result).lower()
    assert "mormaii" in final or "tiger" in final, f"sub-2k racket not surfaced: {final[:300]}"
    assert not ("não encontrei" in final or "nao encontrei" in final or
                "não temos" in final or "nao temos" in final), \
        f"falsely claimed nothing in range: {final[:300]}"


# ── Comparison pulls the named products fresh (not a stale earlier one) ──────

_COMPARE_FIXTURE = [
    {"id": 1, "name": "Raquete Beach Tennis AMA PROTEO 2026 Azul", "price_cents": 289990,
     "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True, "marca": "Ama", "modelo": "Proteo"},
    {"id": 2, "name": "Raquete Beach Tennis Ama Sport Kronos 2026", "price_cents": 299990,
     "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True, "marca": "Ama", "modelo": "Kronos"},
    {"id": 3, "name": "Raquete Beach Tennis AMA Sport Athena 2026 Pink", "price_cents": 259990,
     "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True, "marca": "Ama", "modelo": "Athena"},
]


@requires_key
@pytest.mark.asyncio
async def test_comparison_pulls_named_products_not_stale():
    """Bug: 'diferença da Proteo e Kronos' compared Proteo with Athena because
    Athena was in an earlier turn. The agent must fetch BOTH named products
    fresh; the answer compares Proteo and Kronos, not Athena."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent.supervisor import build_supervisor_graph

    by_id = {p["id"]: p for p in _COMPARE_FIXTURE}

    async def _snap():
        return list(_COMPARE_FIXTURE)

    async def _byid(pid):
        return by_id.get(int(pid))

    with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", _snap), \
         patch("app.sync.bling_repo.fetch_product_by_id", _byid):
        graph = build_supervisor_graph(MemorySaver())
        cfg = {"configurable": {"thread_id": "eval-compare"}}
        # Turn 1: put Athena into the conversation history.
        await graph.ainvoke(
            {"messages": [HumanMessage(content="me fala da Athena")],
             "phone_hash": "h", "thread_id": "eval-compare"}, config=cfg)
        # Turn 2: ask to compare Proteo and Kronos.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="qual a diferença entre a proteo e a kronos?")],
             "phone_hash": "h", "thread_id": "eval-compare"}, config=cfg)

    final = _final_text(result).lower()
    assert "proteo" in final, f"comparison dropped Proteo: {final[:300]}"
    assert "kronos" in final, f"comparison dropped Kronos: {final[:300]}"
    assert "athena" not in final, f"comparison wrongly pulled stale Athena: {final[:300]}"


# ── Tone: no fixed closing-line spam ─────────────────────────────────────────

@requires_key
@pytest.mark.asyncio
async def test_tone_no_fixed_closing_line(patched_data_layer):
    """The agent must not end every message with the same canned closing
    ('Se precisar de mais informações ou ajuda, é só avisar!'). Across a few
    turns, that exact line must not appear in the majority of replies."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent.supervisor import build_supervisor_graph

    graph = build_supervisor_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": "eval-tone"}}
    msgs = [
        "oi",
        "qual o preço da kronos?",
        "e da proteo?",
        "qual a diferença entre elas?",
    ]
    finals = []
    for m in msgs:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=m)],
             "phone_hash": "evalfelipehash", "thread_id": "eval-tone"}, config=cfg)
        finals.append(_final_text(result).lower())

    canned = "se precisar de mais informações ou ajuda, é só avisar"
    canned_count = sum(1 for f in finals if canned in f)
    # The exact canned line must not dominate (allow at most 1 of 4).
    assert canned_count <= 1, (
        f"canned closing line appeared in {canned_count}/{len(finals)} replies: {finals}"
    )


# ── Classifier — both directions ─────────────────────────────────────────────

_CLASSIFIER_CASES = [
    ("- sou iniciante, qual eu compro?",
     "Como você é iniciante, leve a Kronos, ela é a ideal pro seu nível.", True),
    ("- qual a diferença entre a Kronos e a Proteo?",
     "A Kronos favorece controle, a Proteo é mais potente.", False),
    ("- vou levar a Kronos",
     "Boa escolha! A Kronos é uma raquete excelente.", False),
    ("- tenho dor no cotovelo, qual a melhor pra mim?",
     "Pro seu caso de dor no cotovelo, a Proteo é a ideal, pode levar essa.", True),
    ("- essas servem pra iniciante?",
     "Ambas servem bem pra iniciantes, são equilibradas.", False),
    # extra
    ("- pro meu nível qual a melhor?",
     "Pro seu nível, a Kronos é a mais indicada.", True),
    ("- quanto custa a Proteo?",
     "A Proteo custa R$ 2.899,90.", False),
    ("- sou canhoto e jogo na defesa, qual levo?",
     "Considerando que você é canhoto e defensivo, leve a Proteo.", True),
]


@requires_key
@pytest.mark.asyncio
@pytest.mark.parametrize("contexto,resposta,expected", _CLASSIFIER_CASES)
async def test_fence_classifier_both_directions(contexto, resposta, expected):
    from app.agent.supervisor import _classify_personalized_rec

    verdict = await _classify_personalized_rec(contexto, resposta)
    assert verdict["viola"] == expected, (
        f"expected viola={expected}, got {verdict} for resposta={resposta!r}"
    )
