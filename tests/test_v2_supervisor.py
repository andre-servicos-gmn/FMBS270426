"""V2 supervisor — deterministic test suite (no network, no real LLM).

This is the automated safety net for the supervisor architecture built in
Phases 0–2 (behind ``use_v2=False``). Four batteries:

1. buscar_catalogo typo tolerance — proves difflib+phonetic folding
   generalizes beyond the two Phase-1 examples.
2. PII masking — masked in user/assistant/system, NOT in ToolMessage; the
   11-digit collision regression (product id survives, CPF is masked).
3. WhatsApp sanitization — bold/id/UUID removed, prices+years preserved,
   rewrite is by-id (update not append).
4. escalar_humano — invokes the dossier pipeline (delivery mocked).

All marked ``deterministic`` and run without touching production or the flag.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

pytestmark = pytest.mark.deterministic


# ════════════════════════════════════════════════════════════════════════════
# 1. buscar_catalogo typo battery — difflib + phonetic folding
# ════════════════════════════════════════════════════════════════════════════

# In-memory fixture catalog: real product names + noise that must NOT match.
_FIXTURE_CATALOG = [
    {"id": 16623454022, "name": "Raquete De Beach Tennis Ama Sport Kronos 6th Generation 2026",
     "marca": "Ama Sports", "modelo": "Kronos 2026", "price_cents": 299990, "categoria_nome": "Raquetes de Praia"},
    {"id": 16522624615, "name": "Raquete Beach Tennis Ama Sport Kronos 2025 Hugo Russo Capa",
     "marca": "AMA", "modelo": "Kronos", "price_cents": 239990, "categoria_nome": "Raquetes de Praia"},
    {"id": 16652244726, "name": "Raquete Beach Tennis AMA PROTEO 2026 Azul",
     "marca": "Ama Sports", "modelo": "Proteo 2026", "price_cents": 289990, "categoria_nome": "Raquetes de Praia"},
    {"id": 16587972391, "name": "Raquete Beach Tennis Fobel Macaw Gustavo Russo 12k Eva Soft",
     "marca": "Fobel", "modelo": "Macaw", "price_cents": 249990, "categoria_nome": "Raquetes de Praia"},
    {"id": 16161363602, "name": "Película Vidro Tela Drone Dji Mavic Mini 3 e 4 Pro Rc Protetora",
     "marca": "Dji", "modelo": "Mavic", "price_cents": 9801, "categoria_nome": "Acessorios"},
    {"id": 16376771838, "name": "Manguito Esquerdo Solar Drop Shot Proteção UV Beach Tennis",
     "marca": "Drop Shot", "modelo": "Manguito", "price_cents": 8900, "categoria_nome": "Manguitos"},
]


async def _run_buscar(consulta: str) -> list[dict]:
    """Call buscar_catalogo with the fixture catalog patched in."""
    from app.agent import tools_v2

    async def fake_snapshot():
        return list(_FIXTURE_CATALOG)

    with patch.object(tools_v2, "get_catalog_snapshot", fake_snapshot, create=True):
        # get_catalog_snapshot is imported lazily inside the tool; patch the
        # source module function instead.
        with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", fake_snapshot):
            raw = await tools_v2.buscar_catalogo.ainvoke({"consulta": consulta})
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def _names(results: list[dict]) -> list[str]:
    return [r.get("nome", "") for r in results]


# (query, substring that must appear in the TOP-1 result) — typos that should
# resolve to the right product as the best hit.
_TOP1_CASES = [
    ("kronus", "Kronos"),
    ("cronus", "Kronos"),
    ("kronnos", "Kronos"),
    ("Kronos", "Kronos"),
    ("ama proteo", "PROTEO"),
    ("ama proteu", "PROTEO"),
    ("macaw", "Macaw"),
    ("Proteo", "PROTEO"),
]

# Typos that should at least put the right product in the TOP-N (looser).
_TOPN_CASES = [
    ("proteu", "PROTEO"),
    ("protheu", "PROTEO"),
]

# Queries that must NOT surface a racket (pure noise / unrelated).
_NEGATIVE_CASES = [
    "guarda-sol",
    "bicicleta",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query,expect_substr", _TOP1_CASES)
async def test_buscar_catalogo_typo_resolves_top1(query, expect_substr):
    results = await _run_buscar(query)
    assert results, f"{query!r} returned nothing"
    assert expect_substr.lower() in _names(results)[0].lower(), (
        f"{query!r} top-1 was {_names(results)[0]!r}, expected substr {expect_substr!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("query,expect_substr", _TOPN_CASES)
async def test_buscar_catalogo_typo_resolves_topn(query, expect_substr):
    results = await _run_buscar(query)
    assert results, f"{query!r} returned nothing"
    assert any(expect_substr.lower() in n.lower() for n in _names(results)), (
        f"{query!r} top-N missing {expect_substr!r}: {_names(results)}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("query", _NEGATIVE_CASES)
async def test_buscar_catalogo_noise_does_not_match_racket(query):
    results = await _run_buscar(query)
    racket_hits = [n for n in _names(results) if "raquete" in n.lower()]
    assert not racket_hits, f"{query!r} spuriously matched rackets: {racket_hits}"


# ════════════════════════════════════════════════════════════════════════════
# 2. PII masking in the supervisor payload
# ════════════════════════════════════════════════════════════════════════════

def _payload_by_role(messages):
    from app.agent.supervisor import _to_openai_messages
    payload = _to_openai_messages(messages)
    return payload


def test_pii_masked_in_user_and_assistant(monkeypatch):
    monkeypatch.setenv("PII_MASK_ENABLED", "true")
    monkeypatch.setenv("APP_ENV", "production")  # avoid is_clean raising mid-test
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.supervisor import build_system_prompt
    msgs = [
        HumanMessage(content="meu cpf 123.456.789-00 e zap (11) 99999-8888"),
        AIMessage(content="te ligo no (11) 99999-8888"),
    ]
    from langchain_core.messages import SystemMessage
    payload = _payload_by_role([SystemMessage(content=build_system_prompt())] + msgs)

    user = next(p for p in payload if p["role"] == "user")
    assistant = next(p for p in payload if p["role"] == "assistant")
    assert "[CPF]" in user["content"]
    assert "[FONE]" in user["content"]
    assert "123.456.789-00" not in user["content"]
    assert "[FONE]" in assistant["content"]
    get_settings.cache_clear()


def test_system_prompt_not_masked_store_address_survives(monkeypatch):
    """The system message is OUR fixed prompt (store identity) — it must NOT be
    masked, or the canonical store address gets rewritten to [ENDERECO] and the
    model echoes that to the customer. Regression guard for the address fix."""
    monkeypatch.setenv("PII_MASK_ENABLED", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("STORE_ADDRESS", "Av. Beira Mar, 1234, Florianópolis")
    from app.config import get_settings
    get_settings.cache_clear()

    from langchain_core.messages import SystemMessage

    from app.agent.supervisor import build_system_prompt
    payload = _payload_by_role([SystemMessage(content=build_system_prompt())])
    system = next(p for p in payload if p["role"] == "system")
    assert "Beira Mar, 1234" in system["content"]
    assert "[ENDERECO]" not in system["content"]
    get_settings.cache_clear()


def test_system_prompt_empty_address_injects_no_address(monkeypatch):
    """Safety rule: with store_address EMPTY (unconfigured .env), the prompt
    must NOT state any address and must tell the agent to ask the customer to
    confirm it — never assert a false address."""
    monkeypatch.setenv("STORE_NAME", "")
    monkeypatch.setenv("STORE_ADDRESS", "")
    monkeypatch.setenv("STORE_HOURS", "")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.supervisor import build_system_prompt
    prompt = build_system_prompt()
    assert "DADOS FIXOS DA LOJA" not in prompt
    assert "Beira Mar" not in prompt
    # The empty-address branch must instruct asking the customer to confirm.
    assert "confirmar o endereço" in prompt.lower() or "confirme o endereço" in prompt.lower()
    assert "NÃO tem o endereço" in prompt or "não tem o endereço" in prompt.lower()
    get_settings.cache_clear()


def test_system_prompt_configured_address_is_pinned(monkeypatch):
    """With store_address set, the canonical address+hours are pinned in the
    DADOS FIXOS block."""
    monkeypatch.setenv("STORE_NAME", "Base Sports")
    monkeypatch.setenv("STORE_ADDRESS", "Rua Tal, 99, Cidade")
    monkeypatch.setenv("STORE_HOURS", "seg a sex, 8h-18h")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.supervisor import build_system_prompt
    prompt = build_system_prompt()
    assert "DADOS FIXOS DA LOJA" in prompt
    assert "Rua Tal, 99, Cidade" in prompt
    assert "seg a sex, 8h-18h" in prompt
    get_settings.cache_clear()


def test_toolmessage_not_masked_id_survives(monkeypatch):
    """The 11-digit-collision regression: a Bling product id (11 digits) in a
    ToolMessage must survive (NOT become [CPF]); a CPF in a user message must
    still be masked."""
    monkeypatch.setenv("PII_MASK_ENABLED", "true")
    monkeypatch.setenv("APP_ENV", "production")
    from app.config import get_settings
    get_settings.cache_clear()

    msgs = [
        HumanMessage(content="meu cpf 123.456.789-00"),
        ToolMessage(content='[{"id": "16623454022", "nome": "Kronos"}]',
                    tool_call_id="c1", name="buscar_catalogo"),
    ]
    payload = _payload_by_role(msgs)
    user = next(p for p in payload if p["role"] == "user")
    tool = next(p for p in payload if p["role"] == "tool")

    assert "[CPF]" in user["content"]                 # user PII masked
    assert "16623454022" in tool["content"]           # product id preserved
    assert "[CPF]" not in tool["content"]              # NOT mangled
    get_settings.cache_clear()


def test_is_clean_raises_in_dev_if_pii_survives(monkeypatch):
    """In development, a content field that still has PII after masking raises
    rather than leaking. We simulate by disabling the masker's substitution but
    keeping the dev assertion path active."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("PII_MASK_ENABLED", "true")
    from app.config import get_settings
    get_settings.cache_clear()

    # Patch mask_pii to a no-op so masked content still contains the CPF — the
    # dev-only is_clean assertion in _to_openai_messages must then raise.
    with patch("app.agent.supervisor.mask_pii", side_effect=lambda t: t):
        with pytest.raises(ValueError, match="PII leak"):
            _payload_by_role([HumanMessage(content="cpf 123.456.789-00")])
    get_settings.cache_clear()


# ════════════════════════════════════════════════════════════════════════════
# 3. WhatsApp sanitization
# ════════════════════════════════════════════════════════════════════════════

def test_sanitize_strips_bold_and_ids():
    from app.agent.supervisor import _sanitize_for_whatsapp
    out = _sanitize_for_whatsapp("A **Kronos** é boa. ID: 16623454022")
    assert "**" not in out
    assert "16623454022" not in out
    assert "Kronos" in out


def test_sanitize_removes_uuid():
    from app.agent.supervisor import _sanitize_for_whatsapp
    out = _sanitize_for_whatsapp("ref 619aeec0-163d-4f9b-89fb-4ce867f537e6 aqui")
    assert "619aeec0" not in out


def test_sanitize_preserves_prices_and_years():
    from app.agent.supervisor import _sanitize_for_whatsapp
    out = _sanitize_for_whatsapp("A Proteo 2026 custa R$ 2.899,90.")
    assert "R$ 2.899,90" in out
    assert "2026" in out


def test_sanitize_node_rewrites_by_id_no_duplicate():
    from langgraph.graph.message import add_messages

    from app.agent.supervisor import sanitize_node
    original = AIMessage(content="Veja a **Kronos** (ID: 16623454022)", id="ai-1")
    state = {"messages": [HumanMessage(content="oi"), original]}
    out = sanitize_node(state)
    assert out.get("messages"), "sanitize_node should return a replacement"
    replacement = out["messages"][0]
    assert replacement.id == "ai-1"
    # add_messages must UPDATE in place, not append.
    merged = add_messages([HumanMessage(content="oi"), original], out["messages"])
    ai_msgs = [m for m in merged if isinstance(m, AIMessage)]
    assert len(ai_msgs) == 1
    assert "**" not in ai_msgs[0].content
    assert "16623454022" not in ai_msgs[0].content


# ════════════════════════════════════════════════════════════════════════════
# 4. escalar_humano invokes the dossier pipeline
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_escalar_humano_invokes_dossier_pipeline():
    from app.agent import tools_v2

    state = {
        "messages": [HumanMessage(content="quero falar com uma pessoa")],
        "phone_hash": "escalartest" * 5,
        "thread_id": "t1",
    }
    with patch.object(
        tools_v2, "handoff_dossier_pipeline", new_callable=AsyncMock, create=True
    ):
        # The tool imports the pipeline lazily from app.agent.dossier; patch
        # the source so the real summarize/persist/send never run.
        with patch("app.agent.dossier.handoff_dossier_pipeline", new_callable=AsyncMock) as pipe:
            result = await tools_v2.escalar_humano.ainvoke(
                {"motivo": "pedido_humano", "resumo": "cliente quer atendente", "state": state}
            )
    pipe.assert_awaited_once()
    # The reason passed to the pipeline reflects the motivo.
    called_state = pipe.await_args.args[0]
    assert called_state.get("handoff_reason") == "pedido_humano"
    assert "atendente" in result.lower() or "acionad" in result.lower()
