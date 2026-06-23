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
    # Sub-2k rackets — the production bug: "raquete até 2k" must surface these.
    {"id": 16700000001, "name": "Raquete Beach Tennis Mormaii Sunset Plus",
     "marca": "Mormaii", "modelo": "Sunset Plus", "price_cents": 179990,
     "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True},
    {"id": 16700000002, "name": "Raquete Beach Tennis Drop Shot Tiger 2.0 Iniciante",
     "marca": "Drop Shot", "modelo": "Tiger 2.0", "price_cents": 46900,
     "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True},
    # A cheap NON-racket near the same price, to prove racket-restriction works.
    {"id": 16700000003, "name": "Bolsa Raqueteira Beach Tennis Generica",
     "marca": "Generica", "modelo": "Bag", "price_cents": 120000,
     "categoria_nome": "RAQUETEIRAS MOCHILA"},
    # A tennis racket (NOT beach tennis) — categoria text says "raquete" but the
    # curated flag is False. Must NEVER show for categoria="beach tennis".
    {"id": 16700000006, "name": "Raquete de Tenis Wilson Pro Staff",
     "marca": "Wilson", "modelo": "Pro Staff", "price_cents": 89900,
     "categoria_nome": "RAQUETE TENIS", "is_raquete_praia": False},
    # A pickleball racket near the cheap band — same trap, different sport.
    {"id": 16700000007, "name": "Raquete de Pickleball Joola Ben Johns",
     "marca": "Joola", "modelo": "Ben Johns", "price_cents": 79900,
     "categoria_nome": "Raquete de pickleball", "is_raquete_praia": False},
    # A frescobol kit MIS-FLAGGED is_raquete_praia=True (real catalog data gap):
    # the cheapest "racket" by price, but not a single beach-tennis racket. Must
    # be excluded so it never leads a "mais barata" list.
    {"id": 16700000008, "name": "Frescobol Kit Tenis Praia 2 Raquetes 1 Bolinha",
     "marca": "Head", "modelo": "Kit", "price_cents": 16900,
     "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True},
    # Products with tokens that 'baran' transposes into ("branco"/"branca") —
    # the production false-match. Must NOT be returned for "Baran".
    {"id": 16700000004, "name": "Boné Drop Shot Trucker Cor Branco",
     "marca": "Drop Shot", "modelo": "Trucker", "price_cents": 12900,
     "categoria_nome": "Bonés"},
    {"id": 16700000005, "name": "Raquete Beach Tennis Heroes Sofia Chow Branca 2024",
     "marca": "Heroes", "modelo": "Sofia Chow", "price_cents": 159990,
     "categoria_nome": "Raquetes de Praia", "is_raquete_praia": True},
]


async def _run_buscar(
    consulta: str, preco_min=None, preco_max=None, categoria=None, ordenacao=None
) -> list[dict]:
    """Call buscar_catalogo with the fixture catalog patched in."""
    from app.agent import tools_v2

    async def fake_snapshot():
        return list(_FIXTURE_CATALOG)

    args: dict = {"consulta": consulta}
    if preco_min is not None:
        args["preco_min"] = preco_min
    if preco_max is not None:
        args["preco_max"] = preco_max
    if categoria is not None:
        args["categoria"] = categoria
    if ordenacao is not None:
        args["ordenacao"] = ordenacao

    with patch.object(tools_v2, "get_catalog_snapshot", fake_snapshot, create=True):
        # get_catalog_snapshot is imported lazily inside the tool; patch the
        # source module function instead.
        with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", fake_snapshot):
            raw = await tools_v2.buscar_catalogo.ainvoke(args)
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


@pytest.mark.asyncio
async def test_buscar_catalogo_baran_returns_empty_not_irrelevant():
    """Production bug: 'Baran' (not in catalog) falsely matched 'Branca/Branco'
    via letter transposition and dumped Heroes Sofia Chow as 'the only one'.
    With threshold 0.82, no token clears the bar → empty, not an irrelevant
    product."""
    results = await _run_buscar("Baran")
    names = [n.lower() for n in _names(results)]
    assert not any("branco" in n or "branca" in n or "sofia chow" in n for n in names), \
        f"'Baran' dumped an irrelevant transposition match: {_names(results)}"
    # Ideally empty; certainly must not surface the false 'branca' racket.
    assert results == [], f"'Baran' should return empty, got: {_names(results)}"


# ── 1b. Price-range filter ───────────────────────────────────────────────────

def _price_to_float(preco_str: str) -> float:
    """'R$ 1.799,90' → 1799.90."""
    digits = preco_str.replace("R$", "").strip().replace(".", "").replace(",", ".")
    return float(digits)


@pytest.mark.asyncio
async def test_buscar_catalogo_price_max_includes_sub2k_racket():
    """The production bug: 'raquete até 2k' must surface the Mormaii Sunset
    Plus (R$ 1.799,90), not conclude there's nothing under 2000."""
    results = await _run_buscar("raquete", preco_max=2000)
    assert results, "price_max=2000 returned nothing"
    names = [n.lower() for n in _names(results)]
    assert any("mormaii sunset" in n for n in names), f"missing Mormaii Sunset: {_names(results)}"
    # Every result is within the price ceiling.
    assert all(_price_to_float(r["preco"]) <= 2000 for r in results), \
        f"a result exceeds 2000: {results}"
    # And every result is a racket (racket-restriction applied). The bag
    # ("Bolsa Raqueteira") must NOT leak — "raqueteira" contains "raquete" as a
    # substring but is not a racket.
    assert not any("bolsa" in n for n in names), \
        f"a bag leaked into 'raquete até 2k': {_names(results)}"
    assert all(("raquete" in r["nome"].lower().split())
               or ("raquete de" in r["nome"].lower())
               or r["nome"].lower().startswith("raquete")
               for r in results), \
        f"non-racket leaked into 'raquete até 2k': {_names(results)}"


@pytest.mark.asyncio
async def test_buscar_catalogo_price_range_only_in_band():
    """'entre 1000 e 1500' returns only products inside the band."""
    results = await _run_buscar("raquete", preco_min=1000, preco_max=1500)
    for r in results:
        price = _price_to_float(r["preco"])
        assert 1000 <= price <= 1500, f"{r['nome']} at {price} is outside [1000,1500]"


@pytest.mark.asyncio
async def test_buscar_catalogo_price_filter_sorts_ascending():
    """With a price filter, results come ordered cheapest-first."""
    results = await _run_buscar("raquete", preco_max=3000)
    prices = [_price_to_float(r["preco"]) for r in results]
    assert prices == sorted(prices), f"not ascending: {prices}"


@pytest.mark.asyncio
async def test_buscar_catalogo_name_only_unchanged_no_regression():
    """A pure name search (no price) behaves exactly as before."""
    results = await _run_buscar("Kronos")
    assert results
    assert "kronos" in _names(results)[0].lower()
    # No price params → name-relevance top-5 (the legacy shape).
    assert len(results) <= 5


# ── 1c. Category filter + ordenacao ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_buscar_catalogo_beach_tennis_category_excludes_other_sports():
    """categoria='beach tennis' filters on the curated is_raquete_praia flag —
    a tennis racket, a pickleball racket, and a bag (all with 'raquete' in
    their name/category) must NOT appear."""
    results = await _run_buscar("raquete de beach tennis", categoria="beach tennis")
    names = [n.lower() for n in _names(results)]
    assert names, "category browse returned nothing"
    assert not any("tenis wilson" in n or "pickleball" in n for n in names), \
        f"other-sport racket leaked: {_names(results)}"
    assert not any("bolsa" in n or "raqueteira" in n for n in names), \
        f"bag leaked into beach tennis: {_names(results)}"


@pytest.mark.asyncio
async def test_buscar_catalogo_pure_category_and_sort_no_name():
    """'as mais baratas de beach tennis' — no distinctive name to match, just
    categoria + ordenacao=preco_asc. Must return beach tennis rackets ordered
    cheapest-first, WITHOUT depending on the fuzzy name match."""
    results = await _run_buscar(
        "as mais baratas", categoria="beach tennis", ordenacao="preco_asc"
    )
    assert results, "pure category+sort returned nothing"
    prices = [_price_to_float(r["preco"]) for r in results]
    assert prices == sorted(prices), f"not ascending: {prices}"
    # The mis-flagged Frescobol kit (cheapest by price) must NOT lead the list.
    assert "frescobol" not in _names(results)[0].lower(), \
        f"frescobol kit led the 'mais baratas' list: {_names(results)}"


@pytest.mark.asyncio
async def test_buscar_catalogo_frescobol_kit_excluded_from_beach_tennis():
    """The R$169 Frescobol kit is mis-flagged is_raquete_praia=True; it must be
    excluded from beach-tennis-racket results entirely."""
    results = await _run_buscar("raquete de beach tennis", categoria="beach tennis")
    assert not any("frescobol" in n.lower() for n in _names(results)), \
        f"frescobol kit leaked: {_names(results)}"


@pytest.mark.asyncio
async def test_buscar_catalogo_price_only_no_category_no_name_defaults_beach_tennis():
    """PRODUCTION REGRESSION: the LLM called buscar_catalogo(preco_max=1000)
    with NO categoria and NO consulta, and a tennis racket ('Raquete Tenis
    Tecnifibre') leaked. A bare price query in a beach-tennis store must default
    to beach tennis rackets — no tennis/pickleball/bag, even when the LLM omits
    categoria."""
    results = await _run_buscar("", preco_max=1000)
    names = [n.lower() for n in _names(results)]
    assert results, "price-only query returned nothing"
    assert not any(
        w in n for n in names
        for w in ("tenis wilson", "pickleball", "bolsa", "raqueteira", "frescobol")
    ), f"junk leaked into price-only query: {_names(results)}"
    assert all(_price_to_float(r["preco"]) <= 1000 for r in results)


@pytest.mark.asyncio
async def test_buscar_catalogo_named_product_with_price_still_findable():
    """The default-to-beach-tennis must NOT block a specific named search: a
    customer asking for a named product within a budget still finds it (name
    relevance kept when distinctive tokens are present)."""
    results = await _run_buscar("Kronos", preco_max=5000)
    assert any("kronos" in n.lower() for n in _names(results)), \
        f"named product lost under price filter: {_names(results)}"


@pytest.mark.asyncio
async def test_buscar_catalogo_category_plus_price_combined():
    """'raquete de beach tennis até 1500' = categoria + preco_max, every result
    a beach tennis racket within the band, cheapest-first, no junk."""
    results = await _run_buscar(
        "raquete de beach tennis", categoria="beach tennis", preco_max=1500
    )
    names = [n.lower() for n in _names(results)]
    for r in results:
        assert _price_to_float(r["preco"]) <= 1500, f"{r['nome']} over 1500"
    assert not any(
        w in n for n in names
        for w in ("bolsa", "raqueteira", "pickleball", "tenis wilson", "frescobol")
    ), f"junk leaked: {_names(results)}"


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
    monkeypatch.setenv("ECOMMERCE_URL", "")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.supervisor import build_system_prompt
    prompt = build_system_prompt()
    assert "Beira Mar" not in prompt
    # The empty-address branch must instruct asking the customer to confirm.
    assert "confirmar o endereço" in prompt.lower() or "confirme o endereço" in prompt.lower()
    assert "não tem o endereço" in prompt.lower()
    get_settings.cache_clear()


def test_system_prompt_configured_address_is_pinned(monkeypatch):
    """With store_address set, the canonical address+hours are pinned in the
    physical-store block (use SEMPRE estes dados)."""
    monkeypatch.setenv("STORE_NAME", "Base Sports")
    monkeypatch.setenv("STORE_ADDRESS", "Rua Tal, 99, Cidade")
    monkeypatch.setenv("STORE_HOURS", "seg a sex, 8h-18h")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.supervisor import build_system_prompt
    prompt = build_system_prompt()
    assert "LOJA FÍSICA" in prompt
    assert "Rua Tal, 99, Cidade" in prompt
    assert "seg a sex, 8h-18h" in prompt
    get_settings.cache_clear()


def test_system_prompt_ecommerce_url_pinned_when_set(monkeypatch):
    """With ecommerce_url set, the canonical link is pinned (PIX + discount)."""
    monkeypatch.setenv("ECOMMERCE_URL", "https://loja.basesports.com.br")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.supervisor import build_system_prompt
    prompt = build_system_prompt()
    assert "https://loja.basesports.com.br" in prompt
    assert "pix" in prompt.lower()
    get_settings.cache_clear()


def test_system_prompt_empty_ecommerce_url_no_invented_link(monkeypatch):
    """Safety rule: with ecommerce_url EMPTY, the e-commerce is mentioned but no
    URL is invented — the agent asks the customer to confirm the link."""
    monkeypatch.setenv("ECOMMERCE_URL", "")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent.supervisor import build_system_prompt
    prompt = build_system_prompt()
    assert "http" not in prompt.lower()  # no link of any kind
    assert "e-commerce" in prompt.lower()
    assert "confirmar o link" in prompt.lower() or "confirme o link" in prompt.lower()
    get_settings.cache_clear()


def test_system_prompt_no_online_only_claim(monkeypatch):
    """The prompt must NOT claim sale is 'only in the physical store'."""
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent.supervisor import build_system_prompt
    prompt = build_system_prompt().lower()
    assert "não online" not in prompt and "nao online" not in prompt
    assert "apenas na loja" not in prompt and "só na loja" not in prompt
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


def test_sanitize_converts_markdown_link_to_url():
    """WhatsApp doesn't render [text](url) — convert to the plain url."""
    from app.agent.supervisor import _sanitize_for_whatsapp
    out = _sanitize_for_whatsapp(
        "Acesse [loja.basesports.com.br](https://loja.basesports.com.br) e pague com PIX"
    )
    assert "](" not in out
    assert "https://loja.basesports.com.br" in out
    assert "PIX" in out


def test_sanitize_removes_uuid():
    from app.agent.supervisor import _sanitize_for_whatsapp
    out = _sanitize_for_whatsapp("ref 619aeec0-163d-4f9b-89fb-4ce867f537e6 aqui")
    assert "619aeec0" not in out


def test_sanitize_preserves_prices_and_years():
    from app.agent.supervisor import _sanitize_for_whatsapp
    out = _sanitize_for_whatsapp("A Proteo 2026 custa R$ 2.899,90.")
    assert "R$ 2.899,90" in out
    assert "2026" in out


def test_sanitize_strips_canned_closing_line():
    """Felipe's complaint: gpt-4o-mini keeps appending the fixed closing offer
    no matter what the prompt says. The sanitizer must strip it deterministically
    — but only the TRAILING generic offer, never the product info above it."""
    from app.agent.supervisor import _sanitize_for_whatsapp
    cases = [
        "A Mormaii Sunset sai por R$ 1.799,90.\nSe precisar de mais informações ou ajuda, é só avisar!",
        "Temos a Drop Shot Pentax a R$ 449. Se precisar de ajuda com outras faixas de preço ou produtos, é só avisar!",
        "Encontrei 3 opções. Qualquer dúvida, estou à disposição!",
        "A Kronos é ótima pra controle. Fico à disposição.",
        # Real variants gpt-4o-mini emitted in production replay:
        "Raquete A - R$ 449,00\nSe quiser saber mais sobre alguma delas ou verificar a disponibilidade, é só avisar!",
        "Raquete X - R$ 469,00\nSe alguma delas te interessar, posso verificar a disponibilidade ou fornecer mais detalhes!",
        "Raquete Y - R$ 469,00\nSe alguma dessas opções te interessar, posso verificar a disponibilidade em estoque ou ajudar com mais informações.",
    ]
    for raw in cases:
        out = _sanitize_for_whatsapp(raw)
        low = out.lower()
        assert "é só avisar" not in low and "so avisar" not in low
        assert "à disposição" not in low and "a disposicao" not in low
        assert "qualquer dúvida" not in low
        assert "verificar a disponibilidade" not in low
        assert "fornecer mais detalhes" not in low and "mais informações" not in low
        # The real content above the closing survives.
        assert any(tok in out for tok in ("R$", "Kronos", "opções", "Drop Shot", "Raquete"))


def test_sanitize_keeps_contextual_question_offer():
    """A specific, contextual next-step question is NOT a canned closing and must
    survive — we only strip the generic 'ask me anything' boilerplate."""
    from app.agent.supervisor import _sanitize_for_whatsapp
    out = _sanitize_for_whatsapp(
        "A Mormaii é a mais em conta das duas. Quer que eu veja o estoque dela?"
    )
    assert "estoque" in out.lower()
    assert "Quer que eu" in out


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
# 3b. Anti-hallucination: forced search when the model claims "we don't have it"
#     without calling buscar_catalogo (the production tool_calls=0 bug).
# ════════════════════════════════════════════════════════════════════════════


def _fake_openai_message(content="", tool_calls=None):
    """Build a stub object shaped like an OpenAI chat completion message."""
    from types import SimpleNamespace
    tcs = []
    for tc in (tool_calls or []):
        tcs.append(SimpleNamespace(
            id=tc.get("id", "call_1"),
            function=SimpleNamespace(
                name=tc["name"],
                arguments=json.dumps(tc.get("args", {})),
            ),
        ))
    return SimpleNamespace(content=content, tool_calls=tcs or None)


def _fake_completion(message):
    from types import SimpleNamespace
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_should_force_search_detects_ungrounded_unavailability():
    from app.agent.supervisor import _should_force_search
    state = {"messages": [HumanMessage(content="tem raquetes até 1k?")]}
    ai = AIMessage(content="Não encontrei raquetes de Beach Tennis abaixo de R$ 1000.", tool_calls=[])
    assert _should_force_search(state, ai) is True


def test_should_force_search_on_price_question_regardless_of_phrasing():
    """The production bug: a PRICE question answered without searching must
    force a search no matter how the model phrased the (non-)answer — we don't
    depend on recognizing a 'não temos' string."""
    from app.agent.supervisor import _should_force_search
    price_questions = [
        "tem raquetes até 1k?",
        "e até 2 mil reais?",
        "quero as mais baratas",
        "qual a mais em conta?",
        "quanto custa uma raquete boa?",
        "tem alguma abaixo de 800?",
    ]
    # An answer that does NOT contain any unavailability phrase — pure memory guess.
    vague = AIMessage(content="As raquetes de beach tennis variam bastante de preço.", tool_calls=[])
    for q in price_questions:
        state = {"messages": [HumanMessage(content=q)]}
        assert _should_force_search(state, vague) is True, f"did not force search for: {q!r}"


def test_should_not_force_when_model_answered_with_tool_call():
    from app.agent.supervisor import _should_force_search
    state = {"messages": [HumanMessage(content="tem raquetes até 1k?")]}
    ai = AIMessage(content="", tool_calls=[{"name": "buscar_catalogo", "args": {}, "id": "c1"}])
    assert _should_force_search(state, ai) is False


def test_should_not_force_when_already_searched_this_turn():
    """If buscar_catalogo already ran this turn and came back empty, an honest
    'não temos' is allowed — don't loop."""
    from app.agent.supervisor import _should_force_search
    state = {"messages": [
        HumanMessage(content="tem raquetes até 1k?"),
        AIMessage(content="", tool_calls=[{"name": "buscar_catalogo", "args": {}, "id": "c1"}]),
        ToolMessage(content="[]", tool_call_id="c1", name="buscar_catalogo"),
    ]}
    ai = AIMessage(content="Não encontrei raquetes nessa faixa.", tool_calls=[])
    assert _should_force_search(state, ai) is False


def test_should_not_force_on_nonprice_smalltalk():
    from app.agent.supervisor import _should_force_search
    state = {"messages": [HumanMessage(content="oi, tudo bem?")]}
    ai = AIMessage(content="Tudo ótimo! Como posso ajudar?", tool_calls=[])
    assert _should_force_search(state, ai) is False


@pytest.mark.asyncio
async def test_supervisor_node_forces_buscar_catalogo_on_hallucinated_unavailability(monkeypatch):
    """End-to-end of the guard: model first answers 'não encontrei' with NO
    tool call (the prod bug); supervisor_node rejects it and re-runs FORCING
    buscar_catalogo, so the returned message carries a tool call."""
    from app.config import get_settings
    get_settings.cache_clear()

    from app.agent import supervisor as sup

    calls = {"n": 0, "forced_request": None}

    class _FakeCompletions:
        async def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # First pass: ungrounded unavailability, no tool call.
                return _fake_completion(_fake_openai_message(
                    content="Não encontrei raquetes de Beach Tennis abaixo de R$ 1000."
                ))
            # Second pass must be the forced one.
            calls["forced_request"] = kwargs.get("tool_choice")
            return _fake_completion(_fake_openai_message(
                tool_calls=[{"name": "buscar_catalogo", "args": {"preco_max": 1000}, "id": "c1"}]
            ))

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(sup, "AsyncOpenAI", _FakeClient)

    state = {"messages": [HumanMessage(content="tem raquetes até 1k?")],
             "phone_hash": "h", "thread_id": "t"}
    out = await sup.supervisor_node(state)

    assert calls["n"] == 2, "supervisor did not retry after ungrounded claim"
    assert calls["forced_request"] == {
        "type": "function", "function": {"name": "buscar_catalogo"},
    }, "retry did not force buscar_catalogo via tool_choice"
    msg = out["messages"][0]
    assert msg.tool_calls and msg.tool_calls[0]["name"] == "buscar_catalogo"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_supervisor_node_no_retry_when_answer_is_grounded(monkeypatch):
    """A normal answer (no unavailability claim) must NOT trigger a second call."""
    from app.config import get_settings
    get_settings.cache_clear()
    from app.agent import supervisor as sup

    calls = {"n": 0}

    class _FakeCompletions:
        async def create(self, **kwargs):
            calls["n"] += 1
            return _fake_completion(_fake_openai_message(
                content="A Drop Shot Pentax sai por R$ 449."
            ))

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(sup, "AsyncOpenAI", _FakeClient)
    state = {"messages": [HumanMessage(content="quanto custa a pentax?")],
             "phone_hash": "h", "thread_id": "t"}
    out = await sup.supervisor_node(state)
    assert calls["n"] == 1, "should not retry a grounded answer"
    get_settings.cache_clear()


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
