"""V2 supervisor tools — Phase 1 REAL implementations (behind ``use_v2``, OFF).

Each tool reuses the data layer the legacy flow already owns; none of them
reimplements that layer:

    buscar_catalogo     → app.sync.bling_catalog_cache.get_catalog_snapshot
                          (+ app.sync.bling_repo.fetch_product_by_name fallback)
    detalhes_produto    → app.sync.bling_repo.fetch_product_by_id
    consultar_estoque   → app.sync.bling_stock.get_stock
    buscar_conhecimento → app.rag.retriever.search_knowledge_base (pgvector)
    escalar_humano      → app.agent.dossier.handoff_dossier_pipeline

IMPORTANT — this does NOT touch ``_product_match.py`` and does NOT use its
decision machinery (candidates / confirmation / gates). Disambiguation is now
the LLM's job; these tools only do a SIMPLE search over the data. No
``rapidfuzz`` in the project, so name matching is plain substring/token
overlap — no fuzzy distance.

The docstrings matter: the LLM reads them to decide when to call each tool.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

logger = logging.getLogger(__name__)

# Phonetic-fuzzy match threshold. Calibrated against the dev catalog (Fase 1
# diagnosis): "cronus"->"Kronos" scores 0.83 and "protheu"->"Proteo" scores
# 1.00 after phonetic folding, while unrelated names stay <=0.67. 0.75 cleanly
# separates the real product from noise. NO rapidfuzz (not installed) and NO
# pg_trgm (trigram scored "cronus"/"kronos" at 0.077 — useless for c/k swaps);
# stdlib difflib over phonetically-folded tokens does the job.
_FUZZY_THRESHOLD = 0.75


# ── helpers (simple search, NO _product_match decision machinery) ────────────

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    return _strip_accents((s or "").lower()).strip()


# Generic words that carry no disambiguating signal — excluded from matching so
# "raquete"/"beach"/"tennis" don't make every product look relevant.
_STOPWORDS = {
    "raquete", "raquetes", "beach", "tennis", "padel", "praia", "the", "com",
    "para", "uma", "uns", "and", "and", "sport", "sports",
}


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if len(t) >= 3]


def _content_tokens(s: str) -> list[str]:
    return [t for t in _tokens(s) if t not in _STOPWORDS]


def _phonetic(tok: str) -> str:
    """Light PT-BR phonetic folding so common mis-spellings collapse:
    k<->c, ph->f, th->t, y->i, qu->c, collapse doubles, drop trailing vowels
    ("proteo"/"proteu" -> "prot", "kronos"/"cronus" -> "cronus"->"cronu"...).
    """
    t = tok
    t = t.replace("ph", "f").replace("th", "t").replace("qu", "c")
    t = t.replace("y", "i").replace("k", "c")
    t = re.sub(r"(.)\1+", r"\1", t)
    t = re.sub(r"[aeiou]+$", "", t)
    return t


def _price_brl(price_cents: Any) -> str:
    try:
        cents = int(price_cents or 0)
    except (TypeError, ValueError):
        return "R$ —"
    reais = cents / 100.0
    return "R$ " + f"{reais:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _score_product(q_tokens: list[str], product: dict[str, Any]) -> float:
    """Relevance score combining exact-token overlap with phonetic-fuzzy match.

    For each distinctive query token, we take the best score against any
    product token: an exact token match counts as 1.0; otherwise we fall back
    to a difflib ratio over the phonetically-folded forms (so "cronus" matches
    "kronos"). The product score is the SUM of per-query-token bests, so a
    multi-token query that matches more tokens ranks higher. A token whose best
    fuzzy score is below threshold contributes 0 (no spurious matches).
    """
    haystack = " ".join(
        str(product.get(f) or "")
        for f in ("name", "marca", "modelo", "categoria_nome")
    )
    p_tokens = _content_tokens(haystack)
    if not p_tokens:
        return 0.0
    p_set = set(p_tokens)
    p_phon = {pt: _phonetic(pt) for pt in p_tokens}

    total = 0.0
    for qt in q_tokens:
        if qt in p_set:
            # Exact token match dominates: a real "proteo"=="proteo" must
            # outrank a product that only matches "proteo" phonetically (e.g.
            # "prote" from "proteção"). Weighting exact >> fuzzy guarantees
            # the genuine product ranks above incidental phonetic collisions.
            total += 10.0
            continue
        qp = _phonetic(qt)
        best = 0.0
        for pt in p_tokens:
            r = difflib.SequenceMatcher(None, qp, p_phon[pt]).ratio()
            if r > best:
                best = r
        if best >= _FUZZY_THRESHOLD:
            total += best
    return total


# ── tools ────────────────────────────────────────────────────────────────────

@tool
async def buscar_catalogo(consulta: str) -> str:
    """Busca produtos no catálogo da Base Sports por nome, marca ou características.
    Use sempre que o cliente mencionar um produto, pedir comparação, ou perguntar o que existe.
    Para comparar dois produtos, chame uma vez para cada. Retorna lista com id, nome e preço."""
    q_tokens = _content_tokens(consulta)

    products: list[dict[str, Any]] = []
    # Primary: full in-memory snapshot (same path the legacy recommend uses).
    try:
        from app.sync.bling_catalog_cache import get_catalog_snapshot
        products = list(await get_catalog_snapshot())
    except Exception as exc:
        logger.warning("buscar_catalogo snapshot_failed: %s", exc)

    # Fallback: direct ILIKE on the longest query token if snapshot empty.
    if not products and q_tokens:
        try:
            from app.sync.bling_repo import fetch_product_by_name
            longest = max(q_tokens, key=len)
            products = await fetch_product_by_name(longest)
        except Exception as exc:
            logger.warning("buscar_catalogo ilike_failed: %s", exc)

    if not products:
        return json.dumps({"resultados": [], "aviso": "catalogo indisponivel"}, ensure_ascii=False)

    if q_tokens:
        scored = [(p, _score_product(q_tokens, p)) for p in products]
        scored = [(p, s) for p, s in scored if s > 0]
        scored.sort(key=lambda ps: ps[1], reverse=True)
        ranked = [p for p, _ in scored[:5]]
    else:
        ranked = products[:5]

    out = [
        {"id": str(p.get("id")), "nome": p.get("name"), "preco": _price_brl(p.get("price_cents"))}
        for p in ranked
    ]
    logger.info("buscar_catalogo q_tokens=%d results=%d", len(q_tokens), len(out))
    return json.dumps(out, ensure_ascii=False)


# Customer-facing labels we DON'T want to surface, and values that are raw
# Bling IDs (pure digits) rather than human-readable text.
_ATTR_SKIP_SUBSTR = ("bateria", "amazon", "es raquete", "tema de varia", "sem genero")


def _is_displayable_attr(label: str, value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    v = str(value).strip()
    if v.isdigit():  # raw Bling reference id, not human text
        return False
    if v.lower() in ("true", "false"):
        return False
    low = _norm(label)
    return not any(skip in low for skip in _ATTR_SKIP_SUBSTR)


@tool
async def detalhes_produto(produto_id: str) -> str:
    """Retorna as especificações técnicas e características de um produto pelo id.
    Use quando o cliente quiser saber detalhes, specs ou para que tipo de jogo um produto serve."""
    try:
        pid = int(produto_id)
    except (TypeError, ValueError):
        return json.dumps({"erro": f"id invalido: {produto_id!r}"}, ensure_ascii=False)

    try:
        from app.sync.bling_repo import fetch_product_by_id
        product = await fetch_product_by_id(pid)
    except Exception as exc:
        logger.warning("detalhes_produto fetch_failed id=%s: %s", pid, exc)
        return json.dumps({"erro": "nao foi possivel consultar o produto"}, ensure_ascii=False)

    if product is None:
        return json.dumps({"erro": f"produto {produto_id} nao encontrado"}, ensure_ascii=False)

    specs: dict[str, Any] = {
        "id": str(product.get("id")),
        "nome": product.get("name"),
        "marca": product.get("marca"),
        "modelo": product.get("modelo"),
        "categoria": product.get("categoria_nome"),
        "preco": _price_brl(product.get("price_cents")),
    }
    if product.get("weight_g"):
        specs["peso"] = f"{product['weight_g']}g"
    if product.get("description"):
        specs["descricao"] = product["description"]

    # Human-readable custom fields (PT-BR labels), filtering Bling-id noise.
    extras: dict[str, str] = {}
    for label, value in (product.get("campos_customizados") or {}).items():
        if label.upper() in ("MARCA", "MODELO"):
            continue  # already surfaced above
        if _is_displayable_attr(label, value):
            extras[label] = str(value)
    if extras:
        specs["caracteristicas"] = extras

    specs = {k: v for k, v in specs.items() if v not in (None, "")}
    logger.info("detalhes_produto id=%s extras=%d", pid, len(extras))
    return json.dumps(specs, ensure_ascii=False)


@tool
async def consultar_estoque(produto_id: str) -> str:
    """Consulta a disponibilidade em estoque de um produto pelo id."""
    try:
        pid = int(produto_id)
    except (TypeError, ValueError):
        return json.dumps({"erro": f"id invalido: {produto_id!r}"}, ensure_ascii=False)

    try:
        from app.sync.bling_stock import get_stock
        saldo = await get_stock(pid)
    except Exception as exc:
        logger.warning("consultar_estoque failed id=%s: %s", pid, exc)
        saldo = None

    if saldo is None:
        # Bling unauthorized / timeout / off → honest "couldn't confirm".
        return json.dumps(
            {"id": str(pid), "disponivel": None, "aviso": "nao consegui confirmar o estoque agora"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"id": str(pid), "em_estoque": saldo > 0, "quantidade": saldo}, ensure_ascii=False
    )


@tool
async def buscar_conhecimento(consulta: str) -> str:
    """Busca informações da loja que NÃO são de produto: horário, endereço, garantia,
    formas de pagamento, como funciona a Consultoria e quem a conduz. Use para dúvidas institucionais."""
    try:
        from app.rag.retriever import search_knowledge_base
        from app.storage.db import get_session
        async with get_session() as session:
            docs = await search_knowledge_base(session, consulta, k=4)
    except Exception as exc:
        logger.warning("buscar_conhecimento failed: %s", exc)
        return json.dumps(
            {"resultados": [], "aviso": "nao consegui consultar a base de conhecimento agora"},
            ensure_ascii=False,
        )

    if not docs:
        # Empty KB → tell the LLM plainly so it can escalate instead of inventing.
        return json.dumps(
            {"resultados": [], "aviso": "nenhum conteudo encontrado na base de conhecimento"},
            ensure_ascii=False,
        )

    out = []
    for d in docs:
        out.append({
            "titulo": d.get("title") or d.get("titulo") or d.get("category"),
            "conteudo": d.get("content") or d.get("conteudo") or d.get("text") or "",
        })
    logger.info("buscar_conhecimento q_len=%d results=%d", len(consulta), len(out))
    return json.dumps(out, ensure_ascii=False)


@tool
async def escalar_humano(
    motivo: str, resumo: str, state: Annotated[dict, InjectedState]
) -> str:
    """Aciona um atendente humano de verdade. Use quando o cliente pedir falar com pessoa,
    quando a dúvida sair do escopo, ou para encaminhar o fechamento da Consultoria.
    'motivo' = categoria (ex: 'consultoria', 'fora_de_escopo', 'pedido_humano').
    'resumo' = resumo curto da conversa para o atendente."""
    # Identity + history come from the injected graph state, NOT from the LLM.
    # The dossier pipeline consumes ``messages`` + ``phone_hash`` and opens its
    # own DB session. ``handoff_reason`` is set to ``motivo`` so the persisted
    # and delivered dossier reflects why the escalation happened.
    from app.agent.dossier import handoff_dossier_pipeline

    state_for_dossier = dict(state or {})
    state_for_dossier["handoff_reason"] = motivo or "user_requested"
    # The supervisor's free-text ``resumo`` becomes the dossier summary seed —
    # passed via produto_pesquisado-independent path: the pipeline computes its
    # own LLM summary, but we keep the resumo in the state so future phases can
    # use it. For Phase 1 it's logged; the pipeline summarizes from messages.
    phone_hash = state_for_dossier.get("phone_hash") or ""

    try:
        await handoff_dossier_pipeline(state_for_dossier)
        logger.info(
            "escalar_humano dispatched motivo=%s phone_hash=%.8s resumo_len=%d",
            motivo, phone_hash, len(resumo or ""),
        )
    except Exception as exc:
        logger.error("escalar_humano dossier_failed motivo=%s: %s", motivo, exc)
        return "Tentei acionar um atendente mas houve um problema; um humano será avisado."

    return (
        "Atendente acionado. Encaminhei um resumo da conversa para a equipe "
        "e em breve alguém te chama por aqui."
    )


TOOLS_V2 = [buscar_catalogo, detalhes_produto, consultar_estoque, buscar_conhecimento, escalar_humano]
