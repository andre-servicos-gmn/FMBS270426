"""Shared helpers for the ``tests/evals`` gate suite (T1–T6).

Canonical decisions baked in here (confirmed with the lead / dev-tools):

- Mirrored-stock snapshot: the product dict carries ``stock`` (int) and
  ``created_at``. Stock filter is GLOBAL — explicit ``0`` excludes; a missing
  key or ``None`` KEEPS the product (fail-open). Ordering by ``created_at``
  DESC applies ONLY to the browse/default path (a price-range or
  ``preco_asc`` query keeps price order).
- Persona lives in the V2 supervisor: ``build_system_prompt()`` /
  ``SYSTEM_SUPERVISOR_TEMPLATE`` (app/agent/supervisor.py). Travessão guard is
  a deterministic backstop in ``_sanitize_for_whatsapp``.

These helpers patch the SAME data-layer entry point the legacy flow owns
(``app.sync.bling_catalog_cache.get_catalog_snapshot``), so the tool runs its
real filter/rank logic over a fixture catalog with no DB or network.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import get_settings

# Snapshot product dict keys - CANONICAL, verified against the shipped code
# end-to-end: migration 0009 adds column ``stock``, BlingProduct.stock,
# bling_repo._row_to_dict exposes ``stock``, tools_v2._has_stock reads ``stock``.
# None/absent = KEEP (fail-open); stock <= 0 (zero or negative) = EXCLUDE;
# unparseable = keep. Recency is ``created_at``. Final: the shipped column and
# dict key are ``stock`` (English, matches the codebase convention).
STOCK_FIELD = "stock"
CREATED_FIELD = "created_at"

# Sentinel meaning "leave this key out of the dict entirely" — the genuine
# missing-field case (distinct from an explicit ``None`` value).
OMIT: Any = object()


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    """Tiny constructor for distinct, ordered ``created_at`` values."""
    return datetime(year, month, day)


def make_racket(
    pid: int,
    name: str,
    price_reais: float = 599.0,
    *,
    stock: Any = 10,
    created_at: Any = OMIT,
    is_raquete_praia: bool = True,
    categoria: str = "Raquetes de Praia",
    marca: str = "Drop Shot",
    modelo: str | None = None,
) -> dict[str, Any]:
    """Build a fixture product dict shaped like ``get_catalog_snapshot`` output.

    Pass ``stock=OMIT`` to omit the stock key entirely (the missing-field case)
    or ``stock=None`` for the explicit-None case — both must be KEPT (fail-open).
    """
    d: dict[str, Any] = {
        "id": pid,
        "name": name,
        "price_cents": int(round(price_reais * 100)),
        "is_raquete_praia": is_raquete_praia,
        "categoria_nome": categoria,
        "marca": marca,
        "modelo": modelo or name,
        "external_id": str(pid),
    }
    if stock is not OMIT:
        d[STOCK_FIELD] = stock
    if created_at is not OMIT:
        d[CREATED_FIELD] = created_at
    return d


async def run_buscar(
    catalog: list[dict[str, Any]],
    consulta: str = "",
    *,
    preco_min: float | None = None,
    preco_max: float | None = None,
    categoria: str | None = None,
    ordenacao: str | None = None,
) -> list[dict[str, Any]]:
    """Invoke ``buscar_catalogo`` with ``catalog`` patched in as the snapshot.

    Returns the parsed list of result dicts (``{id, nome, preco}``). When the
    tool returns the empty-catalog dict shape, returns ``[]``.
    """
    from app.agent import tools_v2

    async def _fake_snapshot() -> list[dict[str, Any]]:
        return list(catalog)

    args: dict[str, Any] = {"consulta": consulta}
    if preco_min is not None:
        args["preco_min"] = preco_min
    if preco_max is not None:
        args["preco_max"] = preco_max
    if categoria is not None:
        args["categoria"] = categoria
    if ordenacao is not None:
        args["ordenacao"] = ordenacao

    # buscar_catalogo imports get_catalog_snapshot lazily from the source module,
    # so patching the source attribute is what actually takes effect.
    with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", _fake_snapshot):
        raw = await tools_v2.buscar_catalogo.ainvoke(args)
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def result_ids(results: list[dict[str, Any]]) -> list[str]:
    """Result ids as strings, in returned order (the tool stringifies ids)."""
    return [str(r.get("id")) for r in results]


def parse_brl(s: str) -> float:
    """'R$ 1.799,90' → 1799.90."""
    return float(s.replace("R$", "").strip().replace(".", "").replace(",", "."))


@asynccontextmanager
async def _dummy_session():
    yield MagicMock()


async def run_buscar_conhecimento(docs: list[dict[str, Any]], consulta: str = "x") -> Any:
    """Invoke ``buscar_conhecimento`` with the KB search patched to return ``docs``.

    Patches both the lazily-imported ``search_knowledge_base`` and the DB
    session factory so no Postgres is touched. Proves the KB→tool wiring.
    """
    from app.agent import tools_v2

    with patch("app.storage.db.get_session", lambda: _dummy_session()), patch(
        "app.rag.retriever.search_knowledge_base", new=AsyncMock(return_value=docs)
    ):
        raw = await tools_v2.buscar_conhecimento.ainvoke({"consulta": consulta})
    return json.loads(raw)


def live_prompt() -> str:
    """The system prompt the V2 agent actually sends (store identity rendered)."""
    from app.agent.supervisor import build_system_prompt

    return build_system_prompt()


def has_openai_key() -> bool:
    return bool(get_settings().openai_api_key)


# Skip marker for the LLM evals (mirrors tests/test_v2_llm_eval.py).
requires_key = pytest.mark.skipif(
    not has_openai_key(), reason="OPENAI_API_KEY not configured"
)
